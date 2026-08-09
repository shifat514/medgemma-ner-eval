"""MedGemma zero-shot term-level billing-NER on MDACE (MIMIC-III) notes.

Per note:
    text -> overlapping 400-word windows (chunking.chunk_windows)
         -> per chunk: prompt MedGemma -> parse JSON -> collect term strings
         -> union the normalized terms across the note's chunks
    then compare that set against the gold term set.

Usage:
    python -m src.evaluate_mdace --oracle         # no GPU: harness sanity check
    python -m src.evaluate_mdace --smoke 5        # 15 chunks, ~12 min
    python -m src.evaluate_mdace --limit 50       # phase 1: 122 chunks, ~1.6h
    python -m src.evaluate_mdace                  # phase 2: +80 chunks, ~1.1h

TWO PHASES, BY DESIGN. The sample file is ordered stratified-first, so
``--limit 50`` runs exactly the 50-note stratified draw and yields the headline
view on its own. The unlimited run then adds the 23 remaining notes, reusing
phase 1's cache, and completes the ladder. A view is withheld until every note
it needs has run, so a partial pass cannot render a 1-note table beside a
50-note one.

ONE INFERENCE PASS, THREE SCORED VIEWS. The views differ only in which notes they
count and which answer key they use — neither touches the model — so re-running
inference per view would cost hours and buy nothing. Decoding is greedy
(do_sample=False), so a second pass with the same prompt is byte-identical
anyway; a second pass with a *different* prompt would make the views
non-comparable, which is worse than expensive.

Incremental + resumable: each note's terms and counts are appended as soon as the
note finishes, and a rerun skips note_ids already present. A ~2.7h free-Colab run
will disconnect, so this is not optional. Point MDACE_OUTPUT_DIR at a mounted
Drive path there.

FILE SPLIT IS DELIBERATE (see docs/mdace-term-ner-plan.md, decision 2):
    per_note.jsonl        integer counts only. No note-derived text at all.
    extracted_terms.jsonl the per-note term lists, for the downstream term->ICD
                          lookup. These ARE phrases copied out of patient notes.
Both live under the gitignored output dir; splitting them means the counts file
can be opened, pasted and shared without a second thought.
"""

import argparse
import json
import os

from .chunking import chunk_windows, tokenize_with_spans
from .datasets.mdace import load_sample, normalize_term
from .mdace_config import (
    CAP_MARGIN,
    CHUNK_WORDS,
    GEN_CONFIG,
    LOAD_IN_4BIT,
    MODEL_ID,
    MODEL_NAME,
    OUTPUT_DIR,
    OVERLAP_WORDS,
    RESULTS_DIR,
    SAMPLE_FILE,
    SEED,
)
from .prompt_mdace import build_messages, parse_terms_diag
from .report_mdace import write_report
from .term_scoring import score_by_chart_type, score_view


def run_tag(seed, chunk_words, overlap_words, model_name, max_new_tokens):
    """Run directory name. Excludes note count so runs share cached work."""
    safe = model_name.replace("/", "_")
    return f"{safe}_seed{seed}_cw{chunk_words}_ov{overlap_words}_mnt{max_new_tokens}"


def make_oracle_run_fn(record):
    """Stand-in for the model that returns this note's gold terms verbatim.

    Measures the harness ceiling with no GPU: recall must come out at 1.000 and
    precision at 1.000 for the full-gold views. Anything less is a bug in
    chunking, normalization, or parsing — not a model result.
    """
    def run(pipe, chunk_text, char_lo=0, char_hi=None):
        hi = len(record["text"]) if char_hi is None else char_hi
        window = record["text"][char_lo:hi]
        terms = [g["term"] for g in record.get("gold", ())
                 if g["term"] and g["term"] in window]
        return json.dumps({"terms": [{"text": t, "type": "Condition"}
                                     for t in terms]})
    run.wants_char_range = True
    return run


def predict_note(pipe, record, chunk_words=CHUNK_WORDS,
                 overlap_words=OVERLAP_WORDS, run_fn=None, count_fn=None,
                 gen_config=None, diag_sink=None, reply_sink=None):
    """Predict the term set for one note. Returns ``(terms, raw_terms, stats)``.

    `terms` is the set of normalized strings actually scored. `raw_terms` keeps
    one verbatim spelling per normalized term so the downstream lookup can use
    the model's own casing. Any inference or parse failure degrades that chunk to
    zero terms rather than killing the note or the run.
    """
    text = record["text"]
    if run_fn is None:
        from .model import run_messages

        def run_fn(p, chunk_text):
            return run_messages(p, build_messages(chunk_text),
                                gen_config=gen_config, default_gen=GEN_CONFIG)
    if count_fn is None:
        from .model import count_tokens
        count_fn = count_tokens

    gen = dict(GEN_CONFIG)
    if gen_config:
        gen.update(gen_config)
    cap = gen.get("max_new_tokens")

    tokens, char_spans = tokenize_with_spans(text)
    windows = chunk_windows(len(tokens), chunk_words, overlap_words)

    st = {
        "n_tokens": len(tokens), "n_chars": len(text),
        "n_chunks": len(windows), "n_chunk_failures": 0, "n_cap_hits": 0,
        "n_items_seen": 0, "n_items_kept": 0, "n_items_no_text": 0,
        "n_chunks_no_json": 0, "n_chunks_empty_list": 0,
        "n_chunks_zero_terms": 0, "n_mentions": 0, "n_empty_after_norm": 0,
    }

    terms, raw = set(), {}
    for start, end in windows:
        char_lo, char_hi = char_spans[start][0], char_spans[end - 1][1]
        chunk_text = text[char_lo:char_hi]

        try:
            if getattr(run_fn, "wants_char_range", False):
                reply = run_fn(pipe, chunk_text, char_lo=char_lo, char_hi=char_hi)
            else:
                reply = run_fn(pipe, chunk_text)
            parsed, pdiag = parse_terms_diag(reply)
        except Exception as e:  # noqa: BLE001 - one bad chunk must not kill the run
            print(f"[warn] chunk {start}-{end} failed on {record['note_id']}: {e}")
            st["n_chunk_failures"] += 1
            st["n_chunks_zero_terms"] += 1
            continue

        st["n_items_seen"] += pdiag["n_items"]
        st["n_items_kept"] += pdiag["n_kept"]
        st["n_items_no_text"] += pdiag["n_no_text"]
        if pdiag["shape"] in ("no-json", "empty-reply", "no-item-list"):
            st["n_chunks_no_json"] += 1
        if pdiag["empty_list"]:
            st["n_chunks_empty_list"] += 1
        if not parsed:
            st["n_chunks_zero_terms"] += 1
        if diag_sink is not None:
            diag_sink["shapes"][pdiag["shape"]] = \
                diag_sink["shapes"].get(pdiag["shape"], 0) + 1
            for k, v in pdiag["types"].items():
                diag_sink["types"][k] = diag_sink["types"].get(k, 0) + v
        if reply_sink is not None:
            reply_sink.append({
                "note_id": record["note_id"], "chunk": [start, end],
                "shape": pdiag["shape"], "n_kept": pdiag["n_kept"],
                "reply": reply,
            })

        if cap and isinstance(reply, str):
            n_gen = count_fn(pipe, reply)
            if n_gen is not None and n_gen >= cap - CAP_MARGIN:
                st["n_cap_hits"] += 1

        st["n_mentions"] += len(parsed)
        for raw_text, _etype in parsed:
            norm = normalize_term(raw_text)
            if not norm:
                st["n_empty_after_norm"] += 1
                continue
            terms.add(norm)
            raw.setdefault(norm, raw_text)

    st["n_pred_terms"] = len(terms)
    return terms, raw, st


def select_smoke_notes(records, n=5):
    """Pick `n` notes that span the risky dimensions, not the first `n` in file order.

    The sample file is ordered stratified-first, and the stratified draw starts
    with Profee — so a naive head-of-file smoke test runs five short
    single-chunk notes and never exercises a long one. That is precisely
    backwards: OOM, truncation and the multi-chunk union path all live in the
    long notes.

    So: the longest note overall, the longest note of the other chart type, then
    the shortest remaining ones. Both chart types are covered, the worst-case
    memory and generation-length path runs first, and the single-chunk path
    still gets exercised — for the same handful of model calls.
    """
    if n >= len(records):
        return list(records)

    by_size = sorted(records, key=lambda r: (-(r.get("n_chunks") or 1), r["note_id"]))
    picked, seen = [], set()

    def take(record):
        if record is not None and record["note_id"] not in seen:
            picked.append(record)
            seen.add(record["note_id"])

    take(by_size[0])
    other = picked[0]["chart_type"] if picked else None
    take(next((r for r in by_size if r.get("chart_type") != other), None))
    for record in reversed(by_size):
        if len(picked) >= n:
            break
        take(record)
    return picked[:n]


def _load_done(path):
    """note_ids already scored, from a previous (possibly killed) run."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # a partial final line from a killed run
            if "note_id" in rec:
                done.add(rec["note_id"])
    return done


def _load_terms(path):
    """``{note_id: [normalized terms]}`` from a previous run."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "note_id" in rec:
                out[rec["note_id"]] = rec.get("terms") or []
    return out


def _print_diagnostics(per_note, diag_sink):
    """Explain under-extraction instead of leaving a silent zero."""
    def tot(key):
        return sum(r.get(key, 0) or 0 for r in per_note)

    chunks = tot("n_chunks")
    if not chunks:
        return
    print("\n--- extraction diagnostics " + "-" * 40)
    print(f"  chunks                              {chunks:,}")
    print(f"    returned no usable JSON           {tot('n_chunks_no_json'):,}"
          f"  ({100.0 * tot('n_chunks_no_json') / chunks:.1f}%)")
    print(f"    returned an explicitly empty list {tot('n_chunks_empty_list'):,}")
    print(f"    yielded zero terms                {tot('n_chunks_zero_terms'):,}"
          f"  ({100.0 * tot('n_chunks_zero_terms') / chunks:.1f}%)")
    print(f"    generation hit max_new_tokens     {tot('n_cap_hits'):,}"
          "   <- must be 0; if not, raise MDACE_MAX_NEW_TOKENS back to 1024")
    print(f"    inference/parse exception         {tot('n_chunk_failures'):,}")
    print(f"  JSON items emitted                  {tot('n_items_seen'):,}")
    print(f"    dropped: no usable text           {tot('n_items_no_text'):,}")
    print(f"    dropped: empty after normalizing  {tot('n_empty_after_norm'):,}")
    print(f"  term mentions kept                  {tot('n_mentions'):,}")
    print(f"  distinct terms per note             "
          f"{tot('n_pred_terms') / max(len(per_note), 1):.1f}")

    shapes = diag_sink.get("shapes") or {}
    if shapes:
        print("  reply shapes: " + ", ".join(
            f"{k}={v}" for k, v in sorted(shapes.items(), key=lambda kv: -kv[1])))
    types = diag_sink.get("types") or {}
    if types:
        print("  type labels the model used (not scored):")
        for name, count in sorted(types.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {count:>6}  {name!r}")
    print("-" * 67)


def build_views(records, preds):
    """The three scored views. `preds` maps note_id -> set of normalized terms.

    `records` is the WHOLE sample, not only the notes that have predictions, so
    that a partially-complete run can tell "this view is finished" apart from
    "this view happens to have one note in it".

    A view is emitted only when every note it needs has been run. Phase 1
    (``--limit 50``) covers the stratified draw but touches just the single note
    it shares with the 100-row cut, so it emits B2 alone; rendering a 1-note A1
    beside a 50-note B2 would invite exactly the comparison the ladder exists to
    prevent. Phase 2 fills the rest in and all three appear.

    Each step changes one thing from the step above it:
        A1 -> B1  what the answer-key column is worth
        B1 -> B2  what balanced sampling changes
    """
    def split(predicate):
        wanted = [r for r in records if predicate(r)]
        ready = [(r, preds[r["note_id"]]) for r in wanted if r["note_id"] in preds]
        return wanted, ready

    shipped_all, shipped = split(lambda r: r.get("in_sample100"))
    strat_all, strat = split(lambda r: r.get("in_stratified"))

    views = {}
    if shipped and len(shipped) == len(shipped_all):
        n = len(shipped_all)
        views["A1"] = {
            "title": f"The 100-row sample cut ({n} notes), scored as originally asked",
            "detail": "answer key = gold_code_description from that file",
            "metrics": score_view(shipped, gold_key="descr"),
        }
        views["B1"] = {
            "title": f"The same {n} notes, corrected method",
            "detail": "answer key = every evidence phrase those notes carry",
            "metrics": score_view(shipped, gold_key="full"),
        }
    elif shipped:
        print(f"[info] views A1/B1 skipped: {len(shipped)}/{len(shipped_all)} "
              "of their notes have been run (finish phase 2 to get them)")

    if strat and len(strat) == len(strat_all):
        views["B2"] = {
            "title": f"Stratified {len(strat_all)} notes — THE HEADLINE",
            "detail": "25 Profee + 25 Inpatient, seed 13, full gold",
            "metrics": score_view(strat, gold_key="full"),
            "by_chart_type": score_by_chart_type(strat, gold_key="full"),
        }
    elif strat:
        print(f"[info] view B2 skipped: {len(strat)}/{len(strat_all)} of the "
              "stratified notes have been run")
    return views


def run_eval(limit=None, smoke=None, sample_file=None, chunk_words=CHUNK_WORDS,
             overlap_words=OVERLAP_WORDS, seed=SEED, model_id=None,
             model_name=None, resume=True, results_dir=None, output_dir=None,
             load_model=True, oracle=False, dump_replies=False,
             max_new_tokens=None):
    model_id = model_id or MODEL_ID
    model_name = model_name or MODEL_NAME
    if oracle:
        model_name = f"{model_name}-ORACLE"
        load_model = False
    results_dir = results_dir or RESULTS_DIR
    output_dir = output_dir or OUTPUT_DIR

    gen_config = {"max_new_tokens": max_new_tokens} if max_new_tokens else None
    cap = (max_new_tokens or GEN_CONFIG["max_new_tokens"])

    # The unlimited sample is kept so build_views can tell "this view is
    # complete" from "this view happens to have one note in it" — --limit 50
    # slices `records` down to the stratified draw, which also contains the
    # single note shared with the 100-row cut.
    all_records = load_sample(sample_file or SAMPLE_FILE)
    records = all_records
    if smoke:
        records = select_smoke_notes(records, smoke)
        label = f"smoke_{len(records)}"
    elif limit:
        # File order is stratified-first, so --limit 50 is exactly the 50-note
        # stratified sample: phase 1 of a two-phase run. The remaining notes are
        # picked up by a later unlimited run, which reuses this one's cache.
        records = records[:limit]
        label = f"first{len(records)}"
    else:
        label = str(len(records))
    if oracle:
        label = f"oracle_{label}"

    tag = run_tag(seed, chunk_words, overlap_words, model_name, cap)
    run_dir = os.path.join(output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    per_note_path = os.path.join(run_dir, "per_note.jsonl")
    terms_path = os.path.join(run_dir, "extracted_terms.jsonl")
    errors_path = os.path.join(run_dir, "errors.jsonl")
    replies_path = os.path.join(run_dir, "raw_replies.jsonl")
    diag_path = os.path.join(run_dir, "diagnostics.json")

    diag_sink = {"shapes": {}, "types": {}}
    done = _load_done(per_note_path) if resume else set()
    cached_terms = _load_terms(terms_path) if resume else {}
    todo = [r for r in records if r["note_id"] not in done]

    print(f"sample:   {sample_file or SAMPLE_FILE}")
    print(f"notes:    {len(records)}  (cached {len(records) - len(todo)}, "
          f"to run {len(todo)})")
    print(f"chunks:   {sum(r.get('n_chunks', 0) for r in todo)} to run")
    print(f"chunking: {chunk_words} words / {overlap_words} overlap")
    print(f"model:    {model_id}  4bit={LOAD_IN_4BIT}  max_new_tokens={cap}")
    print(f"run dir:  {run_dir}")

    pipe = None
    if todo and load_model:
        from .model import load_medgemma
        print("loading model ...")
        pipe = load_medgemma(model_id, load_in_4bit=LOAD_IN_4BIT)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(todo, total=len(todo), desc="MedGemma MDACE term-NER")
    except ImportError:
        iterator = todo

    for record in iterator:
        reply_sink = [] if dump_replies else None
        run_fn = make_oracle_run_fn(record) if oracle else None
        terms, raw, st = predict_note(
            pipe, record, chunk_words=chunk_words, overlap_words=overlap_words,
            run_fn=run_fn, count_fn=(lambda p, r: None) if oracle else None,
            gen_config=gen_config, diag_sink=diag_sink, reply_sink=reply_sink,
        )

        # Terms first, then counts: the counts file is the resume marker, so
        # writing it last means its presence always implies the terms landed.
        with open(terms_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "note_id": record["note_id"],
                "chart_type": record.get("chart_type"),
                "terms": sorted(terms),
                "terms_verbatim": [raw[t] for t in sorted(terms)],
            }) + "\n")

        out = {"note_id": record["note_id"],
               "chart_type": record.get("chart_type"),
               "n_gold_terms": len(record.get("gold_terms") or ()),
               **st}
        with open(per_note_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")

        cached_terms[record["note_id"]] = sorted(terms)
        done.add(record["note_id"])

        if reply_sink:
            # Raw model output. QUOTES NOTE TEXT (the prompt echoes the chunk).
            with open(replies_path, "a", encoding="utf-8") as f:
                for rec in reply_sink:
                    f.write(json.dumps(rec) + "\n")

    preds = {nid: set(t) for nid, t in cached_terms.items()}
    scored = [r for r in records if r["note_id"] in preds]
    missing = [r["note_id"] for r in records if r["note_id"] not in preds]
    if missing:
        print(f"[warn] {len(missing)} notes have no result and are excluded")

    per_note_stats = []
    with open(per_note_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                per_note_stats.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag_sink, f, indent=2, sort_keys=True)
    _print_diagnostics(per_note_stats, diag_sink)

    views = build_views(all_records, preds)
    _dump_errors(errors_path, scored, preds)

    run_meta = {
        "label": label,
        "oracle": oracle,
        "seed": seed,
        "model_id": model_id,
        "model_name": model_name,
        "load_in_4bit": LOAD_IN_4BIT,
        "max_new_tokens": cap,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "n_notes_scored": len(scored),
        "notes_missing": missing,
        "run_dir": run_dir,
        "sample_file": sample_file or SAMPLE_FILE,
        "n_cap_hits": sum(r.get("n_cap_hits", 0) or 0 for r in per_note_stats),
        "n_mixed_chart_type": sum(1 for r in scored if r.get("mixed_chart_type")),
    }
    report_path, json_path = write_report(views, run_meta, results_dir=results_dir,
                                          label=label)

    _print_summary(views)
    print(f"\nWrote {report_path}")
    print(f"Wrote {json_path}")
    print(f"Per-note counts (gitignored, no note text): {per_note_path}")
    print(f"Extracted terms for the ICD lookup (gitignored): {terms_path}")
    print(f"Error detail (gitignored, CONTAINS NOTE TEXT): {errors_path}")
    if dump_replies:
        print(f"Raw replies (gitignored, CONTAINS NOTE TEXT): {replies_path}")
    return views


def _dump_errors(path, records, preds):
    """Per-note FN and bucketed FP terms. CONTAINS NOTE-DERIVED TEXT.

    This is what makes the strict-vs-fuzzy matching decision evidence-based after
    the run instead of a guess before it: the false negatives are the actual
    phrases strict matching threw away.
    """
    from .term_scoring import classify_fp, note_gold, padded_note_norm

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            pred = preds.get(record["note_id"], set())
            gold = note_gold(record, "full")
            note_norm = padded_note_norm(record.get("text", ""))
            adm = set(record.get("admission_gold_terms") or ())
            fps = {}
            for term in sorted(pred - gold):
                fps.setdefault(classify_fp(term, note_norm, adm, gold),
                               []).append(term)
            f.write(json.dumps({
                "note_id": record["note_id"],
                "chart_type": record.get("chart_type"),
                "false_negatives": sorted(gold - pred),
                "false_positives": fps,
            }) + "\n")


def _print_summary(views):
    for key in ("A1", "B1", "B2"):
        view = views.get(key)
        if not view:
            continue
        m = view["metrics"]
        print(f"\n[{key}] {view['title']}")
        print(f"      {view['detail']}")
        print(f"      notes {m['n_notes']}  gold {m['n_gold']}  pred {m['n_pred']}"
              f"   P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}")
        if key == "B2":
            for ct, sub in (view.get("by_chart_type") or {}).items():
                print(f"        {ct:10} gold {sub['n_gold']:4}  pred {sub['n_pred']:5}"
                      f"   P {sub['precision']:.4f}  R {sub['recall']:.4f}"
                      f"  F1 {sub['f1']:.4f}   (ceiling P {sub['precision_ceiling']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma zero-shot term-level billing NER on MDACE notes"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N notes of the sample. "
                             "File order is stratified-first, so --limit 50 is "
                             "exactly the 50-note stratified sample (~1.6h) and "
                             "a later unlimited run adds the rest (~1.1h)")
    parser.add_argument("--smoke", type=int, default=None,
                        help="quick check on N notes chosen to span short and "
                             "long, both chart types — not the first N")
    parser.add_argument("--sample-file", default=None)
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help=f"override the generation cap "
                             f"(default {GEN_CONFIG['max_new_tokens']})")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dump-replies", action="store_true",
                        help="write every raw model reply to the gitignored run "
                             "dir (CONTAINS NOTE TEXT). Use on the smoke run")
    parser.add_argument("--oracle", action="store_true",
                        help="no model: feed the gold terms back through the "
                             "pipeline. Recall must come out 1.000")
    args = parser.parse_args()

    run_eval(
        limit=args.limit, smoke=args.smoke, sample_file=args.sample_file,
        chunk_words=args.chunk_words, overlap_words=args.overlap_words,
        seed=args.seed, model_id=args.model, model_name=args.model_name,
        resume=not args.no_resume, oracle=args.oracle,
        dump_replies=args.dump_replies, max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
