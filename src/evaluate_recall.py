"""MedGemma zero-shot recall benchmark on MDACE billing evidence.

The question is one question: how much of the billed evidence does MedGemma-4B
recover from a note, zero-shot, on `8-07-mdace-ner-eval_sample_100-LOCAL.jsonl`
and nothing else.

Per note:
    text -> overlapping 400-word windows (chunking.chunk_windows)
         -> per chunk: prompt MedGemma -> parse JSON -> {span, name} findings
         -> pool and dedupe across the note's chunks
    then run the matching ladder against the note's accept-sets.

Usage:
    python -m src.evaluate_recall --oracle       # no GPU: harness check, ~10s
    python -m src.evaluate_recall --smoke 3      # the longest notes first
    python -m src.evaluate_recall                # all 24 notes / 82 chunks

VERIFY THE HARNESS BEFORE SPENDING GPU TIME. `--oracle` feeds the gold
accept-sets back through chunking, parsing, normalization and matching as if
they were model output. Every source must read 1.0000 at L1 and zero false
positives. Anything less is a bug in the harness, not a result. This caught real
bugs twice on the term-NER branch and costs ten seconds.

Incremental + resumable: each note's findings and counts are appended as soon as
the note finishes, and a rerun skips note_ids already present.

KNOWN TRAP, and it has bitten before. Re-running after a prompt edit prints
`cached N, to run 0` and reports the old prompt's numbers in about a second with
no error. The prompt hash is part of the run-directory name so this cannot
happen silently any more — but if a run finishes suspiciously fast, check the
`run dir:` line for the hash before believing anything.

FILE SPLIT IS DELIBERATE:
    per_note.jsonl     integer counts only. No note-derived text at all.
    findings.jsonl     the per-note {span, name} lists. These ARE phrases copied
                       out of patient notes.
    new_pairs_L*.jsonl what each ladder level newly accepted, for audit and L5.
Everything lives under the gitignored output dir; splitting them means the
counts file can be opened and shared without a second thought.
"""

import argparse
import json
import os

from .chunking import chunk_windows, tokenize_with_spans
from .datasets.mdace_recall import load, normalize_term
from .prompt_recall import (
    DEFAULT_VARIANT,
    VARIANTS,
    build_messages,
    parse_findings_diag,
    prompt_fingerprint,
)
from .recall_config import (
    CAP_MARGIN,
    CHUNK_WORDS,
    COSINE_MIN,
    DICE_MIN,
    GEN_CONFIG,
    LEVELS,
    LOAD_IN_4BIT,
    MODEL_ID,
    MODEL_NAME,
    OUTPUT_DIR,
    OVERLAP_WORDS,
    RATIO_MIN,
    RESULTS_DIR,
    SAMPLE_100_FILE,
)
from .recall_matching import Embedder
from .recall_scoring import dedupe_findings, score_run, trailing_repeat_len
from .report_recall import write_report


def run_tag(chunk_words, overlap_words, model_name, max_new_tokens,
            prompt_id=None):
    """Run directory name. Excludes note count so runs share cached work.

    Every input that changes what the model produces belongs here, because the
    resume cache is keyed on this directory. The matching thresholds are NOT
    here on purpose: they change scoring, not generation, so re-scoring an
    existing run with different thresholds should reuse its inference.
    """
    safe = model_name.replace("/", "_")
    tag = f"{safe}_cw{chunk_words}_ov{overlap_words}_mnt{max_new_tokens}"
    return f"{tag}_p{prompt_id}" if prompt_id else tag


def make_oracle_run_fn(record):
    """Stand-in for the model that returns this note's accept-sets verbatim.

    Measures the harness ceiling with no GPU. One finding per accepted form —
    the evidence phrase as the span, the form as the name — emitted in whichever
    chunk that row's evidence phrase falls in. So ALL THREE sources must come
    out at 1.0000, not only the evidence column: a weaker oracle emitting just
    the evidence phrase would leave the description and SNOMED lines
    unverified, which is exactly where a per-source bug would hide.

    One finding per form rather than one per (row, form) pair, so the combined
    matching has an exact 1:1 solution and false positives must come out at 0
    too. The per-source lines still show false positives, and that is correct
    rather than a defect — see the per-source trap in report_recall.

    The span for a form is fixed ONCE for the whole note, not chosen per chunk.
    Nine codes are evidenced twice, so a form reachable from two rows would
    otherwise be emitted with one row's phrasing in one window and the other's
    in the next, and the two would survive deduping as two findings for one gold
    form — sixteen false positives that say nothing about the harness.
    """
    span_for = {}
    for form in record["forms"]:
        for entry in record["rows"]:
            if form in entry["accept"] and entry["evidence_text"]:
                span_for[form] = entry["evidence_text"]
                break

    def run(pipe, chunk_text, char_lo=0, char_hi=None):
        hi = len(record["text"]) if char_hi is None else char_hi
        window = record["text"][char_lo:hi]
        return json.dumps({"findings": [
            {"span": span, "name": form}
            for form, span in sorted(span_for.items()) if span in window
        ]})
    run.wants_char_range = True
    return run


def predict_note(pipe, record, chunk_words=CHUNK_WORDS,
                 overlap_words=OVERLAP_WORDS, run_fn=None, count_fn=None,
                 gen_config=None, diag_sink=None, reply_sink=None,
                 variant=None):
    """Predict the finding list for one note. Returns ``(findings, stats)``.

    Any inference or parse failure degrades that chunk to zero findings rather
    than killing the note or the run.
    """
    text = record["text"]
    if run_fn is None:
        from .model import run_messages

        def run_fn(p, chunk_text):
            return run_messages(p, build_messages(chunk_text, variant),
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
        "n_chunks_zero_findings": 0, "n_mentions": 0,
        "n_chunks_salvaged": 0, "n_items_salvaged": 0,
        "n_span_only": 0, "n_name_only": 0, "n_bare_string": 0,
        # Repeats WITHIN one reply, which is the repetition-loop signal. Repeats
        # ACROSS a note's chunks are expected — windows overlap by design — so
        # the pooled duplicate rate cannot tell the two apart and a run that is
        # looping looks the same as one that is merely overlapping. Both are
        # counts, so neither leaves the diagnostics PHI-free.
        "n_items_dup_in_chunk": 0,
        # Chunks cut off at the token cap that still parsed cleanly. The object
        # format degrades gracefully — the scanner recovers the complete
        # {"span":..,"name":..} objects written before the cut — so `shape`
        # stays "json" and n_chunks_salvaged never fires. Without this counter
        # the diagnostics read as though nothing was lost, when in fact
        # everything after each cut is gone.
        "n_chunks_cut_but_parsed": 0, "n_cap_hits_while_repeating": 0,
    }

    collected = []
    for start, end in windows:
        char_lo, char_hi = char_spans[start][0], char_spans[end - 1][1]
        chunk_text = text[char_lo:char_hi]

        try:
            if getattr(run_fn, "wants_char_range", False):
                reply = run_fn(pipe, chunk_text, char_lo=char_lo, char_hi=char_hi)
            else:
                reply = run_fn(pipe, chunk_text)
            parsed, pdiag = parse_findings_diag(reply)
        except Exception as e:  # noqa: BLE001 - one bad chunk must not kill the run
            print(f"[warn] chunk {start}-{end} failed on {record['note_id']}: {e}")
            st["n_chunk_failures"] += 1
            st["n_chunks_zero_findings"] += 1
            continue

        st["n_items_seen"] += pdiag["n_items"]
        st["n_items_kept"] += pdiag["n_kept"]
        st["n_items_no_text"] += pdiag["n_no_text"]
        st["n_span_only"] += pdiag["n_span_only"]
        st["n_name_only"] += pdiag["n_name_only"]
        st["n_bare_string"] += pdiag["n_bare_string"]
        if pdiag["shape"] in ("no-json", "empty-reply", "no-item-list"):
            st["n_chunks_no_json"] += 1
        if pdiag.get("n_salvaged"):
            # Recovered from a reply cut off at the token cap. Counted apart
            # from clean parses so the report never implies the chunk was fine.
            st["n_chunks_salvaged"] += 1
            st["n_items_salvaged"] += pdiag["n_salvaged"]
        if pdiag["empty_list"]:
            st["n_chunks_empty_list"] += 1
        if not parsed:
            st["n_chunks_zero_findings"] += 1

        # Did this one reply say the same thing twice? Counted per chunk, before
        # the note-level pool, so window overlap cannot be mistaken for a loop.
        seen_here = {(normalize_term(f.get("span")), normalize_term(f.get("name")))
                     for f in parsed}
        st["n_items_dup_in_chunk"] += len(parsed) - len(seen_here)
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
                if parsed and not pdiag.get("n_salvaged"):
                    st["n_chunks_cut_but_parsed"] += 1
                # Two very different truncations. Cut while still producing new
                # findings means content was genuinely lost and recall is
                # understated. Cut while replaying its own list means nothing
                # was lost — pooling would have collapsed those duplicates
                # anyway. Counting them together made every cap hit look like
                # damage.
                if trailing_repeat_len(parsed):
                    st["n_cap_hits_while_repeating"] += 1

        st["n_mentions"] += len(parsed)
        collected.extend(parsed)

    findings = dedupe_findings(collected)
    st["n_findings"] = len(findings)
    return findings, st


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


def _load_findings(path):
    """``{note_id: [findings]}`` from a previous run."""
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
                out[rec["note_id"]] = dedupe_findings(rec.get("findings") or [])
    return out


# Counters added after the first smoke run. A note cached before they existed
# has no value for them, and summing a missing key as 0 would report "no
# repetition" for a run that was never measured for it — the same shape of lie
# as a stale prompt replaying old numbers.
_LATE_COUNTERS = ("n_items_dup_in_chunk", "n_chunks_cut_but_parsed",
                  "n_cap_hits_while_repeating")


def _print_diagnostics(per_note, diag_sink):
    """Explain under-extraction instead of leaving a silent zero."""
    def tot(key):
        return sum(r.get(key, 0) or 0 for r in per_note)

    chunks = tot("n_chunks")
    if not chunks:
        return
    stale = sum(1 for r in per_note
                if any(k not in r for k in _LATE_COUNTERS))
    print("\n--- extraction diagnostics " + "-" * 40)
    if stale:
        print(f"  [warn] {stale} of {len(per_note)} notes were cached before the "
              "repetition and\n         truncation counters existed and report 0 "
              "for them regardless of\n         what actually happened. Re-run "
              "those notes with --no-resume to measure.")
    print(f"  chunks                              {chunks:,}")
    print(f"    returned no usable JSON           {tot('n_chunks_no_json'):,}"
          f"  ({100.0 * tot('n_chunks_no_json') / chunks:.1f}%)")
    print(f"    returned an explicitly empty list {tot('n_chunks_empty_list'):,}")
    print(f"    yielded zero findings             {tot('n_chunks_zero_findings'):,}"
          f"  ({100.0 * tot('n_chunks_zero_findings') / chunks:.1f}%)")
    caps = tot("n_cap_hits")
    print(f"    generation hit max_new_tokens     {caps:,}"
          f"  ({100.0 * caps / chunks:.1f}%)"
          + ("   <- recall is UNDERSTATED" if caps else ""))
    looping = tot("n_cap_hits_while_repeating")
    print(f"      cut while REPLAYING its own list {looping:,}"
          + ("   <- nothing lost; pooling drops those" if caps else ""))
    print(f"      cut while producing NEW findings {caps - looping:,}"
          + ("   <- this is the part that understates recall" if caps else ""))
    print(f"    inference/parse exception         {tot('n_chunk_failures'):,}")
    print(f"    nothing parsed, prefix salvaged   {tot('n_chunks_salvaged'):,}"
          f"  ({tot('n_items_salvaged'):,} findings recovered)")
    print(f"  JSON items emitted                  {tot('n_items_seen'):,}")
    print(f"    dropped: no usable text           {tot('n_items_no_text'):,}")
    print(f"    repeated within their own reply   {tot('n_items_dup_in_chunk'):,}"
          "   <- repetition loop, not window overlap")
    print(f"    span but no standard name         {tot('n_span_only'):,}")
    print(f"    standard name but no span         {tot('n_name_only'):,}"
          "   <- these cannot be checked against the note")
    print(f"  findings after pooling per note     {tot('n_findings'):,}")
    print(f"  distinct findings per note          "
          f"{tot('n_findings') / max(len(per_note), 1):.1f}")

    shapes = diag_sink.get("shapes") or {}
    if shapes:
        print("  reply shapes: " + ", ".join(
            f"{k}={v}" for k, v in sorted(shapes.items(), key=lambda kv: -kv[1])))
    print("-" * 67)


def run_eval(limit=None, smoke=None, sample_file=None, chunk_words=CHUNK_WORDS,
             overlap_words=OVERLAP_WORDS, model_id=None, model_name=None,
             resume=True, results_dir=None, output_dir=None, load_model=True,
             oracle=False, dump_replies=False, max_new_tokens=None,
             dice_min=DICE_MIN, ratio_min=RATIO_MIN, cosine_min=COSINE_MIN,
             embed=True, embed_model=None, score_only=False,
             prompt_variant=None):
    model_id = model_id or MODEL_ID
    model_name = model_name or MODEL_NAME
    if oracle:
        model_name = f"{model_name}-ORACLE"
        load_model = False
    results_dir = results_dir or RESULTS_DIR
    output_dir = output_dir or OUTPUT_DIR

    gen_config = {"max_new_tokens": max_new_tokens} if max_new_tokens else None
    cap = max_new_tokens or GEN_CONFIG["max_new_tokens"]

    records, data_stats = load(sample_file or SAMPLE_100_FILE,
                               chunk_words, overlap_words)
    if smoke:
        # Records are ordered longest-note-first, so the smoke run exercises the
        # multi-chunk path and the worst case for truncation first. Taking the
        # head of a file ordered any other way would run short single-chunk
        # notes and never touch where the failures live.
        records = records[:smoke]
        label = f"smoke_{len(records)}"
    elif limit:
        records = records[:limit]
        label = f"first{len(records)}"
    else:
        label = str(len(records))
    if oracle:
        label = f"oracle_{label}"
    # The prompt variant is part of the label, not just the run directory.
    # Without it both arms of an A/B write the same results/ filename and the
    # second silently overwrites the first -- which then makes the comparison
    # tool pick up whatever unrelated run happens to be next-most-recent.
    #
    # It carries the FULL configuration, not just the variant, and the long
    # filename is the point. The first version keyed on note count alone, so
    # both arms of the prompt A/B wrote one file and the second destroyed the
    # first. Keying on the variant fixed that axis and left every other one --
    # chunk geometry, token cap, an edit to a variant's own text -- with exactly
    # the same hole. This mirrors the run tag, so two runs that differ in
    # anything the model can see cannot share a results file.
    label = (f"{label}_{prompt_variant or DEFAULT_VARIANT}"
             f"_cw{chunk_words}_ov{overlap_words}_mnt{cap}"
             f"_p{prompt_fingerprint(prompt_variant)}")

    tag = run_tag(chunk_words, overlap_words, model_name, cap,
                  prompt_id=prompt_fingerprint(prompt_variant))
    run_dir = os.path.join(output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    per_note_path = os.path.join(run_dir, "per_note.jsonl")
    findings_path = os.path.join(run_dir, "findings.jsonl")
    replies_path = os.path.join(run_dir, "raw_replies.jsonl")
    diag_path = os.path.join(run_dir, "diagnostics.json")

    diag_sink = {"shapes": {}, "types": {}}
    done = _load_done(per_note_path) if resume else set()
    cached = _load_findings(findings_path) if resume else {}
    todo = [] if score_only else [r for r in records if r["note_id"] not in done]

    print(f"input:    {sample_file or SAMPLE_100_FILE}")
    print(f"gold:     {data_stats['n_rows']} rows, {data_stats['n_codes']} "
          f"distinct codes, {data_stats['forms_combined']} accepted forms "
          f"on {data_stats['n_notes']} notes")
    print(f"notes:    {len(records)}  (cached {len(records) - len(todo)}, "
          f"to run {len(todo)})")
    print(f"chunks:   {sum(r.get('n_chunks', 0) for r in todo)} to run "
          f"of {sum(r.get('n_chunks', 0) for r in records)}")
    print(f"chunking: {chunk_words} words / {overlap_words} overlap")
    print(f"model:    {model_id}  4bit={LOAD_IN_4BIT}  max_new_tokens={cap}")
    print(f"prompt:   {prompt_variant or DEFAULT_VARIANT} "
          f"{prompt_fingerprint(prompt_variant)}  "
          "(a prompt edit starts a fresh cache)")
    print(f"run dir:  {run_dir}")

    pipe = None
    if todo and load_model:
        from .model import load_medgemma
        print("loading model ...")
        pipe = load_medgemma(model_id, load_in_4bit=LOAD_IN_4BIT)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(todo, total=len(todo), desc="MedGemma MDACE recall")
    except ImportError:
        iterator = todo

    for record in iterator:
        reply_sink = [] if dump_replies else None
        run_fn = make_oracle_run_fn(record) if oracle else None
        findings, st = predict_note(
            pipe, record, chunk_words=chunk_words, overlap_words=overlap_words,
            run_fn=run_fn, count_fn=(lambda p, r: None) if oracle else None,
            gen_config=gen_config, diag_sink=diag_sink, reply_sink=reply_sink,
            variant=prompt_variant,
        )

        # Findings first, then counts: the counts file is the resume marker, so
        # its presence always implies the findings landed.
        with open(findings_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "note_id": record["note_id"],
                "findings": [{"span": x["span"], "name": x["name"]}
                             for x in findings],
            }) + "\n")

        out = {"note_id": record["note_id"],
               "n_gold_rows": len(record["rows"]),
               "n_gold_forms": len(record["forms"]),
               **st}
        with open(per_note_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")

        cached[record["note_id"]] = findings
        done.add(record["note_id"])

        if reply_sink:
            # Raw model output. QUOTES NOTE TEXT (the prompt echoes the chunk).
            with open(replies_path, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(rec) + "\n" for rec in reply_sink)

    preds = {nid: f for nid, f in cached.items()}
    missing = [r["note_id"] for r in records if r["note_id"] not in preds]
    if missing:
        print(f"[warn] {len(missing)} notes have no result and are excluded")

    per_note_stats = []
    if os.path.exists(per_note_path):
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

    # L4's backend is optional by design: L1-L3 must stay runnable on a
    # CPU-only laptop, and the report says which levels are present rather than
    # printing a silent zero for a level that never ran.
    embedder = Embedder.load(embed_model) if embed else None
    levels = tuple(LEVELS) if embedder is not None else tuple(
        lv for lv in LEVELS if lv != "L4")
    if embedder is None and embed:
        print("[info] scoring L1-L3 only.")

    result = score_run(records, preds, embedder=embedder, dice_min=dice_min,
                       ratio_min=ratio_min, cosine_min=cosine_min,
                       levels=levels)

    # If L5 has run over this run directory, score a second ladder with the
    # rejected pairs removed. Reported ALONGSIDE the unadjudicated figure, never
    # in place of it: the gap between them is how much of the ladder's gain was
    # real, and that gap is the point of having a ladder at all.
    from .recall_judge import load_rejected

    rejected = load_rejected(run_dir)
    adjudicated = None
    if rejected:
        print(f"[info] L5 verdicts found: {len(rejected)} pairs rejected. "
              "Scoring an adjudicated ladder alongside.")
        adjudicated = score_run(records, preds, embedder=embedder,
                                dice_min=dice_min, ratio_min=ratio_min,
                                cosine_min=cosine_min, levels=levels,
                                rejected=rejected)
        result["adjudicated"] = adjudicated["by_source"]

    for level, pairs in result["new_pairs"].items():
        path = os.path.join(run_dir, f"new_pairs_{level}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(pair) + "\n" for pair in pairs)

    run_meta = {
        "label": label,
        "oracle": oracle,
        "model_id": model_id,
        "model_name": model_name,
        "load_in_4bit": LOAD_IN_4BIT,
        "max_new_tokens": cap,
        "prompt_id": prompt_fingerprint(prompt_variant),
        "prompt_variant": prompt_variant or DEFAULT_VARIANT,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "n_notes_scored": len(records) - len(missing),
        "notes_missing": missing,
        # Names, not paths. Both of these are printed in full at run start,
        # where they are useful for finding the files; the committed artifact is
        # public and has no business carrying somebody's directory layout — and
        # on Colab the output dir is a Drive path. The run tag still identifies
        # model, chunk geometry, token cap and prompt hash, and the filename
        # still identifies the input, which is all the artifact needs.
        "run_tag": tag,
        "input_file": os.path.basename(sample_file or SAMPLE_100_FILE),
        "levels": list(levels),
        "thresholds": {"dice_min": dice_min, "ratio_min": ratio_min,
                       "cosine_min": cosine_min,
                       "embed_model": embedder.name if embedder else None},
        "n_chunks": sum(r.get("n_chunks", 0) or 0 for r in per_note_stats),
        "n_cap_hits": sum(r.get("n_cap_hits", 0) or 0 for r in per_note_stats),
        "n_rejected_pairs": len(rejected),
        "n_chunks_salvaged": sum(
            r.get("n_chunks_salvaged", 0) or 0 for r in per_note_stats),
    }
    report_path, json_path = write_report(result, run_meta, data_stats,
                                          results_dir=results_dir, label=label)

    _print_summary(result, run_meta)
    if adjudicated:
        _print_adjudicated(result, adjudicated)
    print(f"\nWrote {report_path}")
    print(f"Wrote {json_path}")
    print(f"Per-note counts (gitignored, no note text): {per_note_path}")
    print(f"Extracted findings (gitignored, CONTAINS NOTE TEXT): {findings_path}")
    print(f"Newly-matched pairs per level (gitignored, CONTAINS NOTE TEXT): "
          f"{run_dir}/new_pairs_L*.jsonl")
    if dump_replies:
        print(f"Raw replies (gitignored, CONTAINS NOTE TEXT): {replies_path}")

    if oracle:
        _check_oracle(result, run_meta)
    return result


def _print_summary(result, run_meta):
    volume = result["volume"]
    print("\n--- recall by ladder level (combined accept-set) " + "-" * 18)
    print(f"  {'level':6} {'rows':>13} {'codes':>13} {'forms':>13} "
          f"{'FP':>7} {'FP rate':>8}")
    for level in result["levels"]:
        m = result["by_source"]["combined"][level]
        print(f"  {level:6} "
              f"{m['rows_matched']:4}/{m['rows_total']:<4} {m['row_recall']:.4f}  "
              f"{m['codes_matched']:4}/{m['codes_total']:<4} {m['code_recall']:.4f}  "
              f"{m['forms_matched']:4}/{m['forms_total']:<4} {m['form_recall']:.4f}  "
              f"{m['fp']:6}  {m['fp_rate']:.4f}")
    top = result["levels"][-1]
    if result.get("by_field"):
        print("\n--- which of the model's two fields matched " + "-" * 22)
        print(f"  {'matched on':34} " + " ".join(f"{lv:>8}" for lv in result["levels"]))
        for field, label in (("span", "span (copied from the note)"),
                             ("name", "name (standard clinical name)"),
                             ("both", "either field  <- the headline")):
            cells = " ".join(f"{result['by_field'][field][lv]['row_recall']:8.4f}"
                             for lv in result["levels"])
            print(f"  {label:34} {cells}")
        print("  Rows do not sum: one prediction claims at most one gold form.")

    # The three gold columns, side by side. This lived only in the markdown
    # report, and asking someone to open a file to see the comparison they asked
    # for is how a result goes unread.
    print("\n--- the three ways gold spells each answer " + "-" * 24)
    print(f"  {'gold column':34} {'answers':>8} "
          + " ".join(f"{lv:>8}" for lv in result["levels"]))
    labels = {"combined": "any of the three (ignore - see below)",
              "evidence": "what the note says",
              "description": "the official billing name",
              "snomed": "the medical dictionary name"}
    for source in ("combined", "evidence", "description", "snomed"):
        by_level = result["by_source"][source]
        total = by_level[result["levels"][0]]["forms_total"]
        cells = " ".join(f"{by_level[lv]['form_recall']:8.4f}"
                         for lv in result["levels"])
        print(f"  {labels[source]:34} {total:>8} {cells}")
    print("  Each row: of that column's phrasings, how many the model produced.")
    print("  Top row is NOT a score: it counts all 318 phrasings, but a gold")
    print("  answer only needs ONE of its ~4 to count as found.")

    m = result["by_source"]["combined"][top]
    buckets = m.get("fp_buckets") or {}
    if buckets:
        print(f"\n--- what the {top} false positives ARE " + "-" * 27)
        total = max(m["fp"], 1)
        print(f"  in the note, nothing billed for it  "
              f"{buckets['in_note_unbilled']:6}  "
              f"({100 * buckets['in_note_unbilled'] / total:5.1f}%)"
              "   <- correct extraction, unbilled")
        print(f"  not in the note at all              "
              f"{buckets['not_in_note']:6}  "
              f"({100 * buckets['not_in_note'] / total:5.1f}%)"
              "   <- the only real model error")
        print(f"  no span, cannot be checked          "
              f"{buckets['no_span']:6}")
        print("  MDACE marks only codes that were BILLED, so most misses are")
        print("  correct findings nobody billed. Judge the model on line 2.")

    print(f"\n  findings per note   {volume['pred_per_note']:.1f}")
    print(f"  not in the note     {volume['n_not_in_note']}/"
          f"{volume['n_span_checked']} = {volume['not_in_note_rate']:.4f}"
          "   <- hallucination")
    print("\n  Recall is not quotable on its own: a model that lists every "
          "phrase\n  scores near 1.00. Quote it with the two lines above it.")


def _print_adjudicated(result, adjudicated):
    """The ladder with judge-rejected pairs removed, beside the raw one."""
    print("\n--- after L5 adjudication " + "-" * 41)
    print(f"  {'level':6} {'rows raw':>10} {'rows judged':>12} {'lost':>6}")
    for level in result["levels"]:
        raw = result["by_source"]["combined"][level]
        adj = adjudicated["by_source"]["combined"][level]
        print(f"  {level:6} {raw['row_recall']:10.4f} {adj['row_recall']:12.4f} "
              f"{raw['rows_matched'] - adj['rows_matched']:6}")
    print("\n  A rejected pair only demotes a row that had no other accepted")
    print("  form, so rows fall by less than pairs do. Quote both.")
    print("-" * 67)


def _check_oracle(result, run_meta):
    """The harness check, stated as a pass/fail rather than left to the reader."""
    bad = []
    for source, levels in result["by_source"].items():
        m = levels["L1"]
        for unit in ("form", "row", "code"):
            if m[f"{unit}s_total"] and m[f"{unit}_recall"] < 1.0:
                bad.append(f"{source}: L1 {unit} recall "
                           f"{m[f'{unit}_recall']:.4f} "
                           f"({m[f'{unit}s_matched']}/{m[f'{unit}s_total']})")
    # Only the COMBINED line can reach zero false positives. Per-source FP means
    # "matched nothing in that source", so a finding naming the catalogue
    # wording is an FP on the evidence-text line by construction.
    combined_fp = result["by_source"]["combined"]["L1"]["fp"]
    if combined_fp:
        bad.append(f"combined: {combined_fp} false positives at L1")
    print("\n--- harness check " + "-" * 49)
    if bad:
        print("  FAIL — a bug in chunking, normalization or matching, not a "
              "result:")
        for line in bad:
            print(f"    {line}")
    else:
        print("  PASS — every source reads 1.0000 at L1, and the combined "
              "matching has zero false positives.")
    print("-" * 67)


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma zero-shot recall benchmark on MDACE billing evidence"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N notes (longest first)")
    parser.add_argument("--smoke", type=int, default=None,
                        help="quick check on the N longest notes — the "
                             "multi-chunk path, where truncation and OOM live")
    parser.add_argument("--prompt", default=None, choices=sorted(VARIANTS),
                        help="which prompt to run. 'scoped' names the "
                             "categories to exclude; 'billable' replaces "
                             "them with one positive criterion. They hash "
                             "differently, so the runs cannot mix")
    parser.add_argument("--sample-file", default=None)
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help=f"override the generation cap "
                             f"(default {GEN_CONFIG['max_new_tokens']})")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dump-replies", action="store_true",
                        help="write every raw model reply to the gitignored run "
                             "dir (CONTAINS NOTE TEXT). Use on the smoke run")
    parser.add_argument("--oracle", action="store_true",
                        help="no model: feed the gold accept-sets back through "
                             "the pipeline. Every source must read 1.0000 at L1")
    parser.add_argument("--score-only", action="store_true",
                        help="re-score a finished run without calling the "
                             "model — for changing thresholds")
    parser.add_argument("--dice-min", type=float, default=DICE_MIN)
    parser.add_argument("--ratio-min", type=float, default=RATIO_MIN)
    parser.add_argument("--cosine-min", type=float, default=COSINE_MIN)
    parser.add_argument("--no-embed", action="store_true",
                        help="skip L4 even when the backend is installed")
    parser.add_argument("--embed-model", default=None,
                        help="sentence-transformers model id for L4. Must be a "
                             "BIOMEDICAL encoder; a general MiniLM does not "
                             "know HTN")
    args = parser.parse_args()

    run_eval(
        limit=args.limit, smoke=args.smoke, sample_file=args.sample_file,
        chunk_words=args.chunk_words, overlap_words=args.overlap_words,
        model_id=args.model, model_name=args.model_name,
        resume=not args.no_resume, oracle=args.oracle,
        dump_replies=args.dump_replies, max_new_tokens=args.max_new_tokens,
        dice_min=args.dice_min, ratio_min=args.ratio_min,
        cosine_min=args.cosine_min, embed=not args.no_embed,
        embed_model=args.embed_model, score_only=args.score_only,
        prompt_variant=args.prompt,
    )


if __name__ == "__main__":
    main()
