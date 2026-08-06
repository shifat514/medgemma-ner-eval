"""MedGemma zero-shot medication-NER evaluation on MIMIC-IV discharge summaries.

Per note:
    text -> whitespace tokens (+ char spans)
         -> gold BIO from the label CSV's char offsets
         -> overlapping token windows (chunking.chunk_windows)
         -> per chunk: prompt MedGemma -> parse JSON -> align spans to THAT
            window's tokens only (align.align_entities_to_bio)
         -> merge chunk BIO into note-level spans, dedupe on (start, end, type)
         -> paint the final predicted BIO
    then seqeval over all notes.

Everything except the data loading and the chunk/merge step is the same code the
NCBI/BC5CDR evaluation uses: src/prompt.py's parser, src/align.py, src/scoring.py.

Usage:
    python -m src.evaluate_mimic --limit 5        # smoke test
    python -m src.evaluate_mimic --n 50
    python -m src.evaluate_mimic --n 100          # reuses the 50 already done

Incremental + resumable: every note's result is appended to
outputs/mimic/<tag>/per_note.jsonl as soon as it finishes, and a rerun skips
note_ids already present. The tag encodes seed + chunk settings but NOT the
sample size, so the 100-note run reuses the 50-note run's work (the 50-note set
is the first 50 of the same seeded draw). Free-Colab disconnects cost only the
note in flight.

The per_note.jsonl holds BIO tag arrays and counts — no note text. Only the
optional --dump-errors file quotes note text; both live under the gitignored
outputs/ directory.
"""

import argparse
import json
import os

from .align import align_entities_to_bio
from .chunking import (
    bio_to_spans,
    chunk_windows,
    merge_chunk_spans,
    spans_to_bio,
    token_spans_to_char,
)
from .datasets.mimic_meds import build_gold, load_sample
from .mimic_config import (
    CAP_MARGIN,
    CHUNK_WORDS,
    ENTITY_TYPES,
    GEN_CONFIG,
    LOAD_IN_4BIT,
    MODEL_ID,
    MODEL_NAME,
    OUTPUT_DIR,
    OVERLAP_WORDS,
    RESULTS_DIR,
    SAMPLE_FILE,
    SEED,
    TYPE_PRIORITY,
)
from .prompt_mimic import build_messages, parse_entities
from .report_mimic import write_report
from .scoring import score, write_results


def run_tag(seed, chunk_words, overlap_words, model_name):
    """Cache/run directory name. Excludes sample size so runs share work."""
    safe_model = model_name.replace("/", "_")
    return f"{safe_model}_seed{seed}_cw{chunk_words}_ov{overlap_words}"


def _norm_key(text, etype):
    return (" ".join(text.lower().split()), etype)


def make_oracle_run_fn(text, gold_char_spans):
    """A stand-in for the model that returns the exact gold strings per chunk.

    Used by ``--oracle`` to measure the pipeline's structural ceiling: how well
    a *perfect* extractor could score given string-matching alignment, chunking,
    and whitespace tokenization. Whatever it loses is a property of the harness,
    not of MedGemma, and precision loss here is the multi-occurrence expansion
    made concrete.

    The window's char range is passed in explicitly rather than recovered by
    searching for the chunk text, which would mislocate on repeated boilerplate.
    """
    def run(pipe, chunk_text, char_lo=0, char_hi=None):
        hi = len(text) if char_hi is None else char_hi
        ents = [
            {"text": text[s:e], "type": t}
            for s, e, t in gold_char_spans
            if s >= char_lo and e <= hi
        ]
        return json.dumps({"entities": ents})
    run.wants_char_range = True
    return run


def predict_note(pipe, gold, text, chunk_words=CHUNK_WORDS,
                 overlap_words=OVERLAP_WORDS, run_fn=None, count_fn=None,
                 gen_config=None):
    """Predict BIO for one note. Returns ``(pred_bio, stats, pred_char_spans)``.

    `run_fn(pipe, chunk_text) -> reply` and `count_fn(pipe, reply) -> int|None`
    are injectable for testing. Any inference/parse failure on a chunk degrades
    that chunk to no entities rather than killing the note or the run.

    Stats collected for honest reporting:
      n_chunks, n_chunk_failures, n_cap_hits (generation hit max_new_tokens),
      n_pred_mentions (entities the model emitted), n_pred_unique (distinct
      normalized text+type), n_aligned_spans (spans align produced, pre-dedupe),
      n_unmatched (emitted spans that matched no tokens in their window),
      n_overlap_duplicates (spans removed by the overlap dedupe),
      n_pred_spans (final, after dedupe), n_pred_dropped_overlap.
    """
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

    tokens = gold["tokens"]
    char_spans = gold["char_spans"]
    windows = chunk_windows(len(tokens), chunk_words, overlap_words)

    st = {
        "n_chunks": len(windows), "n_chunk_failures": 0, "n_cap_hits": 0,
        "n_pred_mentions": 0, "n_pred_unique": 0, "n_pred_unique_chunksum": 0,
        "n_aligned_spans": 0, "n_unmatched": 0, "n_overlap_duplicates": 0,
        "n_pred_spans": 0, "n_pred_dropped_overlap": 0, "tokens_covered": 0,
    }

    chunk_results = []
    unique_keys = set()
    covered = set()

    for start, end in windows:
        covered.update(range(start, end))
        # Feed the original substring, not " ".join(tokens): discharge summaries
        # are heavily list-formatted and the line structure carries meaning.
        char_lo, char_hi = char_spans[start][0], char_spans[end - 1][1]
        chunk_text = text[char_lo:char_hi]

        try:
            if getattr(run_fn, "wants_char_range", False):
                reply = run_fn(pipe, chunk_text, char_lo=char_lo, char_hi=char_hi)
            else:
                reply = run_fn(pipe, chunk_text)
            entities = parse_entities(reply)
        except Exception as e:  # noqa: BLE001 - one bad chunk must not kill the run
            print(f"[warn] chunk {start}-{end} failed on {gold['note_id']}: {e}")
            st["n_chunk_failures"] += 1
            chunk_results.append((start, ["O"] * (end - start)))
            continue

        if cap and isinstance(reply, str):
            n_gen = count_fn(pipe, reply)
            if n_gen is not None and n_gen >= cap - CAP_MARGIN:
                st["n_cap_hits"] += 1

        st["n_pred_mentions"] += len(entities)
        chunk_keys = {_norm_key(t, ty) for t, ty in entities}
        unique_keys |= chunk_keys
        st["n_pred_unique_chunksum"] += len(chunk_keys)

        chunk_bio = align_entities_to_bio(tokens[start:end], entities)
        st["n_aligned_spans"] += len(bio_to_spans(chunk_bio))

        # Emitted (text, type) pairs that match nothing in this window: the model
        # paraphrased, hallucinated, or the span straddled the window edge.
        # Probed one at a time on purpose — the question is "is this string in the
        # text at all", not "did it survive first-come-first-served claiming".
        for norm_text, etype in chunk_keys:
            probe = align_entities_to_bio(tokens[start:end], [(norm_text, etype)])
            if not bio_to_spans(probe):
                st["n_unmatched"] += 1

        chunk_results.append((start, chunk_bio))

    st["n_pred_unique"] = len(unique_keys)
    st["tokens_covered"] = len(covered)

    merged, duplicates = merge_chunk_spans(chunk_results)
    st["n_overlap_duplicates"] = duplicates

    pred_bio, dropped = spans_to_bio(len(tokens), merged, priority=TYPE_PRIORITY)
    st["n_pred_spans"] = len(merged) - len(dropped)
    st["n_pred_dropped_overlap"] = len(dropped)

    pred_char_spans = token_spans_to_char(merged, char_spans)
    return pred_bio, st, pred_char_spans


def _load_done(path):
    """Existing per-note results, keyed by note_id (for --resume)."""
    done = {}
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
                continue  # a partial final line from a killed run
            if "note_id" in rec:
                done[rec["note_id"]] = rec
    return done


def _dump_errors(path, note_id, text, gold_char, pred_char):
    """Append per-example FP/FN with the offending snippets.

    CONTAINS NOTE TEXT. `path` must be under the gitignored outputs/ directory.
    """
    gold_set, pred_set = set(map(tuple, gold_char)), set(map(tuple, pred_char))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "note_id": note_id,
            "false_negatives": [
                {"start": s, "end": e, "type": t, "text": text[s:e]}
                for s, e, t in sorted(gold_set - pred_set)
            ],
            "false_positives": [
                {"start": s, "end": e, "type": t, "text": text[s:e]}
                for s, e, t in sorted(pred_set - gold_set)
            ],
        }) + "\n")


def run_eval(n=None, limit=None, sample_file=None, chunk_words=CHUNK_WORDS,
             overlap_words=OVERLAP_WORDS, seed=SEED, model_id=None,
             model_name=None, resume=True, dump_errors=False,
             results_dir=None, output_dir=None, load_model=True, oracle=False):
    model_id = model_id or MODEL_ID
    model_name = model_name or MODEL_NAME
    if oracle:
        # Distinct model_name so the oracle's run dir and output files never
        # collide with a real MedGemma run.
        model_name = f"{model_name}-ORACLE"
        load_model = False
    results_dir = results_dir or RESULTS_DIR
    output_dir = output_dir or OUTPUT_DIR

    records = load_sample(sample_file or SAMPLE_FILE)
    n_notes = limit if limit else (n or len(records))
    n_notes = min(n_notes, len(records))
    # The sample file is in seeded selection order, so the first N is a strict
    # subset of the first M for any N < M.
    records = records[:n_notes]
    label = f"smoke_{n_notes}" if limit else str(n_notes)
    if oracle:
        label = f"oracle_{label}"

    tag = run_tag(seed, chunk_words, overlap_words, model_name)
    run_dir = os.path.join(output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    per_note_path = os.path.join(run_dir, "per_note.jsonl")
    errors_path = os.path.join(run_dir, "errors.jsonl")

    done = _load_done(per_note_path) if resume else {}
    todo = [r for r in records if r["note_id"] not in done]

    print(f"sample:   {sample_file or SAMPLE_FILE}")
    print(f"notes:    {len(records)}  (cached {len(records) - len(todo)}, "
          f"to run {len(todo)})")
    print(f"chunking: {chunk_words} tokens / {overlap_words} overlap")
    print(f"model:    {model_id}  4bit={LOAD_IN_4BIT}  {GEN_CONFIG}")
    print(f"run dir:  {run_dir}")

    pipe = None
    if todo and load_model:
        from .model import load_medgemma
        print("loading model ...")
        pipe = load_medgemma(model_id, load_in_4bit=LOAD_IN_4BIT)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(todo, total=len(todo), desc="MedGemma MIMIC med-NER")
    except ImportError:
        iterator = todo

    for record in iterator:
        gold = build_gold(record)
        gold_char = token_spans_to_char(bio_to_spans(gold["bio"]), gold["char_spans"])

        run_fn = make_oracle_run_fn(record["text"], gold_char) if oracle else None
        pred_bio, st, pred_char = predict_note(
            pipe, gold, record["text"],
            chunk_words=chunk_words, overlap_words=overlap_words,
            run_fn=run_fn, count_fn=(lambda p, r: None) if oracle else None,
        )

        # PHI-free: BIO tag arrays and integer counts only.
        out = {
            "note_id": record["note_id"],
            "gold_bio": gold["bio"],
            "pred_bio": pred_bio,
            "n_chars": gold["n_chars"],
            "n_tokens": gold["n_tokens"],
            "n_gold_raw": gold["n_gold_raw"],
            "n_gold_spans": gold["n_gold_spans"],
            "n_gold_out_of_bounds": gold["n_gold_out_of_bounds"],
            "n_gold_dropped_overlap": gold["n_gold_dropped_overlap"],
            "n_gold_no_token": gold["n_gold_no_token"],
            "n_gold_boundary_snapped": gold["n_gold_boundary_snapped"],
            "n_type_conflicts": gold["n_type_conflicts"],
            "n_label_rows_raw": gold["n_label_rows_raw"],
            "n_unique_triples": gold["n_unique_triples"],
            **st,
        }
        with open(per_note_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        done[record["note_id"]] = out

        if dump_errors:
            _dump_errors(errors_path, record["note_id"], record["text"],
                         gold_char, pred_char)

    # Score in sample order over exactly the requested notes.
    ordered = [done[r["note_id"]] for r in records if r["note_id"] in done]
    missing = [r["note_id"] for r in records if r["note_id"] not in done]
    if missing:
        print(f"[warn] {len(missing)} notes have no result and are excluded")

    gold_bio = [r["gold_bio"] for r in ordered]
    pred_bio = [r["pred_bio"] for r in ordered]
    report = score(gold_bio, pred_bio)

    df, csv_path = write_results(
        report, model_name, results_dir=results_dir,
        csv_name=f"mimic_ner_{label}.csv",
        json_name=f"mimic_ner_{label}_full_report.json",
    )

    run_meta = {
        "sample_size": len(ordered),
        "requested": n_notes,
        "label": label,
        "oracle": oracle,
        "seed": seed,
        "model_id": model_id,
        "model_name": model_name,
        "load_in_4bit": LOAD_IN_4BIT,
        "gen_config": dict(GEN_CONFIG),
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "entity_types": list(ENTITY_TYPES),
        "notes_missing": missing,
        "run_dir": run_dir,
        "sample_file": sample_file or SAMPLE_FILE,
    }
    report_path = write_report(report, ordered, run_meta, results_dir=results_dir)

    print("\n" + df.to_string(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {report_path}")
    print(f"Per-note state (gitignored): {per_note_path}")
    if dump_errors:
        print(f"Error dump (gitignored, CONTAINS NOTE TEXT): {errors_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma zero-shot medication NER on MIMIC-IV discharge summaries"
    )
    parser.add_argument("--n", type=int, default=None,
                        help="sample size to score (e.g. 50 or 100)")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test: evaluate only the first N notes")
    parser.add_argument("--sample-file", default=None,
                        help=f"sample JSONL (default {SAMPLE_FILE})")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="seed used to build the sample (recorded in the report)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore cached per-note results and rerun every note")
    parser.add_argument("--dump-errors", action="store_true",
                        help="write per-example FP/FN to the gitignored run dir "
                             "(CONTAINS NOTE TEXT)")
    parser.add_argument("--oracle", action="store_true",
                        help="no model: feed the gold spans back through the "
                             "pipeline to measure its structural ceiling")
    args = parser.parse_args()

    run_eval(
        n=args.n, limit=args.limit, sample_file=args.sample_file,
        chunk_words=args.chunk_words, overlap_words=args.overlap_words,
        seed=args.seed, model_id=args.model, model_name=args.model_name,
        resume=not args.no_resume, dump_errors=args.dump_errors,
        oracle=args.oracle,
    )


if __name__ == "__main__":
    main()
