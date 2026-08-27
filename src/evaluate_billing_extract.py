"""Did MedGemma FIND the conditions it failed to code? A separate step.

Usage:
    python -m src.evaluate_billing_extract --ceiling   # no GPU, no model
    python -m src.evaluate_billing_extract             # 12 calls
    python -m src.evaluate_billing_extract --variant leakage_cut

THE QUESTION THIS ANSWERS, AND WHY IT IS NOT THE ONE evaluate_billing ASKS.
That module reports 0 of 16 billed codes recovered. On its own that number
cannot separate

    the model never found the influenza          -> a bigger model might help
    it found the influenza and coded it J11.9    -> a code book would help

and those point at completely different work. So this runs the EXTRACTION
prompt that already exists — `prompt_recall`'s `billable`, measured at 78 of 100
billed phrases on MDACE — and asks, per gold code, whether the condition was
surfaced at all.

DELIBERATELY A SEPARATE MODULE, NOT A FLAG ON THE OTHER ONE. Different prompt,
different unit of measurement (phrases, not codes), different answer key
(`billing_evidence`, not the DX lines). Folding it in would put two metrics
behind one command and invite quoting the friendlier one.

THE CEILING IS PART OF THE RESULT, AND IT CORRECTS A NUMBER ALREADY REPORTED.
`--ceiling` needs no GPU and says how many gold codes are evidenced in each
variant's input at all:

    full             16 of 16
    assessment_cut   16 of 16
    leakage_cut      12 of 16

Four codes lose their only evidence when the Problem List goes — D18.00,
L20.9, and note 55688's J30.2 and R06.2. They are chronic problems carried
forward on the chart, which is how a coder bills them. So `leakage_cut`'s
0.0000 recall was quoted against 16 when 12 was the reachable maximum. It does
not rescue the result, because the model also missed all twelve that WERE
reachable, but the denominator was wrong and the report should say so.

REAL PATIENT DATA. Extracted phrases are copied out of the notes. per_note.jsonl
holds counts and matched gold codes only; phrases.jsonl holds the phrases
themselves. Both gitignored.
"""

import argparse
import json
import os

from .billing_config import (
    GEN_CONFIG,
    LOAD_IN_4BIT,
    MODEL_ID,
    MODEL_NAME,
    OUTPUT_DIR,
    SAMPLE_FILE,
    VARIANTS,
    variant_label,
)
from .billing_evidence import ceiling, evidence_for, matched_by, present_in
from .datasets.billing import load_sample
from .evaluate_billing import gen_fingerprint
from .prompt_recall import build_messages, parse_findings, prompt_fingerprint

# The extraction prompt under test. `billable` beat `scoped` on hallucinated
# spans 9.22% -> 1.02% at identical row recall (see prompt_recall), so it is the
# one that gets carried here rather than a new prompt written for this file.
EXTRACT_VARIANT = "billable"


def phrases_of(findings):
    """Both fields of every finding — the note's wording and the standard name.

    `billable` returns {"span", "name"} per finding precisely so a finding can
    match either the note's phrasing or the catalogue's. Collapsing to one here
    would throw away half of what the prompt was built to produce.
    """
    out = []
    for f in findings:
        for key in ("span", "name"):
            val = f.get(key)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
    return out


def score_note(record, variant, phrases):
    """Which of this note's gold codes the extraction surfaced."""
    text = record["variants"][variant]
    rows = []
    for code in record["gold_codes"]:
        entry = evidence_for(record["note_id"], code)
        hits = matched_by(phrases, record["note_id"], code)
        rows.append({
            "code": code,
            "condition": entry["condition"] if entry else "(not catalogued)",
            "reachable": bool(entry) and bool(
                present_in(text, record["note_id"], code)
            ),
            "found": bool(hits),
            "matched_phrases": hits[:4],
        })
    return {
        "note_id": record["note_id"],
        "variant": variant,
        "n_gold": len(rows),
        "n_reachable": sum(1 for r in rows if r["reachable"]),
        "n_found": sum(1 for r in rows if r["found"]),
        "n_phrases": len({p.lower() for p in phrases}),
        "rows": rows,
    }


def _append(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_done(path):
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                done[(rec["note_id"], rec["variant"])] = rec
    return done


def print_ceiling(records):
    print("STRUCTURAL CEILING — how many gold codes are evidenced in the input")
    print("=" * 74)
    print("No model involved. This is the denominator recall should be quoted")
    print("against, and it is NOT 16 for every variant.\n")
    for variant in VARIANTS:
        c = ceiling(records, variant)
        print(f"  {variant:<16} {c['n_reachable']:>2} of 16   "
              f"{variant_label(variant)}")
        for note_id, code in c["unreachable"]:
            entry = evidence_for(note_id, code)
            cond = entry["condition"] if entry else "?"
            print(f"       unreachable: {note_id} {code:<9} {cond}")
    print()


def run(sample_file=SAMPLE_FILE, variants=None, model_id=MODEL_ID,
        model_name=MODEL_NAME, output_dir=OUTPUT_DIR, resume=True,
        dump_phrases=False):
    records = load_sample(sample_file)
    variants = list(variants or VARIANTS)

    gen = dict(GEN_CONFIG)
    tag = (f"extract_{model_name}_tok{gen['max_new_tokens']}"
           f"_g{gen_fingerprint(gen)}_p{prompt_fingerprint(EXTRACT_VARIANT)}")
    run_dir = os.path.join(output_dir, tag)
    per_note_path = os.path.join(run_dir, "per_note.jsonl")
    phrases_path = os.path.join(run_dir, "phrases.jsonl")

    done = _load_done(per_note_path) if resume else {}
    todo = [(r, v) for v in variants for r in records
            if (r["note_id"], v) not in done]

    print(f"sample     {sample_file}  ({len(records)} notes)")
    print(f"prompt     {EXTRACT_VARIANT} / {prompt_fingerprint(EXTRACT_VARIANT)}")
    print(f"run dir    {run_dir}")
    print(f"cached {len(done)}, to run {len(todo)}\n")

    pipe = None
    if todo:
        from .model import load_medgemma
        print(f"loading {model_id} (4bit={LOAD_IN_4BIT}) ...")
        pipe = load_medgemma(model_id)
        print("loaded.\n")

    from .model import run_messages

    for i, (rec, variant) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] note {rec['note_id']}  {variant} ...",
              flush=True)
        reply = run_messages(
            pipe, build_messages(rec["variants"][variant], EXTRACT_VARIANT),
            default_gen=gen,
        )
        phrases = phrases_of(parse_findings(reply))
        scored = score_note(rec, variant, phrases)
        _append(per_note_path, scored)
        done[(rec["note_id"], variant)] = scored
        if dump_phrases:
            _append(phrases_path, {"note_id": rec["note_id"],
                                   "variant": variant, "phrases": phrases})
        print(f"        {scored['n_phrases']} phrases   "
              f"found {scored['n_found']} of {scored['n_gold']} gold conditions "
              f"(reachable {scored['n_reachable']})")

    by_variant = {
        v: [done[(r["note_id"], v)] for r in records
            if (r["note_id"], v) in done]
        for v in variants
    }
    return records, by_variant


def print_report(records, by_variant):
    print("\n" + "=" * 74)
    print("DID IT FIND THE CONDITION? (extraction, not coding)")
    print("=" * 74)
    print(f"{'variant':<17} {'phrases':>8} {'reachable':>10} {'found':>7} "
          f"{'find rate':>10}")
    print("-" * 74)
    for variant, rows in by_variant.items():
        if not rows:
            continue
        n_ph = sum(r["n_phrases"] for r in rows)
        n_re = sum(r["n_reachable"] for r in rows)
        n_fd = sum(r["n_found"] for r in rows)
        rate = n_fd / n_re if n_re else 0.0
        print(f"{variant:<17} {n_ph:>8} {n_re:>10} {n_fd:>7} {rate:>10.4f}")
    print("\n'find rate' is out of REACHABLE codes, not out of 16. A code whose")
    print("only evidence was removed cannot be found and is not counted against")
    print("the model.\n")

    for variant, rows in by_variant.items():
        if not rows:
            continue
        print("-" * 74)
        print(f"{variant}  —  {variant_label(variant)}")
        print(f"  {'note':<8} {'code':<9} {'found':<6} {'reach':<6} condition")
        for row in rows:
            for r in row["rows"]:
                mark = "YES" if r["found"] else "no"
                reach = "yes" if r["reachable"] else "NO"
                print(f"  {row['note_id']:<8} {r['code']:<9} {mark:<6} "
                      f"{reach:<6} {r['condition'][:38]}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Did MedGemma find the conditions it failed to code?"
    )
    parser.add_argument("--sample-file", default=SAMPLE_FILE)
    parser.add_argument("--variant", action="append", choices=list(VARIANTS))
    parser.add_argument("--ceiling", action="store_true",
                        help="print the structural ceiling and exit; no GPU")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dump-phrases", action="store_true",
                        help="keep the extracted phrases (they quote the note)")
    args = parser.parse_args()

    records = load_sample(args.sample_file)
    print_ceiling(records)
    if args.ceiling:
        return

    records, by_variant = run(
        sample_file=args.sample_file, variants=args.variant,
        model_id=args.model, model_name=args.model_name,
        output_dir=args.output_dir, resume=not args.no_resume,
        dump_phrases=args.dump_phrases,
    )
    print_report(records, by_variant)


if __name__ == "__main__":
    main()
