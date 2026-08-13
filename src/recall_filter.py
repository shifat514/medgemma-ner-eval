"""Second pass: ask per finding whether a coder would bill it, and drop the noes.

WHY A SECOND PASS AND NOT A BETTER PROMPT. Extraction and filtering are currently
one call, and the model is measurably good at the first and bad at the second: it
recovers 78 of 100 billed phrases while emitting 51 findings per note against
about 4 billed. Two prompt versions were built to fix that inside the extraction
call and neither moved volume at all. Asking "extract only billable findings" in
the middle of a long instruction is a different task from asking "is this one
phrase billable" about a single phrase, and the second is far easier.

The failure analysis is what ruled out the alternative. False positives are not
concentrated in droppable sections — the largest single section holds 8% of them,
half sit in a long tail, and the sections producing the most false positives are
the same ones holding the most gold. Cutting input cannot fix a model that
extracts everything medical-looking wherever it looks.

IS THIS CHEATING? No, and the distinction matters. Tuning the model to emit
SNOMED-shaped strings would be, because the three SNOMED terms this file ships
are an accident of the data. Billability is not an accident — it is the task. The
product produces bills, so a term nobody bills is not a correct output of the
system even when it names a real condition.

TWO THINGS THAT KEEP IT HONEST.

**The default question is bare.** `bare` asks only whether a coder would assign a
code. `guided` adds the categories the model was observed getting wrong —
medications, vital signs, blood products, bare anatomy. If `bare` works, the
model knows what is billable. If only `guided` works, that is a finding about the
model rather than a fix, and the report should say so. Do not iterate on the
wording: set it once, measure, and report what happened.

**Both numbers are always reported.** Filtering changes what is being
benchmarked, from MedGemma to MedGemma-plus-a-filter. Raw and filtered appear
side by side so the filter's contribution is never folded into the model's.

An unreadable answer KEEPS the finding, the same rule as L5. A model that failed
to answer has not said the finding is unbillable, and defaulting to drop would
let a parse failure look like a precision win.

WEAKNESS, STATED: the filter is MedGemma judging MedGemma. `--judge none` writes
the questions out for a human or a stronger model instead.
"""

import argparse
import json
import os

from .recall_config import (
    COSINE_MIN,
    DICE_MIN,
    GEN_CONFIG,
    LEVELS,
    LOAD_IN_4BIT,
    MODEL_ID,
    OUTPUT_DIR,
    RATIO_MIN,
    SAMPLE_100_FILE,
)
from .recall_judge import parse_verdict

_SYSTEM = (
    "You are an experienced medical coder. You decide whether a phrase from a "
    "hospital note is something you would assign a billing code to. You answer "
    "with one word."
)

_BARE = (
    "A phrase was extracted from a hospital note:\n"
    "  {phrase}\n"
    "\n"
    "Would you, as a medical coder, assign a billing code to this — an ICD-10 "
    "diagnosis code, or a CPT or ICD-10-PCS procedure code?\n"
    "\n"
    "Answer with exactly one word: YES or NO."
)

# `guided` names the categories the model was observed extracting wrongly on the
# 24-note run: medications and blood products, vital signs and labs, bare
# anatomy. It exists to tell "the model knows what is billable" apart from "we
# told it", which is a fact about the model worth having either way.
_GUIDED = (
    "A phrase was extracted from a hospital note:\n"
    "  {phrase}\n"
    "\n"
    "Would you, as a medical coder, assign a billing code to this — an ICD-10 "
    "diagnosis code, or a CPT or ICD-10-PCS procedure code?\n"
    "\n"
    "Answer NO if it is:\n"
    "  - a medication, dose, IV fluid, bowel prep or blood product\n"
    "  - a vital sign or a lab value\n"
    "  - a body part or location with no finding attached\n"
    "  - a device, tube, drain, line or piece of equipment\n"
    "  - normal, negative, or explicitly absent\n"
    "  - a statement about care rather than a finding\n"
    "\n"
    "Answer YES if a coder could look it up in a code book as a diagnosis, a "
    "procedure performed on the patient, or an injury.\n"
    "\n"
    "Answer with exactly one word: YES or NO."
)

VARIANTS = {"bare": _BARE, "guided": _GUIDED}

# Note sections that cannot contain a billable finding BY CATEGORY, not because
# they happened to hold no gold in these 24 notes. That distinction is the whole
# point: "had no gold in our sample" measured against the same sample shows a
# zero recall cost by construction, which is not evidence of anything.
#
# Radiology is deliberately NOT here. FINDINGS, IMPRESSION and Imaging carry no
# gold in this sample and produced 114 false positives between them, but a
# radiology impression genuinely can name a billable diagnosis. Dropping them
# would be fitting to 24 notes.
SECTION_BLOCKLIST = (
    "medications on admission", "discharge medications", "medications",
    "discharge instructions", "followup instructions", "follow up instructions",
    "discharge disposition", "discharge condition", "order date", "disp",
    "activity", "allergies", "social history", "family history",
    "tablet refills", "capsule refills", "facility", "completed by",
)


def blocked_section(name):
    """True when `name` is a section that cannot hold a billable finding."""
    if not name:
        return False
    low = " ".join(name.lower().replace("-", " ").split())
    return any(low == b or low.startswith(b) for b in SECTION_BLOCKLIST)


def drop_blocked_sections(records, preds):
    """``(kept, n_dropped, by_section)`` — findings from non-diagnostic sections.

    Applied to findings already extracted, so it needs no GPU and no re-run. It
    does not save extraction time the way filtering the input would; it only
    removes false positives that were already produced.
    """
    from .recall_failures import section_of, sections

    by_id = {r["note_id"]: r for r in records}
    kept, dropped, by_section = {}, 0, {}
    for note_id, findings in preds.items():
        record = by_id.get(note_id)
        if record is None:
            kept[note_id] = findings
            continue
        index = sections(record["text"])
        out = []
        for finding in findings:
            name = section_of(record["text"], finding.get("span"), index)
            if blocked_section(name):
                dropped += 1
                by_section[name] = by_section.get(name, 0) + 1
                continue
            out.append(finding)
        kept[note_id] = out
    return kept, dropped, dict(sorted(by_section.items(), key=lambda kv: -kv[1]))
DEFAULT_VARIANT = os.environ.get("RECALL_FILTER_PROMPT", "bare")


def question(variant=None):
    name = variant or DEFAULT_VARIANT
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown filter variant {name!r}; expected one of {sorted(VARIANTS)}"
        ) from None


def phrase_of(finding):
    """How one finding is shown to the filter.

    Both fields when they differ, so the judgement sees the note's wording and
    the standard name together — `HTN (hypertension)` is easier to rule on than
    either alone.
    """
    span = (finding.get("span") or "").strip()
    name = (finding.get("name") or "").strip()
    if span and name and span.lower() != name.lower():
        return f"{span} ({name})"
    return span or name


def build_messages(finding, variant=None):
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user", "content": [{"type": "text",
                                      "text": question(variant).format(
                                          phrase=phrase_of(finding))}]},
    ]


def finding_key(note_id, finding):
    """Stable identity for resuming a partial pass."""
    return f"{note_id}|{finding.get('span', '')}|{finding.get('name', '')}"


def filter_findings(preds, run_fn, variant=None, resume=None):
    """``{key: verdict}`` for every finding. True = keep, False = drop."""
    resume = resume or {}
    verdicts = {}
    total = sum(len(v) for v in preds.values())
    done = 0
    for note_id, findings in preds.items():
        for finding in findings:
            key = finding_key(note_id, finding)
            done += 1
            if key in resume:
                verdicts[key] = resume[key]
                continue
            try:
                reply = run_fn(build_messages(finding, variant))
            except Exception as e:  # noqa: BLE001 - one bad call must not kill the pass
                print(f"[warn] filter failed on finding {done}/{total}: {e}")
                reply = None
            verdicts[key] = parse_verdict(reply)
            if done % 200 == 0:
                print(f"  {done}/{total}")
    return verdicts


def apply_filter(preds, verdicts):
    """``(kept, n_dropped, n_unreadable)``. An unreadable verdict keeps."""
    kept, dropped, unreadable = {}, 0, 0
    for note_id, findings in preds.items():
        out = []
        for finding in findings:
            verdict = verdicts.get(finding_key(note_id, finding))
            if verdict is None:
                unreadable += 1
            if verdict is False:
                dropped += 1
                continue
            out.append(finding)
        kept[note_id] = out
    return kept, dropped, unreadable


def compare(records, sides, rejected, embedder=None, levels=LEVELS,
            dice_min=DICE_MIN, ratio_min=RATIO_MIN, cosine_min=COSINE_MIN):
    """Score every named side. `sides` is ``[(name, preds), ...]`` in order."""
    from .recall_scoring import score_run

    out = {}
    for name, preds in sides:
        result = score_run(records, preds, embedder=embedder, levels=levels,
                           dice_min=dice_min, ratio_min=ratio_min,
                           cosine_min=cosine_min, rejected=rejected)
        top = result["levels"][-1]
        m = result["by_source"]["combined"][top]
        out[name] = {
            "level": top,
            "n_pred": m["n_pred"],
            "tp": m["tp"],
            "precision": m["precision"],
            "row_recall": m["row_recall"],
            "row_recall_ci": m["row_recall_ci"],
            "rows_matched": m["rows_matched"],
            "rows_total": m["rows_total"],
            "fp": m["fp"],
            "precision_ceiling": m["precision_ceiling"],
            "pred_per_note": result["volume"]["pred_per_note"],
        }
    return out


def wrongly_dropped(records, raw, kept, rejected, embedder=None, levels=LEVELS,
                    dice_min=DICE_MIN, ratio_min=RATIO_MIN,
                    cosine_min=COSINE_MIN):
    """The findings that DID match gold and were dropped anyway.

    The filter's mistakes, listed rather than counted. If they share a shape --
    all procedures, all status codes -- that is recoverable. Nobody knows until
    they are looked at, which is the whole reason for this function.
    """
    from .recall_scoring import match_notes

    ladders = match_notes(records, raw, source=None, levels=levels,
                          rejected=rejected, embedder=embedder,
                          dice_min=dice_min, ratio_min=ratio_min,
                          cosine_min=cosine_min)
    top = levels[-1]
    out = []
    for record in records:
        note_id = record["note_id"]
        if note_id not in ladders:
            continue
        survivors = {(f.get("span", ""), f.get("name", ""))
                     for f in kept.get(note_id, [])}
        for index, (form, rule, score) in ladders[note_id][top]["pairs"].items():
            finding = raw[note_id][index]
            if (finding.get("span", ""), finding.get("name", "")) in survivors:
                continue
            slot = record["forms"].get(form, {})
            out.append({
                "note_id": note_id, "span": finding.get("span", ""),
                "name": finding.get("name", ""), "gold_form": form,
                "rule": rule, "score": round(float(score), 4),
                "gold_sources": slot.get("sources", []),
                "gold_codes": slot.get("codes", []),
            })
    return out


def report(scored, meta):
    """Every operating point side by side. Both units kept separate."""
    names = list(scored)
    print(f"\n--- operating points ({meta['variant']}) " + "-" * 28)
    head = f"  {'':26}" + "".join(f"{n:>14}" for n in names)
    print(head)

    def row(label, key, fmt):
        cells = "".join(format(scored[n][key], fmt).rjust(14) for n in names)
        print(f"  {label:26}{cells}")

    row("findings", "n_pred", ",")
    row("per note", "pred_per_note", ".1f")
    row("precision", "precision", ".4f")
    row("best precision possible", "precision_ceiling", ".4f")
    row("row recall", "row_recall", ".4f")
    row("rows found", "rows_matched", "")

    print(f"\n  the billability filter dropped {meta['n_dropped']:,} findings "
          f"({meta['n_unreadable']:,} unreadable answers were kept)")
    # Findings and rows are different units and mixing them makes the numbers
    # look wrong: dropped == removed_fp + removed_tp, while rows lost is smaller
    # because a row survives if it kept any other matching form.
    print(f"    {meta['removed_fp']:,} were false positives  (correct to drop)")
    print(f"    {meta['removed_tp']:,} were real matches      (should have been kept)")
    if meta["n_dropped"]:
        print(f"  so {100.0 * meta['removed_fp'] / meta['n_dropped']:.1f}% of what "
              "it dropped was correct to drop")
    print(f"  those {meta['removed_tp']} lost matches cost {meta['lost_rows']} "
          "actual answers, because some\n  rows kept another matching form")

    if meta.get("n_section_dropped"):
        print(f"\n  section filtering then dropped {meta['n_section_dropped']:,} "
              "more, from sections that\n  cannot hold a billable finding by "
              "category:")
        for name, n in list(meta["section_dropped_by"].items())[:8]:
            print(f"    {n:5,}  {name}")
        print("  Radiology is deliberately NOT in that list: it carries no gold")
        print("  in these 24 notes, but a radiology impression genuinely can.")

    print("\n  Every column is reported because filtering changes what is being")
    print("  benchmarked: MedGemma, versus MedGemma plus each filter.")
    print("-" * 67)


def run(run_dir=None, sample_file=None, variant=None, judge="medgemma",
        model_id=None, embed=True, levels=LEVELS, dice_min=DICE_MIN,
        ratio_min=RATIO_MIN, cosine_min=COSINE_MIN, resume=True):
    from .datasets.mdace_recall import load
    from .recall_failures import _read_jsonl
    from .recall_judge import load_rejected
    from .recall_matching import Embedder
    from .recall_scoring import dedupe_findings

    run_dir = run_dir or _latest_run_dir()
    variant = variant or DEFAULT_VARIANT
    records, _stats = load(sample_file or SAMPLE_100_FILE)

    rows = _read_jsonl(os.path.join(run_dir, "findings.jsonl"))
    if not rows:
        print(f"No findings.jsonl in {run_dir}. Run the benchmark first.")
        return {}
    raw = {r["note_id"]: dedupe_findings(r.get("findings") or [])
           for r in rows if "note_id" in r}

    verdict_path = os.path.join(run_dir, f"filter_{variant}.jsonl")
    prior = {}
    if resume and os.path.exists(verdict_path):
        for rec in _read_jsonl(verdict_path):
            if rec.get("verdict") is not None:
                prior[rec["key"]] = bool(rec["verdict"])

    print(f"run dir:  {run_dir}")
    print(f"variant:  {variant}")
    print(f"findings: {sum(len(v) for v in raw.values()):,}")
    if prior:
        print(f"resumed:  {len(prior):,} verdicts already on disk")

    if judge == "none":
        def run_fn(messages):
            """No-op: writes the questions out with no verdicts."""
            return
    elif judge == "medgemma":
        from .model import load_medgemma, run_messages
        print("loading model ...")
        pipe = load_medgemma(model_id or MODEL_ID, load_in_4bit=LOAD_IN_4BIT)

        def run_fn(messages):
            return run_messages(pipe, messages,
                                gen_config={"max_new_tokens": 8},
                                default_gen=GEN_CONFIG)
    else:
        raise ValueError(f"unknown judge {judge!r}; use medgemma or none")

    verdicts = filter_findings(raw, run_fn, variant, prior)
    with open(verdict_path, "w", encoding="utf-8") as f:
        for note_id, findings in raw.items():
            for finding in findings:
                key = finding_key(note_id, finding)
                f.write(json.dumps({
                    "key": key, "note_id": note_id,
                    "span": finding.get("span", ""),
                    "name": finding.get("name", ""),
                    "verdict": verdicts.get(key), "variant": variant,
                }) + "\n")

    filtered, dropped, unreadable = apply_filter(raw, verdicts)
    stacked, sec_dropped, sec_by = drop_blocked_sections(records, filtered)

    embedder = Embedder.load() if embed else None
    if embedder is None:
        levels = tuple(lv for lv in levels if lv != "L4")
    rejected = load_rejected(run_dir)

    scored = compare(records, [("raw", raw), ("filtered", filtered),
                               ("+ sections", stacked)],
                     rejected, embedder, levels, dice_min, ratio_min,
                     cosine_min)
    meta = {
        "variant": variant, "judge": judge, "levels": list(levels),
        "n_dropped": dropped, "n_unreadable": unreadable,
        "removed_fp": scored["raw"]["fp"] - scored["filtered"]["fp"],
        "removed_tp": scored["raw"]["tp"] - scored["filtered"]["tp"],
        "lost_rows": (scored["raw"]["rows_matched"]
                      - scored["filtered"]["rows_matched"]),
        "n_section_dropped": sec_dropped,
        "section_dropped_by": sec_by,
    }
    report(scored, meta)

    mistakes = wrongly_dropped(records, raw, filtered, rejected, embedder,
                               levels, dice_min, ratio_min, cosine_min)
    mistakes_path = os.path.join(run_dir, f"filter_{variant}_mistakes.jsonl")
    with open(mistakes_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row_) + "\n" for row_ in mistakes)
    print(f"\nWrote {mistakes_path}  ({len(mistakes)} real matches the filter "
          "dropped, CONTAINS NOTE TEXT)")
    print("  Read it: if they share a shape, the lost recall is recoverable.")

    out = {**meta, "sides": scored}
    counts_path = os.path.join(run_dir, f"filter_{variant}_summary.json")
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {counts_path}  (counts only, shareable)")
    print(f"Wrote {verdict_path}  (CONTAINS NOTE TEXT)")
    return out


def _latest_run_dir():
    import glob

    dirs = [d for d in glob.glob(os.path.join(OUTPUT_DIR, "*"))
            if os.path.isdir(d)
            and os.path.exists(os.path.join(d, "findings.jsonl"))]
    if not dirs:
        raise FileNotFoundError(
            f"no finished run under {OUTPUT_DIR}. Run "
            "`python -m src.evaluate_recall` first.")
    return max(dirs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Second pass: drop findings a coder would not bill")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--sample-file", default=None)
    parser.add_argument("--variant", default=None, choices=sorted(VARIANTS),
                        help="'bare' asks only whether a coder would bill it. "
                             "'guided' also names the categories the model was "
                             "observed getting wrong")
    parser.add_argument("--judge", default="medgemma",
                        choices=("medgemma", "none"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dice-min", type=float, default=DICE_MIN)
    parser.add_argument("--ratio-min", type=float, default=RATIO_MIN)
    parser.add_argument("--cosine-min", type=float, default=COSINE_MIN)
    args = parser.parse_args()

    run(run_dir=args.run_dir, sample_file=args.sample_file,
        variant=args.variant, judge=args.judge, model_id=args.model,
        embed=not args.no_embed, resume=not args.no_resume,
        dice_min=args.dice_min, ratio_min=args.ratio_min,
        cosine_min=args.cosine_min)


if __name__ == "__main__":
    main()
