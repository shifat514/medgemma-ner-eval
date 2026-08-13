"""Failure analysis: where the false positives come from, why the misses missed.

WHY THIS EXISTS. Every suggestion for improving the benchmark so far has been
inference from totals. "51 findings per note against 4 billed" says there is too
much volume; it does not say where the volume comes from. "22 rows missed" says
recall is incomplete; it does not say whether the model never saw them, said
something close that the matcher rejected, or produced them in a reply that ran
out of output space. Those have opposite fixes.

TWO OUTPUTS, DELIBERATELY SPLIT, the same rule as everywhere else in this
benchmark:

    failures.json       counts only. Safe to open, paste and share.
    failures_detail.jsonl  the actual phrases. CONTAINS NOTE TEXT.

FALSE POSITIVES ARE ATTRIBUTED TO A NOTE SECTION. Locate the model's span in the
note, walk back to the nearest section header, and count. If labs and medication
sections dominate then dropping those sections is the fix and it can be sized
before anyone spends GPU time on it. If the false positives are spread evenly
across the note then section filtering is not the lever and the second-pass
filter is.

MISSES ARE BUCKETED BY CAUSE:

    truncated      the gold phrase sits in a window that ran out of output
                   space. The model may well have found it and been cut off.
    not_extracted  no finding resembles any accepted form at all. The model
                   never produced it.
    near_miss      something scores above zero but under every threshold.
                   Matching is the problem, not extraction.
    rejected_by_l5 matched, then thrown out by the judge.

`truncated` is checked first because it is not really a model failure, and
`not_extracted` last of the causes because it is the residual — the bucket that
means the model genuinely did not see it.
"""

import argparse
import json
import os
import re

from .datasets.mdace_recall import load, normalize_term
from .recall_config import (
    COSINE_MIN,
    DICE_MIN,
    LEVELS,
    OUTPUT_DIR,
    RATIO_MIN,
    SAMPLE_100_FILE,
)
from .recall_matching import char_ratio, dice, string_rule, token_contains

# MIMIC-III notes label sections with a capitalised phrase then a colon at the
# start of a line. Deliberately generic rather than a hardcoded list of MIMIC
# header names, so the same code works if the production notes differ.
_HEADER = re.compile(r"^[ \t]*([A-Z][A-Za-z /\-]{3,40}):", re.MULTILINE)

# Anything above this against some accepted form means the model said something
# related. Below it, nothing it produced is even in the neighbourhood.
_NEAR_MISS_FLOOR = 0.34


def sections(text):
    """``[(char_start, char_end, header)]`` covering the whole note.

    Text before the first header is attributed to ``"(no section)"`` rather than
    dropped, so the counts always add up to the number of findings.
    """
    marks = [(m.start(), m.group(1).strip()) for m in _HEADER.finditer(text)]
    if not marks or marks[0][0] > 0:
        marks.insert(0, (0, "(no section)"))
    out = []
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((start, end, name))
    return out


def locate(text, span):
    """Character offset of `span` in `text`, or None.

    Tries the verbatim string first, then a whitespace-tolerant search, because
    the model reproduces a phrase that ran across a line break with the newline
    collapsed to a space.
    """
    if not span:
        return None
    at = text.find(span)
    if at != -1:
        return at
    loose = re.compile(r"\s+".join(re.escape(w) for w in span.split()),
                       re.IGNORECASE)
    m = loose.search(text)
    return m.start() if m else None


def section_of(text, span, index=None):
    """Which section a model span came from. ``None`` if it is not in the note."""
    at = locate(text, span)
    if at is None:
        return None
    for start, end, name in (index or sections(text)):
        if start <= at < end:
            return name
    return "(no section)"


def best_similarity(findings, forms):
    """Highest similarity any finding reaches against any accepted form.

    Used only to separate "the model said something related" from "the model
    said nothing related". Not a matching rule and not thresholded the same way.
    """
    best = 0.0
    for finding in findings:
        for cand in {normalize_term(finding.get("span")),
                     normalize_term(finding.get("name"))} - {""}:
            for form in forms:
                if string_rule(cand, form) is not None or token_contains(cand, form):
                    return 1.0
                best = max(best, dice(cand, form), char_ratio(cand, form))
    return best


def analyse(records, preds, per_note, rejected, levels=LEVELS,
            embedder=None, dice_min=DICE_MIN, ratio_min=RATIO_MIN,
            cosine_min=COSINE_MIN):
    """Returns ``(counts, detail)``. `counts` is PHI-free."""
    top = levels[-1]
    caps = {r.get("note_id"): r.get("cap_hit_windows") or []
            for r in per_note}
    has_cap_data = any(caps.values()) or all(
        "cap_hit_windows" in r for r in per_note)

    fp_by_section, fp_total = {}, 0
    fp_not_in_note = 0
    miss_causes = {"truncated": 0, "rejected_by_l5": 0, "near_miss": 0,
                   "not_extracted": 0, "unknown_truncation": 0}
    detail = []

    from .recall_scoring import match_notes

    # Matched WITH the judge's rejections applied, so this analyses the
    # adjudicated result. Without that, a row whose only match L5 threw out
    # still looks like a hit and the rejected_by_l5 bucket can never fire.
    # The embedder and the thresholds MUST be passed through. Without them this
    # silently scored L1-L3 at default thresholds while the report scored L1-L4
    # at loosened ones, so the two disagreed by 6 misses and 50 false positives
    # and there was no way to tell which was right.
    ladders = match_notes(records, preds, source=None, levels=levels,
                          rejected=rejected, embedder=embedder,
                          dice_min=dice_min, ratio_min=ratio_min,
                          cosine_min=cosine_min)

    for record in records:
        note_id = record["note_id"]
        findings = preds.get(note_id)
        if findings is None or note_id not in ladders:
            continue
        text = record["text"]
        index = sections(text)
        matched_idx = set(ladders[note_id][top]["pairs"])
        matched_forms = ladders[note_id][top]["matched_forms"]

        # --- false positives, attributed to a section --------------------
        for i, finding in enumerate(findings):
            if i in matched_idx:
                continue
            fp_total += 1
            name = section_of(text, finding.get("span"), index)
            if name is None:
                fp_not_in_note += 1
                name = "(not in the note)"
            fp_by_section[name] = fp_by_section.get(name, 0) + 1
            detail.append({"kind": "false_positive", "note_id": note_id,
                           "section": name, "span": finding.get("span", ""),
                           "name": finding.get("name", "")})

        # --- misses, bucketed by cause ----------------------------------
        cap_windows = caps.get(note_id) or []
        for entry in record["rows"]:
            accept = set(entry["accept"])
            if accept & matched_forms:
                continue
            cause = _miss_cause(entry, accept, findings, rejected, note_id,
                                text, cap_windows, has_cap_data)
            miss_causes[cause] += 1
            detail.append({"kind": "miss", "note_id": note_id, "cause": cause,
                           "code": f"{entry['code_system']} {entry['code']}",
                           "evidence": entry["evidence_text"],
                           "accepted_forms": sorted(accept)})

    counts = {
        "level": top,
        "false_positives": fp_total,
        "fp_not_in_note": fp_not_in_note,
        "fp_by_section": dict(sorted(fp_by_section.items(),
                                     key=lambda kv: -kv[1])),
        "misses": sum(miss_causes.values()),
        "miss_causes": miss_causes,
        "cap_window_data_available": bool(has_cap_data),
    }
    return counts, detail


def _miss_cause(entry, accept, findings, rejected, note_id, text,
                cap_windows, has_cap_data):
    """Why this gold row was not recalled. Order matters, see the docstring."""
    # 1. Was it inside a window that ran out of output space? Checked first
    #    because a cut-off reply is not the model failing to see the phrase.
    if cap_windows:
        at = entry["evidence_char_begin"]
        word_at = len(text[:at].split())
        if any(lo <= word_at < hi for lo, hi in cap_windows):
            return "truncated"
    elif not has_cap_data:
        return "unknown_truncation"

    # 2. Did the judge throw away a match this row had?
    if any(r[0] == note_id and r[3] in accept for r in rejected):
        return "rejected_by_l5"

    # 3. Did the model say anything in the neighbourhood?
    if best_similarity(findings, accept) >= _NEAR_MISS_FLOOR:
        return "near_miss"

    # 4. Residual: the model never produced anything like it.
    return "not_extracted"


def _read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def report(counts):
    """Console summary. Counts only."""
    print("\n--- where the false positives come from " + "-" * 27)
    print(f"  {counts['false_positives']:,} false positives at {counts['level']}")
    top = list(counts["fp_by_section"].items())[:12]
    width = max((len(k) for k, _ in top), default=10)
    for name, n in top:
        share = 100.0 * n / max(counts["false_positives"], 1)
        print(f"  {n:6,}  ({share:5.1f}%)  {name:{width}}")
    other = counts["false_positives"] - sum(n for _k, n in top)
    if other:
        print(f"  {other:6,}            (all other sections)")

    print("\n--- why the misses missed " + "-" * 40)
    labels = {
        "truncated": "the reply ran out of output space",
        "rejected_by_l5": "matched, then the judge rejected it",
        "near_miss": "model said something close, matcher refused",
        "not_extracted": "model never produced anything like it",
        "unknown_truncation": "cannot tell - run predates cap-window logging",
    }
    for cause, n in sorted(counts["miss_causes"].items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {n:4}  {labels.get(cause, cause)}")
    if not counts["cap_window_data_available"]:
        print("\n  [warn] This run did not record which windows hit the token")
        print("  cap, so truncation cannot be separated out. The next run will.")


def run(run_dir=None, sample_file=None, levels=LEVELS, embed=True,
        dice_min=DICE_MIN, ratio_min=RATIO_MIN, cosine_min=COSINE_MIN):
    from .recall_judge import load_rejected
    from .recall_matching import Embedder

    run_dir = run_dir or _latest_run_dir()
    records, _stats = load(sample_file or SAMPLE_100_FILE)

    findings_raw = _read_jsonl(os.path.join(run_dir, "findings.jsonl"))
    if not findings_raw:
        print(f"No findings.jsonl in {run_dir}. Run the benchmark first.")
        return {}

    from .recall_scoring import dedupe_findings

    preds = {r["note_id"]: dedupe_findings(r.get("findings") or [])
             for r in findings_raw if "note_id" in r}
    per_note = _read_jsonl(os.path.join(run_dir, "per_note.jsonl"))
    rejected = load_rejected(run_dir)

    print(f"run dir:  {run_dir}")
    print(f"notes:    {len(preds)}")
    print(f"findings: {sum(len(v) for v in preds.values()):,}")
    print(f"rejected by L5: {len(rejected)}")

    embedder = Embedder.load() if embed else None
    if embedder is None:
        levels = tuple(lv for lv in levels if lv != "L4")
        print("[info] no embedding backend, scoring L1-L3 only. Pass the same "
              "levels the report used or the two will not reconcile.")
    print(f"levels:   {', '.join(levels)}  "
          f"(dice {dice_min}, ratio {ratio_min}, cosine {cosine_min})")

    counts, detail = analyse(records, preds, per_note, rejected, levels,
                             embedder=embedder, dice_min=dice_min,
                             ratio_min=ratio_min, cosine_min=cosine_min)
    report(counts)

    counts_path = os.path.join(run_dir, "failures.json")
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, sort_keys=True)
    detail_path = os.path.join(run_dir, "failures_detail.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in detail)

    print(f"\nWrote {counts_path}  (counts only, shareable)")
    print(f"Wrote {detail_path}  (CONTAINS NOTE TEXT)")
    return counts


def _latest_run_dir():
    import glob

    dirs = [d for d in glob.glob(os.path.join(OUTPUT_DIR, "*"))
            if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(
            f"no run directories under {OUTPUT_DIR}. Run "
            "`python -m src.evaluate_recall` first.")
    return max(dirs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Failure analysis of a finished run (no GPU)")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--sample-file", default=None)
    parser.add_argument("--dice-min", type=float, default=DICE_MIN)
    parser.add_argument("--ratio-min", type=float, default=RATIO_MIN)
    parser.add_argument("--cosine-min", type=float, default=COSINE_MIN)
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()
    run(run_dir=args.run_dir, sample_file=args.sample_file,
        embed=not args.no_embed, dice_min=args.dice_min,
        ratio_min=args.ratio_min, cosine_min=args.cosine_min)


if __name__ == "__main__":
    main()
