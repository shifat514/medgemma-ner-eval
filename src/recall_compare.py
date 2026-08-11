"""Compare two finished runs side by side. No GPU, no model, no re-scoring.

WHY THIS EXISTS. The prompt is the one part of this benchmark with no measured
justification for most of its content. Every other choice — the chunk geometry,
the token cap, the ladder thresholds, the L4 encoder — was set against a table
somebody can look at. The prompt's exclusion rules were set against "we saw the
model do this once", and they accumulate: medications, then vital signs, then
lab values, then blood products, then bare anatomy, with no principle saying
when the list is finished.

So the two variants get run against the same harness and compared here, and the
argument gets settled the way every other argument in this benchmark was.

READ THE VOLUME COLUMNS BEFORE THE RECALL COLUMN. A prompt that extracts more
will score higher recall almost regardless of quality, so a variant winning on
recall while emitting twice the findings has not won anything. The comparison
that matters is recall held at lower volume.
"""

import argparse
import glob
import json
import os

from .recall_config import COMBINED, OUTPUT_DIR, RESULTS_DIR


def load_metrics(path):
    """Read one committed metrics JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_metrics(results_dir=RESULTS_DIR, label=None):
    """Every recall metrics file, newest first."""
    pattern = f"mdace_recall_{label}_metrics.json" if label \
        else "mdace_recall_*_metrics.json"
    paths = glob.glob(os.path.join(results_dir, pattern))
    return sorted(paths, key=os.path.getmtime, reverse=True)


def _per_note_totals(run_dir):
    """Pooled extraction diagnostics for a run, straight from per_note.jsonl."""
    path = os.path.join(run_dir, "per_note.jsonl")
    totals, n = {}, 0
    if not os.path.exists(path):
        return totals, n
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            for key, value in rec.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
    return totals, n


def compare(paths, output_dir=OUTPUT_DIR):
    """Build the comparison table. `paths` are metrics JSON files."""
    rows = []
    for path in paths:
        data = load_metrics(path)
        run, levels = data["run"], data["levels"]
        top = levels[-1]
        combined = data["by_source"][COMBINED]
        volume = data["volume"]
        totals, _n = _per_note_totals(
            os.path.join(output_dir, run.get("run_tag", "")))
        caps = totals.get("n_cap_hits", run.get("n_cap_hits", 0))
        rows.append({
            "label": run.get("label", "?"),
            "prompt": run.get("prompt_variant", "?"),
            "hash": run.get("prompt_id", "?"),
            "notes": run.get("n_notes_scored", 0),
            "chunks": run.get("n_chunks", 0),
            "per_note": volume["pred_per_note"],
            "not_in_note": volume["not_in_note_rate"],
            "cap_hits": caps,
            "cap_looping": totals.get("n_cap_hits_while_repeating", 0),
            "dup_in_chunk": totals.get("n_items_dup_in_chunk", 0),
            "row_recall_l1": combined[levels[0]]["row_recall"],
            "row_recall_top": combined[top]["row_recall"],
            "code_recall_top": combined[top]["code_recall"],
            "fp_top": combined[top]["fp"],
            "fp_rate_top": combined[top]["fp_rate"],
            "top": top,
        })
    return rows


def render(rows):
    """Markdown, aggregate-only and safe to paste anywhere."""
    if not rows:
        return "No runs to compare.\n"
    top = rows[0]["top"]
    out = [
        "# Prompt variant comparison\n",
        ("**Read the volume block before the recall block.** A prompt that "
         "extracts more scores higher recall almost regardless of quality, so a "
         "variant winning on recall while emitting twice the findings has not "
         "won anything. What counts is recall held at lower volume.\n"),
        "| | " + " | ".join(f"{r['prompt']} (`{r['hash']}`)" for r in rows) + " |",
        "|---|" + "---|" * len(rows),
        "| notes / chunks | " + " | ".join(
            f"{r['notes']} / {r['chunks']}" for r in rows) + " |",
        "| **findings per note** | " + " | ".join(
            f"{r['per_note']:.1f}" for r in rows) + " |",
        "| **false positives** (" + top + ") | " + " | ".join(
            f"{r['fp_top']} ({r['fp_rate_top']:.2f})" for r in rows) + " |",
        "| not in the note | " + " | ".join(
            f"{r['not_in_note']:.4f}" for r in rows) + " |",
        "| chunks cut at the cap | " + " | ".join(
            f"{r['cap_hits']}" for r in rows) + " |",
        "| …of those, cut mid-replay | " + " | ".join(
            f"{r['cap_looping']}" for r in rows) + " |",
        "| items repeated in one reply | " + " | ".join(
            f"{r['dup_in_chunk']}" for r in rows) + " |",
        "| row recall L1 | " + " | ".join(
            f"{r['row_recall_l1']:.4f}" for r in rows) + " |",
        "| row recall " + top + " | " + " | ".join(
            f"{r['row_recall_top']:.4f}" for r in rows) + " |",
        "| code recall " + top + " | " + " | ".join(
            f"{r['code_recall_top']:.4f}" for r in rows) + " |",
    ]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Compare finished recall runs (no GPU, no re-scoring)")
    parser.add_argument("metrics", nargs="*",
                        help="metrics JSON files. Default: the two most recent")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--out", default=None,
                        help="write the markdown here as well as printing it")
    args = parser.parse_args()

    paths = args.metrics or find_metrics(args.results_dir)[:2]
    if len(paths) < 2:
        print("Need two finished runs. Run both variants first:\n"
              "  python -m src.evaluate_recall --smoke 3 --prompt scoped\n"
              "  python -m src.evaluate_recall --smoke 3 --prompt billable")
        return

    text = render(compare(paths, args.output_dir))
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
