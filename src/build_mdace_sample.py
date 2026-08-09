"""Build the MDACE evaluation sample file. LOCAL MACHINE ONLY.

Reads the two MDACE source files from outside the repo and writes one small
gitignored JSONL that the GPU run consumes:

    python -m src.build_mdace_sample
    python -m src.build_mdace_sample --source s3

Output (both gitignored — they contain note text):
    data/samples/mdace_sample.jsonl
    data/samples/mdace_sample_stats.json

The file holds the UNION of the stratified 50-note draw and Ehtesham Bhai's
24-note cut (73 notes, 202 chunks). Every scored view is derived from one
inference pass over that union; the views differ only in which notes they count
and which answer key they use, and neither of those touches the model.
"""

import argparse
import json

from .datasets.mdace import build_sample, write_sample
from .mdace_config import (
    DATASET_FILE,
    NOTES_FILE,
    S3_DATASET_FILE,
    S3_NOTES_FILE,
    S3_SAMPLE_100_FILE,
    SAMPLE_100_FILE,
    SAMPLE_FILE,
    SEED,
    STRATUM_N,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract the MDACE evaluation sample from local disk or S3"
    )
    parser.add_argument("--n-per-stratum", type=int, default=STRATUM_N,
                        help=f"notes per chart type (default {STRATUM_N})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"sampling seed (default {SEED})")
    parser.add_argument("--source", choices=("local", "s3"), default="local")
    parser.add_argument("--aws-profile", default=None)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--notes-file", default=None)
    parser.add_argument("--sample-100-file", default=None)
    parser.add_argument("--no-sample-100", action="store_true",
                        help="stratified notes only; drops views A1 and B1")
    parser.add_argument("--out", default=SAMPLE_FILE)
    args = parser.parse_args()

    if args.source == "s3":
        dataset = args.dataset_file or S3_DATASET_FILE
        notes = args.notes_file or S3_NOTES_FILE
        shipped = args.sample_100_file or S3_SAMPLE_100_FILE
    else:
        dataset = args.dataset_file or DATASET_FILE
        notes = args.notes_file or NOTES_FILE
        shipped = args.sample_100_file or SAMPLE_100_FILE

    if args.aws_profile:
        from . import s3_io
        s3_io.set_profile(args.aws_profile)

    print(f"dataset:   {dataset}")
    print(f"notes:     {notes}")
    print(f"sample100: {'(skipped)' if args.no_sample_100 else shipped}")
    print(f"sampling {args.n_per_stratum} per chart type, seed {args.seed} ...\n")

    records, stats = build_sample(
        dataset, notes,
        sample_100_file=None if args.no_sample_100 else shipped,
        n_per_stratum=args.n_per_stratum, seed=args.seed,
    )
    path, stats_path = write_sample(records, stats, args.out)

    print(f"annotation rows            {stats['n_rows']:,}")
    print(f"notes joined               {stats['n_notes_joined']:,}"
          f"  (missing text: {stats['n_notes_missing_text']})")
    print(f"offset mismatches          {stats['n_offset_mismatches']}"
          "   <- must be 0")
    print(f"distinct (note, term)      {stats['n_distinct_note_term_pairs']:,}")
    print(f"notes billed both ways     {stats['n_mixed_chart_type']}")
    print()
    print(f"selected (union)           {stats['n_selected']} notes,"
          f" {stats['chunks_total']} chunks")
    print(f"  stratified draw          {stats['n_stratified']} notes,"
          f" {stats['chunks_stratified']} chunks,"
          f" {stats['gold_terms_stratified']} gold terms"
          f"   {stats['stratified_by_type']}")
    print(f"  sample_100 notes         {stats['n_sample100']} notes,"
          f" {stats['chunks_sample100']} chunks"
          f"   {stats['sample100_by_type']}")
    print(f"  overlap between them     {stats['n_overlap']} note(s)")
    print()
    print("sample_100 as an answer key:")
    print(f"  gold terms those notes really have  "
          f"{stats['gold_terms_sample100_full']}")
    print(f"  gold terms the file ships           "
          f"{stats['gold_terms_sample100_shipped']}")
    print(f"  MISSING from the shipped file       "
          f"{stats['gold_terms_sample100_full'] - stats['gold_terms_sample100_shipped']}"
          "   <- view A1 measures what this costs")
    print()
    print(f"mixed-chart-type notes in the stratified draw: "
          f"{stats['n_mixed_in_stratified']}/{stats['n_stratified']}")
    print(f"\nwrote {path}")
    print(f"wrote {stats_path}")
    print("\nBOTH FILES CONTAIN NOTE TEXT — gitignored. Never commit them.")

    if stats["n_offset_mismatches"]:
        raise SystemExit(
            f"ABORT: {stats['n_offset_mismatches']} evidence offsets do not "
            "slice their own note. The two source files have drifted apart."
        )


if __name__ == "__main__":
    main()
