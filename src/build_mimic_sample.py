"""Build the small MIMIC medication-NER sample file. LOCAL MACHINE ONLY.

Reads the credentialed source data from outside the repo, extracts just the
sampled notes, and writes a gitignored JSONL inside the repo:

    python -m src.build_mimic_sample                 # 100 notes, seed 13
    python -m src.build_mimic_sample --n 100 --seed 13

Output (both gitignored — they contain note text):
    data/samples/mimic_med_sample.jsonl
    data/samples/mimic_med_sample_stats.json

Upload the .jsonl to Colab manually. The 1.1GB discharge.csv.gz is never copied.

Draw MAX_SAMPLE (=100) once: the 50-note run uses the first 50 records of this
file, making it a strict subset of the 100-note run.
"""

import argparse

from . import s3_io
from .datasets.mimic_meds import build_sample, write_sample
from .mimic_config import (
    DISCHARGE_CSV,
    LABEL_DIR,
    MAX_SAMPLE,
    S3_DISCHARGE_CSV,
    S3_LABEL_DIR,
    SAMPLE_FILE,
    SEED,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract the MIMIC medication-NER sample from local disk or S3"
    )
    parser.add_argument("--n", type=int, default=MAX_SAMPLE,
                        help=f"notes to sample (default {MAX_SAMPLE})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"sampling seed (default {SEED})")
    parser.add_argument("--source", choices=("local", "s3"), default="local",
                        help="where to read the source data (default local). "
                             "s3 streams the 1.1GB discharge gzip without "
                             "writing it to disk")
    parser.add_argument("--aws-profile", default=None,
                        help="AWS profile for --source s3 (default: boto3 chain)")
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--discharge-csv", default=None)
    parser.add_argument("--out", default=SAMPLE_FILE)
    args = parser.parse_args()

    # Explicit --label-dir/--discharge-csv win; otherwise --source picks the pair.
    # Either may be an s3:// URI regardless of --source.
    if args.source == "s3":
        label_dir = args.label_dir or S3_LABEL_DIR
        discharge_csv = args.discharge_csv or S3_DISCHARGE_CSV
    else:
        label_dir = args.label_dir or LABEL_DIR
        discharge_csv = args.discharge_csv or DISCHARGE_CSV

    if args.aws_profile:
        s3_io.set_profile(args.aws_profile)

    args.label_dir, args.discharge_csv = label_dir, discharge_csv

    print(f"labels:    {s3_io.describe(label_dir)}")
    print(f"discharge: {s3_io.describe(discharge_csv)}")
    print(f"sampling {args.n} notes with seed {args.seed} ...")

    records, stats = build_sample(
        n=args.n, seed=args.seed,
        label_dir=args.label_dir, discharge_csv=args.discharge_csv,
    )
    path, stats_path = write_sample(records, stats, args.out)

    print(f"\nlabel files:        {stats['label_files']}")
    print(f"eligible pool:      {stats['pool_size']}")
    print(f"excluded:           {len(stats['excluded'])}")
    for note_id, reason in sorted(stats["excluded"].items()):
        print(f"  - {note_id}: {reason}")
    print(f"discharge rows read: {stats['discharge_rows_scanned']}")
    print(f"notes found:        {stats['notes_found']}/{stats['requested']}")
    if stats["notes_missing"]:
        print(f"NOTES MISSING:      {stats['notes_missing']}")
    print(f"gold type conflicts resolved by priority: {stats['type_conflicts']}")

    n_spans = sum(len(r["spans"]) for r in records)
    print(f"gold spans (deduped): {n_spans}")
    print(f"\nwrote {path}")
    print(f"wrote {stats_path}")
    print("\nBOTH FILES CONTAIN NOTE TEXT — gitignored. Never commit them.")


if __name__ == "__main__":
    main()
