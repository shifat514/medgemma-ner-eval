"""Build the pediatric-billing sample file. LOCAL MACHINE ONLY.

Reads the four encounter PDFs from outside the repo and writes one small
gitignored JSONL that the GPU run consumes:

    python -m src.build_billing_sample
    python -m src.build_billing_sample --show 112976    # dump one note's variants

Output (both gitignored — they contain note text and patient identifiers):
    data/samples/billing_sample.jsonl
    data/samples/billing_sample_stats.json

EVERYTHING THIS PRINTS IS THERE TO BE READ, NOT SKIPPED. Four notes is few
enough to verify by eye and too few to catch a parsing bug statistically. Three
of the printed numbers are checks with a right answer:

    gold lines vs unique     17 and 16. The gap is note 96176 listing Z68.51
                             twice. Any other gap means the DX regex changed.
    sections found           every note must show "Assessment". If one does not,
                             its gold is empty and its recall would read 0.0000
                             for a parsing reason.
    codes still visible      full: 16 (all of them, by design — that variant is
                             the harness check). assessment_cut: 2, both in note
                             26819's Problem List. leakage_cut: 0. A non-zero
                             leakage_cut number means a third copy of the answer
                             exists somewhere this parser does not know about.
"""

import argparse

from .billing_config import BILLING_PDF_DIR, SAMPLE_FILE, VARIANTS, variant_label
from .datasets.billing import build_sample, load_sample, write_sample


def _print_report(records, stats):
    print(f"notes parsed               {stats['n_notes']}")
    print(f"gold DX lines              {stats['n_gold_lines']}")
    print(f"gold unique codes          {stats['n_gold_unique']}"
          f"   (duplicates collapsed: {stats['n_gold_duplicates']})")
    print()

    print("per note:")
    header = f"  {'note':<10} {'visit':<10} {'lines':>5} {'uniq':>5}  codes"
    print(header)
    for rec in records:
        codes = ", ".join(rec["gold_codes"])
        print(f"  {rec['note_id']:<10} {rec['visit_kind']:<10} "
              f"{rec['n_gold_lines']:>5} {rec['n_gold_unique']:>5}  {codes}")
    print()

    print("sections found:")
    for rec in records:
        has_assessment = "Assessment" in rec["sections_found"]
        flag = "" if has_assessment else "   <- NO ASSESSMENT BLOCK"
        print(f"  {rec['note_id']:<10} {len(rec['sections_found']):>2} sections"
              f"{flag}")
        print(f"             {', '.join(rec['sections_found'])}")
    print()

    print("input size and leakage, per variant:")
    for name in VARIANTS:
        words = stats["words_by_variant"][name]
        leaked = stats["leaked_by_variant"][name]
        print(f"  {name:<16} {words:>6} words   "
              f"gold codes still visible: {leaked:>2}   {variant_label(name)}")
    print()

    print("which codes are still visible, per note:")
    for rec in records:
        for name in VARIANTS:
            leaked = rec["leaked_codes"][name]
            if leaked and name != "full":
                print(f"  {rec['note_id']:<10} {name:<16} {', '.join(leaked)}")
    print()

    print("lines removed by each cut:")
    for rec in records:
        for name in ("assessment_cut", "leakage_cut"):
            rm = rec["removed"][name]
            print(f"  {rec['note_id']:<10} {name:<16} "
                  f"assessment {rm['assessment_lines']:>3}   "
                  f"problem list {rm['problem_list_lines']:>3}")


def _show(records, note_id):
    rec = next((r for r in records if r["note_id"] == str(note_id)), None)
    if rec is None:
        raise SystemExit(f"no note {note_id!r}; have "
                         f"{[r['note_id'] for r in records]}")
    print(f"note {rec['note_id']}  ({rec['source_pdf']})")
    print(f"gold: {', '.join(rec['gold_codes'])}\n")
    for name in VARIANTS:
        print("=" * 78)
        print(f"VARIANT: {name}  —  {variant_label(name)}")
        print("=" * 78)
        print(rec["variants"][name])
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Parse the pediatric billing PDFs into a sample file"
    )
    parser.add_argument("--pdf-dir", default=BILLING_PDF_DIR)
    parser.add_argument("--out", default=SAMPLE_FILE)
    parser.add_argument("--show", default=None, metavar="NOTE_ID",
                        help="print one note's three variants and exit "
                             "(reads the built sample; does not rebuild)")
    args = parser.parse_args()

    if args.show:
        _show(load_sample(args.out), args.show)
        return

    print(f"pdf dir: {args.pdf_dir}\n")
    records, stats = build_sample(args.pdf_dir)
    path, stats_path = write_sample(records, stats, args.out)

    _print_report(records, stats)

    print(f"\nwrote {path}")
    print(f"wrote {stats_path}")
    print("\nBOTH FILES CONTAIN NOTE TEXT AND PATIENT IDENTIFIERS — gitignored.")
    print("Never commit them.")

    missing = [r["note_id"] for r in records if not r["gold_codes"]]
    if missing:
        raise SystemExit(
            f"ABORT: no gold codes parsed for note(s) {missing}. The Assessment "
            "block was not found or the DX lines did not match. Fix the parser "
            "before spending GPU time."
        )


if __name__ == "__main__":
    main()
