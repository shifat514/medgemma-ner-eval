"""Markdown report writer for the MIMIC medication-NER evaluation.

AGGREGATE METRICS ONLY. This file must never emit note text, span text, patient
identifiers, or per-example snippets — the report is committed to a public repo.
Everything written here is derived from integer counts and seqeval scores.

Per-example detail lives in the gitignored outputs/ directory (see
evaluate_mimic._dump_errors).
"""

import os

from .mimic_config import (
    ENTITY_TYPES,
    EXCLUDED_NOTE_IDS,
    GOLD_TYPE_MAP,
    TYPE_PRIORITY,
)


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _total(rows, key):
    return sum(r.get(key, 0) or 0 for r in rows)


def _ratio(num, den):
    return f"{num / den:.2f}x" if den else "n/a"


def _metrics_table(report):
    """Per-type rows then the averages, from the seqeval report dict."""
    lines = [
        "| entity | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for etype in ENTITY_TYPES:
        s = report.get(etype)
        if not isinstance(s, dict):
            lines.append(f"| {etype} | — | — | — | 0 |")
            continue
        lines.append(
            f"| {etype} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | "
            f"{_fmt(s['f1-score'])} | {int(s['support'])} |"
        )
    lines.append("| | | | | |")
    for avg in ("micro avg", "macro avg", "weighted avg"):
        s = report.get(avg)
        if isinstance(s, dict):
            lines.append(
                f"| **{avg}** | {_fmt(s['precision'])} | {_fmt(s['recall'])} | "
                f"{_fmt(s['f1-score'])} | {int(s['support'])} |"
            )
    return "\n".join(lines)


_ALIGN_NOTES = {
    "first-per-chunk": (
        "Only the **first** occurrence of each distinct predicted string is tagged "
        "within each chunk. A drug named once by the model yields one span per "
        "chunk rather than one per mention, which is what keeps the predicted-span "
        "count close to the gold count. The 80-token chunk overlap still lets a "
        "genuinely repeated entity be caught in more than one window, so recall "
        "survives. Residual expansion above 1.00x below is that overlap effect "
        "plus multi-chunk notes."
    ),
    "all-per-chunk": (
        "**Every** non-overlapping occurrence of a predicted string is tagged "
        "within the chunk that produced it, so one prediction of a common drug "
        "name becomes several predicted spans. This inflates the predicted-span "
        "count and depresses precision by an amount that is not the model's fault "
        "— measured at 1.72x expansion, costing ~16 points of micro F1 against "
        "`first-per-chunk`. Not the default; kept for comparison."
    ),
    "first-note": (
        "Predictions from every chunk are pooled and only the **first** occurrence "
        "in the whole note is tagged — one span per distinct string per note. Gold "
        "legitimately annotates the same string many times in one note (five `IV` "
        "events are five gold spans), so recall is capped near 0.70 by "
        "construction. Not the default; kept for comparison."
    ),
}


def _align_mode_note(mode):
    return _ALIGN_NOTES.get(mode, f"Alignment mode `{mode}`.")


def _mapping_table():
    lines = [
        "| gold annotation | our label | tie-break priority |",
        "|---|---|---|",
    ]
    for gold, ours in GOLD_TYPE_MAP.items():
        rank = TYPE_PRIORITY.index(ours) + 1 if ours in TYPE_PRIORITY else "—"
        lines.append(f"| `{gold}` | `{ours}` | {rank} |")
    return "\n".join(lines)


def build_report(report, per_note, meta):
    """Render the markdown report string."""
    n = len(per_note)
    tok = _total(per_note, "n_tokens")
    covered = _total(per_note, "tokens_covered")
    chunks = _total(per_note, "n_chunks")

    gold_raw = _total(per_note, "n_gold_raw")
    gold_spans = _total(per_note, "n_gold_spans")
    rows_raw = _total(per_note, "n_label_rows_raw")
    uniq_triples = _total(per_note, "n_unique_triples")
    gold_oob = _total(per_note, "n_gold_out_of_bounds")
    gold_dropped = _total(per_note, "n_gold_dropped_overlap")
    conflicts = _total(per_note, "n_type_conflicts")
    gold_no_token = _total(per_note, "n_gold_no_token")
    gold_snapped = _total(per_note, "n_gold_boundary_snapped")
    micro = report.get("micro avg") or {}
    gold_scored = int(micro.get("support", 0))

    mentions = _total(per_note, "n_pred_mentions")
    uniq = _total(per_note, "n_pred_unique")
    uniq_chunksum = _total(per_note, "n_pred_unique_chunksum")
    aligned = _total(per_note, "n_aligned_spans")
    unmatched = _total(per_note, "n_unmatched")
    dupes = _total(per_note, "n_overlap_duplicates")
    pred_spans = _total(per_note, "n_pred_spans")
    pred_dropped = _total(per_note, "n_pred_dropped_overlap")

    cap_hits = _total(per_note, "n_cap_hits")
    failures = _total(per_note, "n_chunk_failures")
    notes_with_cap = sum(1 for r in per_note if (r.get("n_cap_hits") or 0) > 0)

    gen = meta.get("gen_config", {})
    oracle = bool(meta.get("oracle"))

    title = (
        f"Harness ceiling (ORACLE, no model) — MIMIC-IV medication NER (n={n})"
        if oracle else
        f"MedGemma zero-shot medication NER — MIMIC-IV discharge summaries (n={n})"
    )
    banner = (
        "> **This is not a model evaluation.** No model was run. The gold span\n"
        "> strings were fed back through the identical pipeline to measure the\n"
        "> harness's structural ceiling — the best score any perfect extractor\n"
        "> could achieve under string-matching alignment, chunking, and\n"
        "> whitespace tokenization. Compare a real run's numbers against these.\n"
        if oracle else ""
    )

    md = f"""# {title}

{banner}
Aggregate metrics only. No note text, patient data, or example snippets appear in
this file.

## Sample

| | |
|---|---|
| notes evaluated | **{n}** |
| sampling seed | `{meta.get('seed')}` |
| eligible pool | 598 of 600 annotated notes (2 excluded, below) |
| selection | `random.Random(seed).sample(sorted(pool), 100)`; the n=50 set is the **first 50 of that same 100-note draw**, so it is a strict subset and the two runs are directly comparable |
| source notes | MIMIC-IV-Note v2.2 `discharge.csv.gz` |
| source labels | Medication Extraction Labels for MIMIC-IV-Note v1.0.0 |
| whitespace tokens scored | {tok:,} |

Only the ~600 notes that carry gold labels were eligible; the other ~331,000
discharge summaries have no annotations and were never considered.

## Model and configuration

| | |
|---|---|
| model | {"**none — oracle stand-in**" if oracle else f"`{meta.get('model_id')}`"} |
| quantization | {"n/a" if oracle else "4-bit NF4 (`bnb_4bit_compute_dtype=bfloat16`)"} |
| decoding | {"n/a" if oracle else "`do_sample=False` (greedy)"} |
| `max_new_tokens` | {"n/a" if oracle else gen.get('max_new_tokens')} |
| chunk size | {meta.get('chunk_words')} whitespace tokens |
| chunk overlap | {meta.get('overlap_words')} whitespace tokens |
| alignment mode | `{meta.get('align_mode')}` |
| chunks run | {chunks:,} (avg {0.0 if not n else chunks / n:.1f} per note) |
| scoring | `seqeval` entity-level `classification_report`, exact span match |

## Results

{_metrics_table(report)}

## Gold type → label mapping

All six gold annotation types pass through 1:1 with no renaming or merging.

{_mapping_table()}

Tie-break priority applies when one `(start, end)` char span carries two
different annotation types: the lowest-numbered type wins. It also fixes the
deterministic order in which gold spans are painted onto tokens.

## Chunking, truncation, and content coverage

Discharge summaries run 181–5,425 whitespace tokens (median 1,521) against the
single sentences the original NCBI/BC5CDR pipeline handled, and a single note
carries up to 381 gold annotations. Notes are therefore split into overlapping
token windows.

| | |
|---|---|
| tokens in sample | {tok:,} |
| tokens actually {"processed" if oracle else "sent to the model"} | {covered:,} |
| **note content dropped** | **{tok - covered:,} tokens ({0.0 if not tok else 100.0 * (tok - covered) / tok:.2f}%)** |
| chunks where generation hit `max_new_tokens` | {cap_hits:,} / {chunks:,} ({0.0 if not chunks else 100.0 * cap_hits / chunks:.1f}%) |
| notes affected by ≥1 capped generation | {notes_with_cap} / {n} |
| chunks that failed inference or parsing | {failures:,} |
| notes with no result (excluded from scoring) | {len(meta.get('notes_missing') or [])} |

Windows tile the full token list with the final window clamped to the note end,
so coverage is total by construction — no note content is truncated away. A
capped generation is different: it means the model was still emitting entities
when it hit `max_new_tokens`, so entities in that chunk may be missing and
recall for the affected chunks is understated.

## Prediction alignment and dedupe

Predictions arrive as entity *strings*, not character offsets, so each must be
located in the text again. How that is done — `--align-mode`, here
**`{meta.get('align_mode')}`** — is the single biggest lever on achievable
precision, and it was chosen by measurement (see
`mimic_ner_align_mode_comparison.md`).

{_align_mode_note(meta.get('align_mode'))}

| | |
|---|---|
| entity mentions emitted | {mentions:,} |
| distinct (text, type) pairs, note-level | {uniq:,} |
| distinct (text, type) pairs, summed per chunk | {uniq_chunksum:,} |
| spans produced by alignment, before dedupe | {aligned:,} |
| **multi-occurrence expansion (within chunk)** | **{_ratio(aligned, uniq_chunksum)}** |
| emitted strings matching nothing in their window | {unmatched:,} ({0.0 if not uniq_chunksum else 100.0 * unmatched / uniq_chunksum:.1f}%) |
| duplicate spans removed by overlap dedupe | {dupes:,} |
| spans dropped as partially overlapping another | {pred_dropped:,} |
| **final predicted spans scored** | **{pred_spans:,}** |

Compare **final predicted spans scored** against the gold support in the Results
table: the closer those two numbers, the better calibrated the alignment. Any gap
is systematic over- or under-prediction that no prompt change will fix.

Overlapping windows re-read the same tokens, so an entity in an overlap region
gets predicted twice; predictions are deduped on `(start, end, type)` after
stitching, and the count removed is reported above.

## Gold label processing

| | |
|---|---|
| annotation rows read from the label CSVs | {rows_raw:,} |
| − exact duplicate rows (same span+type, different `Group`) | −{max(0, rows_raw - uniq_triples):,} |
| = unique `(start, end, type)` | {uniq_triples:,} |
| − **type conflicts resolved by priority** | **−{conflicts:,}** |
| = one type per `(start, end)` | {gold_raw:,} |
| − spans out of bounds for the note text | −{gold_oob:,} |
| − spans covering no whitespace token | −{gold_no_token:,} |
| = spans mapped onto tokens | {gold_spans:,} |
| − spans dropped as partially overlapping another | −{gold_dropped:,} |
| **= gold spans scored (support)** | **{gold_scored:,}** |
| *of those, boundary snapped >1 char to token edges* | *{gold_snapped:,} ({0.0 if not gold_spans else 100.0 * gold_snapped / gold_spans:.1f}%)* |

Counts above are for the {n} sampled notes. Corpus-wide across all 600 label
files: 64,106 annotation rows, of which 4,408 are exact `(start, end, type)`
duplicates carrying a different `Group` (one shared `Mode` span reused by several
medication groups), and ~58 `(start, end)` pairs carry two different annotation
types. `Group` is dropped after deduping — it ties one medication event's spans
together and is irrelevant to token-level scoring.

Gold and predicted spans are flattened onto tokens by the same all-or-nothing
rule, so neither side gets an advantage from how overlaps are resolved.

Gold span *starts* land on whitespace-token boundaries throughout this corpus,
but some span *ends* fall inside a token. Most are a trailing `.` or `,`
(harmless — alignment normalizes surrounding punctuation away); the rest are
run-together strings such as `DAILY:PRN`, where the whole token gets tagged. The
row above counts only boundaries that moved by more than one character.

## Structural ceiling

The harness cannot reach F1 = 1.0 even with a perfect extractor, because
predictions are located by string matching, notes are chunked, and spans are
flattened onto whitespace tokens. `--oracle` measures that ceiling by feeding the
gold span strings back through the identical pipeline:

```
python -m src.evaluate_mimic --oracle --n {n}
```

Reading MedGemma's scores against that ceiling separates model error from harness
error. Oracle precision in particular lands well below 1.0 — that gap *is* the
multi-occurrence expansion above, and no model can avoid it.

The `first-per-chunk` default was chosen by running that comparison across all
three alignment modes; it beat the alternatives by +6.5 micro F1 and won on all
six entity types. See `mimic_ner_align_mode_comparison.md`. Results computed
under a different `--align-mode` are **not** comparable with these.

### Notes excluded from the sampling pool

| note_id | reason |
|---|---|
"""
    for note_id, reason in sorted(EXCLUDED_NOTE_IDS.items()):
        md += f"| `{note_id}` | {reason} |\n"

    md += f"""
Median annotations per note across the corpus is 97; a label file with 0 or 1
annotations is far more likely to be a gold-labeling failure than a genuinely
medication-free discharge summary. Including them would have added ~0 support
while contributing false negatives against a note the labeler evidently failed
on. Both were excluded before sampling, leaving a pool of 598.

## Caveats

- **Small sample.** n={n} notes. Per-type confidence intervals are wide,
  especially for `Duration` (the rarest type, 2,797 of 64,106 annotations
  corpus-wide). Differences of a few F1 points between the n=50 and n=100 runs
  should not be read as signal.
- **The gold labels are LLM-generated, not human-annotated.** The Medication
  Extraction Labels dataset was produced by prompting an LLM with a
  `direct-group-yaml` prompt over the i2b2 2009 medication-challenge schema (the
  prompt name is encoded in every label filename). These numbers therefore
  measure agreement between MedGemma and another model's output, not accuracy
  against clinician ground truth. Both the ceiling and the error attribution are
  affected: an apparent MedGemma error may be a gold-label error.
- **Exact-span scoring.** `seqeval` requires the predicted span to match the gold
  span token-for-token. A prediction of `325 mg` where gold is `325` scores as
  both a false positive and a false negative. This penalizes the attribute types
  (`Dose`, `Frequency`, `Duration`, `Reason`) hardest, since their boundaries are
  the most arguable — `Reason` spans in particular are long free-text phrases
  where exact agreement is unlikely even between two careful humans. Read the
  per-type numbers as a lower bound, and `Medication` (short, well-bounded spans)
  as the most trustworthy row.
- **String-matching alignment.** Predictions are located by matching strings back
  onto tokens rather than by asking the model for character offsets. See the
  expansion factor above for the size of this effect.
- **Zero-shot, no prompt tuning.** One prompt, greedy decoding, no few-shot
  examples, no post-processing beyond JSON parsing. This is a floor, not a
  ceiling.
"""
    return md


def write_report(report, per_note, meta, results_dir=None):
    """Write the markdown report; return its path."""
    results_dir = results_dir or meta.get("results_dir") or "results"
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"mimic_ner_{meta.get('label')}_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report(report, per_note, meta))
    return path
