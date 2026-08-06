# Harness ceiling (ORACLE, no model) — MIMIC-IV medication NER (n=50)

> **This is not a model evaluation.** No model was run. The gold span
> strings were fed back through the identical pipeline to measure the
> harness's structural ceiling — the best score any perfect extractor
> could achieve under string-matching alignment, chunking, and
> whitespace tokenization. Compare a real run's numbers against these.

Aggregate metrics only. No note text, patient data, or example snippets appear in
this file.

## Sample

| | |
|---|---|
| notes evaluated | **50** |
| sampling seed | `13` |
| eligible pool | 598 of 600 annotated notes (2 excluded, below) |
| selection | `random.Random(seed).sample(sorted(pool), 100)`; the n=50 set is the **first 50 of that same 100-note draw**, so it is a strict subset and the two runs are directly comparable |
| source notes | MIMIC-IV-Note v2.2 `discharge.csv.gz` |
| source labels | Medication Extraction Labels for MIMIC-IV-Note v1.0.0 |
| whitespace tokens scored | 79,769 |

Only the ~600 notes that carry gold labels were eligible; the other ~331,000
discharge summaries have no annotations and were never considered.

## Model and configuration

| | |
|---|---|
| model | **none — oracle stand-in** |
| quantization | n/a |
| decoding | n/a |
| `max_new_tokens` | n/a |
| chunk size | 400 whitespace tokens |
| chunk overlap | 80 whitespace tokens |
| alignment mode | `first-note` |
| chunks run | 262 (avg 5.2 per note) |
| scoring | `seqeval` entity-level `classification_report`, exact span match |

## Results

| entity | precision | recall | f1 | support |
|---|---|---|---|---|
| Medication | 0.8385 | 0.6621 | 0.7399 | 1811 |
| Dose | 0.8127 | 0.6143 | 0.6997 | 897 |
| Mode | 0.6926 | 0.2730 | 0.3916 | 784 |
| Frequency | 0.7772 | 0.4803 | 0.5937 | 864 |
| Duration | 0.7953 | 0.7246 | 0.7583 | 236 |
| Reason | 0.7530 | 0.6995 | 0.7253 | 619 |
| | | | | |
| **micro avg** | 0.7974 | 0.5724 | 0.6664 | 5211 |
| **macro avg** | 0.7782 | 0.5756 | 0.6514 | 5211 |
| **weighted avg** | 0.7898 | 0.5724 | 0.6554 | 5211 |

## Gold type → label mapping

All six gold annotation types pass through 1:1 with no renaming or merging.

| gold annotation | our label | tie-break priority |
|---|---|---|
| `MEDICATION` | `Medication` | 1 |
| `DOSE` | `Dose` | 2 |
| `MODE` | `Mode` | 3 |
| `FREQUENCY` | `Frequency` | 4 |
| `DURATION` | `Duration` | 5 |
| `REASON` | `Reason` | 6 |

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
| tokens in sample | 79,769 |
| tokens actually processed | 79,769 |
| **note content dropped** | **0 tokens (0.00%)** |
| chunks where generation hit `max_new_tokens` | 0 / 262 (0.0%) |
| notes affected by ≥1 capped generation | 0 / 50 |
| chunks that failed inference or parsing | 0 |
| notes with no result (excluded from scoring) | 0 |

Windows tile the full token list with the final window clamped to the note end,
so coverage is total by construction — no note content is truncated away. A
capped generation is different: it means the model was still emitting entities
when it hit `max_new_tokens`, so entities in that chunk may be missing and
recall for the affected chunks is understated.

## Extraction health

Whether the model's output was understood at all. A zero score for an entity type
is only meaningful if its items were reaching the scorer in the first place — an
earlier revision silently discarded any item whose `type` string was not one of
the six exact labels, which put `Duration` and `Reason` at exactly 0.0000.

| | |
|---|---|
| chunks returning no usable JSON | 0 / 262 (0.0%) |
| chunks returning an explicitly empty list | 9 |
| chunks yielding zero entities | 9 / 262 (3.4%) |
| JSON items emitted | 6,380 |
| items dropped — unrecognized `type` | 0 |
| items dropped — no usable span text | 0 |
| items rescued by type normalization | 0 |
| **entities extracted** | **6,380** |
| entities extracted per chunk | 24.4 |

A non-zero "dropped — unrecognized type" count means entities the model found
were thrown away; the offending type strings are listed in the run's
`parse_diag.json` and should be added to `prompt_mimic._TYPE_ALIASES`.

## Prediction alignment and dedupe

Predictions arrive as entity *strings*, not character offsets, so each must be
located in the text again. How that is done — `--align-mode`, here
**`first-note`** — is the single biggest lever on achievable
precision, and it was chosen by measurement (see
`mimic_ner_align_mode_comparison.md`).

Predictions from every chunk are pooled and only the **first** occurrence in the whole note is tagged — one span per distinct string per note. Gold legitimately annotates the same string many times in one note (five `IV` events are five gold spans), so recall is capped near 0.70 by construction. Not the default; kept for comparison.

| | |
|---|---|
| entity mentions emitted | 6,380 |
| distinct (text, type) pairs, note-level | 3,785 |
| distinct (text, type) pairs, summed per chunk | 5,075 |
| spans produced by alignment, before dedupe | 3,741 |
| **multi-occurrence expansion (within chunk)** | **0.74x** |
| emitted strings matching nothing in their window | 33 (0.7%) |
| duplicate spans removed by overlap dedupe | 0 |
| spans dropped as partially overlapping another | 0 |
| **final predicted spans scored** | **3,741** |

Compare **final predicted spans scored** against the gold support in the Results
table: the closer those two numbers, the better calibrated the alignment. Any gap
is systematic over- or under-prediction that no prompt change will fix.

Overlapping windows re-read the same tokens, so an entity in an overlap region
gets predicted twice; predictions are deduped on `(start, end, type)` after
stitching, and the count removed is reported above.

## Gold label processing

| | |
|---|---|
| annotation rows read from the label CSVs | 5,526 |
| − exact duplicate rows (same span+type, different `Group`) | −154 |
| = unique `(start, end, type)` | 5,372 |
| − **type conflicts resolved by priority** | **−3** |
| = one type per `(start, end)` | 5,369 |
| − spans out of bounds for the note text | −0 |
| − spans covering no whitespace token | −3 |
| = spans mapped onto tokens | 5,366 |
| − spans dropped as partially overlapping another | −155 |
| **= gold spans scored (support)** | **5,211** |
| *of those, boundary snapped >1 char to token edges* | *462 (8.6%)* |

Counts above are for the 50 sampled notes. Corpus-wide across all 600 label
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
python -m src.evaluate_mimic --oracle --n 50
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
| `12619324-DS-22` | 1 annotation in label file (median 97) — likely labeling failure |
| `19933834-DS-2` | 0 annotations in label file (median 97) — likely labeling failure |

Median annotations per note across the corpus is 97; a label file with 0 or 1
annotations is far more likely to be a gold-labeling failure than a genuinely
medication-free discharge summary. Including them would have added ~0 support
while contributing false negatives against a note the labeler evidently failed
on. Both were excluded before sampling, leaving a pool of 598.

## Caveats

- **Small sample.** n=50 notes. Per-type confidence intervals are wide,
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
