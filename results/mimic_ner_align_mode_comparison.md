# Alignment-mode comparison — why `first-per-chunk` is the default

Aggregate metrics only. No note text, patient data, or example snippets.

MedGemma returns entity **strings**, not character offsets, so the harness has to
find each predicted string in the text again. How it does that is the single
biggest lever on the achievable precision — bigger than any prompt change.

This was decided by measurement, not intuition, using `--oracle`: the gold spans
are fed back through the identical pipeline, so the score is the ceiling a
*perfect* extractor could reach. Costs ~7 s per mode and needs no GPU.

```bash
for m in all-per-chunk first-per-chunk first-note; do
  python -m src.evaluate_mimic --oracle --n 100 --align-mode $m
done
```

## The three modes

| mode | rule | spans per distinct predicted string |
|---|---|---|
| `all-per-chunk` | tag every occurrence within the chunk that produced it | many |
| **`first-per-chunk`** | tag only the first occurrence within each chunk | one per chunk |
| `first-note` | pool all chunks' predictions, tag first occurrence in the note | one per note |

## Oracle ceiling, n=100 (11,002 gold spans)

| mode | micro P | micro R | micro F1 | macro F1 |
|---|---|---|---|---|
| `all-per-chunk` | 0.6976 | **0.9505** | 0.8046 | 0.7996 |
| **`first-per-chunk`** | **0.8583** | 0.8817 | **0.8698** | **0.8644** |
| `first-note` | 0.7917 | 0.5524 | 0.6507 | 0.6417 |

`first-per-chunk` beats the previous `all-per-chunk` default by **+6.5 micro F1**,
trading 6.9 points of recall for 18.6 points of precision.

## Per-type F1 — `first-per-chunk` wins all six

| type | `all-per-chunk` | **`first-per-chunk`** | `first-note` |
|---|---|---|---|
| Medication | 0.8580 | **0.9084** | 0.7221 |
| Dose | 0.8338 | **0.8920** | 0.6841 |
| Mode | 0.7204 | **0.7880** | 0.3685 |
| Frequency | 0.7562 | **0.8398** | 0.5774 |
| Duration | 0.8246 | **0.8853** | 0.7794 |
| Reason | 0.8047 | **0.8729** | 0.7188 |

Winning on every type individually — not just on the aggregate — is what makes
this a clear result rather than a wash. There is no type for which the old
default was preferable.

## Pipeline stats

| stat | `all-per-chunk` | `first-per-chunk` | `first-note` |
|---|---|---|---|
| entity mentions emitted | 13,434 | 13,434 | 13,434 |
| distinct (text, type), note-level | 7,772 | 7,772 | 7,772 |
| distinct (text, type), summed per chunk | 10,440 | 10,440 | 10,440 |
| spans produced by alignment | 17,940 | 13,342 | 7,676 |
| **expansion** (aligned ÷ per-chunk distinct) | **1.718x** | **1.278x** | **0.735x** |
| duplicates removed by overlap dedupe | 2,750 | 2,041 | 0 |
| spans dropped as partial overlap | 199 | 0 | 0 |
| **final predicted spans scored** | 14,991 | 11,301 | 7,676 |
| gold spans (support) | 11,002 | 11,002 | 11,002 |

Expansion uses one consistent denominator throughout: alignment-produced spans
divided by the per-chunk sum of distinct predicted (text, type) pairs. For
`first-note`, which aligns once over the whole note rather than per chunk, that
denominator overcounts — the ratio is below 1.0 because it produces one span per
distinct string *per note* (7,676) against a *per-chunk* tally (10,440).

## Why each mode lands where it does

**`all-per-chunk` over-predicts.** 14,991 predicted spans against 11,002 gold.
A drug named once by the model gets tagged at all ~4 of its occurrences in that
chunk, but gold annotates only the occurrences that are genuinely medication
events. Precision 0.70 is the direct consequence. It is worst for `Mode`
(P=0.572), where `IV` and `PO` recur constantly.

**`first-note` under-predicts badly.** Only 7,676 spans for 11,002 gold — it
*cannot* reach recall above ~0.70 by construction, because gold legitimately
annotates the same string many times in one note (five separate `IV` events are
five separate gold spans). Recall 0.55 confirms it. `Mode` collapses to F1=0.369.

**`first-per-chunk` is close to calibrated.** 11,301 predicted against 11,002
gold — within 3%. The 80-token chunk overlap means a genuinely-repeated entity
can still be caught in more than one window, which recovers much of the recall
that `first-note` throws away, while the per-chunk cap prevents the runaway
expansion of `all-per-chunk`.

## What this does and does not change

It raises the **ceiling**, so model results computed under this default are not
comparable with results computed under `all-per-chunk`. Anything measured before
this change must be re-run or relabelled. The resume cache is keyed on the mode
(`run_tag` includes it), so the two can never silently mix.

It does **not** eliminate the effect: 1.278x expansion remains, meaning a perfect
extractor still loses ~14 points of precision. Closing that gap requires asking
the model to emit character offsets rather than strings — a prompt-and-parse
change, not an alignment change, and a larger piece of work.

## Reproducing

Committed alongside this file, for both sample sizes:

- `mimic_ner_oracle_{50,100}.csv` — the default, `first-per-chunk`
- `mimic_ner_oracle_{50,100}_all-per-chunk.csv`
- `mimic_ner_oracle_{50,100}_first-note.csv`

The ranking holds at both sample sizes, so it is not an artefact of n (micro
P/R/F1):

| mode | n=50 | n=100 |
|---|---|---|
| `all-per-chunk` | 0.6956 / 0.9491 / 0.8029 | 0.6976 / 0.9505 / 0.8046 |
| **`first-per-chunk`** | **0.8557 / 0.8783 / 0.8669** | **0.8583 / 0.8817 / 0.8698** |
| `first-note` | 0.7974 / 0.5724 / 0.6664 | 0.7917 / 0.5524 / 0.6507 |
