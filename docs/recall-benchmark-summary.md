# MedGemma-4B on MDACE billing evidence — results

**The question.** Given a hospital note, how much of the evidence a human coder
highlighted to justify their billing codes can MedGemma-4B find on its own — no
fine-tuning, one prompt?

**The data.** 24 notes, 100 highlighted phrases, 91 distinct billing codes.
4-bit, greedy, 400-word windows. One-shot (the prompt carries one synthetic
example).

## Headline

**78 of 100 phrases recovered** — 95% CI 69–85%. Ceiling on this file is 98, not
100, because 2 rows are status/history and the prompt excludes them by design.

Sanity check: the previous branch scored **0.5278** using exact matching against
note wording only. The equivalent measurement here is **0.5100** — 1.8 points
apart. Two independent implementations landing that close means this harness
measures what the old one measured, so the rest reads as a real gain rather than
a changed yardstick.

## What each matching level bought

| level | rule | found | gain |
|---|---|---|---|
| L1 | identical text after normalisation | 57 | — |
| **L2** | **whole-token containment, either direction** | **74** | **+17** |
| L3 | token-set Dice / difflib, thresholded | 75 | +1 |
| L4 | biomedical embedding cosine | 79 | +4 |
| L5 | LLM judge deletes wrong matches | **78** | −1 |

**L2 did almost all the work.** Most L1 misses were the model writing a shorter
or longer form of the same phrase — `back pain` for `chronic back pain`.

**L3 is not worth keeping.** +1 row, and still +1 after its thresholds were
loosened. L2 already catches what it would.

**L4 earned its place but not its reputation.** +4 rows, and it is the only level
that reaches abbreviations at all. But no cosine threshold separates real
synonyms from near-misses: `acute renal failure` vs `chronic renal failure`
scores 0.833, *above* 8 of the 10 abbreviation pairs L4 exists to catch. L4 is
not usable without L5.

**L5 rejected 28 of 85 loose matches (33%), including 43% of L4's** — and cost
only 1 row of recall, because each answer has ~4 accepted spellings and usually
keeps another.

## Why the three-column accept-set mattered

Gold spells each answer three ways:

| gold spelling | exact match | loose match |
|---|---|---|
| what the note says | 56% | 76% |
| official billing name | **1%** | 65% |
| medical dictionary name | 7% | 47% |

**The model writes the note's words, not the catalogue's.** `HTN` versus
`Essential (primary) hypertension` — same condition, no shared characters.
Scoring against catalogue wording alone gives the model 1%.

## Precision, recall and F1 per source

| gold source | entries | precision | recall | F1 | max precision possible |
|---|---|---|---|---|---|
| any of the three | 318 | 0.1159 | 0.4465 | 0.1841 | **0.2571** |
| what the note says | 99 | 0.0612 | 0.7576 | 0.1133 | **0.0808** |
| official billing name | 91 | 0.0482 | 0.6484 | 0.0897 | **0.0743** |
| medical dictionary name | 142 | 0.0547 | 0.4718 | 0.0980 | **0.1151** |

**Precision and F1 here are diagnostics, not quality**, for two reasons that have
nothing to do with the model.

**1. The annotation scope.** 1,225 findings produced, 142 matched, so 1,083
"false positives" — but:

| | count | |
|---|---|---|
| in the note, never billed | 1,033 (95.4%) | a correct extraction |
| **not in the note at all** | **50 (4.6%)** | **real error** |

MDACE annotates only codes that were actually *billed*, and a note is full of
real conditions nobody billed. Those are counted against precision anyway,
because subtracting your own false positives before dividing raises precision by
construction. **The true error rate is 4.6%, not 88%.**

**2. Per-source, "false positive" means "matched nothing *in that source*".** A
prediction that correctly hits the catalogue wording counts as a false positive
on the note-wording row, so every single-source row carries almost the whole
prediction set as false positives. Only the combined row counts predictions that
matched nothing anywhere.

Hence the **max precision possible** column — the arithmetic ceiling given how
many findings were produced. With 1,225 predictions against 318 accepted
phrasings, **no extractor of any quality exceeds 0.2571 on the combined row.**
F1 blends precision with recall and sits near the worse of them, so it inherits
all of this and should not be quoted as a headline.

The one precision-side figure that is *not* distorted is the **4.2%
not-in-the-note rate**, because it does not depend on what was billed at all.

## Caveats

- **51 findings per note** against ~4 correct: about 12:1. Recall is real,
  precision is unsolved, and this is the blocker for any downstream code lookup.
- **0.78 is a floor.** 21 of 82 chunks hit the 1024-token cap; 7 were cut
  mid-production and lost findings outright.
- **L5's judge was MedGemma itself.** It rejected a third of its own matcher's
  work — not how a self-serving judge fails — but an independent judge would be
  stronger evidence.

## Before fine-tuning — what is still untried

1. **Smaller windows, 400 → 250 words.** Directly targets the 7 truncated
   chunks. Cheapest remaining recall gain. ~45 min GPU.
2. **Diagnose the 22 misses.** Nobody yet knows whether they are truncation, the
   model never mentioning them, or matching still too strict. Free, and it
   decides whether (1) is worth the GPU at all.
3. **Volume control.** 12:1 has had no attempt made on it.
4. **A "would a coder bill this?" prompt.** Cut invented spans 9x on a 2-note
   test (9.2% → 1.0%) at equal recall. Untested at scale.
5. **An independent L5 judge** instead of the model grading itself.

None of these require training. Fine-tuning is the step after they run out.
