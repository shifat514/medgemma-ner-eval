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

Each level is a looser rule for "did the model find this phrase". Every level
includes the one above it.

| level | rule | example it catches | found | gain |
|---|---|---|---|---|
| L1 | text must match exactly (lowercased, punctuation stripped) | `sepsis` = `sepsis` | 57 | — |
| **L2** | **one phrase contains the other, whole words** | `back pain` in `chronic back pain` | **74** | **+17** |
| L3 | fuzzy — typos and reordered words (Dice / difflib) | `hyperlipidema` ≈ `hyperlipidemia` | 75 | +1 |
| L4 | means the same thing (biomedical embeddings) | `CHF` ≈ `congestive heart failure` | 79 | +4 |
| L5 | an LLM reviews every loose match and deletes the wrong ones | rejects `diabetes` ≠ `diabetes insipidus` | **78** | −1 |

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

## Verdict — a good extractor, a poor filter

**What works:**

- Finds **78%** of billed evidence, and only **4.6%** of its output is invented —
  low for a 4B model.
- **It knows the vocabulary.** It expands `HTN` → hypertension, `CABG` →
  coronary artery bypass graft, `BRBPR` → bright red blood per rectum. That is
  why the catalogue column went from 1% to 65%.
- **Format compliance is solid.** Zero JSON parse failures in 82 calls, and both
  fields populated on all 1,706 emitted items.

**What does not work:**

- **12:1 over-extraction.** It cannot tell a billable finding from any medical
  phrase. It extracted vital signs, blood products and bowel preps until the
  prompt named each one explicitly — and a second prompt built on a positive
  criterion instead of a blacklist did not fix the volume either.
- **Repetition loops.** 413 of 1,706 emitted items (24%) were duplicates *inside
  a single reply*. This caused most of the 21 truncated chunks.
- **No self-limiting.** It ran out of output budget on 26% of calls.

It sees what is in the note. It has no idea what is billable.

## Is it worth fine-tuning for the billing pipeline?

**Yes — and the volume problem is precisely what fine-tuning is for.**

The model already demonstrates the two hard parts: it locates findings in free
text, and it names them in standard clinical vocabulary. What it lacks is
billing-scope judgment, and that is exactly what the 9,499 labelled MDACE rows
encode. Prompt engineering could not teach it — that was attempted twice — but
supervised examples plausibly can.

Two things to be clear about:

- **This benchmark does not test code assignment.** It tests whether the
  *evidence* is found. Code recall of **77% (70 of 91)** is the ceiling on the
  downstream term → ICD lookup, since you cannot retrieve a code whose evidence
  was never extracted. Whether MedGemma picks the *right* ICD code is untested.
- **SNOMED reads as its weakest column at 47%, and that number is a floor.**
  This file ships at most 3 SNOMED terms per code against 1,894 mapped — 157 of
  them, or **8%**. One row maps to 64 concepts and ships 3. A real lookup has an
  order of magnitude more terms to match against, so SNOMED is probably stronger
  than 47% suggests.

Suggested order: exhaust the items below first, since they are hours rather than
days. Then fine-tune on the full 1,074-note corpus with **volume**, not recall,
as the target metric.

## The second-pass filter — what actually fixed precision

Extraction and filtering were one call, and the model is good at the first and
bad at the second. So they were split: after extraction, each finding gets its
own call asking **"would a coder assign a billing code to this?"** and the noes
are dropped.

| | model alone | **+ filter** |
|---|---|---|
| findings per note | 51.0 | **21.5** |
| precision | 0.1135 | **0.2311** |
| best precision possible | 0.2571 | **0.5010** |
| recall | 0.7800 | **0.6700** |

It dropped 710 findings. **690 were false positives and 20 were real matches, so
97.2% of the dropping was correct.** Those 20 cost 11 of the 100 phrases, because
a row survives if it keeps any other matching form.

**The question was bare — no hints, no list of categories to avoid.** So
MedGemma already knows what is billable; it simply does not apply that knowledge
while extracting. That is why splitting into two steps worked where two prompt
rewrites had not.

Recall falls. That is the price of the precision, and both columns are reported
so the operating point is a choice rather than a default.

## Section filtering — measured, and ruled out

Dropping irrelevant sections looked like the obvious lever and is not. The false
positives are spread thin: the largest single section holds 8% of them, half sit
in a long tail, and **the sections producing the most false positives are the
same ones holding the most gold** — Brief Hospital Course has 19 of the 100 gold
phrases and 94 false positives. Applied on top of the filter it bought +0.6
precision points and cost one answer.

One measurement worth keeping from it: stripping medications, labs and admin
sections *before* the model reads them removes 18% of the text at **zero gold
cost**. Social History is deliberately kept — it holds 3 of the 100 phrases, the
smoking and alcohol status codes.

## Before fine-tuning — what is still untried

1. **Smaller windows, 400 → 250 words.** Targets the 21 chunks that ran out of
   output room, 7 of which were cut while still producing findings.
2. **A cap on findings per call.** 413 of 1,706 emitted items were repeats
   *within a single reply*, and that looping is what hits the token cap. Built
   and untested.
3. **Showing the filter the surrounding sentence.** Its 20 mistakes cluster on
   procedure codes, where the billed evidence is a substance or an observation
   rather than a diagnosis — `platelets` for a platelet transfusion, `sinus
   rhythm` for an ECG. A bare phrase cannot be judged; a sentence can.
4. **An independent judge** instead of MedGemma grading itself.

None of these require training.

## The ceiling, and why it matters

**Even a perfect filter over the current findings tops out near 0.55 precision.**
That is arithmetic: the model still emits ~17 findings per note against ~4
billed, and precision cannot exceed the ratio of possible answers to produced
answers.

So the items above are worth a few points each. Closing the remaining gap needs
the model to extract less and better, which is fine-tuning. Everything short of
that is cleanup on a measurement that is already defensible.
