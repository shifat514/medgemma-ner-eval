# MDACE term-level NER evaluation — plan

Branch: `mdace-term-ner` (off `mimic-medication-ner`)
Status: plan agreed, decisions 1–7 settled. Building next.

One correction up front, plainly: nothing in the fact sheet was wrong — the 122 chunks /
324 terms figure was checked and reproduced exactly. The three things raised below are
gaps the sheet didn't cover, plus one instruction that contradicts itself (Decision 2).

## Decisions at a glance

| # | decision | why |
|---|---|---|
| 1 | Split false positives into "in the note but unbilled" vs "made up" | precision on Inpatient notes is capped near 0.15 by annotation scope, not by model quality |
| 2 | Two output files: counts, and terms | extracted terms are patient-derived text; counts alone are not |
| 3 | Generation cap 512, not 1024 | the medication run's cap hits were a repetition loop, not length; verified on the smoke run |
| 4 | Score against `mdace_gold_evidence_text`, not `gold_code_description` | the two match in only 4.5% of rows; the note says "depression", the catalogue says "Major depressive disorder, single episode, unspecified" |
| 5 | Report code-level recall alongside term-level recall | codes are the business outcome, and this is the ceiling for the experiment-2 lookup |
| 6 | Prompt asks for conditions **and** procedures/injuries/status | 16% of the answer key is not a plain disease; a conditions-only prompt caps recall at 0.84 |
| 7 | Run both note sets in one pass, score separately | Ehtesham Bhai's 24-note sample and the 50-note stratified sample overlap by only 1 note |

---

# DECISIONS I NEED FROM YOU

## Background you need for all three

**What this evaluation actually measures.** MedGemma reads a clinical note and lists the
medical conditions it finds. We compare that list against the "gold" list — the reference
answer, here the exact phrases human medical coders highlighted in that note.

**The domain words.** Hospitals bill insurers using standardized codes. *ICD-10-CM* codes
diagnoses and conditions — 93% of our data. *CPT* codes procedures and services a
clinician performed; *ICD-10-PCS* codes inpatient procedures; *SNOMED* is a separate
medical vocabulary used for concept mapping, not billing. The MDACE dataset is real
hospital notes where coders highlighted the exact phrase justifying each code they billed
— that highlighted phrase is our target. *Profee* ("professional fee") is billing for an
individual doctor's work: short notes, median 148 words. *Inpatient* is billing for the
hospital stay itself: long notes, median 1,113 words.

**The one fact that drives everything below:** coders highlighted evidence only for codes
that were **actually billed** — not for every condition mentioned in the note. A note can
describe a condition in detail and have no highlight on it, simply because nobody billed
for it.

**Precision, recall, F1.** The model produces a list of terms; the gold is a list of terms.

- **Recall** = of the gold terms, what fraction the model found. "How much of the right
  answer did it get." Missing things hurts recall.
- **Precision** = of the terms the model listed, what fraction were in the gold list.
  "How much of what it said was right." Over-listing hurts precision.
- **F1** = a single blended score of the two. It's the harmonic mean, which means it sits
  closer to whichever of the two is worse — so you can't earn a good F1 by being excellent
  at one and terrible at the other.
- **Micro** averaging (what the spec asks for) = pool every note's hits and misses into one
  big pile, then compute the ratio once. The alternative, *macro*, scores each note
  separately and averages those scores; that would let a note with 2 gold terms count as
  much as one with 30. Micro is the right choice here.

All three run 0 to 1, higher is better.

---

## Decision 1 — Split false positives into two kinds

**The concept:** a **false positive** is a term the model listed that isn't in the gold
list. Normally that means the model was wrong. Here, usually it doesn't.

**The problem, concretely.** An Inpatient note is ~1,113 words but has only ~6 gold terms,
because only 6 things got billed. Suppose the model reads it and correctly lists 40 real
conditions that are genuinely written in that note. Six match gold; the other 34 score as
false positives. Precision = 6/40 = 0.15, F1 ≈ 0.26. That is the score a *perfect*
extractor gets. It is arithmetic about what was annotated, not a measurement of MedGemma.

**What I want to do.** For every false positive, check whether that phrase actually appears
in the note text (a plain substring check after normalization). That splits them:

- **(a) phrase is in the note** — the model found a real condition that simply wasn't
  billed. An artifact of annotation scope.
- **(b) phrase is not in the note** — the model made it up. This is **hallucination**:
  output not grounded in the input. A real model error.

I'd also report the **precision ceiling** — gold count divided by predicted count — which
is the best precision anyone could achieve given how many terms the model chose to list.

**Why it matters.** It turns the caveat from a sentence people skip into a number they
can't skip. And bucket (b) is a hallucination rate, which is a real model-quality signal
that isn't distorted by the billing-scope problem at all. When your lead asks "is it wrong,
or is it right about things nobody billed?" — that's the number that answers him.

**One thing I will deliberately not do:** I won't remove bucket (a) before computing
precision. Deleting your own false positives raises precision by definition and makes the
number meaningless. Measured and reported next to the real score, never folded into it.

**If we don't:** the Inpatient table reads F1 ≈ 0.25 with a footnote, someone screenshots
it as "MedGemma fails on inpatient notes," and you have no evidence to push back with.

**Default if you say "go":** I do it.

---

## Decision 2 — Two output files instead of one

**Here I'm contradicting an instruction, and I want to be direct about it.** You asked for
the per-note results file to be "PHI-free (counts and term lists only)." *PHI* is protected
health information — patient data that's legally restricted. Those two halves conflict: the
extracted terms **are** patient-derived clinical text, phrases copied verbatim out of real
notes. A file containing them is not PHI-free in the strict sense, even though it holds no
free-form note prose.

Nothing leaks either way — the whole output directory is already excluded from git, and I'm
not proposing any change there.

**What I want:** two files side by side. One holds only integer counts, genuinely free of
patient text, so you can open it, paste from it, or share a number without stopping to
think. The other holds the per-note term lists for the experiment-2 lookup. Same directory,
same resume behavior, and experiment 2 still gets everything it needs without a rerun.

**If we don't:** one file you must treat as patient data every single time you touch it,
including when you just want a count.

**Default if you say "go":** two files. Say the word if you'd rather have one — it's a
trivial difference.

---

## Decision 3 — Shorten the generation limit from 1024 to 512

**The concept.** `max_new_tokens` caps how many tokens (word-fragments) the model may
produce in a single answer. If it hits the cap, the answer is chopped off mid-sentence — in
our case mid-JSON, so the reply often can't be read at all and that chunk contributes
nothing. A bigger cap is safer against truncation but slower, because generation time
scales with how much text is produced.

**The evidence is already in the existing config.** The medication run hit the cap on *the
same 5 of 21 chunks* at both 1024 and 1536. A limit that binds identically at two very
different values isn't a length problem — the model is stuck in a **repetition loop**,
emitting the same item over and over until something cuts it off. Raising the cap just buys
more of the loop.

**Why 512 here.** That run asked for six kinds of medication detail and expected ~30
entities per chunk. This one asks for conditions only and expects ~6 gold terms per note.
512 tokens is roughly ten times the expected output — generous. It halves worst-case
generation time and cuts repetition loops off sooner. The 1.6h estimate is based on 1024,
so the real run may well come in under that.

**The risk, and how I cover it.** If I've misjudged output length, replies get truncated
and recall drops — the exact failure I'm trying to avoid. The harness already counts how
many chunks hit the cap. I run the 5-note smoke test first, check that count is zero, and
only start the full run if it is. If it isn't, I go back to 1024 and report back before
spending the GPU time.

**If we don't:** longer run, and truncation-by-repetition-loop stays exactly as likely as
it was in the medication eval.

**Default if you say "go":** 512, verified on the smoke run before the real one.

---

## If you just reply "go"

All three defaults. Then build in this order: data loading and the sampler (with the
reproducibility test), then the prompt and reply parser, then scoring, then the runner.
Stop after the 5-note smoke test, report the cap-hit count and what the model's replies
actually look like, and only then start the 50-note run.

---

# FYI, NO ACTION NEEDED

## What was checked before designing anything

The fact sheet was accurate. These are three things it didn't cover that had to be settled.

**1. `chart_type` is a property of each annotation row, not of the note.** Each row in the
annotation file is one billing code plus its evidence phrase, and each row carries its own
chart_type. 118 of the 1,074 notes have both Profee and Inpatient rows — the same note was
billed both ways. So "is this a Profee note or an Inpatient note?" needed a rule, and
different rules give different groups.

The rule "if any row on this note is Inpatient, call the note Inpatient" reproduces the
fact sheet's numbers exactly — 604 Inpatient / 470 Profee, medians 1,108 and 148 words, 6
and 4 gold terms. That's a strong signal it's the rule behind the fact sheet, so it's what
will be used, with a comment explaining why.

**2. The "122 chunks / 324 gold terms" figure points to one specific set of 50 notes, and
it was located.**

*Seed:* random selection in code is actually deterministic — you give it a starting number
(the seed) and the same seed with the same code always picks the same items. *Stratified
sample:* rather than picking 50 notes at random (which by bad luck could hand you 44 Profee
and 6 Inpatient), pick 25 from each group separately, so both are guaranteed represented.

The subtlety that bit here: it isn't enough to fix the seed — the *order and grouping* of
the draws changes the result too. With one random generator drawing the 25 Profee notes
first, the Inpatient draw continues from wherever the Profee draw left the generator. With
a fresh generator per group, both start from the same place and produce a different set of
Inpatient notes. Same seed, same code, different sample.

- One generator, Profee drawn first → **122 chunks, 324 gold terms** ← matches exactly
- Fresh generator per group → 167 chunks, 432 gold terms

So the first is what was measured. That gets locked in, plus a test that fails loudly if
anyone changes it, because this is precisely the kind of thing that drifts silently in a
refactor and quietly invalidates a comparison. The split is Profee 28 chunks / 110 gold
terms, Inpatient 94 chunks / 214 gold terms.

*Chunk:* MedGemma can't read a 1,113-word note and reliably list everything in one pass, so
long notes are cut into overlapping 400-word windows — 80 words of overlap so a phrase
sitting on a boundary isn't lost — and each window is a separate model call. 122 chunks
means 122 calls, which is the real unit of runtime. Most Profee notes fit in a single
window; the long Inpatient notes are why that side is 94 rather than 25.

**3. The normalizer reproduces 7,663 exactly.** *Normalization* means flattening away
differences that shouldn't count, so "COPD," and "copd" are treated as the same term:
lowercase everything, drop punctuation, collapse all whitespace including line breaks. The
fact sheet says the 9,499 rows collapse to 7,663 distinct (note, term) pairs. Mine gives
exactly 7,663, which means the normalizer matches the one behind those numbers rather than
merely resembling it. No term normalizes to an empty string.

## Module layout

Below is what gets built. Everything is a new file — no existing file is edited, and no
existing test changes, so the medication evaluation cannot break as a side effect.

| new file | what it does |
|---|---|
| `src/mdace_config.py` | all the settings in one place: where the data lives, seed 13, 25+25 split, 400/80 chunking, generation settings, output location |
| `src/datasets/mdace.py` | reads and joins the two data files, decides each note's group, normalizes terms, builds the gold term sets, draws the sample |
| `src/build_mdace_sample.py` | writes the 50-note sample out so the GPU run reads one small file |
| `src/prompt_mdace.py` | the new conditions/diagnoses prompt, plus the reply reader |
| `src/term_scoring.py` | the set comparison and the precision/recall/F1 maths, per group |
| `src/report_mdace.py` | the written report, Profee and Inpatient tables kept apart |
| `src/evaluate_mdace.py` | the runner, including resume-after-disconnect |
| `tests/test_mdace_*.py` | tests that run without a GPU |

Reused as-is: the chunking code and the model loading/calling code.

**Deliberately not reused, and this is worth being able to explain out loud** — it's the
main structural difference from the medication evaluation.

That eval scores *positions*. It has to locate where in the note each predicted phrase
sits, label the note word by word using **BIO tagging** (Beginning / Inside / Outside — the
standard way to mark multi-word entities across individual words), and score with
**seqeval**, the standard scorer for that format. That approach created a genuinely nasty
problem: if the model says "aspirin" once and "aspirin" appears 15 times in the note, is
that 1 prediction or 15? Answering it took a whole investigation and left behind a
three-way configuration knob.

This eval scores *sets of terms*. Position never enters the picture — the only question is
"does the phrase 'atrial fibrillation' appear anywhere in what the model produced for this
note," yes or no. The fact about 4,770 gold terms appearing more than once in their own
note is exactly why that matters: under position scoring it's a problem that needs a
policy, and under set scoring it evaporates. So the alignment and seqeval code isn't used
at all. The instruction to score terms rather than positions is what buys that, and it's
the cleanest thing in the design.

## Per-note flow

Note text → cut into overlapping 400-word windows → for each window: prompt the model, read
the reply, collect terms → merge all the windows' terms into one set for the note → compare
that set against the gold set.

## What gets written

Everything below lands in the run directory, which is excluded from git. Results are
appended note by note as each finishes, and a rerun skips notes already done — so a Colab
disconnect costs only the note in flight.

- counts file — integers only, safe to open and share
- terms file — the per-note extracted term list, for the experiment-2 lookup
- errors file — the missed gold terms and the false positives, split by the
  grounded/hallucinated check from Decision 1
- the report and metrics — aggregate only, safe to commit

## Remaining points

**Strict matching will miss a long tail, and that's the right call for run 1.** Strict exact
means that after normalization, the model's phrase must match the gold phrase character for
character. 88% of gold terms are 1–3 words and will behave fine. The remaining 12% are
longer clause-like snippets a model won't reproduce word-perfectly, so those will score as
misses even where the model substantially got it right. The alternative is **fuzzy
matching** — counting near-misses as hits, either by string similarity or by comparing
meaning numerically rather than characters. Strict first is right, because fuzzy matching
has thresholds to set, and setting them before seeing real errors is guessing. The errors
file gives the actual missed phrases so that call is evidence-based after run 1 rather than
a guess before it.

**These sample sizes are small — here's how much to trust them.** Profee recall rests on
110 gold terms and Inpatient on 214. As a rough guide, a rate measured on ~110 items
carries about ±9 percentage points of sampling noise, and on ~214 about ±7. In practice: if
Profee recall comes out 0.62 and Inpatient 0.58, that gap means nothing and won't be
reported as a difference — the two would need to differ by roughly 15 points or more before
claiming they genuinely behave differently. Same caution on the headline number: recall
over 324 terms is a real signal but not a precise one, so it should be said as "around
0.6," never "0.61." The counts get printed beside every rate so this stays visible instead
of being something you have to hold in your head.

**The 118 mixed notes slightly blur the Profee-vs-Inpatient comparison.** They're scored as
Inpatient, but their gold terms include some Profee-billed evidence. The report will say
how many of the 25 sampled Inpatient notes are mixed, so if that number turns out to be
large you know the contrast between the two groups is muddier than the table makes it look.

**Minor:** note IDs in this dataset are plain integers, unlike the string form the
medication eval uses. Changes nothing, just don't be surprised the two run directories look
different.

---

# Round 2 — decisions 4 to 7

Added after reviewing Ehtesham Bhai's original requirement message.

## Decision 4 — Score against the note's own words, not the code description

Ehtesham Bhai's message describes `descr` as the "expected NER from doc notes." That is
not correct as a scoring target, and scoring against it would have made the evaluation
measure nothing.

Two different columns exist. `mdace_gold_evidence_text` is the phrase the human coder
highlighted *in the note*. `gold_code_description` is the official ICD-10-CM catalogue
wording, written by a standards body rather than by the treating clinician.

| what the note says | `gold_code_description` |
|---|---|
| `depression` | Major depressive disorder, single episode, unspecified |
| `HTN` | Essential (primary) hypertension |
| `HCV` | Unspecified viral hepatitis C without hepatic coma |
| `Diabetes` | Type 2 diabetes mellitus without complications |
| `asthma` | Unspecified asthma, uncomplicated |
| `Biliary obstruction` | Obstruction of bile duct |

The two match word-for-word in **424 of 9,499 rows — 4.5%**. Scored against `descr`,
MedGemma would read "depression", output "depression", and be marked wrong. Recall would
approach zero for reasons unrelated to the model.

**Resolved:** score against `mdace_gold_evidence_text`. The 4.5% figure goes in the report
so the choice does not have to be re-argued.

## Decision 5 — Report code-level recall as well as term-level recall

Two distinct questions, and they differ because one phrase can justify several codes:

- **Term recall** — of the phrases coders highlighted, how many did the model find? This
  measures the NER step in isolation.
- **Code recall** — of the codes actually billed, how many had at least one of their
  evidence phrases found? This measures the business outcome.

Code recall costs nothing to compute — the gold data already links each phrase to its
code, so no lookup system is required. It is the honest **ceiling for experiment 2**: the
lookup can never retrieve a code whose evidence phrase was never extracted.

**Resolved:** report both, per group.

## Decision 6 — The prompt must ask for more than conditions

What the answer key is actually made of:

| share | what it is |
|---|---|
| 84.0% | real diseases and conditions |
| 5.5% | ICD-10-CM Z-codes: status and history (smoking history, long-term drug therapy) |
| 4.0% | injury and poisoning |
| 3.6% | CPT — a procedure or service performed by a clinician |
| 2.9% | ICD-10-PCS — an inpatient procedure |

16% of the answer key is not a plain disease. A prompt saying "extract diagnoses and
conditions" caps recall at 0.84 before the model does anything, and that cap would read as
a model weakness in the report.

**Resolved:** broaden the prompt to conditions, procedures, injuries, and relevant
history/medication status — and report recall separately for the 84% disease slice as well
as overall, so both the clean "conditions" number and the honest total are available.

Cost: more extracted terms, so precision falls further on Inpatient notes. Decision 1's
false-positive split is what keeps that readable.

## Decision 7 — Run both note sets, one pass, scored separately

Ehtesham Bhai's `sample_100` file and the 50-note stratified sample are almost disjoint:

| | notes | chunks (model calls) |
|---|---|---|
| his `sample_100` | 24 | 82 |
| stratified sample | 50 | 122 |
| overlap | **1 note** | — |
| union (what actually runs) | 73 | 202 |

Running both costs 202 model calls against 122 for the stratified sample alone — roughly
2.7 hours instead of 1.6 on a free Colab T4. Deduplicating the overlap saves only 2 calls.

**Resolved:** one pass over the 73-note union, with three scored views produced from the
same predictions at no extra GPU cost.

### His shipped file is an incomplete answer key — this is the important part

For his 24 notes, the full dataset holds **195** distinct gold terms. His `sample_100` file
ships only **99** of them. The missing 96 are real billed evidence for those same notes,
left out because the file was cut to 100 annotation rows rather than to whole notes.

Scored against his file alone, MedGemma would extract terms that genuinely *are* billed
gold, find them absent from the answer key, and be penalised for being right. Precision
would be understated by roughly a factor of two.

So the three views are:

1. **His 24 notes, scored against his shipped 99 terms** — comparable with any number he
   has already computed himself.
2. **His 24 notes, scored against all 195 gold terms** — the correct answer key.
3. **The 50-note stratified sample** — the headline result, Profee and Inpatient separate.

The gap between views 1 and 2 is itself worth reporting: it quantifies how much the
truncated sample file distorts the picture.

### His sample is also skewed toward the unreadable case

His 24 notes are **17 Inpatient / 7 Profee**. Inpatient is exactly the group where
precision is bounded by billing scope rather than model quality (Decision 1), so a
precision number computed on his sample will look worse than the model deserves. And 7
Profee notes is far too few to read anything into — treat any Profee rate from his sample
as indicative only.

The stratified sample (25 / 25) exists precisely to avoid this.
