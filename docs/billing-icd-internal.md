# Pediatric billing ICD-code evaluation — internal decision log

Started 2026-08-27. Branch `billing-icd-eval`, cut from `mdace-recall-benchmark`.

Internal. Reasoning, dead ends, and the things that should be said out loud
before any number from this leaves the building.

---

## The ask, verbatim

Ehtesham Bhai, 2026-08-27, with four PDFs in `../ai-medical-billing/`:

> hi, these are some real data - notes and expected billing codes. can u run ur
> megemma model to check final ICD code prediction - precision and recall
> Expected answer under Assessment. Make sure to remove that before sending to
> model

Four things follow from those three sentences, and only the first one is
obvious.

1. **The metric is precision and recall on ICD codes.** Not phrases.
2. **Gold is the Assessment block.** Specifically the `DX n:` lines in it.
3. **The Assessment must be removed from the input.** He is right that it has to
   go, and he is describing a leak. He has not seen the second one — see
   "Removing the Assessment is not enough".
4. **This repo could not do it.** Everything built so far extracts phrases. That
   gap is the first thing this branch closes.

---

## Decision 1 — build code assignment, do not adapt the phrase extractor

**The gap.** `evaluate_mdace.py` writes `extracted_terms.jsonl` "for the
downstream term→ICD lookup". That lookup does not exist. It was always going to
be someone else's step, and the recall benchmark measured *how many billed
phrases MedGemma recovers* precisely because the code half was out of scope.

So there were two ways to answer the ask:

- **(a) Build the missing lookup.** Extract phrases with the existing prompt,
  then map phrase → ICD-10 code. This needs a code catalogue, a matching
  strategy, and a decision about what to do when one phrase maps to six codes
  that differ only by laterality and episode. It is the real product shape, and
  it is at minimum a week.
- **(b) Ask the model for the code directly.** One prompt, one call per note.

**Chose (b).** Not because it is better — because it answers the question that
was asked, today, and because (a)'s hard part is the catalogue rather than the
model, so (b) measures the thing under test more cleanly anyway. If MedGemma
cannot name `B08.5` from a note about coxsackie, no lookup layer built on top of
its phrases will rescue that.

What (b) gives up: it cannot say *why* a code was assigned, so a wrong code
cannot be traced back to the phrase that caused it. Accepted. With 16 gold codes
the per-code table is readable in full, which covers most of that need.

**This should be said to Ehtesham Bhai plainly.** The number he gets is
"MedGemma asked directly for codes", not "the pipeline you have been hearing
about, now with codes". Those are different systems and the second one still
does not exist.

---

## Decision 2 — gold is the `DX` lines, and nothing else

The Assessment block has two parts:

```
Assessment
Influenza                                    <- clinician's free-text impression
Right radius fracture                        <- ditto
DX 1: J11.1 Influenza due to unidentified flu virus with other resp manifestations
DX 2: Z68.52 Body mass index pediatric, 5th percentile to < 85% for age
...
```

The `DX` lines are what gets submitted on the claim. The free-text lines are the
clinician thinking out loud; they are not codes, they do not always appear
(note 26819 has none), and scoring them would be scoring a different task.

**The code is scored, the description is not.** The description is the ICD
catalogue's wording for that code, so a correct code implies a correct
description. Scoring both would double-count one decision. The description is
still *requested* from the model — it makes a wrong answer readable (a code
carrying the description "Influenza" means the model found the condition and
missed a digit, which is a completely different failure from an invented code)
and it costs ~10 tokens against a 512-token budget.

**CPT is out of scope.** 99213/99214/99394 sit under Procedures. He asked about
ICD. They are left in the input — a real chart has them and they leak nothing —
but a model that returns one is counted as a false positive, not filtered out.
See decision 6.

**The duplicate is real.** Note 96176 lists `Z68.51` as DX 3 *and* DX 4. It is a
data-entry duplicate. Deduplicated to 3 codes for that note, 16 unique overall
against 17 lines. If this is not deduplicated, recall is capped below 1.0 for a
reason that has nothing to do with the model.

---

## Decision 3 — removing the Assessment is not enough

This is the finding of the day and it is not in the ask.

Note 26819, in the Problem List, inside Patient History:

```
- J30.2 OTHER SEASONAL ALLERGIC RHINITIS
- L20.9 ATOPIC DERMATITIS, UNSPECIFIED
- UNDERACHIEVEMENT IN SCHOOL
- ATTENTION AND CONCENTRATION DEFICIT
```

`J30.2` and `L20.9` are gold codes for that note, printed with the code string
itself. `UNDERACHIEVEMENT IN SCHOOL` and `ATTENTION AND CONCENTRATION DEFICIT`
are the descriptions of `Z55.3` and `R41.840`, word for word. That is four of
that note's six gold codes visible in the input after the Assessment is cut.

Notes 55688 (`SEASONAL ALLERGIC RHINITIS`, `WHEEZING`) and 112976
(`HEMANGIOMA`) leak the same way, one and two descriptions respectively.

**Not patched silently.** A real coder does read the Problem List — it is part of
the chart and it is legitimately how a chronic problem gets carried onto a
claim. Deciding unilaterally that it is contamination would be substituting our
judgement for the coder's. So it is measured instead: `assessment_cut` is what
he asked for, `leakage_cut` is the same run without the Problem List, and both
numbers get reported with the difference between them stated.

**Prediction, recorded before the run:** `assessment_cut` scores meaningfully
higher than `leakage_cut`, and most of the gap is note 26819. If the two come
back identical, the model is not using the Problem List at all, which is worth
knowing for a different reason.

---

## Decision 4 — one prompt, three inputs

The recall branch ran a prompt A/B (`scoped` vs `billable`) because the question
there was which framing extracted better. Here the experiment is on the **input**.

Three prompts against one input would confound "the model got better" with "the
prompt got better". Three inputs against one prompt makes every difference
between the numbers attributable to the text that was removed. That is the whole
design and it is why `prompt_billing.py` has no `VARIANTS` dict.

`full` — nothing removed, `DX` lines still on the page — is a **harness check,
not a result**, and the notebook says so twice. The model is being asked to read
codes off the page. A low score there means the prompt or the parser is wrong,
and the other two numbers would be measuring the harness rather than the model.
Run it first.

That is two harness checks, not one:

| check | cost | what it rules out |
|---|---|---|
| `--oracle` | ~1 s, no GPU | parser or scorer cannot reproduce gold from gold |
| variant `full` | 4 calls | prompt or parser cannot survive the model |

Both exist because the equivalent check caught real bugs twice on the previous
branches, and both times the broken version returned a plausible-looking number
instead of an error.

---

## Decision 5 — the prompt names the criterion, and that is not a hint

The prompt says: code what the clinician **assessed and addressed at this
visit**.

That looks like it might be leaning on the scale. It is not, and the alternative
is worse. Gold averages 4 codes per note. A note mentions far more codeable
things than that — note 26819's past medical history alone names asthma, a heart
murmur, constipation and enuresis, none of them billed for this visit. A prompt
saying "extract every codeable finding" would answer a different question and
score precision near zero for a reason that is the prompt's fault.

The framing names the task. It says nothing about which conditions, how many, or
where in the note to look. The negative example in the prompt is synthetic
(a made-up otitis media visit) and demonstrates the one distinction that matters:
the condition treated today is coded, the eczema in the history is not.

**Rule 4 says "most visits have between one and six".** That is the closest thing
to a thumb on the scale in the whole prompt, and it is deliberate: the recall
branch measured this model extracting 15–17× gold volume and looping until it
hit the token cap. Without a sense of scale a 4B model will list thirty codes and
precision becomes a measure of its verbosity. It is a range, not a cap, and it is
stated here so it can be argued with.

---

## Decision 6 — a malformed code is a false positive, never a dropped row

`parse_codes` is permissive about **shape** — bare strings, wrong container key,
wrong field name, code and description crammed into one field — because losing a
real prediction to a formatting quirk understates precision and recall at once.

It is strict about **content** in exactly one way: a returned code that is not
shaped like ICD-10-CM (a CPT code like `99213`, an invented string, a
description with no code) is **kept and flagged**, never dropped.

This is the one bug in this branch that would have produced a *better-looking*
number rather than a crash. Dropping a returned `99213` deletes a false positive
and raises precision for free. `test_malformed_code_is_kept_as_a_false_positive`
exists for that and says so.

---

## Decision 7 — micro-average, and print the whole answer key anyway

Micro, not macro: pooling the counts then dividing answers "what share of billed
codes came back". Macro would weight note 112976's two codes as heavily as note
26819's six, which is not the question.

But the aggregate is **not** the headline. 16 gold codes means one code is 6.25
recall points — wider than most differences anyone would want to report. So the
report prints every gold code with its per-variant hit/miss. At this size the
whole answer key fits on one screen and reading it is strictly more informative
than reading its mean.

Three of the 16 are not extraction problems and the report separates them:

- `Z68.51` / `Z68.52` (3 codes) — BMI-percentile. The note prints
  `BMI: 17.8 (24 %ile)`; the code depends on knowing the 24th percentile maps to
  `Z68.52`. A code-book lookup, not reading.
- `Z00.121` (1 code) — needs "well visit *with abnormal findings*", a judgement
  about the visit type rather than about a condition.

If the misses concentrate in those four, the story is "reads the notes fine,
fails at code-book mechanics", which is a very different report — and a very
different fix — from "missed the influenza".

---

## Decision 8 — the patient banner stays, the page furniture goes

Each of the three pages repeats the clinic address, the report title, the
patient's name/sex/DOB, the visit date, a `Generated ... Page N of M` footer and
a Connexin copyright.

The furniture is noise and is dropped. The banner is **not** noise: `Z68.5x` is
*pediatric* BMI-for-age and `Z00.121` is a *child* health exam, so sex and date
of birth are load-bearing for the codes actually in gold. Dropping the banner
would remove information a coder genuinely uses and would depress recall for a
preprocessing reason.

Kept on first occurrence, repeats dropped. The model sees it once.

Same reasoning, different direction, for the Problem List: it is a **sub-block
inside Patient History**, not a section of its own, so `leakage_cut` removes the
block and stops at the next `... Reviewed by` marker. Removing the whole Patient
History section would have taken the past medical, family and social history with
it — none of which leaks, all of which a coder reads.

---

## What is deliberately absent

- **No chunking.** The longest note is 1,163 words and fits in one prompt.
  Windowing would reintroduce the pooling and double-counting problems that cost
  real time on the recall branch, in exchange for nothing.
- **No fuzzy matching ladder.** `B08.5` either equals `B08.5` or it does not.
  `recall_matching.py` does not apply here and importing it would be cargo cult.
- **No section-drop experiment.** Ruled out on the recall branch; nothing here
  changes that, and these notes are short enough that input budget is not
  scarce.
- **No fine-tuning, no few-shot beyond the one synthetic example.** Zero-shot is
  the baseline being asked for.
- **No CPT/E&M evaluation.** He asked about ICD. The Procedures block is left in
  the input but is not scored. It is the obvious next question if he asks.

---

## Results, 2026-08-27

Four configurations: two prompts x two decoding configs. **Recall on
`leakage_cut` is 0 of 16 in every one of them.**

| prompt | penalty | full | assessment_cut | leakage_cut |
|---|---|---|---|---|
| `e78d3dd8` ear example | off | P .9412 / R 1.0000 | P .0625 / R .1250 | P 0 / R 0 |
| `e78d3dd8` ear example | 1.15 | P .9412 / R 1.0000 | P 0 / R 0 | P 0 / R 0 |
| `4af27169` skin example | off | P .3111 / R .8750 | P .0175 / R .0625 | P 0 / R 0 |
| `4af27169` skin example | 1.15 | P .7778 / R .8750 | P .1000 / R .0625 | P 0 / R 0 |

**The harness is proven.** `full` reached 16/16 recall with one false positive
under the ear prompt: given a note with `DX 1: B08.5` printed on it, the pipeline
scores a perfect answer. Nothing downstream of the model is hiding a result.

**The two codes ever recovered without the Assessment were printed in the input.**
`assessment_cut` peaked at 2 of 16, both in note 26819, both `J30.2` and `L20.9`
— the two codes its Problem List spells out in full. Remove the Problem List and
it is 0 of 16. The model copied; it did not derive.

**Precision is not worth quoting.** It ranges 0.00 to 0.94 across configurations,
driven entirely by how badly the model looped on a given run rather than by
anything about coding. Recall is the number that survives.

---

## Three bugs found by running it, in the order they were found

Each one produced a plausible-looking number rather than an error, which is the
only reason they needed finding rather than noticing.

**1. A truncated reply scored as one code.** A reply cut off mid-array leaves the
outer object unclosed; `_find_json_object` returns the first *balanced* brace,
which is the first code object, and it parses cleanly as a single unwrapped code.
6 of 12 replies truncated and every one scored exactly one prediction.
`leakage_cut` read 0.0000 from four notes whose answers were never read. Fixed by
walking every balanced object and advancing past unbalanced ones.

**2. Generation arguments never reached `generate`.** `ImageTextToTextPipeline`
routes exactly four things to the model: `max_new_tokens`, `generate_kwargs`, and
two return-shape flags. Everything else goes to the *processor* and is dropped.
The repetition-penalty A/B came back byte-identical — same counts, same false
positives, same order, one code out of sequence in both — because the penalty had
never been applied. `do_sample=False` had been dropped the same way since
`model.py` was written, which affects the MDACE and recall branches too; their
outputs were deterministic regardless, so this appears to have cost nothing
there, but it was luck rather than the flag working.

**3. The prompt's example contaminated the answers, in both directions.** The
first example was an otitis media visit coded `H66.001`; note 112976 opens with
"Ear infection, fever, ear pulling" and `H66.001` came back as a false positive
on it in every configuration. The *shape* leaked too: 10 of 16 false positives
under the penalty carried three digits after the decimal point against 2 of 16
gold codes shaped that way.

Replacing it with two skin conditions moved the contamination rather than
removing it — `B35.4` then appeared as a false positive on note 26819, and note
112976's harness check collapsed from 2/2 to 0/2 with 31 predictions. **A 4B
model copies whatever example it is given.** `tests/test_billing_prompt_hygiene.py`
now asserts the example shares no term, code or ICD category with the corpus, but
no test can stop the model from copying something.

---

## What this does and does not license saying

**Says:** MedGemma-4B, zero-shot, recovered none of the 16 billed codes from
these four notes once the answer was not in the input. That is robust to the
token cap, to a repetition penalty, and to a full prompt rewrite.

**Does not say:** that MedGemma cannot code. 16 codes is a spot check; one code
is 6.25 recall points. It does not speak to the 27B model, to a fine-tune, to a
retrieval step over an ICD catalogue, or to any note type other than pediatric
outpatient.

**The obvious next question, unanswered:** the phrase-extraction work on MDACE
recovered 78 of 100 billed phrases. This run says the code half is where it
breaks. A term to code lookup over a real ICD catalogue is the architecture that
follows from both results, and it is still not built.

---

## What Ehtesham Bhai has been told, and what he has not

**Told, 2026-08-27:** the headline (0 of 16), the Problem List leak and what the
two recovered codes were, the harness check at 16/16, the four configurations,
and the sample-size caveat. Also the `Z68.51` duplicate in note 96176, as
something to check on their side.

**Not told, and not currently worth a message:**

1. The three harness bugs above. They are the reason today took as long as it
   did, and the numbers are only trustworthy because they were found — but the
   final numbers do not depend on knowing that.
2. That the repo could not answer his question at all before today; the
   phrase to code step never existed.
3. That three of the 16 gold codes are BMI-percentile lookups and one is a
   visit-type judgement, so even a working model needs a code-book step.
4. That the notes are not de-identified and this ran on Colab.

Items 3 and 4 should go out if he asks for next steps or for more data.

Related: [`recall-benchmark-internal.md`](recall-benchmark-internal.md),
[`precision-plan-internal.md`](precision-plan-internal.md).
