# MedGemma recall benchmark — MDACE billing evidence

Model **medgemma-4b-it-ORACLE** (`google/medgemma-4b-it`), 4-bit, greedy decoding, `max_new_tokens=1024`, prompt variant `billable`. **One-shot, not zero-shot:** the prompt carries a single synthetic worked example containing no MDACE content. It is not the thing under test — the comments in `prompt_recall` record that abstract prohibitions failed on a 4B model where a demonstrated one worked — but it is an example, and calling the run zero-shot would be wrong. Chunking 400 words / 80 overlap. 24 notes, 82 chunks.
Prompt `1d16939a`, run `medgemma-4b-it-ORACLE_cw400_ov80_mnt1024_p1d16939a`.

**ORACLE RUN — no model involved.** The gold accept-sets were fed back through the pipeline to check the harness. Every source below must read 1.0000 at L1 and the combined line must show zero false positives; anything less is a bug in chunking, normalization or matching, not a result.

---
## What is being measured

MedGemma reads a clinical note and lists the findings in it. That list is compared against what human medical coders recorded as the justification for the billing codes they submitted. **Recall is the metric**: how much of the billed evidence the model recovers. Precision is secondary and appears here as an explicit false-positive count rather than as a ratio.

The input is one file: 100 annotation rows on 24 notes, carrying 91 distinct `(note, code system, code)` triples. Note text is embedded in that file, so there is no join and no separate notes file. Gold is whatever that file says it is.

### The accept-set

Scoring against the note's own wording alone made `HTN` unable to match `Essential (primary) hypertension` however right the model was. So per billed code the accept-set is the **union of three columns**: the evidence text the coder highlighted, the ICD code description, and every SNOMED concept term the file ships for that code. A prediction matching any of them recalls that row.

That gives a median of 4 accepted forms per row (min 2, max 5), against 1 under evidence-text-only scoring.

### The model's side

Each finding carries two fields: `span`, the phrase as written in the note, and `name`, the standard clinical name. Either may match the accept-set. MedGemma already knows the expansions, and using its own expansion beats making the matcher infer one. `span` is also what the not-in-note check below is run against.
---
## The matching ladder

Each level is a **superset** of the one above, so recall is monotonically non-decreasing and the gain at each level is attributable to that level. Read the jumps, not just the top row.

Matching is one prediction to at most one gold form. Without that rule a single vague prediction satisfies several gold entries at once and recall measures vagueness.

| level | rule | rows recalled | codes recalled | accepted forms matched | false positives | FP rate |
|---|---|---|---|---|---|---|
| **L1** | exact after normalization | 100/100 1.0000 (96.3%–100.0%) | 91/91 1.0000 | 318/318 1.0000 | 0 | 0.0000 |
| **L2** | whole-token containment, either direction | 100/100 1.0000 (96.3%–100.0%) | 91/91 1.0000 | 318/318 1.0000 | 0 | 0.0000 |
| **L3** | token-set Dice and/or difflib ratio, thresholded | 100/100 1.0000 (96.3%–100.0%) | 91/91 1.0000 | 318/318 1.0000 | 0 | 0.0000 |

**Quote the rows or codes column, not the forms column.** A row is recalled by matching any ONE of its accepted forms, so a model that recalls every row while producing a single phrasing each still leaves most forms unmatched — the forms column has a ceiling set by how many phrasings the model emits, not by how much it found. It is here because the per-source breakdown below is measured in forms and the two must reconcile.

- **L2** added 0 rows and 0 accepted forms over the level above it.
- **L3** added 0 rows and 0 accepted forms over the level above it.

**L5 has not been applied to these figures.** Every level above L1 still includes matches nobody has checked, so **quote L1 as the number that needs no caveat.**

**L3 earned almost nothing at these thresholds** — 0 accepted forms beyond L2. Dice 0.80 and character ratio 0.90 are strict enough that whole-token containment has already taken everything they would reach. That is a finding about the thresholds, not about the model: either loosen them and re-score (`--score-only --dice-min …`, no GPU), or read the ladder as three effective levels rather than four.

**Thresholds are reported, never silently chosen.** Dice ≥ `0.8`, character ratio ≥ `0.9`, cosine ≥ `0.6`.

They come from measured pairs, not from taste. Dice 0.80 keeps *acute kidney injury* against *kidney injury, acute* (1.00) and drops *acute renal failure* against *chronic renal failure* (0.67); character ratio 0.90 keeps *hyperlipidema* against *hyperlipidemia* (0.96) and drops that same acute/chronic pair (0.75). No string threshold separates the good pairs from the bad ones on its own — anything loose enough to catch *CHF* against *congestive heart failure* (0.22) accepts the acute/chronic pair several times over. That is the argument for L4, and for dumping every pair each level newly accepted.

**L2 knowingly admits two bad shapes.** *diabetes* inside *diabetes insipidus* is a false match, and *sepsis* inside *no evidence of sepsis* is a negation. The pairs L2 newly accepted are written to the run directory for exactly this reason; L5 adjudicates them.

**L4 did not run in this report.** The biomedical embedding backend was not available, so the ladder stops at L3 and abbreviations like *CHF* remain unreachable. The recall figures above are correspondingly a floor.
---
## Which of the model's two fields did the work?

Every finding carries two strings: `span`, copied from the note character for character, and `name`, the standard clinical name with abbreviations expanded. Either is allowed to match, which means the headline pools two different abilities — faithful copying and medical vocabulary — into one number. Splitting them is the only way to read the ladder as intended.

| matched on | L1 | L2 | L3 |
|---|---|---|---|
| **span** — the phrase copied from the note | 1.0000 | 1.0000 | 1.0000 |
| **name** — the standard clinical name | 1.0000 | 1.0000 | 1.0000 |
| either field (the headline) | 1.0000 | 1.0000 | 1.0000 |

**The rows do not add up, and that is correct.** One prediction claims at most one gold form, so a finding whose `span` reaches one accepted phrase and whose `name` reaches another still scores a single match. `either` is therefore never the sum of the two rows above it — only never less than the larger of them.

**Read the `span` row as the conservative result.** It is the half that can be verified against the note, since the not-in-note check can only see `span`. A `name` match is the model asserting a vocabulary fact that nothing in the note confirms.

**And read the `span` row at L1 as the closest thing to the old method.** Exact matching on note wording only is what produced the 0.53 this work set out to improve; everything above it is what the accept-set, the ladder and the second field bought.
---
## Recall is never quotable on its own

With loose matching and no volume control, a model that lists every phrase in the note scores near 1.00. A bare recall figure therefore cannot rank MedGemma against a larger model later, which is the entire purpose of this benchmark. These are the numbers that must travel with it.

| | |
|---|---|
| findings per note | 13.2 |
| false positives at L3 | 0 of 318 findings (0.0%) |
| span not found in the note | 0 of 318 checked — 0.0000 (0.0%–1.2%) |
| findings with no span to check | 0 |

The last two lines are the hallucination signal: a span the model claims to have copied from the note that is not in the note. It is the one precision-side number the billing-scope problem does not distort, because it does not depend on what was billed at all.

A finding the model returned with no `span` cannot be checked and is excluded from that denominator rather than counted as clean.
---
## Whose wording does the model produce?

Recall broken out by which column of the answer key was matched. This is the genuinely useful question behind the benchmark: does the model speak in note wording, catalogue wording, or SNOMED wording. Each source is scored by its own independent matching.

### Recall per source, by level

| source | entries | L1 | L2 | L3 |
|---|---|---|---|---|
| all three combined | 318 | 1.0000 | 1.0000 | 1.0000 |
| evidence text (what the note says) | 99 | 1.0000 | 1.0000 | 1.0000 |
| code description (ICD catalogue wording) | 91 | 1.0000 | 1.0000 | 1.0000 |
| SNOMED concept terms | 142 | 1.0000 | 1.0000 | 1.0000 |

Recall per source is unambiguous: of that source's accepted forms, how many were matched.

### Denominators, which differ per source

| source | accepted forms | billed rows reachable | distinct codes reachable |
|---|---|---|---|
| all three combined | 318 | 100 | 91 |
| evidence text (what the note says) | 99 | 100 | 91 |
| code description (ICD catalogue wording) | 91 | 100 | 91 |
| SNOMED concept terms | 142 | 53 | 48 |

SNOMED terms are shipped for only some rows, so a SNOMED recall computed out of all rows would be measuring the file's coverage and calling it model performance. Every denominator is printed for that reason.

### False positives per source, by level

**This table is easy to misread.** Per-source FP means *matched nothing in that source*. A prediction matching only the catalogue wording is a hit on the description line and a false positive on the evidence-text line. So every individual source line reads high, and **only the combined line counts predictions that matched nothing anywhere**. Counts, with the rate over all findings in brackets.

| source | L1 | L2 | L3 |
|---|---|---|---|
| all three combined | 0 (0.00) | 0 (0.00) | 0 (0.00) |
| evidence text (what the note says) | 219 (0.69) | 219 (0.69) | 219 (0.69) |
| code description (ICD catalogue wording) | 227 (0.71) | 227 (0.71) | 227 (0.71) |
| SNOMED concept terms | 176 (0.55) | 176 (0.55) | 176 (0.55) |

### What the L3 false positives actually are

A count on its own cannot say whether a false positive is the model's fault, so it is split three ways. **Only the middle row is model error.**

| | count | share of FPs | |
|---|---|---|---|
| in the note, nothing billed for it | 0 | 0.0% | a correct extraction of something nobody billed |
| **not in the note at all** | **0** | **0.0%** | **the model invented or paraphrased it — real error** |
| no verbatim span to check | 0 | 0.0% | unverifiable either way |

**None of this is subtracted from the false-positive count above.** Removing your own false positives before dividing raises precision by construction and measures nothing. It is reported alongside so the headline FP figure of 0 can be read for what it is: overwhelmingly correct extractions of findings that were never billed, plus 0 genuine mistakes.

The one bucket that would strengthen this is unavailable here. On the previous branch a false positive could also be checked against *other notes from the same hospital admission* — a phrase billed elsewhere in the encounter is demonstrably correct. This file carries 24 notes with no siblings, so that check cannot be run.
---
## Precision, recall and F1 per source (at L3)

| gold source | entries | precision | recall | F1 | best precision possible |
|---|---|---|---|---|---|
| all three combined | 318 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| evidence text (what the note says) | 99 | 0.3113 | 1.0000 | 0.4748 | 0.3113 |
| code description (ICD catalogue wording) | 91 | 0.2862 | 1.0000 | 0.4450 | 0.2862 |
| SNOMED concept terms | 142 | 0.4465 | 1.0000 | 0.6174 | 0.4465 |

**Read the precision and F1 columns as diagnostics, not as quality.** Two separate things push them down for reasons that have nothing to do with the model:

1. **The annotation scope.** MDACE marks evidence only for codes that were actually billed, and a note is full of real conditions nobody billed. 100.0% of this run's false positives are phrases genuinely present in the note. They are counted against precision anyway, because subtracting your own false positives before dividing raises precision by construction and measures nothing.
2. **Per-source false positives mean "matched nothing *in that source*".** A prediction that correctly matches the catalogue wording is a false positive on the evidence-text row. Every single-source row therefore carries almost the whole prediction set as false positives, and only the combined row counts predictions that matched nothing anywhere.

The **best precision possible** column is the arithmetic ceiling given how many findings were produced: with 318 predictions against 318 accepted phrasings, no extractor of any quality exceeds 1.0000 on the combined row. F1 blends precision with recall and sits near the worse of them, so it inherits all of this and should not be quoted as a headline.

**The precision-side number that is not distorted** is the not-in-the-note rate, because it does not depend on what was billed at all. It is in the section above.
---
## What the SNOMED column is, and is not

| | |
|---|---|
| rows with any SNOMED term | 53 of 100 |
| concept terms shipped | 157 of 1894 reported by `gold_snomed_concept_count` — **8.3%** |
| CPT rows with SNOMED | 0 of 15 |
| ICD-10-CM rows with SNOMED | 53 of 70 |
| ICD-10-PCS rows with SNOMED | 0 of 15 |

The list is capped at 3 entries per code and the survivors are not the top-ranked ones — an HCV row ships two pregnancy-related concepts and omits plain *hepatitis C*.

This is stated as **a limit on the number, not a request to anyone to fix the file**. One file was provided and the benchmark works with it. The same applies to the SNOMED lookup itself: the matching ladder approximates it, and that approximation is what L4 is.
---
## Reference numbers

The earlier `mdace-term-ner` run, strict exact matching against evidence text alone, on a different 50-note sample:

| | |
|---|---|
| recall | 0.5278 (324 gold terms) |
| precision | 0.0679 (2,520 predicted) |
| code recall | 165 of 302 = 54.6% |
| terms per note | ~50 |
| independent figure computed separately | ≈0.5 |

**The comparable figure from this run is 1.0000, not 1.0000.** The previous branch matched the model's copied phrasing against the note's own wording and nothing else, which is the `span` row at L1 — not the combined row, which also counts cases where the model's expansion of an abbreviation happened to equal the catalogue name.

1.0000 against 0.5278 is a gap of 47.2 points on samples of 100 and 324, well inside both intervals. **Two independent implementations landing that close is the strongest evidence available that this harness measures what the previous one measured** — and it is what makes the rest of the table readable as a gain rather than a change of yardstick.
---
## How much to trust these numbers

The sample is small. Brackets are 95% Wilson intervals — the range the true rate plausibly sits in. A recall measured over ~100 rows carries roughly ±10 percentage points, so **quote the headline as "around 0.6", never as "0.61"**, and treat a gap of less than about 10 points between two levels as noise.

Generation hit the token cap on **0** of 82 chunks, so no reply was cut off mid-JSON and recall is not understated on that account.
---
## Known limits

- **L5 has not been applied to these figures.** The pairs L2, L3 and L4 newly accepted are written to the run directory and are adjudicated by a separate step (`python -m src.recall_judge`). Until that has run, every level above L1 includes matches nobody has checked — the negation and the *diabetes insipidus* shapes above are real and unquantified. **Quote L1 as the number that needs no caveat.**
- **The SNOMED lookup is approximated, not performed.** The ladder stands in for a real terminology lookup. L4 reaches abbreviations, which is the part that matters most, but it is a similarity model and not a terminology — and the measurement above shows it cannot tell a synonym from a change of acuity.
- **Medications are not excluded by instruction in this run, and the ceiling is unmeasured.** The `billable` prompt carries no medication rule; whether the model still leaves them alone is an emergent consequence of asking only for codeable findings, not something this run establishes. The `scoped` variant excludes them explicitly and therefore caps at about 0.945; this variant may reach higher or leak medications as false positives, and only a per-source audit of the 5.5% of gold evidenced by medications would say which.
- **MIMIC-III only.** MDACE is built on MIMIC-III notes. It shares no notes with the MIMIC-IV medication evaluation in this repo, so the two sets of numbers must not be pooled.
