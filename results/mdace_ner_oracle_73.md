# MedGemma zero-shot term-level NER — MDACE billing evidence

Model **medgemma-4b-it-ORACLE** (`google/medgemma-4b-it`), 4-bit, greedy decoding, `max_new_tokens=1024`.
Chunking 400 words / 80 overlap. Seed 13. 73 notes scored.
Prompt `090a0072`, run `medgemma-4b-it-ORACLE_seed13_cw400_ov80_mnt1024_p090a0072`.

**ORACLE RUN — no model involved.** Gold terms were fed back through the pipeline to check the harness. Recall below must read 1.0000; anything less is a bug in chunking, normalization or parsing, not a model result.

---
## What is being measured

MedGemma reads a clinical note and lists the medical findings in it. That list is compared against the phrases human medical coders highlighted in the same note as justification for the billing codes they submitted.

Scoring is on **term sets**, not positions: per note, gold is the set of normalized highlighted phrases and prediction is the set of normalized phrases the model produced. Normalization lowercases, drops punctuation and collapses all whitespace including newlines. **Recall** is the fraction of highlighted phrases the model found; **precision** is the fraction of the model's phrases that were highlighted; **F1** blends the two and sits nearer the worse of them. All run 0 to 1, higher is better.

Matching is **strict** — after normalization the strings must be identical. A model that answers "hypertension" where the coder highlighted "HTN" scores a miss. Recall here is therefore a floor on real end-to-end usefulness, never an overstatement.
---
## The ladder

Read this table top to bottom as an argument, not as separate results. Every row uses **the same model output** — one inference pass — and changes exactly one thing from the row above it. Higher is better throughout, but the interesting quantity is the *jump between rows*, because each jump isolates the cost of one decision.

| view | what it is | notes | gold terms | terms predicted | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| **A1** | The 100-row sample cut (24 notes), scored as originally asked | 24 | 91 | 195 | 0.0000 | 0.0000 | 0.0000 |
| **A2** | The same 24 notes, scored on the phrases that file ships | 24 | 99 | 195 | 0.5077 | 1.0000 | 0.6735 |
| **B1** | The same 24 notes, corrected method | 24 | 195 | 195 | 1.0000 | 1.0000 | 1.0000 |
| **B2** | Stratified 50 notes — THE HEADLINE | 50 | 324 | 324 | 1.0000 | 1.0000 | 1.0000 |

- **A1 → A2** is what the answer-key *column* is worth. A1 scores against `gold_code_description`, the ICD catalogue wording, exactly as the original request described. A2 scores the same notes against the note wording that same file carries. The two columns agree in only 4.5% of rows corpus-wide — the note says `depression`, the catalogue says `Major depressive disorder, single episode, unspecified` — so A1 is near-zero by construction. That is a property of the two columns, not a fault in the model or in the data as built.
- **A2 → B1** is what the answer-key *completeness* is worth. Both rows use note wording on the same notes; only the key changes. The 100-row cut carries 99 evidence phrases where those notes actually hold 195, because the file was cut to 100 annotation rows rather than to whole notes. Everything A2 counts as a false positive and B1 counts as a hit is a phrase that really was billed. A2 is the number comparable with any figure computed from that file directly.
- **B1 → B2** is what balanced sampling changes. B1 runs on the 24 notes reachable from the 100-row sample cut, which are 17 Inpatient and 7 Profee. B2 runs on 25 of each, drawn with seed 13.
---
## B2 by chart type — the headline result

Profee and Inpatient are reported separately and must never be pooled: they are different measurement regimes. **Read recall as the real result. Read precision on Inpatient with the caveat below.** "Terms per gold term" is how many phrases the model listed for each one the coders highlighted; a large number there is what drags precision down.

| stratum | notes | gold terms | terms predicted | terms per gold term | precision | best precision possible | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| **Profee** | 25 | 110 | 110 | 1.0x | 1.0000 (96.6%–100.0%) | 1.0000 | 1.0000 (96.6%–100.0%) | 1.0000 |
| **Inpatient** | 25 | 214 | 214 | 1.0x | 1.0000 (98.2%–100.0%) | 1.0000 | 1.0000 (98.2%–100.0%) | 1.0000 |

### Why Inpatient precision is not a model failure

MDACE annotates evidence only for codes that were **actually billed**, not for every condition documented. A Profee note is a single clinician encounter — short, and densely billed. An Inpatient note is the record of a whole hospital stay: a median of 1,113 words carrying a median of 6 highlighted phrases. A model that reads such a note and correctly lists 40 real conditions has done its job, and scores precision 6/40 = 0.15 for it.

The **best precision possible** column above makes that concrete: it is the highest micro precision any extractor could reach given how many terms were predicted. Where that number is low, no model of any quality scores above it, and the F1 in that row should not be read as a quality judgement.
---
## What the false positives actually are

Every phrase the model produced that was not in the gold set, split three ways. The first two columns are extractions that are correct about the note; only the last column is model error. A healthy result is a **small last column** — that is the hallucination rate, and it is the one precision-side number the billing-scope problem does not distort.

| stratum | false positives | already billed, marked on another note of the same admission | in the note, nothing billed for it | not in the note at all |
|---|---|---|---|---|
| **Profee** | 0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Inpatient** | 0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

An ICD code belongs to an *admission*, but MDACE marks its evidence on whichever note the coder happened to be reading. So the first column is a phrase that demonstrably justifies a code that really was billed for this patient's stay — counted as a false positive purely because the annotator marked it elsewhere.

**None of this is subtracted from the precision figures above.** Removing your own false positives before dividing raises precision by construction and measures nothing.
---
## Code recall — the ceiling for the term → ICD lookup

Of the billing codes on these notes, the fraction that had at least one of their evidence phrases extracted. One phrase often justifies several codes, and one code is often evidenced by two phrasings, so this is not the same number as term recall.

This is the **upper bound on the downstream lookup**: it can never retrieve a code whose evidence phrase was never extracted. Higher is better, and it is measured on the same notes as the NER, so unlike an end-to-end code-prediction run it separates the extraction step from the lookup step.

| stratum | codes billed | codes whose evidence was found | code recall |
|---|---|---|---|
| **Profee** | 108 | 108 | 1.0000 (96.6%–100.0%) |
| **Inpatient** | 194 | 194 | 1.0000 (98.1%–100.0%) |
---
## How much to trust these numbers

The samples are small and the intervals in brackets are 95% Wilson intervals — the range the true rate plausibly sits in.

As a rule of thumb, a rate measured on ~110 items carries roughly ±9 percentage points and on ~214 items roughly ±7. **A gap between Profee and Inpatient smaller than about 15 points is not a finding.** Quote the headline as "around 0.6", never as "0.61".

13 of the scored notes were billed both ways (some Profee rows, some Inpatient) and are counted as Inpatient. Their gold therefore includes Profee-billed evidence, which blurs the contrast between the two strata slightly.

Generation hit the token cap on **0** of 202 chunks. Those replies were cut off mid-JSON; the parser salvages the complete part, but anything after the cut is lost, so **recall above is a floor rather than an estimate**. Raising the cap does not fix this — it was already tried, and the same chunks bound at 1024 and 1536, which is a repetition loop rather than a length problem. The lever is output volume: a narrower prompt, or smaller `--chunk-words` so each call has less to describe.
---
## Known limits

- **Strict matching only.** No fuzzy or embedding matching. 88% of gold terms are 1–3 words and behave well; the remaining 12% are clause-length snippets a model will not reproduce word-perfectly, and those score as misses even when the model was substantially right. The gitignored `errors.jsonl` holds the actual missed phrases so that decision can be made on evidence rather than guesswork.
- **MIMIC-III only.** MDACE is built on MIMIC-III notes (integer note IDs, 6-digit admission IDs, `[**...**]` redaction). It shares no notes with the MIMIC-IV medication evaluation in this repo, so the two sets of numbers must not be pooled or compared directly.
- **Recall is capped at 0.945, by choice.** The prompt asks for diagnoses, procedures and injuries, and explicitly excludes medications. That puts the 5.5% of gold carried by ICD-10-CM Z-codes (status and history — smoking history, long-term drug therapy) out of reach. The alternative was measured and was worse: asking for that category too made the model return 33% medication items, which truncated 12 of 15 replies and pushed extraction to 68.6 terms per note against ~10 gold. A 5.5% ceiling costs less than the damage that volume did to the other 94.5%.
- **Type labels are not scored.** The prompt asks the model to classify each term as Condition / Procedure / Injury, but scoring compares strings only. A term labelled wrongly still counts as a hit.
