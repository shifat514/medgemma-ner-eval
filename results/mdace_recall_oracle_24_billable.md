# MedGemma zero-shot recall benchmark — MDACE billing evidence

Model **medgemma-4b-it-ORACLE** (`google/medgemma-4b-it`), 4-bit, greedy decoding, `max_new_tokens=1024`. Chunking 400 words / 80 overlap. 24 notes, 82 chunks.
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
| **L4** | biomedical sentence-embedding cosine | 100/100 1.0000 (96.3%–100.0%) | 91/91 1.0000 | 318/318 1.0000 | 0 | 0.0000 |

**Quote the rows or codes column, not the forms column.** A row is recalled by matching any ONE of its accepted forms, so a model that recalls every row while producing a single phrasing each still leaves most forms unmatched — the forms column has a ceiling set by how many phrasings the model emits, not by how much it found. It is here because the per-source breakdown below is measured in forms and the two must reconcile.

- **L2** added 0 rows and 0 accepted forms over the level above it.
- **L3** added 0 rows and 0 accepted forms over the level above it.
- **L4** added 0 rows and 0 accepted forms over the level above it.

**Thresholds are reported, never silently chosen.** Dice ≥ `0.8`, character ratio ≥ `0.9`, cosine ≥ `0.6` using `NeuML/pubmedbert-base-embeddings`.

They come from measured pairs, not from taste. Dice 0.80 keeps *acute kidney injury* against *kidney injury, acute* (1.00) and drops *acute renal failure* against *chronic renal failure* (0.67); character ratio 0.90 keeps *hyperlipidema* against *hyperlipidemia* (0.96) and drops that same acute/chronic pair (0.75). No string threshold separates the good pairs from the bad ones on its own — anything loose enough to catch *CHF* against *congestive heart failure* (0.22) accepts the acute/chronic pair several times over. That is the argument for L4, and for dumping every pair each level newly accepted.

**L2 knowingly admits two bad shapes.** *diabetes* inside *diabetes insipidus* is a false match, and *sepsis* inside *no evidence of sepsis* is a negation. The pairs L2 newly accepted are written to the run directory for exactly this reason; L5 adjudicates them.

### L4 reaches abbreviations. It does not separate them.

This is a result, not a caveat added for form. Cosine on the default biomedical encoder, sorted, for pairs a matcher should accept and pairs it must reject:

| model says | gold says | should match | cosine |
|---|---|---|---|
| acute kidney injury | kidney injury acute | yes | 0.978 |
| back pain | chronic back pain | yes | 0.933 |
| acute renal failure | chronic renal failure | **no** | 0.833 |
| anemia | sickle cell anemia | **no** | 0.744 |
| hyperlipidema | hyperlipidemia | yes | 0.723 |
| chf | congestive heart failure | yes | 0.721 |
| mi | myocardial infarction | yes | 0.719 |
| sepsis | no evidence of sepsis | **no** | 0.668 |
| htn | hypertension | yes | 0.662 |
| chest pain | abdominal pain | **no** | 0.659 |
| cabg | coronary artery bypass graft | yes | 0.653 |
| afib | atrial fibrillation | yes | 0.646 |
| hypertension | pulmonary hypertension | **no** | 0.642 |
| esrd | end stage renal disease | yes | 0.641 |
| copd | chronic obstructive pulmonary disease | yes | 0.610 |
| diabetes | diabetes insipidus | **no** | 0.545 |
| pneumonia | fracture of left wrist | **no** | 0.177 |

The two columns are interleaved from top to bottom. *acute renal failure* against *chronic renal failure* — a different diagnosis — scores 0.833, above 8 of the 10 pairs L4 exists to catch. No cosine threshold admits every one of those and none of these, so the threshold above is a **floor chosen to reach the abbreviations**, not a cutoff that works. Raising it loses synonyms without buying precision.

**So the L4 row is provisional until L5 has run.** The argument the string table makes about L1-L3 turns out to hold one level higher up as well, which makes L5 not a refinement of L4 but the thing that makes L4's gain interpretable at all. Every pair L4 newly accepted is in the run directory waiting for it.
---
## Recall is never quotable on its own

With loose matching and no volume control, a model that lists every phrase in the note scores near 1.00. A bare recall figure therefore cannot rank MedGemma against a larger model later, which is the entire purpose of this benchmark. These are the numbers that must travel with it.

| | |
|---|---|
| findings per note | 13.2 |
| false positives at L4 | 0 of 318 findings (0.0%) |
| span not found in the note | 0 of 318 checked — 0.0000 (0.0%–1.2%) |
| findings with no span to check | 0 |

The last two lines are the hallucination signal: a span the model claims to have copied from the note that is not in the note. It is the one precision-side number the billing-scope problem does not distort, because it does not depend on what was billed at all.

A finding the model returned with no `span` cannot be checked and is excluded from that denominator rather than counted as clean.
---
## Whose wording does the model produce?

Recall broken out by which column of the answer key was matched. This is the genuinely useful question behind the benchmark: does the model speak in note wording, catalogue wording, or SNOMED wording. Each source is scored by its own independent matching.

### Recall per source, by level

| source | entries | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| all three combined | 318 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| evidence text (what the note says) | 99 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| code description (ICD catalogue wording) | 91 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SNOMED concept terms | 142 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

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

| source | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| all three combined | 0 (0.00) | 0 (0.00) | 0 (0.00) | 0 (0.00) |
| evidence text (what the note says) | 219 (0.69) | 219 (0.69) | 219 (0.69) | 219 (0.69) |
| code description (ICD catalogue wording) | 227 (0.71) | 227 (0.71) | 227 (0.71) | 227 (0.71) |
| SNOMED concept terms | 176 (0.55) | 176 (0.55) | 176 (0.55) | 176 (0.55) |
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

Two implementations landing in the same place is a real cross-check, which is why it is kept here. It is **not** directly comparable with the table above — different notes, a single-column answer key, and exact matching only — so read it as the L1-with-one-source starting point the accept-set and the ladder were built to improve on.
---
## How much to trust these numbers

The sample is small. Brackets are 95% Wilson intervals — the range the true rate plausibly sits in. A recall measured over ~100 rows carries roughly ±10 percentage points, so **quote the headline as "around 0.6", never as "0.61"**, and treat a gap of less than about 10 points between two levels as noise.

Generation hit the token cap on **0** of 82 chunks, so no reply was cut off mid-JSON and recall is not understated on that account.
---
## Known limits

- **L5 has not been applied to these figures.** The pairs L2, L3 and L4 newly accepted are written to the run directory and are adjudicated by a separate step (`python -m src.recall_judge`). Until that has run, every level above L1 includes matches nobody has checked — the negation and the *diabetes insipidus* shapes above are real and unquantified. **Quote L1 as the number that needs no caveat.**
- **The SNOMED lookup is approximated, not performed.** The ladder stands in for a real terminology lookup. L4 reaches abbreviations, which is the part that matters most, but it is a similarity model and not a terminology — and the measurement above shows it cannot tell a synonym from a change of acuity.
- **Medications are out of scope, by measurement.** The prompt excludes them because asking for them produced 33% of extraction for 5.5% of gold and truncated 12 of 15 chunks on the previous branch. Rows whose evidence is a medication are therefore unreachable.
- **MIMIC-III only.** MDACE is built on MIMIC-III notes. It shares no notes with the MIMIC-IV medication evaluation in this repo, so the two sets of numbers must not be pooled.
