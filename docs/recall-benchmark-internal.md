# Recall benchmark — internal reference

Branch: `mdace-recall-benchmark`, off `mdace-term-ner`.
Companion to `medgemma-recall-benchmark-plan.pdf`, which is Ehtesham Bhai's copy and
deliberately carries no project mechanics. This file holds everything that was cut
from it, plus the reasoning behind the cuts.

Status: **complete and reported.** The 24-note run was executed on 2026-08-13 and
the results were sent: 78 of 100 billed phrases, model alone. See "What was
built" at the bottom, which records the one measurement that revised the plan —
L4 does not do what this document assumed it would.

**This phase is closed. The precision phase continues in
`precision-plan-internal.md`, and that is where the outstanding work is.**

---

## Scope, and what it replaced

The task is a **benchmark**, not an evaluation design exercise. Measure how much of
the billed evidence MedGemma-4B recovers from a note, zero-shot, on
`8-07-mdace-ner-eval_sample_100-LOCAL.jsonl` and nothing else.

Dropped from the `mdace-term-ner` work, all of it deliberately:

| dropped | why |
|---|---|
| the 9,499-row dataset and 1,074-note corpus | one file only |
| the stratified 50-note sample, seed 13 | not his data |
| Profee / Inpatient separation | not asked for; benchmark is one number |
| the A1 / A2 / B1 / B2 view ladder | that was answer-key archaeology, not benchmarking |
| precision as headline | recall is the metric; precision stays as a secondary |

The `sample_100` truncation (99 of the 195 phrases those 24 notes carry) **stops
mattering** under this scope. It only ever inflated false positives, and gold is now
whatever his file says it is. That concern is closed, not deferred.

## Data handling

24 notes, 100 rows, 91 distinct `(note, code_system, code)`, 99 distinct
`(note, normalized evidence)`.

Note text is embedded in the file, so there is no join and no separate notes file.
Long notes still need chunking: **82 windows** at 400 words / 80 overlap, terms
pooled per note. At the ~68 s/chunk seen on the last smoke run that is ~1.5 h on a
T4; the rewritten prompt should generate less, so the PDF quotes ~1 h. If the first
smoke run says otherwise, correct the PDF rather than the estimate.

## Gold construction

Per billed code, the accept-set is the union of:

- `mdace_gold_evidence_text`
- `gold_code_description`
- every `gold_snomed_concepts[].term`

Median 4 accepted forms per row (min 2, max 5), against 1 on the previous branch.

Recall reported per row (100) **and** per distinct code (91). They will be close;
reporting both removes a question rather than answering it later.

### SNOMED coverage, measured

| | |
|---|---|
| rows with any SNOMED | 53 of 100 (his file) · 8,271 of 9,499 (full corpus) |
| concept terms shipped | 24,333 of 270,287 reported by `gold_snomed_concept_count` — **9%** |
| CPT rows with SNOMED | 0 of 346 |
| ICD-10-PCS rows with SNOMED | 0 of 272 |
| ICD-10-CM rows without | 608 of 8,879 |

The list is capped at 3 entries and the survivors are not the top-ranked ones — an
`HCV` row ships two pregnancy-related concepts and omits plain "hepatitis C".

**We are not asking him to fix this.** He gave us one file; we work with it. It is
stated in his copy as a limit on the number, not as a request. Same for the real
SNOMED lookup: we approximate it with the matching ladder and say so.

His file is also procedure-heavy (30% CPT+PCS against 6.5% corpus-wide), which is why
its SNOMED coverage reads 53% rather than 87%.

## Matching ladder

Each level is a superset of the one above, so recall is monotonically
non-decreasing and the gain from each level is attributable.

| level | rule | notes |
|---|---|---|
| L1 | exact after normalization | current behaviour |
| L2 | whole-token containment, either direction | guard: `diabetes` ⊂ `diabetes insipidus` is a false match, and `sepsis` ⊂ `no evidence of sepsis` is a negation. Report L2's newly-matched pairs. |
| L3 | token-set Dice and/or `difflib` ratio, thresholded | typos and word order |
| L4 | biomedical sentence embeddings, cosine | the only level that reaches abbreviations |
| L5 | LLM judge | adjudication; he expects to need it |

Measured on real pairs, which is why the ladder exists rather than one threshold:

| model says | gold says | contains | dice | char |
|---|---|---|---|---|
| hyperlipidema | hyperlipidemia | no | 0.00 | **0.96** |
| back pain | chronic back pain | **yes** | 0.80 | 0.69 |
| acute kidney injury | kidney injury, acute | no | **1.00** | 0.68 |
| HTN | hypertension | no | 0.00 | **0.40** |
| CHF | congestive heart failure | no | 0.00 | **0.22** |
| acute renal failure | chronic renal failure | no | 0.67 | **0.75** |
| sepsis | no evidence of sepsis | **yes** | 0.40 | 0.44 |
| diabetes | diabetes insipidus | **yes** | 0.67 | 0.62 |

**No string threshold separates the good rows from the bad ones.** Anything loose
enough to catch `CHF` at 0.22 accepts `acute` vs `chronic renal failure` at 0.75
several times over. That is the whole argument for L4 and for the deferred audit.

Implementation notes:

- Matching must be **one prediction to at most one gold form**, greedy best-first.
  Otherwise a vague prediction satisfies several gold entries at once.
- Embedding model: a biomedical sentence encoder, not general-purpose. A general
  MiniLM does not know `HTN`. Runs on the same Colab GPU after the extraction pass.
- Thresholds are reported, never silently chosen.

## Prompt rewrite

The current prompt (`prompt_mdace.py`, hash `090a0072`) says *copy VERBATIM, keep it
1–3 words*. That was correct when gold was the note's own wording. It is now wrong:
`HTN` can never string-match `Essential (primary) hypertension`.

New shape, two fields per finding:

- `span` — the phrase as written in the note. Needed for the not-in-note check.
- `name` — the standard clinical name. What the lookup would consume.

Either may match the accept-set. MedGemma already knows the expansions; using that is
better than making the matcher infer them.

Cost is ~2x tokens per finding, which is what the flat-string change bought back on
the previous branch. Offset it by asking for one entry per distinct finding and no
repeats — which also serves the false-positive concern.

Carry forward from the previous branch: medications stay out of scope (they produced
33% of extraction for 5.5% of gold and truncated 12 of 15 chunks), the redaction-marker
instruction stays, `max_new_tokens` stays at 1024, and the truncation salvage stays.

**The prompt hash is part of the run-directory name**, so this rewrite starts a fresh
cache automatically and cannot replay old results.

## Metrics

Three additions after Ehtesham Bhai reviewed the plan: report false positives
explicitly, expect to run L5, and break recall and FP out **per ground-truth
source**.

| | |
|---|---|
| recall | per level, per row and per code |
| false positives | count and rate, first-class rather than implied by precision |
| terms per note | the volume the recall was bought with |
| not-in-note rate | hallucination |

**Recall is never quoted bare.** With loose matching and no volume control, a model
that lists every phrase scores near 1.00, so a bare figure cannot rank MedGemma
against a larger model later, which is the entire purpose of the benchmark.

### Per-source breakdown, and its one trap

A 4-row table: evidence text, code description, SNOMED terms, and all three
combined; recall and FP on each.

Recall per source is unambiguous: of that source's entries, how many were matched.
It answers a genuinely useful question, which is *whose wording the model produces*
— note-style, catalogue-style, or SNOMED-style.

**FP per source is easy to misread.** A prediction matching only the catalogue
wording is a hit on the description row and an FP on the evidence-text row, because
per-source FP means "matched nothing *in that source*". So every individual row's FP
count will look high, and only the combined row counts predictions that matched
nothing anywhere. That sentence is in his copy; keep it there.

Denominators differ per source and must be printed: 99 distinct evidence phrases, 91
descriptions, and SNOMED only where the file ships terms (53 of 100 rows).

## Reference numbers

The `mdace-term-ner` run, strict exact matching against evidence text alone, 50 notes:

| | |
|---|---|
| recall | 0.5278 (324 gold terms) |
| precision | 0.0679 (2,520 predicted) |
| code recall | 165 of 302 = 54.6% |
| terms per note | ~50 |
| his independent figure | ≈0.5 |

Two implementations landing in the same place is a real cross-check. Keep it.

## Build order

1. ~~Branch is created. Loader for `sample_100` (self-contained, no join).~~ done
2. ~~Accept-set construction from the three columns.~~ done
3. ~~Matching ladder L1-L4, thresholds configurable, newly-matched pairs dumped per level.~~ done
4. ~~Prompt rewrite, two fields.~~ done, hash `7f93b2f6`
5. ~~Report: recall and FP per level, and the per-source breakdown.~~ done
6. **Run: 82 chunks.** ← next, and the only step needing a GPU
7. ~~L5 adjudication over the pairs L2-L4 newly accepted.~~ built; run it after step 6

L5 is now a planned step rather than a contingency: he expects the assessment to need
it. Build it after L1-L4 are producing numbers, and run it on the pairs each level
newly accepted rather than on everything, so the cost stays bounded.

---

## Starting a fresh session from this file

Everything below lived in the conversation that produced this plan and is not
recoverable from the repo alone.

### Where things are

| | |
|---|---|
| repo | `/home/shifat/zeda_ml_works/medgemma-ner-eval` |
| branch | `mdace-recall-benchmark` (created, 3 doc commits, **not pushed**) |
| the one input file | `/home/shifat/zeda_ml_works/zeda_mimic_datasets/8-07-mdace-ner-eval_sample_100-LOCAL.jsonl` |
| S3 mirror | `s3://zeda-mimic-dataset/eval_datasets/` (credentials live in the repo's `.env`, which the shell does **not** auto-source: `set -a && . ./.env && set +a`) |
| Colab notebook to adapt | `colab_runner_mdace.ipynb` |

### Reuse unchanged

- `src/chunking.py` — `chunk_windows(n_tokens, 400, 80)` and `tokenize_with_spans`.
- `src/model.py` — `load_medgemma`, `run_messages`, `count_tokens`. 4-bit, greedy.
- The resume pattern in `src/evaluate_mdace.py`: append each note's result as it
  finishes, skip note_ids already present, key the run directory on model + seed +
  chunk geometry + token cap + **prompt hash**. That last one is load-bearing; a
  prompt edit without it silently replays the previous prompt's numbers.
- The tolerant reply parser in `src/prompt_mdace.py`, including
  `_salvage_truncated_strings` for replies cut off at the token cap.

### Do not touch

`src/evaluate_mimic.py`, `src/prompt_mimic.py`, `src/mimic_config.py`,
`src/datasets/mimic_meds.py`, `src/report_mimic.py`, `src/align.py`,
`src/scoring.py`. Different corpus (MIMIC-IV), different task. The `mdace_*` and
`term_scoring` modules from the previous branch should also be left alone, so the
0.53 starting point stays reproducible; add new modules alongside them.

### PHI rules, non-negotiable

MDACE is built on MIMIC-III, credentialed PhysioNet data.

- Note text, extracted terms, and any per-example dump stay under the gitignored
  `outputs/` tree. Never commit them, never paste note content anywhere.
- Keep the counts file and the extracted-terms file separate: counts are integers
  and shareable, terms are phrases copied out of patient notes.
- `results/*.md` and `results/*.json` are aggregate-only and safe to commit.
- On Colab, point the output dir at mounted Drive so a disconnect cannot lose a run.

### Two design calls already made

**L4's embedding backend is optional.** Import it lazily and skip L4 with a clear
message if the package is absent, so L1-L3 stay unit-testable on a CPU-only laptop
and L4 runs on Colab. A general-purpose encoder is not enough; it will not know
`HTN`.

**Verify the harness before spending GPU time.** Feed the gold accept-sets back
through the pipeline as if they were model output. Recall must come out 1.0000 at
L1. Anything less is a bug in chunking, normalization or matching, not a result.
This caught real bugs twice on the previous branch and costs ten seconds.

### Known trap

Re-running after a prompt edit prints `cached N, to run 0` and reports the old
prompt's numbers in about a second, with no error. If a run finishes suspiciously
fast, check the `run dir:` line for the prompt hash before believing anything.

---

## What was built

New modules alongside the `mdace_*` and `term_scoring` ones, which are untouched
so the 0.53 starting point stays reproducible.

| module | what it is |
|---|---|
| `src/recall_config.py` | thresholds, paths, chunk geometry, gold sources |
| `src/datasets/mdace_recall.py` | the loader and the accept-set construction |
| `src/recall_matching.py` | the ladder: string rules, embeddings, the matching |
| `src/prompt_recall.py` | the two-field prompt and its parser |
| `src/recall_scoring.py` | aggregation into rows / codes / forms, per source |
| `src/report_recall.py` | the committed markdown + metrics JSON |
| `src/evaluate_recall.py` | the runner |
| `src/recall_judge.py` | L5 |
| `colab_runner_recall.ipynb` | the T4 runner |

Added during the precision phase, after this document was written:

| module | what it is |
|---|---|
| `src/recall_failures.py` | false positives by note section, misses by cause |
| `src/recall_filter.py` | the second-pass billability filter |
| `src/recall_compare.py` | two finished runs side by side |

131 new tests, 470 in the suite, all passing on CPU with no model download.
`make recall-oracle` passes: every source 1.0000 at L1, zero false positives on
the combined matching. Run it before spending GPU time.

Every plan number was checked against the file and holds: 24 notes, 82 chunks,
100 rows, 91 codes, 99 evidence phrases, 91 descriptions, 142 SNOMED forms,
median 4 accepted forms per row (min 2, max 5), 53 of 100 rows with SNOMED, 0 of
15 CPT and 0 of 15 PCS rows with SNOMED. The measured string table in the ladder
section reproduces exactly.

### Three decisions this document did not make

**L4 does not separate synonyms from near-misses.** This is the one that
matters. The plan treats L4 as the level that finally works because it is the
only one that reaches abbreviations. It reaches them and it does not separate
them. On `NeuML/pubmedbert-base-embeddings`, sorted by cosine, the pairs we want
and the pairs we do not are interleaved from top to bottom: `acute renal
failure` against `chronic renal failure` scores **0.833**, above eight of the
ten abbreviation pairs L4 exists to catch. Two other biomedical encoders are
worse — `S-PubMedBert-MS-MARCO` saturates at 0.85-0.99 for everything including
`pneumonia` against `fracture of left wrist`, and `MedEmbed-small-v0.1` ranks
the `no evidence of sepsis` negation above HTN, CHF and MI.

So the sentence in this document — *no string threshold separates the good rows
from the bad ones, that is the whole argument for L4* — turns out to be true one
level further up as well. The default cosine is 0.60, which is a floor chosen to
reach every abbreviation measured rather than a cutoff that works; raising it
loses synonyms without buying precision (0.60 catches 10/10 wanted and 5/7
unwanted, 0.66 catches 6/10 and still admits 3/7). **L5 is therefore not a
refinement of L4, it is what makes L4's gain interpretable at all.** The numbers
are pinned in `recall_matching.MEASURED_COSINE` and the report prints the table
rather than describing it.

**The matching is max-cardinality, not plain greedy.** "One prediction to at
most one gold form, greedy best-first" breaks on ties, and it breaks in exactly
the case the harness check rests on: several findings compete for one form, one
takes a form another needed, and the oracle reads under 1.0000 with no bug
present. Each level now runs greedy by rule strictness then score — best-first
survives wherever it is meaningful — then augments to maximum cardinality over
what is left. Lower-level assignments are frozen, so per-level attribution and
the dumped pairs stay stable.

**Rows and codes are the quotable units; forms is a diagnostic.** A row is
recalled by matching any one of its ~4 accepted forms, and matching is 1:1, so a
model that recalls every row while offering one phrasing each leaves most forms
unmatched by construction. Combined form recall has a ceiling set by how many
phrasings the model emits, not by how much it found. It is carried because the
per-source breakdown is measured in forms and the two have to reconcile.

### Open question

L5's judge defaults to the same 4-bit MedGemma the benchmark runs, which is free
on a box that already has it loaded and is the model under test grading its own
matches. The summary and the report say so out loud. `--judge none` writes the
questions out for a human or a stronger model and reads answers back via
`--verdicts`. Which one the reported figures should use is a call worth making
before the write-up, not after.
