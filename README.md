# medgemma-ner-eval

Evaluates Google's **MedGemma** (`google/medgemma-4b-it`) on clinical Named Entity
Recognition — finding **diseases** and **chemicals** in biomedical text — with
results **directly combinable** with the sibling [`clinical-ner-eval`](../clinical-ner-eval-main)
repo (which benchmarks fine-tuned BioBERT / PubMedBERT / Bio_ClinicalBERT).

MedGemma is a **generative** model, so this is **zero-shot prompting**, not
token classification: we prompt MedGemma for JSON entities, align the returned
spans back onto the token list, and score the resulting BIO tags with the exact
same seqeval scorer and test set as the sibling repo.

> **Phase 1 (this repo):** zero-shot prompting.
> **Phase 2 (later):** fine-tuning — the repo is structured to add it (dataset
> registry, config, scoring, and alignment are all reused unchanged), but it is
> not built yet.

There are **two evaluations** in this repo, sharing all the prompt/parse/align/score
machinery:

| | entrypoint | data | labels |
|---|---|---|---|
| **Disease/Chemical NER** | `python -m src.evaluate` | NCBI Disease + BC5CDR (public, from the Hub) | `Disease`, `Chemical` |
| **Medication NER** | `python -m src.evaluate_mimic` | MIMIC-IV-Note discharge summaries (**credentialed, local only**) | `Medication`, `Dose`, `Mode`, `Frequency`, `Duration`, `Reason` |

See [MIMIC-IV medication NER](#mimic-iv-medication-ner) below. **That evaluation
touches real patient data — read the data-handling rules before running it.**

---

## Why the numbers combine with `clinical-ner-eval`

Comparability rests on three things being identical, and they are:

| | This repo | `clinical-ner-eval` |
|---|---|---|
| **Test set** | NCBI (`ncbi_disease_bigbio_kb`) + BC5CDR (`bc5cdr_bigbio_kb`) test splits, `trust_remote_code=True`, `\S+` whitespace tokenization, harmonized to `Disease`/`Chemical` | same |
| **Gold labels** | `O, B-Disease, I-Disease, B-Chemical, I-Chemical` (~1,363 entities: 707 Disease + 656 Chemical) | same |
| **Scorer** | seqeval entity-level `classification_report` | same |
| **Output** | `results/comparison.csv` with columns `model, entity, precision, recall, f1, support` | same |

The dataset-loading code is **vendored** (copied into `src/datasets/`), not
imported from the sibling repo, so this repo runs standalone in Colab.

**Combining** is just concatenation:

```python
import pandas as pd
med  = pd.read_csv("medgemma-ner-eval/results/comparison.csv")
bert = pd.read_csv("clinical-ner-eval-main/results/comparison.csv")
combined = pd.concat([bert, med], ignore_index=True)
combined[combined.entity == "micro avg"]   # head-to-head across all models
```

---

## The pipeline (per test example)

```
tokens ──" ".join──▶ sentence ──prompt──▶ MedGemma
                                             │
                                   {"entities":[{"text","type"}]}   (JSON reply)
                                             │ parse_entities()
                                    [(span, Disease|Chemical), ...]
                                             │ align_entities_to_bio()  ← onto the SAME tokens
                                    predicted BIO tags
                                             │
              gold BIO + predicted BIO ──▶ seqeval ──▶ comparison.csv
```

The **span → BIO alignment** (`src/align.py`) is the delicate part: spans are
matched case- and punctuation-insensitively, multi-word spans must be a
contiguous token run, every non-overlapping occurrence is tagged, and a token
already claimed by an earlier span is never overwritten.

A malformed model reply degrades gracefully to "no entities" (all-`O`) for that
example rather than crashing the run.

---

## Quickstart

### Colab (primary runner — free T4 GPU)

Open [`colab_runner.ipynb`](colab_runner.ipynb) and run the cells in order:

1. `Runtime → Change runtime type → T4 GPU`
2. Install deps
3. **HF login** via `getpass` (MedGemma is gated — accept the license at
   <https://huggingface.co/google/medgemma-4b-it> first). Your token is read at
   runtime and never stored in the notebook.
4. Clone/upload this repo and `cd` in
5. **Smoke test** (`--limit 10`)
6. **Full eval** → writes `results/comparison.csv`
7. Inspect + combine with the sibling CSV

The 4B model runs in **4-bit** (`BitsAndBytesConfig(load_in_4bit=True)`), ~5–7 GB
VRAM — comfortably within a free T4.

### Local

Running the model locally needs a GPU. The **tests do not**.

```bash
pip install uv && uv sync --extra test

# CPU-only unit tests — no GPU, no model download
uv run pytest -q            # or: make test

# Smoke test / full eval (needs a GPU + HF login)
python -m src.evaluate --limit 10     # make smoke
python -m src.evaluate                # make eval
```

CLI flags: `--limit N` (first N examples), `--model <hf_id>`, `--model-name <label>`.

---

## MIMIC-IV medication NER

Zero-shot medication NER over **real MIMIC-IV-Note discharge summaries**, scored
against the [Medication Extraction Labels for MIMIC-IV-Note](https://physionet.org/content/medication-labels-mimic-iv/1.0.0/)
dataset (i2b2 2009 medication-challenge schema).

### ⚠️ Data handling — read this first

**This repo is public. MIMIC-IV is credentialed PhysioNet data.**

- The source data lives **outside** the repo and stays there. Configure paths via
  `MIMIC_BASE` / `MIMIC_DISCHARGE_CSV` / `MIMIC_LABEL_DIR` (see
  `src/mimic_config.py`). The 1.1 GB `discharge.csv.gz` is never copied in.
- `build_mimic_sample` writes a ~1.5 MB sample to `data/samples/` — **gitignored**.
- Per-note run state goes to `outputs/` — **gitignored**. `--dump-errors` writes
  per-example dumps that **quote note text**; they live there and only there.
- **Safe to commit:** `results/mimic_ner_*.csv` and `results/mimic_ner_*_report.md`.
  These are aggregate metrics and configuration only — no note text, no patient
  data, no example snippets. `.gitignore` un-ignores exactly these two patterns.
- Run `make mimic-check` before committing: it fails if any data file is tracked.

### The extra problem: note length

Discharge summaries are **181–5,425 whitespace tokens** (median 1,521) against the
single sentences the Disease/Chemical pipeline handled, and one note carries up to
**381 gold annotations**. The binding constraint is *generation*, not context: a
note with 100 entities needs ~100 JSON objects emitted, far past any sane
`max_new_tokens`. So notes are split into overlapping token windows:

```
note text ──▶ \S+ tokens (+ char spans) ──▶ gold BIO from label char offsets
                     │
            chunk_windows(400 tokens, 80 overlap)   ← tiles ALL tokens; nothing dropped
                     │
        per chunk:  prompt ──▶ MedGemma ──▶ parse ──▶ align onto THAT window's tokens
                     │
            merge chunk BIO ──▶ dedupe on (start, end, type) ──▶ note-level BIO
                     │
                  seqeval ──▶ mimic_ner_<n>.csv + report.md
```

Two decisions keep the scoring honest:

- **Alignment is per chunk, first-occurrence-only.** The model returns *strings*,
  not offsets, so each must be found in the text again. Tagging every occurrence
  turns one prediction of `aspirin` into ~15 spans; restricting to the first
  occurrence within each chunk keeps the predicted-span count within 3% of the
  gold count. Chosen by measurement — see below.
- **Predictions are deduped, not just gold.** Overlapping windows re-read the same
  tokens, so entities in an overlap get predicted twice. Merging dedupes on
  `(start, end, type)` after stitching, and reports how many were removed.

### The harness has a structural ceiling — measure it

`--oracle` feeds the gold spans back through the identical pipeline. No model, no
GPU, ~7 seconds. Whatever it loses is harness error, not model error, so read
MedGemma's numbers against it rather than against 1.0.

```bash
make mimic-oracle       # python -m src.evaluate_mimic --oracle --n 100
```

It is also how the alignment default was chosen. Oracle ceiling on n=100:

| `--align-mode` | micro P | micro R | micro F1 |
|---|---|---|---|
| **`first-per-chunk`** (default) | **0.8583** | 0.8817 | **0.8698** |
| `all-per-chunk` | 0.6976 | **0.9505** | 0.8046 |
| `first-note` | 0.7917 | 0.5524 | 0.6507 |

`first-per-chunk` wins by **+6.5 micro F1** and wins on all six entity types
individually, trading 6.9 points of recall for 18.6 of precision. Full breakdown
and reasoning: [results/mimic_ner_align_mode_comparison.md](results/mimic_ner_align_mode_comparison.md).

The ceiling is still only 0.87, not 1.0 — 1.28x expansion remains. Closing that
needs the model to emit character offsets instead of strings, which is a
prompt-and-parse change, not an alignment one.

**Results are not comparable across modes.** The resume cache is keyed on the
mode, and non-default modes get a filename suffix, so the two can never silently
mix.

### Sampling

Only the 600 annotated notes are eligible. Two are excluded (0 and 1 annotations
against a median of 97 — labeling failures, not medication-free notes), leaving
598. A single seeded draw of 100 is taken and **sliced**: the n=50 set is the
first 50 of that draw, so it is a strict subset and the two runs are comparable.
`random.sample(pool, 50)` is *not* a subset of `random.sample(pool, 100)` — a
test guards against anyone "simplifying" it that way.

### Source data: local disk or S3

The sample builder reads from either. Any path starting with `s3://` is handled
transparently by [src/s3_io.py](src/s3_io.py); everything else uses plain local
file calls, so local runs are unaffected.

```bash
make mimic-sample                                    # local paths (default)
python -m src.build_mimic_sample --source s3         # s3://zeda-mimic-dataset/...
python -m src.build_mimic_sample --source s3 --aws-profile 615770945455_zeda-dev
```

Or point the env vars at S3 directly: `MIMIC_DISCHARGE_CSV=s3://…`,
`MIMIC_LABEL_DIR=s3://…`. Credentials come from the normal boto3 chain. `boto3`
is an optional dependency (`uv sync --extra s3`) — local-only installs and the
CPU tests don't need it.

The 1.1 GB discharge gzip is **streamed and decompressed on the fly**, never
written to local disk. The 600 label objects are fetched with 16 parallel workers
and cached, since `eligible_note_ids` reads all 600 and `build_sample` re-reads
the sampled 100.

**Only the sample build reads S3.** The evaluation always reads the small local
sample file. That is deliberate: a 1–3 hour GPU run should not depend on network
or on credentials (Zeda's are temporary STS tokens that expire), and it keeps
Colab on a 1.5 MB manual upload rather than holding AWS credentials for a bucket
of PHI.

### Running it

```bash
# 1. On the machine with access to the source data — local disk or S3 (~55 s)
make mimic-sample          # -> data/samples/mimic_med_sample.jsonl (gitignored)

# 2. On a GPU (Colab T4): smoke test, then the two sample sizes
make mimic-smoke           # --limit 5   ~28 chunks
make mimic-50              # ~262 chunks
make mimic-100             # ~561 total, but reuses the 50 above

make mimic-check           # verify no data file is tracked
```

**Runs are incremental and resumable.** Every note is appended to
`outputs/mimic/<tag>/per_note.jsonl` the moment it finishes, and a rerun skips
what's already done — a free-Colab disconnect costs only the note in flight. The
tag encodes seed + chunk settings but *not* sample size, so `--n 100` reuses the
`--n 50` work. Changing `--chunk-words` starts a fresh cache rather than mixing
results from different settings. The resume file holds BIO tag arrays and integer
counts only — no note text.

CLI: `--n N`, `--limit N` (smoke), `--chunk-words`, `--overlap-words`,
`--align-mode {first-per-chunk,all-per-chunk,first-note}`, `--no-resume`,
`--dump-errors`, `--oracle`, `--model`, `--model-name`.

Colab: [`colab_runner_mimic.ipynb`](colab_runner_mimic.ipynb) — T4, deps, HF login
via `getpass`, **manual upload** of the sample file (no Drive mount, no data
clone), then 5 → 50 → 100.

---

## MDACE recall benchmark

*(branch `mdace-recall-benchmark`. Design notes and the decision log live in
[`docs/recall-benchmark-internal.md`](docs/recall-benchmark-internal.md).)*

One question: **how much of the billed evidence does MedGemma-4B recover from a
note, zero-shot?** MDACE pairs MIMIC-III notes with the phrases human medical
coders highlighted as justification for the billing codes they submitted. The
model reads the note, lists the findings it sees, and we measure how much of that
highlighted evidence it got.

**Recall is the metric.** Precision is not the headline and is reported as an
explicit false-positive count rather than as a ratio, because MDACE annotates
evidence only for codes that were *actually billed* — a correct extraction of a
documented-but-unbilled condition is not a model error, and dividing by it
measures the annotation scope rather than the model.

Two things make this harder than string comparison, and most of the code exists
because of them.

**Gold has three spellings, not one.** The note says `HTN`; the ICD catalogue
says `Essential (primary) hypertension`; SNOMED says `Hypertensive disorder`. All
three are correct answers, so the accept-set per billed code is their union —
a median of 4 accepted forms per row against 1 under evidence-text-only scoring.

**No similarity threshold separates a synonym from a near-miss.** Measured on
real pairs: `CHF` against `congestive heart failure` scores 0.22 on character
similarity — a pair we *want*. `acute renal failure` against `chronic renal
failure` scores 0.75 — a different diagnosis. Anything loose enough to catch the
first admits the second several times over. So matching runs a **ladder** of four
levels, each a superset of the one above, and reports what each level bought.

### ⚠️ Data handling — read this first

Same rules as the MIMIC-IV section above, with one difference worth knowing: the
input file **embeds the note text**, so there is no join, no notes file and no
sample-building step.

- The one input, `8-07-mdace-ner-eval_sample_100-LOCAL.jsonl`, lives **outside**
  the repo. Point `RECALL_SAMPLE_FILE` at your copy. Mirror:
  `s3://zeda-mimic-dataset/eval_datasets/`.
- Everything a run writes except `per_note.jsonl` quotes note text, and all of it
  goes to the gitignored `outputs/mdace_recall/`.
- **Safe to commit:** `results/mdace_recall_*.md` and `*_metrics.json`. Counts,
  rates and thresholds only — verified: zero note-text fragments reach them.

### What each file does

**New modules.** All of them sit *alongside* the `mdace_*` and `term_scoring`
modules from the term-NER branch, which are untouched so that branch's 0.53
reference number stays reproducible.

| file | what it does |
|---|---|
| [`src/recall_config.py`](src/recall_config.py) | Every knob in one place: input path, chunk geometry (400/80), token cap (1024), the three ladder thresholds, and the L4 encoder id. Each default carries the measurement that set it. |
| [`src/datasets/mdace_recall.py`](src/datasets/mdace_recall.py) | Reads the file and builds the accept-sets. Groups 100 rows onto 24 notes, unions the three gold columns per billed code, and tags every accepted form with which column it came from. Also measures how much of the SNOMED column the file actually ships. |
| [`src/recall_matching.py`](src/recall_matching.py) | The ladder. L1 exact, L2 whole-token containment, L3 Dice/difflib, L4 biomedical embeddings — plus the bipartite matching that enforces *one prediction to at most one gold form*. Carries both measured pair tables as code constants. |
| [`src/prompt_recall.py`](src/prompt_recall.py) | The two-field prompt (`span` = as written in the note, `name` = standard clinical name) and a deliberately permissive reply parser that survives markdown fences, prose, odd key names, and replies cut off at the token cap. |
| [`src/recall_scoring.py`](src/recall_scoring.py) | Turns per-note matchings into the benchmark's numbers: recall in three units (rows / codes / accepted forms), false positives, and the per-source breakdown. Runs a separate matching per gold source so "recall per source" means what it says. |
| [`src/report_recall.py`](src/report_recall.py) | Writes the committed markdown + metrics JSON. Aggregate only — it never receives a phrase, only counts. Every table is preceded by what would count as good or bad, and every rate carries a 95% Wilson interval. |
| [`src/evaluate_recall.py`](src/evaluate_recall.py) | The runner. Chunk → prompt → parse → pool per note → score → report. Resumable, and carries `--oracle` (the harness check) and `--score-only` (re-score without touching the GPU). |
| [`src/recall_judge.py`](src/recall_judge.py) | L5. Adjudicates the pairs each level *newly* accepted — never everything — so cost stays proportional to what the ladder gained. |
| [`src/recall_failures.py`](src/recall_failures.py) | Failure analysis. Attributes every false positive to the note section it came from, and buckets every miss by cause — truncated, never extracted, near miss, or rejected by the judge. Those four have opposite fixes. |
| [`src/recall_filter.py`](src/recall_filter.py) | The second-pass filter. Asks per finding whether a coder would bill it and drops the noes. Reports raw and filtered side by side, because filtering changes what is being benchmarked. |
| [`src/recall_compare.py`](src/recall_compare.py) | Two finished runs side by side, with guards that refuse to compare runs over different notes or an oracle against a model run. |
| [`colab_runner_recall.ipynb`](colab_runner_recall.ipynb) | The T4 runner: deps, HF login via `getpass`, Drive mount for run state, manual upload of the input, then oracle → smoke → full run → L5. |
| `tests/test_recall_{data,matching,prompt,scoring,judge}.py` | 131 tests, CPU-only, no model download. The measured pair tables are pinned here, so a rule change that invalidates the report's own justification fails the suite. |

**Reused unchanged, by import — never by edit.**

| file | what is borrowed |
|---|---|
| [`src/chunking.py`](src/chunking.py) | `chunk_windows` and `tokenize_with_spans`. Same 400/80 geometry as the term-NER run, so its measured ~68 s/chunk carries over. |
| [`src/model.py`](src/model.py) | `load_medgemma`, `run_messages`, `count_tokens`. 4-bit, greedy. |
| [`src/prompt_mdace.py`](src/prompt_mdace.py) | The tolerant JSON scanner and the truncation salvage. Reimplementing them would have meant two parsers drifting apart. |
| [`src/term_scoring.py`](src/term_scoring.py) | `wilson_ci` only — one 10-line function, imported rather than copied so the two sets of intervals cannot disagree. |

**What a run writes.** The split is deliberate: the counts file can be opened and
shared without a second thought, and everything else cannot.

| path | contents | shareable |
|---|---|---|
| `outputs/mdace_recall/<tag>/per_note.jsonl` | integer counts, the resume marker | yes |
| `outputs/mdace_recall/<tag>/findings.jsonl` | the `{span, name}` lists | **no — note text** |
| `outputs/mdace_recall/<tag>/new_pairs_L*.jsonl` | what each level newly accepted | **no — note text** |
| `outputs/mdace_recall/<tag>/raw_replies.jsonl` | raw model output (`--dump-replies`) | **no — note text** |
| `outputs/mdace_recall/<tag>/verdicts_L*.jsonl` | L5 verdicts | **no — note text** |
| `results/mdace_recall_<label>.md` + `_metrics.json` | the report | yes |

`<tag>` encodes model, chunk geometry, token cap **and a hash of the prompt**.
That last part is load-bearing — see step 6 below.

### How the problem was navigated

The plan document specified *what* to build. These are the decisions taken while
building it, in the order they came up, including the two places the plan turned
out to be wrong.

**1. Check the plan's arithmetic before writing a line of code.** Everything
downstream rests on the input being what the plan says it is, and confirming it
costs thirty seconds. 24 notes, 82 chunks, 100 rows, 91 distinct codes, 99
evidence phrases, 91 descriptions, 142 SNOMED forms, median 4 accepted forms per
row. All held. The plan's measured pair table was also reproduced independently,
to confirm its `contains` / `dice` / `char` columns meant what this
implementation would mean by them — they did, exactly.

**2. Read the existing code first to decide what *not* to write.** The constraint
was that the term-NER modules stay untouched so their 0.53 result stays
reproducible. That makes "reuse" mean *import*, not *copy* and not *edit* — which
is why the parser borrows the previous branch's JSON scanner and the scorer
borrows its confidence-interval function.

**3. Build the data layer, then make an oracle prove it.** The oracle feeds the
gold accept-sets back through chunking, parsing, normalization and matching with
no model involved. Recall must come out at 1.0000. It is worth doing before any
GPU time because it is the only way to tell a harness bug from a model result,
and it caught real bugs twice on the previous branch.

The oracle was deliberately built stronger than the plan asked: it emits every
accepted form, not just the evidence phrase, so all three gold sources have to
come out at 1.0000. An oracle that only ever produced note wording would leave
the description and SNOMED paths unverified — which is exactly where a
per-source bug would hide.

**4. The oracle found a bug on its first run — in the oracle.** 16 false
positives. The cause: nine codes are evidenced twice on their own note, so a gold
form reachable from two rows was emitted with one row's phrasing in one window
and the other's in the next, and both survived deduping. The matcher was right;
the fake model was wrong. Fixing the fake was the correct repair, and resisting
the urge to "fix" the matcher to make the number go green is most of what this
check is for.

**5. The plan's matching rule could not pass its own harness check.** The plan
says *one prediction to at most one gold form, greedy best-first*. The first half
is essential — without it a single vague prediction satisfies several gold
entries and recall measures vagueness. But plain greedy breaks on ties: when
several findings can reach the same form, one takes a form another needed, and
the oracle lands under 1.0000 with no bug present.

So each level now runs greedy by rule strictness then score — best-first survives
wherever it is meaningful — and then augments to maximum cardinality over what is
left. Assignments made at lower levels are frozen, so a pair credited to L2 stays
credited to L2 and per-level attribution still holds. The failing case is pinned
as a test rather than described in a comment.

**6. Key the cache on the prompt, because this has already gone wrong once.**
Re-running after a prompt edit used to print `cached 24, to run 0` and report the
*old* prompt's numbers in about a second, with no error — the worst kind of bug,
because it looks like success. The run directory now carries a hash of the
prompt, and the report header prints it, so a shared artifact can be traced back
to the exact prompt that produced it. The residual rule: **if a run finishes
suspiciously fast, check the `run dir:` line before believing anything.**

**7. Measure L4 instead of assuming it.** The plan treats embeddings as the level
that finally works, since it is the only one that reaches abbreviations at all.
It reaches them — and it does not separate them. Sorted by cosine, the pairs we
want and the pairs we must reject interleave from top to bottom: `acute renal
failure` against `chronic renal failure` scores **0.833**, above eight of the ten
abbreviation pairs L4 exists to catch. Two other biomedical encoders were
measured and are worse.

So the plan's own argument — *no threshold separates the good pairs from the bad
ones* — turns out to hold one level higher up as well. The consequences: the
default cosine is a floor chosen to reach every abbreviation (0.60) rather than a
cutoff that works, the report marks the L4 row **provisional**, and **L5 stops
being a refinement of L4 and becomes the thing that makes L4's gain
interpretable**. The numbers live in a code constant with a test asserting the
property, so the finding cannot decay into prose that has quietly stopped being
true.

**8. Prove the pipeline is not degenerate on imperfect input.** A perfect oracle
only shows that nothing is broken when everything matches. A throwaway stand-in
model — gold with 20% of rows dropped, typos injected, phrases truncated, and
invented noise added — showed recall climbing 0.59 → 0.68 → 0.82 → 0.82 across
the ladder with each level's gain attributed, and the per-source split telling a
coherent story about whose wording the model produced.

That run also surfaced something worth writing down: combined *form* recall has a
ceiling set by how many phrasings the model emits, not by how much it found,
because a row needs only one of its four forms matched and matching is 1:1. So
the report now says in as many words: **quote rows or codes, not forms.**

**9. Verify the PHI boundary rather than assume it.** Checked that no 30-character
window of any note text appears in the committed report, and that every string
value in the metrics JSON is a level name, a model id or a path.

**10. Then measure the fix before building it.** The obvious next move was to stop
sending the model whole notes. The failure analysis killed it: the false positives
are spread thin, the biggest single section holds 8% of them, and the sections
producing the most are the same ones holding the most gold — Brief Hospital Course
has 19 of the 100 gold phrases and 94 false positives. Applied on top of the
filter, section filtering bought +0.6 precision points and cost an answer.

What worked instead was splitting extraction from filtering: a second call per
finding asking whether a coder would bill it. Precision 0.1135 → 0.2311, volume
51 → 21.5 findings per note, recall 0.78 → 0.67. It dropped 710 findings and 690
of them deserved to go.

**11. And ask the bare question first.** The filter's default prompt names no
categories to avoid — only "would a coder assign a billing code to this?". It got
97% of those calls right, which means the model already knows what is billable and
simply does not apply that while extracting. A hinted variant exists, but if only
the hinted one worked that would be a fact about the model rather than a fix, and
the report has to be able to say which.

### Running it

```bash
# No GPU. ~10 s. Every source must read 1.0000 at L1 with zero combined FPs.
make recall-oracle

# On a GPU (Colab T4)
make recall-smoke          # 3 longest notes, keeps raw replies
make recall-run            # 24 notes / 82 chunks, ~1-1.5 h. Resumable.

# No GPU: re-score a finished run with different thresholds
make recall-rescore

# L5 adjudication of what levels 2-4 newly accepted
make recall-judge          # MedGemma grades its own matches, and says so
make recall-questions      # writes the questions out for a human instead
```

**The smoke run picks the longest notes, not the first three.** Records are
ordered longest-first precisely so that the multi-chunk path — where OOM and
truncation actually live — runs before anything is committed to.

CLI: `--limit N`, `--smoke N`, `--oracle`, `--score-only`, `--dice-min`,
`--ratio-min`, `--cosine-min`, `--no-embed`, `--embed-model`, `--chunk-words`,
`--overlap-words`, `--max-new-tokens`, `--dump-replies`, `--no-resume`,
`--model`, `--model-name`.

Colab: [`colab_runner_recall.ipynb`](colab_runner_recall.ipynb).

L4 needs `sentence-transformers` (`uv sync --extra embed`). It is imported
lazily: without it the ladder stops at L3, the report says so, and L1–L3 stay
runnable on a CPU-only laptop.

---

## Tests

CPU-only, no GPU, no model download, no dataset download — and **no real note
text or label files**; every MIMIC fixture is synthetic, so the suite runs
anywhere.

- `tests/test_parse.py` — JSON parsing (plain, fenced, prose-wrapped, malformed,
  type normalization, braces-in-strings).
- `tests/test_align.py` — span→BIO alignment (multi-word, case/punctuation,
  repeated occurrences, no-overwrite, absent spans).
- `tests/test_scoring.py` — seqeval integration + full parse→align→score chain,
  including graceful degradation on bad/failed replies.
- `tests/test_chunking.py` — window tiling (total coverage, stride, clamping),
  BIO↔span round-trips, all-or-nothing overlap painting, chunk-merge dedupe,
  char↔token offset conversion.
- `tests/test_mimic_labels.py` — label filename parsing, CSV reading, dedupe and
  the type-priority tie-break, deterministic sampling and the subset property,
  gold-BIO construction from char offsets.
- `tests/test_mimic_pipeline.py` — the note pipeline with a fake model: chunk
  coverage, overlap dedupe, multi-occurrence expansion, generation-cap logging,
  incremental save, resume, partial-line tolerance, and assertions that the
  resume file and the report contain **no note text**.

MDACE recall benchmark (131 of the tests; the real input file is credentialed, so
the tests that need it skip cleanly without it and pin every headline denominator
when it is present):

- `tests/test_recall_data.py` — normalization identical to the term-NER
  normalizer, accept-set construction from three columns, source tagging,
  per-source denominators, and the guarantee that every gold phrase is a literal
  substring of its own note.
- `tests/test_recall_matching.py` — the two **measured pair tables**, pinned. If a
  rule change moves any of them, the report's own threshold justification has
  stopped being true and the suite fails. Plus the 1:1 constraint, ladder
  monotonicity, per-level attribution, and the tie case that plain greedy loses.
- `tests/test_recall_prompt.py` — every reply shape a 4B model actually produces:
  fences, prose, bare strings, alternative key names, type labels that must never
  become findings, and two kinds of truncation.
- `tests/test_recall_scoring.py` — recall in three units, the per-source
  false-positive trap, the hallucination check, and an assertion that the
  committed report contains **no note text and no gold phrases**.
- `tests/test_recall_judge.py` — verdict parsing, and that an *unreadable* verdict
  keeps its pair rather than rejecting it.

```bash
uv run pytest -q
```

---

## Architecture

Shared by both evaluations:

```
src/
  prompt.py            prompt construction + robust JSON reply parsing
                       (parse_entities takes valid_types, so both label spaces reuse it)
  align.py             predicted span → BIO alignment onto a token list
  model.py             MedGemma pipeline load + inference (lazy torch import)
  scoring.py           seqeval report + metrics-CSV writer (filenames parameterized)
```

Disease/Chemical (NCBI + BC5CDR):

```
  config.py            model id, harmonized label space, generation config
  datasets/
    base.py            adapter ABC (vendored)
    ncbi.py            NCBI Disease adapter (vendored, verbatim)
    bc5cdr.py          BC5CDR adapter (vendored, verbatim)
    __init__.py        build_test_set() → combined NCBI+BC5CDR test split
  evaluate.py          orchestration + CLI (--limit smoke mode)
colab_runner.ipynb     runner (deps, HF login, T4, smoke → full eval)
results/
  comparison.csv       model × entity scores
  full_report.json     full-precision seqeval report
```

MIMIC-IV medication NER — only the data loading and chunking are new:

```
  mimic_config.py      source paths (OUTSIDE the repo), 6-type label space,
                       type priority, excluded notes, seed, chunk + gen settings
  datasets/
    mimic_meds.py      label filename/CSV parsing, dedupe + tie-break, seeded
                       sampling, note extraction, gold BIO from char offsets
  chunking.py          overlapping windows, BIO↔span, chunk merge + dedupe,
                       char↔token offset conversion
  prompt_mimic.py      medication prompt (reuses prompt.parse_entities)
  evaluate_mimic.py    orchestration, incremental save + resume, --oracle, CLI
  report_mimic.py      markdown report writer (aggregate metrics ONLY)
  analyze_replies.py   repetition/truncation analysis of raw_replies.jsonl
                       (structure and counts only — never note text)
  build_mimic_sample.py  LOCAL-ONLY sample extraction from the credentialed data
colab_runner_mimic.ipynb  T4 runner (manual sample upload, 5 → 50 → 100)
results/
  mimic_ner_{50,100}.csv        aggregate metrics — safe to commit
  mimic_ner_{50,100}_report.md  human-readable report — safe to commit
  mimic_ner_align_mode_comparison.md  why first-per-chunk is the default
  mimic_ner_smoke_history.md          n=5 runs, what each fix bought
data/samples/          GITIGNORED — extracted note text
outputs/mimic/         GITIGNORED — per-note run state; error dumps quote notes
```

**Model id / label:** default model `google/medgemma-4b-it`; the `model` column
is `medgemma-4b-it` (`medgemma-4b-it-ORACLE` for `--oracle` runs).

Note the two label spaces are **disjoint**, so `mimic_ner_*.csv` shares the schema
of `comparison.csv` but does not concatenate meaningfully with it.

---

## Security and data handling

Never hardcode or commit a Hugging Face token. The notebooks read it at runtime
via `getpass`; `.gitignore` also excludes common token filenames and caches.

**MIMIC-IV is real patient data and this repo is public.** `.gitignore` blocks
`data/`, `samples/`, `outputs/`, `*.csv.gz`, and note-directory names, and
un-ignores only the aggregate `results/mimic_ner_*` metrics and reports. Run
`make mimic-check` before committing — it fails if any data file became tracked.
When using `colab_runner_mimic.ipynb`, clear all cell outputs before saving.
