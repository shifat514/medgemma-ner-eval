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

- **Alignment is per chunk.** `align_entities_to_bio` tags *every* occurrence of a
  predicted string; run on a whole 10k-char note, one prediction of `aspirin`
  becomes ~15 predicted spans. Per-chunk alignment confines that to the window the
  model actually read. It **narrows but does not eliminate** the effect, so the
  report prints the measured expansion factor.
- **Predictions are deduped, not just gold.** Overlapping windows re-read the same
  tokens, so entities in an overlap get predicted twice. Merging dedupes on
  `(start, end, type)` after stitching, and reports how many were removed.

### The harness has a structural ceiling — measure it

`--oracle` feeds the gold spans back through the identical pipeline. No model, no
GPU, runs in seconds. On n=100 it scores **micro P=0.698, R=0.951, F1=0.805** —
so a *perfect* extractor still loses ~30 points of precision to string-matching
alignment. Read MedGemma's numbers against that, not against 1.0.

```bash
make mimic-oracle       # python -m src.evaluate_mimic --oracle --n 100
```

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
`--no-resume`, `--dump-errors`, `--oracle`, `--model`, `--model-name`.

Colab: [`colab_runner_mimic.ipynb`](colab_runner_mimic.ipynb) — T4, deps, HF login
via `getpass`, **manual upload** of the sample file (no Drive mount, no data
clone), then 5 → 50 → 100.

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
  build_mimic_sample.py  LOCAL-ONLY sample extraction from the credentialed data
colab_runner_mimic.ipynb  T4 runner (manual sample upload, 5 → 50 → 100)
results/
  mimic_ner_{50,100}.csv        aggregate metrics — safe to commit
  mimic_ner_{50,100}_report.md  human-readable report — safe to commit
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
