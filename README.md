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

## Claude Haiku baseline (API, no GPU)

A parallel baseline runs the **exact same** dataset, prompt, JSON parsing,
span→BIO alignment, and seqeval scorer — the **only** variable is the model.
Instead of a local MedGemma pipeline, each test sentence is sent to the
**Anthropic Messages API** (`claude-haiku-4-5`, `temperature=0`) via the official
`anthropic` SDK. No GPU, no `transformers` — it runs on CPU locally.

Results go to **separate** files so the MedGemma output is never overwritten:

| | MedGemma | Haiku |
|---|---|---|
| CSV | `results/comparison.csv` | `results/haiku_comparison.csv` |
| JSON | `results/full_report.json` | `results/haiku_full_report.json` |

The CSV schema is identical (`model, entity, precision, recall, f1, support`) and
the `model` column is `claude-haiku-4-5`, so the Haiku row concatenates with the
MedGemma and sibling BERT tables the same way.

### `.env` setup (public repo — key is never committed)

```bash
cp .env.example .env
# edit .env and set your real key:
#   ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is git-ignored (`.env.example`, a placeholder, is committed). The key is
loaded at runtime via `python-dotenv` from `ANTHROPIC_API_KEY` — it is never
hardcoded, printed, or committed. Verify it is ignored:

```bash
git check-ignore .env        # prints ".env" if correctly ignored
```

### Run it

```bash
pip install uv && uv sync --extra test    # installs anthropic + python-dotenv

# Smoke test: does a single 1-sentence API call to confirm the key + model,
# THEN evaluates the first 10 examples.  (make haiku-smoke)
python -m src.evaluate_haiku --limit 10

# Full eval -> results/haiku_comparison.csv        (make haiku-eval)
python -m src.evaluate_haiku
```

Every run prints estimated token usage and a rough cost at the end. CLI flags:
`--limit N`, `--model <id>`, `--model-name <label>`, `--no-preflight` (skip the
1-sentence check). A small inter-request delay and retry-with-backoff handle API
rate limits.

---

## Tests

CPU-only, no GPU, no model download, no dataset download — they exercise the
pure logic (`prompt`, `align`, `scoring`, and the orchestration glue with an
injected fake reply):

- `tests/test_parse.py` — JSON parsing (plain, fenced, prose-wrapped, malformed,
  type normalization, braces-in-strings).
- `tests/test_align.py` — span→BIO alignment (multi-word, case/punctuation,
  repeated occurrences, no-overwrite, absent spans).
- `tests/test_scoring.py` — seqeval integration + full parse→align→score chain,
  including graceful degradation on bad/failed replies.
- `tests/test_haiku.py` — Claude Haiku backend with a **mocked** Anthropic API
  (no network): reply parsing, usage accumulation, request shape, retry-on-429,
  and the full parse→align pipeline.

```bash
uv run pytest -q
```

---

## Architecture

```
src/
  config.py            model id, harmonized label space, generation config
  datasets/
    base.py            adapter ABC (vendored)
    ncbi.py            NCBI Disease adapter (vendored, verbatim)
    bc5cdr.py          BC5CDR adapter (vendored, verbatim)
    __init__.py        build_test_set() → combined NCBI+BC5CDR test split
  prompt.py            prompt construction + robust JSON reply parsing
  align.py             predicted span → BIO alignment onto the token list
  model.py             MedGemma pipeline load + text-only inference (lazy torch import)
  haiku_model.py       Claude Haiku API inference (anthropic SDK, .env key, retry/backoff)
  scoring.py           seqeval report + comparison.csv writer (sibling-identical)
  evaluate.py          MedGemma orchestration + CLI (--limit smoke mode)
  evaluate_haiku.py    Claude Haiku orchestration + CLI (1-sentence preflight, cost print)
colab_runner.ipynb     PRIMARY runner (deps, HF login, T4, smoke → full eval)
tests/                 CPU-only unit tests
results/
  comparison.csv       MedGemma model × entity scores
  full_report.json     MedGemma full-precision seqeval report
  haiku_comparison.csv MedGemma-schema scores for claude-haiku-4-5 (separate file)
  haiku_full_report.json  Haiku full-precision seqeval report
```

**Model id / label:** default model `google/medgemma-4b-it`; the `model` column
in `comparison.csv` is `medgemma-4b-it`.

---

## Security

Never hardcode or commit a Hugging Face token. The notebook reads it at runtime
via `getpass`; `.gitignore` also excludes common token filenames and caches.

For the Haiku baseline, the **Anthropic API key** is read from a git-ignored
`.env` file via `python-dotenv` (`ANTHROPIC_API_KEY`) — never hardcoded, printed,
or committed. Only the placeholder `.env.example` is committed.
