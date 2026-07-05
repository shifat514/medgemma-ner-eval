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
  scoring.py           seqeval report + comparison.csv writer (sibling-identical)
  evaluate.py          orchestration + CLI (--limit smoke mode)
colab_runner.ipynb     PRIMARY runner (deps, HF login, T4, smoke → full eval)
tests/                 CPU-only unit tests
results/
  comparison.csv       model × entity scores (written by a run)
  full_report.json     full-precision seqeval report
```

**Model id / label:** default model `google/medgemma-4b-it`; the `model` column
in `comparison.csv` is `medgemma-4b-it`.

---

## Security

Never hardcode or commit a Hugging Face token. The notebook reads it at runtime
via `getpass`; `.gitignore` also excludes common token filenames and caches.
