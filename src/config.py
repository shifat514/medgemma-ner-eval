"""Configuration for the MedGemma clinical-NER evaluation.

Kept deliberately parallel to clinical-ner-eval/src/config.py: same harmonized
label space and the same `label_maps` helper, so the two repos score on an
identical target and their comparison.csv files concatenate cleanly.
"""

import os

# Default gated model. Override via --model or MEDGEMMA_MODEL.
MODEL_ID = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")

# Row label written to comparison.csv (matches the sibling repo's `model` column).
MODEL_NAME = os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-4b-it")

# Identical to clinical-ner-eval's HARMONIZED_LABELS — the shared BIO target.
HARMONIZED_LABELS = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]

# The entity types MedGemma is asked to emit (the two harmonized types).
ENTITY_TYPES = ["Disease", "Chemical"]


def label_maps(labels):
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    return label2id, id2label


# Override at runtime — Colab may point this at a Drive path.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# ── Claude Haiku zero-shot backend (parallel API baseline to MedGemma) ──────
# Same dataset / prompt / parsing / alignment / scoring as the MedGemma run; the
# ONLY variable is the model. Instead of a local pipeline, each sentence goes to
# the Anthropic Messages API (no GPU, no transformers). See src/haiku_model.py.

# Anthropic model id — kept here so the run is reproducible.
HAIKU_MODEL = os.environ.get("HAIKU_MODEL", "claude-haiku-4-5")

# Row label written to the "model" column of results/haiku_comparison.csv.
HAIKU_MODEL_NAME = os.environ.get("HAIKU_MODEL_NAME", "claude-haiku-4-5")

# Deterministic decoding (Haiku 4.5 accepts temperature; the 4.7/4.8 family does not).
HAIKU_TEMPERATURE = 0
HAIKU_MAX_TOKENS = 512

# Politeness delay between successful calls + retry-with-backoff for rate limits.
HAIKU_REQUEST_DELAY = 0.2      # seconds slept after each successful call
HAIKU_MAX_RETRIES = 5          # attempts before giving up on one sentence
HAIKU_BACKOFF_BASE = 2.0       # seconds; delay = base * 2**attempt

# claude-haiku-4-5 pricing (USD per 1M tokens) — used only for the cost estimate.
HAIKU_PRICE_INPUT_PER_MTOK = 1.00
HAIKU_PRICE_OUTPUT_PER_MTOK = 5.00

# The env var holding the API key — loaded from .env via python-dotenv.
# NEVER hardcode, print, or commit the key.
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Greedy decoding for reproducibility (do_sample=False, per the MedGemma card).
GEN_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
}

# 4-bit quantization so google/medgemma-4b-it fits a free Colab T4 (~5-7GB VRAM).
LOAD_IN_4BIT = True
