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

# Greedy decoding for reproducibility (do_sample=False, per the MedGemma card).
GEN_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
}

# 4-bit quantization so google/medgemma-4b-it fits a free Colab T4 (~5-7GB VRAM).
LOAD_IN_4BIT = True
