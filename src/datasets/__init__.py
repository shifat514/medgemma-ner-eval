"""Vendored dataset loaders — the combined NCBI + BC5CDR test split.

Self-contained copy of clinical-ner-eval's loading logic so this repo runs
standalone in Colab (no runtime import from the sibling repo). It produces the
SAME test set — same Hub configs, whitespace tokenization, harmonized labels —
so MedGemma's scores are directly comparable to the sibling's comparison.csv.

The sibling's DEFAULT_COMBINATION also lists "maccrobat", but that dataset is
not on the Hub and is skipped there, so it contributes nothing and is omitted
here. The effective test set is identical: NCBI test + BC5CDR test (~1,363 gold
entities: 707 Disease + 656 Chemical).
"""

from datasets import concatenate_datasets

from ..config import HARMONIZED_LABELS
from .bc5cdr import BC5CDRAdapter
from .ncbi import NCBIDiseaseAdapter

# Registry kept so Phase 2 (fine-tuning) can add train/val loading + new corpora.
REGISTRY = {
    "ncbi": NCBIDiseaseAdapter,
    "bc5cdr": BC5CDRAdapter,
}

DEFAULT_COMBINATION = ["ncbi", "bc5cdr"]


def build_test_set(names=None):
    """Return (Dataset, labels) for the combined test split.

    Each example has `tokens` (list[str]) and `bio` (list[str]) over the
    harmonized label space. Splits are concatenated in `names` order to mirror
    the sibling repo.
    """
    names = names or DEFAULT_COMBINATION
    keep = ["tokens", "bio"]
    parts = []
    for n in names:
        adapter = REGISTRY[n](label_mode="harmonized")
        ds = adapter.load()
        if "test" not in ds:
            continue
        test = ds["test"]
        drop = [c for c in test.column_names if c not in keep]
        parts.append(test.remove_columns(drop))
    if not parts:
        raise RuntimeError("no test splits loaded")
    return concatenate_datasets(parts), HARMONIZED_LABELS
