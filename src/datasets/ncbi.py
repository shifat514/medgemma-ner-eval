"""NCBI Disease adapter using bigbio/ncbi_disease (bigbio_kb config).

Vendored verbatim from clinical-ner-eval so this repo produces the identical
test set standalone. Do not "improve" the tokenization or offset logic — any
drift here breaks comparability with the sibling repo's results.
"""

import re

from datasets import Dataset, DatasetDict, load_dataset

from .base import BaseNERAdapter

# All entity types in NCBI Disease corpus map to Disease
_DISEASE_TYPES = {
    "Disease", "SpecificDisease", "CompositeMention", "DiseaseClass", "Modifier",
}
_LABELS = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]
_TOKEN_RE = re.compile(r"\S+")


def _doc_to_example(doc, label_mode):
    passages = sorted(doc["passages"], key=lambda p: p["offsets"][0][0])
    if not passages:
        return None
    text = passages[0]["text"][0]
    base = passages[0]["offsets"][0][0]

    tokens, spans = [], []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((base + m.start(), base + m.end()))
    if not tokens:
        return None

    bio = ["O"] * len(tokens)
    for entity in doc["entities"]:
        etype = entity["type"]
        if label_mode == "harmonized":
            if etype in _DISEASE_TYPES:
                prefix = "Disease"
            else:
                continue
        else:
            prefix = etype

        for char_start, char_end in entity["offsets"]:
            first = True
            for ti, (ts, te) in enumerate(spans):
                if te > char_start and ts < char_end:
                    if bio[ti] == "O":
                        bio[ti] = f"B-{prefix}" if first else f"I-{prefix}"
                    first = False

    return {"tokens": tokens, "bio": bio}


class NCBIDiseaseAdapter(BaseNERAdapter):
    def load(self) -> DatasetDict:
        # bigbio/ncbi_disease requires the `bioc` package (part of the loading script).
        # ncbi_disease_bigbio_kb is the only available bigbio config.
        raw = load_dataset(
            "bigbio/ncbi_disease", name="ncbi_disease_bigbio_kb", trust_remote_code=True
        )

        split_examples = {}
        for split in raw:
            examples = [
                ex for doc in raw[split]
                for ex in [_doc_to_example(doc, self.label_mode)]
                if ex
            ]
            split_examples[split] = Dataset.from_list(examples)

        return DatasetDict(split_examples)

    def label_list(self):
        return _LABELS
