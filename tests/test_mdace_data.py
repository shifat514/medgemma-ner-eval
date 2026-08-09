"""MDACE data loading, the chart-type rule, and sample reproducibility.

The synthetic tests run anywhere. The corpus tests at the bottom need the real
credentialed files and skip cleanly without them — but when they do run they are
the guard that stops a refactor from silently redrawing the evaluation sample.
"""

import json
import os

import pytest

from src.datasets.mdace import (
    build_corpus,
    load_sample_100,
    normalize_term,
    note_chart_type,
    stratified_sample,
)
from src.mdace_config import DATASET_FILE, NOTES_FILE, SAMPLE_100_FILE, code_bucket


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Depression", "depression"),
    ("  COPD,  ", "copd"),
    ("chronic pancreatitis.", "chronic pancreatitis"),
    ("Tobacco history:  quit smoking", "tobacco history quit smoking"),
    ("third-degree  heart   block", "third degree heart block"),
    ("", ""),
    (None, ""),
    (123, ""),
])
def test_normalize_term(raw, expected):
    assert normalize_term(raw) == expected


def test_normalize_collapses_newlines():
    """Some gold spans run across lines; the newline must not survive."""
    assert normalize_term("acute renal\nfailure") == "acute renal failure"
    assert normalize_term("a\n\n  b\tc") == "a b c"


def test_normalize_is_idempotent():
    once = normalize_term("Chronic Kidney Disease, Stage 3")
    assert normalize_term(once) == once


# --------------------------------------------------------------------------
# Note-level chart type
# --------------------------------------------------------------------------

def test_chart_type_single():
    assert note_chart_type([{"chart_type": "Profee"}]) == "Profee"
    assert note_chart_type([{"chart_type": "Inpatient"}]) == "Inpatient"


def test_chart_type_mixed_resolves_to_inpatient():
    """118 corpus notes are billed both ways; Inpatient wins.

    This rule is what reproduces the 604/470 split and both word/term medians.
    Flipping it silently rebuilds both strata.
    """
    rows = [{"chart_type": "Profee"}, {"chart_type": "Inpatient"},
            {"chart_type": "Profee"}]
    assert note_chart_type(rows) == "Inpatient"


# --------------------------------------------------------------------------
# Code bucketing (drives the conditions-only slice)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("system,code,bucket", [
    ("ICD-10-CM", "I10", "conditions"),
    ("ICD-10-CM", "E78.5", "conditions"),
    ("ICD-10-CM", "Z87.891", "status_history"),
    ("ICD-10-CM", "Z79.82", "status_history"),
    ("ICD-10-CM", "S72.001A", "injury"),
    ("ICD-10-CM", "T39.1X1A", "injury"),
    ("CPT", "99213", "procedure"),
    ("ICD-10-PCS", "BF10YZZ", "procedure"),
    ("ICD-9-CM", "401.9", "conditions"),
])
def test_code_bucket(system, code, bucket):
    assert code_bucket(system, code) == bucket


# --------------------------------------------------------------------------
# Sampling determinism
# --------------------------------------------------------------------------

def _fake_records(n_profee=40, n_inpatient=40):
    recs = []
    for i in range(n_profee):
        recs.append({"note_id": 1000 + i, "chart_type": "Profee"})
    for i in range(n_inpatient):
        recs.append({"note_id": 2000 + i, "chart_type": "Inpatient"})
    return recs


def test_stratified_sample_is_deterministic():
    recs = _fake_records()
    a = stratified_sample(recs, n_per_stratum=5, seed=13)
    b = stratified_sample(recs, n_per_stratum=5, seed=13)
    assert a == b


def test_stratified_sample_balances_strata():
    recs = _fake_records()
    picked = set(stratified_sample(recs, n_per_stratum=5, seed=13))
    assert len(picked) == 10
    assert sum(1 for n in picked if n < 2000) == 5
    assert sum(1 for n in picked if n >= 2000) == 5


def test_stratified_sample_ignores_input_order():
    """Sorting the pools means dict/file ordering cannot leak into the draw."""
    recs = _fake_records()
    shuffled = list(reversed(recs))
    assert (stratified_sample(recs, n_per_stratum=5, seed=13)
            == stratified_sample(shuffled, n_per_stratum=5, seed=13))


def test_shared_generator_differs_from_per_stratum_generator():
    """The draw ORDER is part of the spec, not an implementation detail.

    A shared generator makes the second stratum continue from the first's state.
    A fresh generator per stratum is equally defensible and gives a different
    sample from the same seed — 167 chunks / 432 terms instead of 122 / 324.
    This test exists so that difference can never be mistaken for noise.
    """
    import random
    recs = _fake_records()
    shared = stratified_sample(recs, n_per_stratum=5, seed=13)
    inpatient_shared = shared[5:]

    pool = sorted(r["note_id"] for r in recs if r["chart_type"] == "Inpatient")
    inpatient_fresh = random.Random(13).sample(pool, 5)
    assert inpatient_shared != inpatient_fresh


def test_stratified_sample_rejects_undersized_stratum():
    with pytest.raises(ValueError, match="Profee"):
        stratified_sample(_fake_records(n_profee=3), n_per_stratum=5, seed=13)


# --------------------------------------------------------------------------
# sample_100 loading
# --------------------------------------------------------------------------

def test_load_sample_100_normalizes_and_dedupes(tmp_path):
    path = tmp_path / "s100.jsonl"
    rows = [
        {"note_id": 1, "mdace_gold_evidence_text": "Depression",
         "gold_code_description": "Major depressive disorder"},
        {"note_id": 1, "mdace_gold_evidence_text": "depression,",
         "gold_code_description": "Major depressive disorder"},
        {"note_id": 2, "mdace_gold_evidence_text": "HTN",
         "gold_code_description": "Essential (primary) hypertension"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    got = load_sample_100(str(path))
    assert got[1]["terms"] == ["depression"]          # both rows collapse to one
    assert got[1]["descrs"] == ["major depressive disorder"]
    assert got[2]["terms"] == ["htn"]


# --------------------------------------------------------------------------
# Corpus tests — need the real files
# --------------------------------------------------------------------------

_HAVE_DATA = os.path.exists(DATASET_FILE) and os.path.exists(NOTES_FILE)
needs_data = pytest.mark.skipif(
    not _HAVE_DATA, reason="MDACE source files not present on this machine"
)


@needs_data
def test_corpus_joins_cleanly():
    _records, stats = build_corpus(DATASET_FILE, NOTES_FILE)
    assert stats["n_rows"] == 9499
    assert stats["n_notes_joined"] == 1074
    assert stats["n_notes_missing_text"] == 0
    # Every evidence span must slice its own note exactly. If this fails, the
    # annotation file and the notes file have drifted and nothing downstream
    # can be trusted.
    assert stats["n_offset_mismatches"] == 0


@needs_data
def test_normalizer_reproduces_the_corpus_pair_count():
    """7,663 distinct (note_id, term) pairs.

    Matching this exactly is what shows our normalizer is the same rule that
    produced the upstream corpus figures, rather than merely a similar one.
    """
    _records, stats = build_corpus(DATASET_FILE, NOTES_FILE)
    assert stats["n_distinct_note_term_pairs"] == 7663


@needs_data
def test_chart_type_rule_reproduces_the_corpus_split():
    records, stats = build_corpus(DATASET_FILE, NOTES_FILE)
    counts = {}
    for r in records:
        counts[r["chart_type"]] = counts.get(r["chart_type"], 0) + 1
    assert counts == {"Inpatient": 604, "Profee": 470}
    assert stats["n_mixed_chart_type"] == 118


@needs_data
def test_stratified_draw_is_pinned_to_122_chunks_and_324_terms():
    """The evaluation sample, nailed down.

    Seed 13, pools sorted, one shared generator, Profee drawn first. If this
    fails, the sample changed and no result computed before the change is
    comparable with one computed after it.
    """
    from src.datasets.mdace import build_sample

    _records, stats = build_sample(DATASET_FILE, NOTES_FILE, SAMPLE_100_FILE)
    assert stats["n_stratified"] == 50
    assert stats["stratified_by_type"] == {"Profee": 25, "Inpatient": 25}
    assert stats["chunks_stratified"] == 122
    assert stats["gold_terms_stratified"] == 324


@needs_data
@pytest.mark.skipif(not os.path.exists(SAMPLE_100_FILE),
                    reason="sample_100 file not present")
def test_sample_100_is_an_incomplete_answer_key():
    """The 100-row cut ships 99 of the 195 gold terms its 24 notes carry.

    This gap is the entire point of view A1 vs B1, so it is asserted rather
    than described.
    """
    from src.datasets.mdace import build_sample

    _records, stats = build_sample(DATASET_FILE, NOTES_FILE, SAMPLE_100_FILE)
    assert stats["n_sample100"] == 24
    assert stats["gold_terms_sample100_full"] == 195
    assert stats["gold_terms_sample100_shipped"] == 99
    assert stats["n_overlap"] == 1
    assert stats["chunks_total"] == 202
