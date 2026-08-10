"""Loading the 100-row cut, and building an accept-set out of three columns.

The synthetic tests run anywhere. The tests at the bottom need the real
credentialed file and skip cleanly without it — but when they do run they pin
the numbers the report quotes, so a refactor cannot quietly change what the
benchmark is measured against.
"""

import json
import os

import pytest

from src.datasets.mdace_recall import (
    build_notes,
    normalize_term,
    padded_note_norm,
    reachable_codes,
    reachable_rows,
    row_accept_set,
    snomed_coverage,
    source_forms,
)
from src.recall_config import SAMPLE_100_FILE


def _row(note_id=1, code="I10", system="ICD-10-CM", evidence="HTN",
         descr="Essential (primary) hypertension", snomed=(), text=None,
         count=None):
    return {
        "note_id": note_id,
        "hadm_id": 100,
        "chart_type": "Inpatient",
        "code_system": system,
        "gold_code": code,
        "gold_code_description": descr,
        "mdace_gold_evidence_text": evidence,
        "gold_snomed_concepts": [{"concept_id": str(i), "term": t}
                                 for i, t in enumerate(snomed)],
        "gold_snomed_concept_count": len(snomed) if count is None else count,
        "note_text": text if text is not None else f"The patient has {evidence}.",
    }


def _write(tmp_path, rows):
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# Normalization — must stay identical to the term-NER normalizer, or the 0.53
# reference number stops being comparable.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Depression", "depression"),
    ("  COPD,  ", "copd"),
    ("Essential (primary) hypertension", "essential primary hypertension"),
    ("acute renal\nfailure", "acute renal failure"),
    ("", ""),
    (None, ""),
    (123, ""),
])
def test_normalize_term(raw, expected):
    assert normalize_term(raw) == expected


def test_normalizer_matches_the_term_ner_one():
    from src.datasets.mdace import normalize_term as term_ner_normalize

    for raw in ("HTN", "Third-Degree Heart Block", "sepsis.\n", "CO2  retention"):
        assert normalize_term(raw) == term_ner_normalize(raw)


def test_padded_note_norm_gives_whole_token_containment():
    note = padded_note_norm("Patient underwent CABG today.")
    assert " cabg " in note
    assert " ca " not in note


# --------------------------------------------------------------------------
# The accept-set
# --------------------------------------------------------------------------

def test_accept_set_unions_three_columns():
    accept = row_accept_set(_row(snomed=["Hypertensive disorder"]))
    assert set(accept) == {"htn", "essential primary hypertension",
                           "hypertensive disorder"}


def test_every_form_carries_its_source():
    accept = row_accept_set(_row(snomed=["Hypertensive disorder"]))
    assert accept["htn"] == ["evidence"]
    assert accept["essential primary hypertension"] == ["description"]
    assert accept["hypertensive disorder"] == ["snomed"]


def test_a_form_reachable_from_two_columns_keeps_both_sources():
    """It is credited to each source, not to whichever was read first."""
    accept = row_accept_set(_row(evidence="sepsis", descr="Sepsis"))
    assert accept["sepsis"] == ["description", "evidence"]


def test_a_row_with_no_snomed_still_has_two_forms():
    accept = row_accept_set(_row(snomed=()))
    assert len(accept) == 2


def test_blank_columns_are_dropped_not_kept_as_empty_forms():
    accept = row_accept_set(_row(descr="", snomed=["", "   "]))
    assert set(accept) == {"htn"}


# --------------------------------------------------------------------------
# Note records
# --------------------------------------------------------------------------

def test_forms_dedupe_across_rows_and_credit_every_row(tmp_path):
    """One phrase justifying two codes is ONE gold form crediting BOTH rows.

    Without this a single correct prediction would have to satisfy two separate
    gold entries, and the 1:1 matching rule would make that impossible.
    """
    path = _write(tmp_path, [
        _row(code="A1", evidence="sepsis", descr="Sepsis, unspecified"),
        _row(code="A2", evidence="sepsis", descr="Severe sepsis"),
    ])
    records, _ = build_notes(path)
    form = records[0]["forms"]["sepsis"]
    assert form["rows"] == [0, 1]
    assert len(form["codes"]) == 2


def test_rows_and_codes_are_counted_separately(tmp_path):
    """Nine codes in the real file are evidenced twice; both units are reported."""
    path = _write(tmp_path, [
        _row(code="A1", evidence="chest pain"),
        _row(code="A1", evidence="angina"),
    ])
    records, stats = build_notes(path)
    assert stats["n_rows"] == 2
    assert stats["n_codes"] == 1
    assert len(reachable_rows(records[0])) == 2
    assert len(reachable_codes(records[0])) == 1


def test_source_forms_restricts_to_one_column(tmp_path):
    path = _write(tmp_path, [_row(snomed=["Hypertensive disorder"])])
    records, _ = build_notes(path)
    assert source_forms(records[0], "evidence") == {"htn"}
    assert source_forms(records[0], "snomed") == {"hypertensive disorder"}
    assert len(source_forms(records[0])) == 3


def test_snomed_denominator_excludes_rows_with_no_snomed(tmp_path):
    """Scoring SNOMED out of all rows would measure the file, not the model."""
    path = _write(tmp_path, [
        _row(code="A1", snomed=["Hypertensive disorder"]),
        _row(code="A2", evidence="sepsis", snomed=()),
    ])
    records, _ = build_notes(path)
    assert len(reachable_rows(records[0], "snomed")) == 1
    assert len(reachable_rows(records[0], "evidence")) == 2
    assert len(reachable_rows(records[0])) == 2


def test_notes_are_ordered_longest_first(tmp_path):
    """So --smoke exercises the multi-chunk path rather than avoiding it."""
    path = _write(tmp_path, [
        _row(note_id=1, text="short note about HTN"),
        _row(note_id=2, text=" ".join(["word"] * 1200) + " HTN"),
    ])
    records, _ = build_notes(path)
    assert [r["note_id"] for r in records] == [2, 1]
    assert records[0]["n_chunks"] > records[1]["n_chunks"]


def test_evidence_text_is_carried_raw_for_the_oracle(tmp_path):
    path = _write(tmp_path, [_row(evidence="HTN")])
    records, _ = build_notes(path)
    assert records[0]["rows"][0]["evidence_text"] == "HTN"


def test_truncated_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(_row()) + "\n{\"note_id\": 2, \"gold",
                    encoding="utf-8")
    records, stats = build_notes(str(path))
    assert stats["n_rows"] == 1
    assert len(records) == 1


# --------------------------------------------------------------------------
# SNOMED coverage — reported as a limit on the number, not a complaint
# --------------------------------------------------------------------------

def test_snomed_coverage_measures_shipped_against_reported():
    rows = [_row(snomed=["a", "b", "c"], count=64), _row(snomed=(), count=0)]
    cov = snomed_coverage([], rows)
    assert cov["snomed_rows_with_terms"] == 1
    assert cov["snomed_terms_shipped"] == 3
    assert cov["snomed_terms_reported"] == 64
    assert cov["snomed_shipped_frac"] == pytest.approx(3 / 64)


# --------------------------------------------------------------------------
# The real file
# --------------------------------------------------------------------------

needs_file = pytest.mark.skipif(
    not os.path.exists(SAMPLE_100_FILE),
    reason=f"benchmark input not present at {SAMPLE_100_FILE}",
)


@needs_file
def test_the_input_file_is_what_the_report_says_it_is():
    """Every headline denominator, pinned to the file.

    These are the numbers the report prints and the plan quotes. If one of them
    moves, either the file changed or the loader did, and the report is wrong
    either way.
    """
    _records, stats = build_notes(SAMPLE_100_FILE)
    assert stats["n_rows"] == 100
    assert stats["n_notes"] == 24
    assert stats["n_codes"] == 91
    assert stats["n_chunks"] == 82
    assert stats["forms_by_source"]["evidence"] == 99
    assert stats["forms_by_source"]["description"] == 91
    assert stats["forms_by_source"]["snomed"] == 142
    assert stats["accept_median"] == 4
    assert (stats["accept_min"], stats["accept_max"]) == (2, 5)


@needs_file
def test_snomed_coverage_on_the_real_file():
    _records, stats = build_notes(SAMPLE_100_FILE)
    assert stats["snomed_rows_with_terms"] == 53
    assert stats["snomed_by_code_system"]["CPT"]["with_snomed"] == 0
    assert stats["snomed_by_code_system"]["ICD-10-PCS"]["with_snomed"] == 0
    # The list is capped at 3 and the count field reports far more.
    assert stats["snomed_shipped_frac"] < 0.10


@needs_file
def test_note_text_is_embedded_so_there_is_no_join():
    records, _ = build_notes(SAMPLE_100_FILE)
    assert all(r["text"] for r in records)


@needs_file
def test_every_evidence_phrase_is_a_literal_substring_of_its_note():
    """Everything downstream depends on this, so it is checked, not assumed."""
    records, _ = build_notes(SAMPLE_100_FILE)
    for record in records:
        for entry in record["rows"]:
            assert entry["evidence_text"] in record["text"]
