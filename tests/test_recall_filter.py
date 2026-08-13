"""The second-pass billability filter.

The filter changes what is being benchmarked, so the tests are mostly about the
two numbers staying separable and an unanswered question never counting as a
precision win.
"""

import json

import pytest

from src.datasets.mdace_recall import build_notes
from src.recall_filter import (
    VARIANTS,
    apply_filter,
    build_messages,
    compare,
    filter_findings,
    finding_key,
    phrase_of,
    question,
)
from src.recall_scoring import dedupe_findings

LEVELS = ("L1", "L2", "L3")


def _row(code="I10", evidence="HTN", descr="Essential (primary) hypertension",
         text=None):
    return {
        "note_id": 1, "hadm_id": 1, "chart_type": "Inpatient",
        "code_system": "ICD-10-CM", "gold_code": code,
        "gold_code_description": descr, "mdace_gold_evidence_text": evidence,
        "gold_snomed_concepts": [], "gold_snomed_concept_count": 0,
        "note_text": text or "Patient has HTN. Senna 1 TAB PO BID.",
    }


def _notes(tmp_path, rows):
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return build_notes(str(path))


# --------------------------------------------------------------------------
# What the filter is shown
# --------------------------------------------------------------------------

def test_the_filter_sees_both_fields_when_they_differ():
    """`HTN (hypertension)` is easier to rule on than either alone."""
    assert phrase_of({"span": "HTN", "name": "hypertension"}) == \
        "HTN (hypertension)"


def test_a_repeated_field_is_not_shown_twice():
    assert phrase_of({"span": "sepsis", "name": "sepsis"}) == "sepsis"
    assert phrase_of({"span": "sepsis", "name": ""}) == "sepsis"
    assert phrase_of({"span": "", "name": "hypertension"}) == "hypertension"


def test_the_question_reaches_the_model():
    text = build_messages({"span": "HTN", "name": "hypertension"})[1]["content"][0]["text"]
    assert "HTN (hypertension)" in text
    assert "YES or NO" in text


# --------------------------------------------------------------------------
# The two variants — the point is to tell "the model knows" from "we told it"
# --------------------------------------------------------------------------

def test_bare_names_no_categories_and_guided_does():
    bare, guided = question("bare"), question("guided")
    for word in ("medication", "vital sign", "lab value"):
        assert word not in bare
        assert word in guided


def test_bare_is_the_default():
    """If only `guided` works, that is a finding about the model, not a fix."""
    from src.recall_filter import DEFAULT_VARIANT

    assert DEFAULT_VARIANT == "bare"
    assert question() == question("bare")


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="unknown filter variant"):
        question("whatever")


def test_both_variants_ask_the_same_underlying_question():
    for name in VARIANTS:
        assert "assign a billing code" in question(name)


# --------------------------------------------------------------------------
# Applying it
# --------------------------------------------------------------------------

def test_a_no_drops_the_finding_and_a_yes_keeps_it():
    preds = {1: dedupe_findings([{"span": "HTN", "name": "hypertension"},
                                 {"span": "Senna 1 TAB", "name": "senna"}])}
    verdicts = filter_findings(
        preds, run_fn=lambda m: "NO" if "Senna" in str(m) else "YES")
    kept, dropped, unreadable = apply_filter(preds, verdicts)

    assert [f["span"] for f in kept[1]] == ["HTN"]
    assert (dropped, unreadable) == (1, 0)


def test_an_unreadable_answer_keeps_the_finding():
    """A model that failed to answer has not said the finding is unbillable, and
    defaulting to drop would let a parse failure look like a precision win."""
    preds = {1: dedupe_findings([{"span": "HTN", "name": "hypertension"}])}
    verdicts = filter_findings(preds, run_fn=lambda m: "hmm, hard to say")
    kept, dropped, unreadable = apply_filter(preds, verdicts)

    assert len(kept[1]) == 1
    assert (dropped, unreadable) == (0, 1)


def test_a_failed_call_does_not_kill_the_pass():
    def explode(messages):
        raise RuntimeError("CUDA OOM")

    preds = {1: dedupe_findings([{"span": "HTN", "name": ""},
                                 {"span": "sepsis", "name": ""}])}
    verdicts = filter_findings(preds, run_fn=explode)
    assert len(verdicts) == 2
    assert all(v is None for v in verdicts.values())


def test_supplied_verdicts_are_reused_instead_of_asking_again():
    preds = {1: dedupe_findings([{"span": "HTN", "name": "hypertension"}])}
    key = finding_key(1, preds[1][0])
    verdicts = filter_findings(
        preds, run_fn=lambda m: pytest.fail("model was called"),
        resume={key: False})
    assert verdicts[key] is False


# --------------------------------------------------------------------------
# Both numbers, always
# --------------------------------------------------------------------------

def test_raw_and_filtered_are_both_scored(tmp_path):
    """Filtering changes what is benchmarked, so the model's own number has to
    stay visible next to the system's."""
    records, _ = _notes(tmp_path, [_row()])
    raw = {1: dedupe_findings([{"span": "HTN", "name": ""},
                               {"span": "Senna 1 TAB", "name": ""}])}
    filtered = {1: [f for f in raw[1] if f["span"] == "HTN"]}

    sides = compare(records, [("raw", raw), ("filtered", filtered)],
                    rejected=set(), levels=LEVELS)
    assert set(sides) == {"raw", "filtered"}
    assert sides["raw"]["n_pred"] == 2
    assert sides["filtered"]["n_pred"] == 1
    # dropping a false positive raises precision and leaves recall alone
    assert sides["filtered"]["precision"] > sides["raw"]["precision"]
    assert sides["filtered"]["row_recall"] == sides["raw"]["row_recall"]


def test_dropping_a_true_positive_costs_recall_and_shows_up(tmp_path):
    records, _ = _notes(tmp_path, [_row()])
    raw = {1: dedupe_findings([{"span": "HTN", "name": ""}])}
    sides = compare(records, [("raw", raw), ("filtered", {1: []})],
                    rejected=set(), levels=LEVELS)

    assert sides["raw"]["row_recall"] == 1.0
    assert sides["filtered"]["row_recall"] == 0.0


# --------------------------------------------------------------------------
# Section filtering, stacked on top
# --------------------------------------------------------------------------

def test_blocked_sections_are_chosen_by_category_not_by_our_sample():
    """"Had no gold in these 24 notes" measured against the same 24 notes shows
    a zero recall cost by construction. Radiology is excluded on purpose: it
    carries no gold here but genuinely can name a billable diagnosis."""
    from src.recall_filter import blocked_section

    for name in ("Discharge Medications", "Medications on Admission",
                 "Allergies", "Followup Instructions", "Order date"):
        assert blocked_section(name), name
    for name in ("Brief Hospital Course", "Chief Complaint", "FINDINGS",
                 "IMPRESSION", "Imaging", "Pertinent Results"):
        assert not blocked_section(name), name


def test_findings_from_blocked_sections_are_dropped(tmp_path):
    from src.recall_filter import drop_blocked_sections

    text = ("Chief Complaint:\nSepsis.\n\n"
            "Discharge Medications:\n1. Senna 1 TAB PO BID\n")
    records, _ = _notes(tmp_path, [_row(evidence="Sepsis", text=text)])
    preds = {1: dedupe_findings([{"span": "Sepsis", "name": ""},
                                 {"span": "Senna 1 TAB", "name": ""}])}

    kept, dropped, by_section = drop_blocked_sections(records, preds)
    assert [f["span"] for f in kept[1]] == ["Sepsis"]
    assert dropped == 1
    assert by_section == {"Discharge Medications": 1}


def test_the_filters_mistakes_are_listed_not_just_counted(tmp_path):
    """If the wrongly-dropped matches share a shape, the recall is recoverable.
    Nobody knows until they are looked at."""
    from src.recall_filter import wrongly_dropped

    records, _ = _notes(tmp_path, [_row(evidence="HTN")])
    raw = {1: dedupe_findings([{"span": "HTN", "name": ""}])}
    mistakes = wrongly_dropped(records, raw, kept={1: []}, rejected=set(),
                               levels=LEVELS)
    assert len(mistakes) == 1
    assert mistakes[0]["span"] == "HTN"
    assert mistakes[0]["gold_form"] == "htn"

    # nothing dropped, nothing to report
    assert wrongly_dropped(records, raw, kept=raw, rejected=set(),
                           levels=LEVELS) == []
