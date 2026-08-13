"""Failure analysis: section attribution for false positives, causes for misses.

The point of this module is to replace inference-from-totals with a list, so the
tests are mostly about the buckets being separable and the counts file staying
free of note text.
"""

import json

from src.datasets.mdace_recall import build_notes
from src.recall_failures import (
    analyse,
    best_similarity,
    locate,
    section_of,
    sections,
)
from src.recall_scoring import dedupe_findings

LEVELS = ("L1", "L2", "L3")

_NOTE = (
    "Chief Complaint:\n"
    "Biliary obstruction.\n"
    "\n"
    "Past Medical History:\n"
    "HTN, depression, chronic back\npain\n"
    "\n"
    "Discharge Medications:\n"
    "1. Docusate Sodium 100 mg PO BID\n"
    "2. Senna 1 TAB PO BID\n"
)


def _record(rows):
    out = []
    for i, (code, evidence, descr) in enumerate(rows):
        out.append({
            "row_id": i, "code_system": "ICD-10-CM", "code": code,
            "code_key": f"ICD-10-CM|{code}", "evidence_text": evidence,
            "n_snomed_shipped": 0, "n_snomed_reported": 0,
            "accept": {evidence.lower(): ["evidence"],
                       descr.lower(): ["description"]},
        })
    forms = {}
    for entry in out:
        for form, srcs in entry["accept"].items():
            slot = forms.setdefault(form, {"sources": set(), "rows": set(),
                                           "codes": set()})
            slot["sources"].update(srcs)
            slot["rows"].add(entry["row_id"])
            slot["codes"].add(entry["code_key"])
    return {
        "note_id": 1, "hadm_id": 1, "chart_type": "Inpatient", "text": _NOTE,
        "n_chunks": 1, "rows": out,
        "forms": {k: {"sources": sorted(v["sources"]),
                      "rows": sorted(v["rows"]),
                      "codes": sorted(v["codes"])} for k, v in forms.items()},
    }


# --------------------------------------------------------------------------
# Section machinery
# --------------------------------------------------------------------------

def test_sections_cover_the_whole_note():
    """Counts must add up, so no character may fall outside a section."""
    got = sections(_NOTE)
    assert got[0][0] == 0
    assert got[-1][1] == len(_NOTE)
    for i in range(len(got) - 1):
        assert got[i][1] == got[i + 1][0]


def test_text_before_the_first_header_is_named_not_dropped():
    got = sections("no header here\nAllergies:\nnone")
    assert got[0][2] == "(no section)"


def test_a_span_is_attributed_to_its_section():
    assert section_of(_NOTE, "Biliary obstruction") == "Chief Complaint"
    assert section_of(_NOTE, "Senna 1 TAB") == "Discharge Medications"


def test_a_span_not_in_the_note_has_no_section():
    assert section_of(_NOTE, "pneumonia") is None


def test_locate_tolerates_a_span_that_ran_across_a_line_break():
    """The model reproduces "chronic back\\npain" as "chronic back pain"."""
    assert _NOTE.find("chronic back pain") == -1
    assert locate(_NOTE, "chronic back pain") is not None
    assert section_of(_NOTE, "chronic back pain") == "Past Medical History"


# --------------------------------------------------------------------------
# False positives
# --------------------------------------------------------------------------

def test_false_positives_are_counted_per_section():
    record = _record([("K83.1", "Biliary obstruction", "Obstruction of bile duct")])
    preds = {1: dedupe_findings([
        {"span": "Biliary obstruction", "name": ""},   # a hit
        {"span": "Senna 1 TAB", "name": ""},           # FP in medications
        {"span": "Docusate Sodium", "name": ""},       # FP in medications
        {"span": "pneumonia", "name": ""},             # FP, not in the note
    ])}
    counts, _detail = analyse([record], preds, [{"note_id": 1,
                                                 "cap_hit_windows": []}],
                              rejected=set(), levels=LEVELS)
    assert counts["false_positives"] == 3
    assert counts["fp_by_section"]["Discharge Medications"] == 2
    assert counts["fp_by_section"]["(not in the note)"] == 1
    assert counts["fp_not_in_note"] == 1


# --------------------------------------------------------------------------
# Miss causes — the four buckets have opposite fixes, so they must separate
# --------------------------------------------------------------------------

def _analyse_one(preds, rejected=frozenset(), cap_windows=(), per_note=None):
    record = _record([("K83.1", "Biliary obstruction", "Obstruction of bile duct")])
    if per_note is None:
        per_note = [{"note_id": 1, "cap_hit_windows": list(cap_windows)}]
    counts, _d = analyse([record], preds, per_note, rejected=set(rejected),
                         levels=LEVELS)
    return counts["miss_causes"]


def test_a_miss_inside_a_truncated_window_is_blamed_on_truncation():
    """Checked first: a cut-off reply is not the model failing to see it."""
    causes = _analyse_one({1: dedupe_findings([{"span": "unrelated", "name": ""}])},
                          cap_windows=[[0, 50]])
    assert causes["truncated"] == 1


def test_a_miss_the_judge_rejected_is_blamed_on_the_judge():
    preds = {1: dedupe_findings([{"span": "Biliary obstruction", "name": ""}])}
    # The rejection key is (note_id, span, name, gold_form) and must name the
    # actual finding, which is how recall_judge writes it.
    causes = _analyse_one(
        preds,
        rejected={(1, "Biliary obstruction", "", "biliary obstruction")},
        cap_windows=[])
    assert causes["rejected_by_l5"] == 1


def test_a_near_miss_is_separated_from_never_extracted():
    """These have opposite fixes -- one is the matcher, one is the model."""
    near = _analyse_one({1: dedupe_findings([{"span": "obstruction bile", "name": ""}])})
    assert near["near_miss"] == 1

    nothing = _analyse_one({1: dedupe_findings([{"span": "ankle sprain", "name": ""}])})
    assert nothing["not_extracted"] == 1


def test_a_run_that_predates_cap_logging_says_so_rather_than_guessing():
    causes = _analyse_one({1: dedupe_findings([{"span": "ankle sprain", "name": ""}])},
                          per_note=[{"note_id": 1}])       # no cap_hit_windows key
    assert causes["unknown_truncation"] == 1
    assert causes["not_extracted"] == 0


# --------------------------------------------------------------------------
# Records the LOADER builds, not records this file builds
#
# Every test above hand-writes its record dicts, and that is exactly how this
# module came to read a row key the loader never sets. It went unnoticed for two
# phases because the branch that reads it only runs when a run recorded cap-hit
# windows, and none had. The first run that did crashed on contact.
# --------------------------------------------------------------------------

def _loader_record(tmp_path, note_text, evidence="Biliary obstruction",
                   drop_sections=False):
    """One note through the real loader, so the row fields are the real ones."""
    row = {
        "note_id": 1, "hadm_id": 100, "chart_type": "Inpatient",
        "code_system": "ICD-10-CM", "gold_code": "K83.1",
        "gold_code_description": "Obstruction of bile duct",
        "mdace_gold_evidence_text": evidence,
        "gold_snomed_concepts": [], "gold_snomed_concept_count": 0,
        "note_text": note_text,
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    records, _stats = build_notes(str(path), drop_sections=drop_sections)
    return records[0]


def _causes(record, cap_windows):
    preds = {1: dedupe_findings([{"span": "ankle sprain", "name": ""}])}
    counts, _d = analyse([record], preds,
                         [{"note_id": 1, "cap_hit_windows": list(cap_windows)}],
                         rejected=set(), levels=LEVELS)
    return counts["miss_causes"]


def test_miss_causes_work_on_a_record_the_loader_actually_built(tmp_path):
    record = _loader_record(tmp_path, _NOTE)
    assert _causes(record, [[0, 50]])["truncated"] == 1


# A stripped section BEFORE the gold, long enough that the two texts disagree
# about where the phrase sits by more than a window's width.
_SHIFTED_NOTE = (
    "Discharge Medications:\n"
    + "1. Docusate Sodium 100 mg PO BID\n" * 20
    + "\nChief Complaint:\nBiliary obstruction.\n"
)


def test_truncation_is_judged_against_the_text_the_model_was_actually_sent(tmp_path):
    """Cap windows index `model_text`. Using the full note is off by every
    stripped section, and both texts search cleanly, so it fails silently."""
    full = _loader_record(tmp_path, _SHIFTED_NOTE, drop_sections=False)
    stripped = _loader_record(tmp_path, _SHIFTED_NOTE, drop_sections=True)
    # The phrase moved from ~word 140 to ~word 2 once medications were cut.
    assert len(stripped["model_text"].split()) < len(full["text"].split()) / 2

    # A window over the first 50 words of what was SENT contains it.
    assert _causes(stripped, [[0, 50]])["truncated"] == 1
    # ...and the same window over the unstripped note does not.
    assert _causes(full, [[0, 50]])["truncated"] == 0


def test_gold_the_section_filter_removed_is_not_blamed_on_the_model(tmp_path):
    """`--drop-sections` claims zero gold lost. This is what measures it."""
    note = ("Discharge Medications:\n1. Senna 1 TAB PO BID\n"
            "\nChief Complaint:\nAbdominal pain.\n")
    record = _loader_record(tmp_path, note, evidence="Senna 1 TAB",
                            drop_sections=True)
    causes = _causes(record, [])
    assert causes["dropped_by_section_filter"] == 1
    assert causes["not_extracted"] == 0
    # It fires without any truncation in play -- the two are independent.
    assert causes["truncated"] == 0


def test_an_unstripped_run_can_never_report_a_section_filter_drop(tmp_path):
    record = _loader_record(tmp_path, _NOTE, drop_sections=False)
    assert _causes(record, [])["dropped_by_section_filter"] == 0


def test_best_similarity_saturates_on_an_actual_rule_match():
    assert best_similarity([{"span": "biliary obstruction", "name": ""}],
                           {"biliary obstruction"}) == 1.0
    assert best_similarity([{"span": "ankle sprain", "name": ""}],
                           {"biliary obstruction"}) < 0.34


# --------------------------------------------------------------------------
# The counts file is shareable; the detail file is not
# --------------------------------------------------------------------------

def test_the_counts_file_carries_no_phrases(tmp_path):
    record = _record([("K83.1", "Biliary obstruction", "Obstruction of bile duct")])
    preds = {1: dedupe_findings([{"span": "Senna 1 TAB", "name": ""}])}
    counts, detail = analyse([record], preds,
                             [{"note_id": 1, "cap_hit_windows": []}],
                             rejected=set(), levels=LEVELS)

    blob = json.dumps(counts)
    assert "Senna" not in blob
    assert "Biliary obstruction" not in blob
    # Section names are structural, not patient data, so they are allowed.
    assert "Discharge Medications" in blob

    # ...and the detail file is where the phrases live.
    assert any("Senna" in json.dumps(row) for row in detail)
