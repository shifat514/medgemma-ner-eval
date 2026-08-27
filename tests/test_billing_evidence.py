"""The evidence key, negation scoping, and the structural ceiling.

The load-bearing tests here are the negation ones. Without negation handling the
ceiling read 13 of 16 and note 55688's R06.2 counted as reachable on the
strength of an ROS line saying "Denies wheezing or difficulty breathing" — a
sentence stating the patient does not have the thing. These notes are ROS-heavy
and most of the ROS is negative, so that is the common case, not an edge one.

The second round of the same bug is also pinned: `normalize` collapses newlines
so that the PDF's mid-phrase line wraps stop breaking term matches, which means
a clause-break rule keyed on "\\n" has nothing to split on and runs backwards
through half the note looking for a negation cue. That made J30.2 and R06.2 read
as unreachable under `assessment_cut`, where both are printed in the Problem
List.

Corpus-dependent tests skip without the built sample, which is gitignored.
"""

import json
import os

import pytest

from src.billing_config import SAMPLE_FILE
from src.billing_evidence import (
    EVIDENCE,
    _negated,
    ceiling,
    evidence_for,
    matched_by,
    normalize,
    present_in,
)


def _corpus():
    if not os.path.exists(SAMPLE_FILE):
        pytest.skip(f"{SAMPLE_FILE} not built (gitignored); run make billing-sample")
    return [json.loads(line) for line in open(SAMPLE_FILE, encoding="utf-8")
            if line.strip()]


# --- normalize --------------------------------------------------------------


def test_normalize_collapses_the_pdf_line_wrap():
    """Note 26819 reads "Allergic\\n      rhinitis or other allergy"."""
    assert "allergic rhinitis" in normalize("Positive for Allergic\n    rhinitis")


# --- negation ---------------------------------------------------------------


def test_a_denied_term_is_not_evidence():
    text = normalize("Respiratory: Reports daytime cough. Denies wheezing or "
                     "difficulty breathing.")
    assert _negated(text, "wheezing")


def test_one_positive_mention_outweighs_a_denial_elsewhere():
    """A chronic problem listed AND denied on today's ROS is still evidenced."""
    text = normalize("Respiratory: Denies wheezing or difficulty breathing.\n"
                     "Problem List Reviewed\n- WHEEZING\nAlbuterol PRN")
    assert not _negated(text, "wheezing")


def test_negation_does_not_leak_across_a_list_bullet():
    """The bug that made assessment_cut read 14 of 16.

    Newlines are gone by this point, so the Problem List's "- " bullets are what
    bound the clause. Without them the scope reaches back into the ROS.
    """
    text = normalize("ENT: Denies sore throat, ear pain.\n"
                     "Problem List Reviewed and updated by A Doctor\n"
                     "Pertinent for:\n- SEASONAL ALLERGIC RHINITIS\nClaritin")
    assert not _negated(text, "allergic rhinitis")


def test_negation_scope_is_bounded():
    """A cue 400 characters back must not negate anything."""
    text = normalize("Denies fever. " + ("filler words here. " * 40) + "wheezing")
    assert not _negated(text, "wheezing")


def test_negation_covers_a_list_after_one_cue():
    """"Denies nausea, vomiting, diarrhea." negates all three."""
    text = normalize("Gastrointestinal: Denies nausea, vomiting, diarrhea.")
    for term in ("nausea", "vomiting", "diarrhea"):
        assert _negated(text, term), term


# --- the key ----------------------------------------------------------------


def test_every_gold_code_is_catalogued():
    """A missing entry silently makes a code unreachable and flatters nothing."""
    records = _corpus()
    missing = [(r["note_id"], c) for r in records for c in r["gold_codes"]
               if evidence_for(r["note_id"], c) is None]
    assert missing == []


def test_key_has_no_entries_for_codes_that_are_not_gold():
    records = _corpus()
    gold = {(r["note_id"], c) for r in records for c in r["gold_codes"]}
    assert set(EVIDENCE) - gold == set()


def test_terms_are_not_the_icd_catalogue_wording():
    """"Enteroviral vesicular pharyngitis" is not in note 112976; "coxsackie" is.

    Matching the catalogue description would measure whether the model memorised
    ICD, which is the thing already known to fail.
    """
    entry = evidence_for("112976", "B08.5")
    assert "enteroviral vesicular pharyngitis" not in [t.lower() for t in entry["terms"]]
    assert "coxsackie" in entry["terms"]


# --- reachability against the real notes ------------------------------------


def test_ceiling_is_full_when_nothing_is_removed():
    assert ceiling(_corpus(), "full")["n_reachable"] == 16


def test_problem_list_codes_stay_reachable_under_assessment_cut():
    """assessment_cut keeps the Problem List, so nothing should be lost there."""
    assert ceiling(_corpus(), "assessment_cut")["n_reachable"] == 16


def test_leakage_cut_loses_four_codes_and_names_them():
    """The finding this module exists for. leakage_cut recall was quoted /16."""
    c = ceiling(_corpus(), "leakage_cut")
    assert c["n_reachable"] == 12
    assert set(c["unreachable"]) == {
        ("112976", "D18.00"),      # hemangioma — Problem List only
        ("26819", "L20.9"),        # atopic dermatitis — Problem List only
        ("55688", "J30.2"),        # allergic rhinitis — Problem List only
        ("55688", "R06.2"),        # wheezing — Problem List, plus an ROS denial
    }


def test_the_conditions_a_visit_was_about_survive_every_cut():
    """Acute problems are documented in the narrative, not just the Problem List."""
    records = _corpus()
    for note_id, code in (("55688", "J11.1"), ("55688", "S52.501A"),
                          ("112976", "B08.5"), ("96176", "B97.89")):
        rec = next(r for r in records if r["note_id"] == note_id)
        assert present_in(rec["variants"]["leakage_cut"], note_id, code), code


# --- matching extracted phrases ---------------------------------------------


def test_extracted_phrase_matches_when_it_contains_the_term():
    assert matched_by(["Influenza B positive"], "55688", "J11.1")


def test_extracted_phrase_matches_when_the_term_contains_it():
    assert matched_by(["influenza"], "55688", "J11.1")


def test_unrelated_phrase_does_not_match():
    assert matched_by(["nasal congestion"], "55688", "J11.1") == []


def test_short_phrases_cannot_match_by_accident():
    """A 3-character phrase inside a long term is a coincidence, not a find."""
    assert matched_by(["flu"], "112976", "B08.5") == []
