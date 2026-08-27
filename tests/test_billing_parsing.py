"""PDF-text parsing, gold extraction and the three input variants.

No PDF is opened here. Every test runs on a synthetic note built to the Connexin
template the four supplied files use, so the suite stays CPU-only and carries no
patient data. The template details that matter — the three-page repetition of
the banner, the "PROCEDURES" case difference, the Problem List sitting *inside*
Patient History rather than being a section of its own — are all reproduced,
because those are exactly the places a parser quietly loses a block.
"""

from src.datasets.billing import (
    ICD10_RE,
    extract_gold,
    normalize_code,
    note_id_from_filename,
    render_variant,
    split_sections,
    strip_page_furniture,
    visit_kind_from_filename,
)

# A two-page synthetic note. Page furniture repeats exactly as it does in the
# real files: clinic line, title, patient banner, visit date, then a
# Generated/Page footer and the Connexin copyright.
_RAW = """  Confidential Information
                        Desert Valley Pediatrics, LLP,10105 Banburry Cross Dr, Ste 370, Las Vegas, NV
                                                                                7022604525


                                            Encounter Summary
                                    TEST PATIENT (Sex: F, DOB: 01/02/2015)
                                        Date of Visit: 03/04/2026
  Visit Information
  Appointment type: SICK VISIT, EST

  CC/HPI
  Cough and fever for three days.

  Patient History
  Past Medical History: Positive for asthma.
  Problem List Reviewed and updated by A Doctor, MD (1500) 03/04/2026 08:29:03
  - J30.2 OTHER SEASONAL ALLERGIC RHINITIS
  Cats and seasonal
  - WHEEZING

  Allergies Reviewed by A Doctor, MD (1500) 03/04/2026 08:29:47
  - AMOXICILLIN: Rash

Generated: 08/17/2026 09:52 AM               Confidential Information            Page 1 of 2
Copyright (c) 2008 by Connexin Software, Inc. 800-218-9916
v2012.4.12.1
                                            Encounter Summary
                                    TEST PATIENT (Sex: F, DOB: 01/02/2015)
                                        Date of Visit: 03/04/2026
  Exam Findings
  Respiratory: ABNORMAL wheeze.

  Assessment
  Viral illness
  DX 1: J06.9 Acute upper respiratory infection, unspecified
  DX 2: R06.2 Wheezing
  DX 3: R06.2 Wheezing

  Plan
  Supportive care.

  PROCEDURES
  99213 OFFICE/OUTPATIENT VISIT, EST

Generated: 08/17/2026 09:52 AM               Confidential Information            Page 2 of 2
Copyright (c) 2008 by Connexin Software, Inc. 800-218-9916
v2012.4.12.1
"""


def _sections():
    clean, _ = strip_page_furniture(_RAW)
    return split_sections(clean)


# --- page furniture ---------------------------------------------------------


def test_furniture_lines_are_dropped():
    clean, dropped = strip_page_furniture(_RAW)
    assert "Connexin" not in clean
    assert "Page 1 of 2" not in clean
    assert "Desert Valley Pediatrics" not in clean
    assert "7022604525" not in clean
    assert dropped > 0


def test_patient_banner_survives_exactly_once():
    """Sex and DOB are load-bearing for pediatric codes, so the banner stays.

    It appears once per page in the source; keeping all of them would triple-feed
    the same line, and dropping all of them would remove the age information that
    Z68.5x and Z00.12x depend on.
    """
    clean, _ = strip_page_furniture(_RAW)
    assert clean.count("DOB: 01/02/2015") == 1
    assert clean.count("Encounter Summary") == 1
    assert clean.count("Date of Visit: 03/04/2026") == 1


# --- sections ---------------------------------------------------------------


def test_known_headings_become_section_boundaries():
    found = [h for h, _ in _sections() if h]
    assert found == [
        "Visit Information", "CC/HPI", "Patient History", "Exam Findings",
        "Assessment", "Plan", "Procedures",
    ]


def test_procedures_heading_matches_case_insensitively():
    """The real note 26819 writes it "PROCEDURES"; the other three do not."""
    assert "Procedures" in [h for h, _ in _sections() if h]


def test_reviewed_by_lines_are_not_mistaken_for_headings():
    """"Medication List Reviewed by ..." must not open a "Medications" section."""
    headings = [h for h, _ in _sections() if h]
    assert headings.count("Patient History") == 1
    assert "Medications" not in headings


# --- gold -------------------------------------------------------------------


def test_gold_codes_come_from_dx_lines_only():
    codes, rows, dupes = extract_gold(_sections())
    assert codes == ["J06.9", "R06.2"]
    assert len(rows) == 3
    assert dupes == 1


def test_free_text_impression_is_not_gold():
    """"Viral illness" sits above the DX lines and is the clinician's wording."""
    codes, _, _ = extract_gold(_sections())
    assert all(c.startswith(("J", "R")) for c in codes)
    assert "VIRAL ILLNESS" not in codes


def test_cpt_code_under_procedures_is_not_gold():
    codes, _, _ = extract_gold(_sections())
    assert "99213" not in codes


def test_duplicate_dx_line_collapses_but_is_still_reported():
    """Note 96176 really does list Z68.51 as both DX 3 and DX 4."""
    codes, rows, dupes = extract_gold(_sections())
    assert len(rows) - dupes == len(codes)


# --- variants ---------------------------------------------------------------


def test_full_variant_keeps_everything():
    text, removed = render_variant(_sections(), "full")
    assert "DX 1: J06.9" in text
    assert "Problem List Reviewed" in text
    assert removed["assessment_lines"] == 0


def test_assessment_cut_removes_dx_lines_but_keeps_problem_list():
    text, removed = render_variant(_sections(), "assessment_cut")
    assert "DX 1" not in text
    assert "J06.9" not in text
    assert "Problem List Reviewed" in text
    assert removed["assessment_lines"] > 0


def test_assessment_cut_still_leaks_via_the_problem_list():
    """The whole reason leakage_cut exists — this is a real leak, not a worry."""
    text, _ = render_variant(_sections(), "assessment_cut")
    assert "R06.2" not in text          # the code itself is gone ...
    assert "WHEEZING" in text           # ... but its description is not


def test_leakage_cut_removes_the_problem_list_block():
    text, removed = render_variant(_sections(), "leakage_cut")
    assert "Problem List Reviewed" not in text
    assert "J30.2" not in text
    assert "WHEEZING" not in text
    assert removed["problem_list_lines"] > 0


def test_leakage_cut_keeps_the_rest_of_patient_history():
    """The cut must take the Problem List and stop, not the whole section."""
    text, _ = render_variant(_sections(), "leakage_cut")
    assert "Past Medical History: Positive for asthma." in text
    assert "Allergies Reviewed by" in text
    assert "AMOXICILLIN" in text


def test_every_variant_keeps_the_clinical_narrative():
    for variant in ("full", "assessment_cut", "leakage_cut"):
        text, _ = render_variant(_sections(), variant)
        assert "Cough and fever for three days." in text
        assert "ABNORMAL wheeze" in text
        assert "DOB: 01/02/2015" in text


# --- code normalization -----------------------------------------------------


def test_normalize_code_uppercases_and_strips():
    assert normalize_code("  b08.5 ") == "B08.5"
    assert normalize_code("s52.501a") == "S52.501A"
    assert normalize_code("J11.1,") == "J11.1"


def test_normalize_code_keeps_the_decimal_point():
    """B08 and B08.5 are different codes; collapsing the dot would hide that."""
    assert normalize_code("B08.5") != normalize_code("B08")


def test_icd10_regex_covers_every_length_in_the_corpus():
    for code in ("B08.5", "D18.00", "Z00.121", "R41.840", "S52.501A", "J30.2"):
        assert ICD10_RE.findall(code) == [code]


# --- filenames --------------------------------------------------------------


def test_note_id_and_visit_kind_come_off_the_filename():
    assert note_id_from_filename("112976 encounter.pdf") == "112976"
    assert visit_kind_from_filename("112976 encounter.pdf") == "encounter"
    assert note_id_from_filename("26819 well.pdf") == "26819"
    assert visit_kind_from_filename("26819 well.pdf") == "well"
