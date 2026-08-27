"""Configuration for the pediatric-billing ICD-code evaluation.

A DIFFERENT QUESTION FROM EVERY OTHER MODULE IN THIS REPO. The MDACE and MIMIC
work asks "which phrases in this note would a coder bill?" and scores phrase
overlap. This asks the question Ehtesham Bhai actually sent: **given the note,
does the model output the right ICD-10 codes?** Gold is a set of codes, not a
set of phrases, and scoring is exact code match. Nothing in recall_matching.py
applies here — there is no fuzzy ladder, because B08.5 either equals B08.5 or
it does not.

THE DATA. Four pediatric encounter notes from a Las Vegas practice, supplied as
PDFs on 2026-08-21. Three sick visits and one well visit. They are NOT MIMIC:
outpatient rather than ICU, pediatric rather than adult, ~740-1160 words rather
than several thousand. Every prompt and threshold tuned on MIMIC should be
assumed not to transfer until measured.

WHY THERE IS NO CHUNKING. The longest note is 1,163 words, which fits in one
prompt with room to spare. MIMIC discharge summaries needed 400-word windows;
these do not, and windowing them would only reintroduce the pooling and
double-counting problems that cost real time on the recall branch. One note,
one call.

SAMPLE SIZE IS THE HEADLINE CAVEAT, NOT A FOOTNOTE. 16 unique gold codes across
4 notes. One code is 6.25 recall points. Any number this produces is a spot
check and must be reported as one; it cannot carry a decision on its own.

REAL PATIENT DATA. The PDFs carry names, dates of birth, a rendering provider
and a license number. They live OUTSIDE the repo and must stay there, exactly
as the MIMIC sources do. Everything downstream reads the built sample file,
which is gitignored.
"""

import os

# --- Source PDFs (outside the repo; local machine only) ---------------------
#
# Ehtesham Bhai's drop, unmodified. Parsing happens on this machine; the GPU
# box never sees a PDF, only the built sample.
BILLING_PDF_DIR = os.environ.get(
    "BILLING_PDF_DIR", "/home/shifat/zeda_ml_works/ai-medical-billing"
)

# --- Repo-internal, gitignored ----------------------------------------------

SAMPLE_FILE = os.environ.get("BILLING_SAMPLE_FILE", "data/samples/billing_sample.jsonl")

# Per-note replies and predicted codes. Note-derived, gitignored. On Colab
# point this at mounted Drive:
#   os.environ["BILLING_OUTPUT_DIR"] = "/content/drive/MyDrive/billing-icd"
OUTPUT_DIR = os.environ.get("BILLING_OUTPUT_DIR", "outputs/billing_icd")

# Aggregate metrics + markdown report. Counts and rates only, safe to commit.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# --- Model ------------------------------------------------------------------

MODEL_ID = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
MODEL_NAME = os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-4b-it")

LOAD_IN_4BIT = True

# 512 rather than the recall branch's 1024. Gold is 2-6 codes per note and the
# output is one short object per code, so a correct reply is well under 200
# tokens. A reply that runs to 512 is looping, not thorough — and n_cap_hits is
# printed so that shows up as a number rather than as quiet lost recall.
GEN_CONFIG = {
    "max_new_tokens": int(os.environ.get("BILLING_MAX_NEW_TOKENS", "512")),
    "do_sample": False,
}

# A reply within this many tokens of the cap is treated as truncated.
CAP_MARGIN = 8

# --- The three input variants -----------------------------------------------
#
# ONE PROMPT, THREE INPUTS. The variants differ only in how much of the note the
# model is shown. Running three prompts against one input would confound "the
# model got better" with "the prompt got better"; this way every difference
# between the three numbers is attributable to the text that was removed.
#
#   assessment_cut  what Ehtesham Bhai asked for: the Assessment block, which
#                   holds the DX lines, is removed and nothing else is.
#
#   leakage_cut     Assessment AND the Problem List removed. The Problem List is
#                   a second copy of the answer: note 26819 prints
#                   "- J30.2 OTHER SEASONAL ALLERGIC RHINITIS" and
#                   "- L20.9 ATOPIC DERMATITIS, UNSPECIFIED" — the gold code
#                   strings themselves — and notes 55688 and 112976 print three
#                   more gold descriptions word for word. Under assessment_cut
#                   the model can copy those instead of reasoning, so the honest
#                   number is here and the flattered one is there.
#
#   full            nothing removed; the DX lines are still in the input. This
#                   is a harness check, not a result. If the model cannot return
#                   B08.5 while "DX 1: B08.5" is printed on the page, the bug is
#                   in the parsing or the scoring, not in the model, and the
#                   other two numbers mean nothing. Run it FIRST.
VARIANTS = ("full", "assessment_cut", "leakage_cut")

VARIANT_LABELS = {
    "full": "full note (answers left in) — harness check",
    "assessment_cut": "Assessment removed — as requested",
    "leakage_cut": "Assessment + Problem List removed — no leaks",
}

DEFAULT_VARIANT = os.environ.get("BILLING_VARIANT", "assessment_cut")

# --- Section names ----------------------------------------------------------
#
# MEASURED OFF THE FOUR NOTES, NOT GUESSED. These are every heading that appears
# in the supplied PDFs; the parser uses them as block boundaries, so a heading
# missing from this list silently merges two sections. build_billing_sample.py
# prints the sections it found per note for exactly that reason — eyeball it.
#
# "PROCEDURES" appears capitalised in note 26819 and title-cased in the other
# three, so matching is case-insensitive.
SECTION_HEADINGS = (
    "Patient Demographics",
    "Visit Information",
    "CC/HPI",
    "Interval History",
    "ROS Findings",
    "Patient History",
    "Vital Signs",
    "Exam Findings",
    "Anticipatory Guidance",
    "Counseling",
    "Assessment",
    "Plan",
    "Patient Instructions",
    "Encounter Disposition",
    "Medications",
    "Orders",
    "Diagnostic Tests",
    "Care Plan: Goals",
    "Procedures",
    "Providers / Care Team",
)

# The block that holds the gold codes. Removed by assessment_cut and leakage_cut.
ASSESSMENT_HEADING = "Assessment"

# The second copy of the answer. Not a section heading of its own — it is a
# sub-block inside Patient History, introduced by a line like
# "Problem List Reviewed and updated by Rabbi Zia, MD (1500) 02/27/2026 08:29:03"
# and running until the next heading or the next "... Reviewed by" sub-block.
PROBLEM_LIST_MARKER = "Problem List Reviewed"

# --- Scoring ----------------------------------------------------------------
#
# EXACT CODE MATCH, AND THAT IS THE WHOLE LADDER. A predicted code is a hit iff
# its normalized form is in the gold set: uppercased, whitespace stripped, the
# decimal point kept. B08.5 matches "b08.5" and " B08.5 "; it does not match
# B08 or B08.50, and it should not — a coder submitting B08 when B08.5 was
# billed has submitted the wrong claim.
#
# The code description is deliberately NOT scored. It is the ICD catalogue's
# wording for the code, so a correct code implies a correct description and
# scoring both would double-count one decision.
#
# DUPLICATES ARE COLLAPSED ON BOTH SIDES. Note 96176 lists Z68.51 as DX 3 AND as
# DX 4 — a data-entry duplicate in the source, not two billable things. Gold for
# that note is 3 codes, not 4. Across the four notes: 17 DX lines, 16 unique.
STRIP_DOT = os.environ.get("BILLING_STRIP_DOT", "0") == "1"


def variant_label(variant):
    """The one-line description of `variant`, as printed in the report."""
    return VARIANT_LABELS.get(variant, variant)
