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

# 1024, RAISED FROM 512 AFTER THE FIRST RUN MEASURED IT.
#
# The original 512 was reasoned, not measured, and the reasoning was wrong. It
# went: gold is 2-6 codes per note, one short object per code, so a correct
# reply is well under 200 tokens; anything reaching 512 is looping rather than
# thorough.
#
# The 2026-08-27 run truncated 6 of 12 replies — all four `leakage_cut` notes
# and two `assessment_cut` notes. The premise the cap rested on is the thing
# under test: the model does NOT return 2-6 codes. `full` returned 17 across
# four notes without truncating, and the two harder variants ran past 512, so
# the model is over-producing exactly as it did on the recall branch. Sizing the
# budget to gold rather than to observed output made every truncation land on
# the two variants that matter and none on the harness check.
#
# 1024 matches the recall branch. It does not fix over-production — nothing here
# does, and precision is where that will show up — but it stops the measurement
# from being decided by the cap. Truncations are still counted and printed.
# REPETITION PENALTY, ADDED AFTER THE FIRST REAL RUN SHOWED THE MODEL COUNTING.
#
# 20 of 32 `assessment_cut` predictions and 20 of 29 `leakage_cut` predictions
# were enumerations, not answers:
#
#     R51.9, R51.81, R51.82, R51.83, ... R51.89        (note 112976)
#     R11.0, R11.1, R11.2, R11.3, ... R11.9            (note 96176)
#
# It is incrementing the last digit. Most of those codes do not exist. This is
# the same degeneration the recall branch measured, and raising the cap from 512
# to 1024 only bought it more room to loop — 6 of 12 replies still truncated.
#
# NOT `no_repeat_ngram_size`, WHICH IS THE OBVIOUS WRONG ANSWER HERE. The output
# is JSON: `", "description": "` and `{"code": "` repeat on every single item by
# design. Forbidding repeated n-grams would forbid the format, and the run would
# come back malformed rather than fixed. A logit penalty degrades gracefully
# where an n-gram ban does not.
#
# 1.15 is chosen to be *mild* for the same reason — the required JSON keys have
# to survive. Above ~1.2 the penalty starts fighting the format. The malformed
# count is printed, so if this is set too high that shows up as a number.
#
# THIS IS EXPECTED TO MOVE PRECISION AND NOT RECALL. A loop produces false
# positives; it does not produce misses. Recall was 0 of 16 on `leakage_cut` and
# the loop cannot explain that. Recorded here so the result is checked against a
# prediction rather than rationalised after the fact.
REPETITION_PENALTY = float(os.environ.get("BILLING_REPETITION_PENALTY", "1.15"))

GEN_CONFIG = {
    "max_new_tokens": int(os.environ.get("BILLING_MAX_NEW_TOKENS", "1024")),
    "do_sample": False,
    "repetition_penalty": REPETITION_PENALTY,
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
