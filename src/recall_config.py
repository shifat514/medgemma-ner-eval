"""Configuration for the MDACE recall benchmark.

Separate from mdace_config.py on purpose. That file configures the term-NER
evaluation on the 73-note union sample; this one configures a *benchmark* with a
single question — how much of the billed evidence MedGemma-4B recovers from a
note, zero-shot — measured on one file and nothing else. Sharing a config module
would have meant one env var name meaning two different things.

WHAT IS DELIBERATELY ABSENT. No stratified sample, no seed, no chart-type split,
no view ladder. The input is `8-07-mdace-ner-eval_sample_100-LOCAL.jsonl`: 100
annotation rows on 24 notes, note text embedded, so there is no join and no
notes file. Gold is whatever that file says it is.

WHAT IS DELIBERATELY IDENTICAL. Chunk geometry, token cap and 4-bit loading are
the same as the term-NER run, so its measured ~68 s/chunk carries over and the
0.53 reference number stays comparable.
"""

import os

# --- The one input file -----------------------------------------------------
#
# OUTSIDE THE REPO AND MUST STAY THAT WAY: it embeds MIMIC-III note text
# (credentialed PhysioNet data). Mirrored at
# s3://zeda-mimic-dataset/eval_datasets/ — the repo's .env holds the
# credentials, and the shell does not auto-source it:
#     set -a && . ./.env && set +a

MDACE_BASE = os.environ.get(
    "MDACE_BASE", "/home/shifat/zeda_ml_works/zeda_mimic_datasets"
)

SAMPLE_100_FILE = os.environ.get(
    "RECALL_SAMPLE_FILE",
    os.path.join(MDACE_BASE, "8-07-mdace-ner-eval_sample_100-LOCAL.jsonl"),
)

S3_SAMPLE_100_FILE = (
    "s3://zeda-mimic-dataset/eval_datasets/"
    "8-07-mdace-ner-eval_sample_100-LOCAL.jsonl"
)

# --- Output -----------------------------------------------------------------

# Per-note run state, extracted findings, and the per-level pair dumps. All of
# it is note-derived, all of it gitignored. On Colab point this at mounted
# Drive so a disconnect cannot lose a run:
#   os.environ["RECALL_OUTPUT_DIR"] = "/content/drive/MyDrive/mdace-recall"
OUTPUT_DIR = os.environ.get("RECALL_OUTPUT_DIR", "outputs/mdace_recall")

# Aggregate metrics + markdown report. Counts and rates only, safe to commit.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# --- Chunking ---------------------------------------------------------------
#
# 24 notes come to 82 windows at 400/80. Terms are pooled per note, so the
# overlap costs nothing at scoring time — a finding seen in two windows is one
# finding.
CHUNK_WORDS = int(os.environ.get("RECALL_CHUNK_WORDS", "400"))
OVERLAP_WORDS = int(os.environ.get("RECALL_OVERLAP_WORDS", "80"))

# --- Model ------------------------------------------------------------------

MODEL_ID = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
MODEL_NAME = os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-4b-it")

LOAD_IN_4BIT = True

# Carried over from the term-NER branch, where 512 truncated 12 of 15 smoke
# chunks. The two-field output costs ~2x tokens per finding, so the cap matters
# more here, not less; the prompt asks for one entry per distinct finding to pay
# that back. n_cap_hits is counted and printed — non-zero means recall is
# understated, never "close enough".
GEN_CONFIG = {
    "max_new_tokens": int(os.environ.get("RECALL_MAX_NEW_TOKENS", "1024")),
    "do_sample": False,
}

# A reply within this many tokens of the cap is treated as truncated.
CAP_MARGIN = 8

# --- Gold sources -----------------------------------------------------------
#
# The accept-set for a billed code is the union of three columns of the input
# file. Recall and false positives are broken out per source, because the
# genuinely useful question is *whose wording the model produces* — the note's,
# the ICD catalogue's, or SNOMED's.
SOURCES = ("evidence", "description", "snomed")
COMBINED = "combined"
SOURCE_LABELS = {
    "evidence": "evidence text (what the note says)",
    "description": "code description (ICD catalogue wording)",
    "snomed": "SNOMED concept terms",
    COMBINED: "all three combined",
}

# --- Matching ladder --------------------------------------------------------
#
# Each level is a superset of the one above, so recall is monotonically
# non-decreasing and the gain from each level is attributable to that level.
#
# THRESHOLDS ARE REPORTED, NEVER SILENTLY CHOSEN. These defaults are the ones
# that separate the good rows from the bad ones in the measured pair table (see
# recall_matching's docstring): Dice 0.80 admits "acute kidney injury" vs
# "kidney injury, acute" (1.00) and rejects "acute renal failure" vs "chronic
# renal failure" (0.67); char ratio 0.90 admits "hyperlipidema" vs
# "hyperlipidemia" (0.96) and rejects that same acute/chronic pair (0.75).
LEVELS = ("L1", "L2", "L3", "L4")

DICE_MIN = float(os.environ.get("RECALL_DICE_MIN", "0.80"))
RATIO_MIN = float(os.environ.get("RECALL_RATIO_MIN", "0.90"))

# 0.60, and it is a floor chosen against a measured table rather than a cutoff
# that works. See recall_matching.MEASURED_COSINE: on the default encoder the
# pairs L4 exists to catch score 0.61 to 0.98, and the pairs it must reject
# score 0.18 to 0.83 — completely interleaved. `acute renal failure` against
# `chronic renal failure` (0.833) outranks eight of the ten pairs L4 is for.
#
# Raising the threshold does not buy precision, it only loses abbreviations:
# 0.60 catches 10 of 10 wanted and admits 5 of 7 unwanted; 0.66 catches 6 of 10
# and still admits 3 of 7. So the default reaches L4's whole purpose and hands
# the separation problem to L5, which is where it was always going to end up.
COSINE_MIN = float(os.environ.get("RECALL_COSINE_MIN", "0.60"))

# L4 needs a BIOMEDICAL sentence encoder. A general-purpose MiniLM does not know
# that HTN is hypertension, which is the only thing L4 exists to reach. The
# backend is imported lazily and L4 is skipped with a message when it is absent,
# so L1-L3 stay unit-testable on a CPU-only laptop.
#
# Two alternatives were measured and are worse: S-PubMedBert-MS-MARCO scores
# every pair 0.85-0.99 including `pneumonia` against `fracture of left wrist`,
# and MedEmbed-small-v0.1 ranks the `no evidence of sepsis` negation above HTN,
# CHF and MI. Change this only with the same table in hand.
EMBED_MODEL = os.environ.get(
    "RECALL_EMBED_MODEL", "NeuML/pubmedbert-base-embeddings"
)


# --- Input section filtering ------------------------------------------------
#
# Sections dropped BEFORE chunking. Post-hoc filtering of findings bought +0.6
# precision points; doing it on the input instead also frees output budget,
# which is the part that matters — 21 of 82 chunks ran out of room and 7 were
# cut while still producing real findings.
#
# MEASURED ON THE FILE, NOT GUESSED. This list removes 18% of the text and
# costs ZERO gold phrases. Two obvious-looking candidates are deliberately
# absent because they hold gold:
#
#   Social History        203 words, 3 gold — the smoking and alcohol status
#                         codes (F17.200, F10.10) are evidenced here
#   Discharge Instructions 1,426 words, 1 gold
#
# Radiology (FINDINGS, IMPRESSION, Imaging) is also absent: it holds no gold in
# these 24 notes, but a radiology impression genuinely can name a billable
# diagnosis and dropping it would be fitting to the sample.
DROP_SECTIONS = (
    "medications on admission", "discharge medications", "medications",
    "followup instructions", "follow up instructions", "discharge disposition",
    "discharge condition", "order date", "disp", "activity", "allergies",
    "family history", "tablet refills", "capsule refills", "facility",
    "completed by", "pertinent results", "discharge labs", "labs on admission",
    "admission labs", "laboratory data", "physical exam", "discharge exam",
)

# Off by default so the existing runs stay reproducible.
DROP_SECTIONS_ON = os.environ.get("RECALL_DROP_SECTIONS", "0") == "1"


def thresholds():
    """The threshold set, as it is printed into the report."""
    return {"dice_min": DICE_MIN, "ratio_min": RATIO_MIN,
            "cosine_min": COSINE_MIN, "embed_model": EMBED_MODEL}
