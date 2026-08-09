"""Configuration for the MDACE term-level billing-NER evaluation.

Separate from mimic_config.py because almost nothing is shared: a different
corpus (MIMIC-III, not MIMIC-IV), a different label space (billing evidence, not
the six i2b2 medication types), and a different scoring unit (term sets, not
token positions).

WHY TERM SETS AND NOT SPANS. 4,770 of 9,499 gold terms occur more than once in
their own note, so "did the model find this term" and "did the model find this
occurrence" are different questions with very different scores. The billing
lookup downstream consumes terms, not offsets, so terms are what we score. This
removes the whole align/BIO/seqeval layer the medication evaluation needs, and
with it the multi-occurrence expansion problem that forced ALIGN_MODE.

PATHS BELOW POINT OUTSIDE THE REPO AND MUST STAY THAT WAY. MDACE is built on
MIMIC-III notes (credentialed PhysioNet data); build_mdace_sample.py reads from
here and writes a small gitignored sample file.
"""

import os

# --- Source data (outside the repo; local machine only) ---------------------
#
# Mirrors s3://zeda-mimic-dataset/eval_datasets/. Two files that join 1:1 on
# note_id: annotations (no note text) and notes (no annotations).
#
# The third file in that prefix, 8-07-mdace-ner-eval_sample_100-LOCAL.jsonl, is
# a 100-ROW cut that lands on only 24 notes and carries 99 of the 195 gold terms
# those notes actually have. It is NOT an answer key — it is scored as view A1
# only, to show what the truncation costs. See SAMPLE_100_FILE below.

MDACE_BASE = os.environ.get(
    "MDACE_BASE", "/home/shifat/zeda_ml_works/zeda_mimic_datasets"
)

DATASET_FILE = os.environ.get(
    "MDACE_DATASET_FILE",
    os.path.join(MDACE_BASE, "8-07-mdace-ner-eval_dataset-LOCAL.jsonl"),
)
NOTES_FILE = os.environ.get(
    "MDACE_NOTES_FILE",
    os.path.join(MDACE_BASE, "8-07-mdace-notes-LOCAL.jsonl"),
)
SAMPLE_100_FILE = os.environ.get(
    "MDACE_SAMPLE_100_FILE",
    os.path.join(MDACE_BASE, "8-07-mdace-ner-eval_sample_100-LOCAL.jsonl"),
)

# --- S3 source (alternative to the local paths above) -----------------------
S3_BUCKET = os.environ.get("MDACE_S3_BUCKET", "zeda-mimic-dataset")
S3_PREFIX = f"s3://{S3_BUCKET}/eval_datasets"
S3_DATASET_FILE = f"{S3_PREFIX}/8-07-mdace-ner-eval_dataset-LOCAL.jsonl"
S3_NOTES_FILE = f"{S3_PREFIX}/8-07-mdace-notes-LOCAL.jsonl"
S3_SAMPLE_100_FILE = f"{S3_PREFIX}/8-07-mdace-ner-eval_sample_100-LOCAL.jsonl"

# --- Repo-internal, gitignored ---------------------------------------------

SAMPLE_FILE = os.environ.get("MDACE_SAMPLE_FILE", "data/samples/mdace_sample.jsonl")

# Per-note run state and the extracted-term lists. On Colab, point this at a
# mounted Drive path so a disconnect cannot lose a 2.7h run:
#   os.environ["MDACE_OUTPUT_DIR"] = "/content/drive/MyDrive/mdace-outputs"
OUTPUT_DIR = os.environ.get("MDACE_OUTPUT_DIR", "outputs/mdace")

# Aggregate metrics + markdown report. PHI-free, safe to commit.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# --- Note-level chart type --------------------------------------------------
#
# chart_type is a property of each ANNOTATION ROW, not of the note: 118 of the
# 1,074 notes carry both Profee and Inpatient rows because the same encounter
# was billed both ways. So a note-level rule is required, and different rules
# give different strata.
#
# "any Inpatient row wins" reproduces the measured corpus figures exactly:
#   Inpatient 604 notes, median 1,108 words, median 6 gold terms
#   Profee    470 notes, median   148 words, median 4 gold terms
# Do not change this without rechecking those four numbers.
CHART_TYPES = ("Profee", "Inpatient")
INPATIENT_WINS = True

# --- Sampling ---------------------------------------------------------------
#
# REPRODUCIBILITY IS LOAD-BEARING. Fixing the seed is not enough: the draw order
# changes the result. One shared Random(seed), pools sorted, Profee drawn FIRST,
# then Inpatient continuing from the same generator state, yields
#   122 chunks / 324 gold terms   (Profee 28/110, Inpatient 94/214)
# A fresh Random(seed) per stratum instead yields 167 chunks / 432 terms — same
# seed, same code, different sample. tests/test_mdace_sample.py pins 122/324.
SEED = int(os.environ.get("MDACE_SEED", "13"))
STRATUM_N = int(os.environ.get("MDACE_STRATUM_N", "25"))
STRATUM_ORDER = ("Profee", "Inpatient")

# --- Chunking ---------------------------------------------------------------
#
# Reused from the medication evaluation unchanged (src/chunking.chunk_windows).
# Inpatient notes run to a median 1,108 words, well past what a 4B model will
# exhaustively enumerate in one call. Terms are unioned across a note's chunks,
# so the overlap costs nothing at scoring time — a term found twice is one term.
CHUNK_WORDS = int(os.environ.get("MDACE_CHUNK_WORDS", "400"))
OVERLAP_WORDS = int(os.environ.get("MDACE_OVERLAP_WORDS", "80"))

# --- Model ------------------------------------------------------------------

MODEL_ID = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
MODEL_NAME = os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-4b-it")

LOAD_IN_4BIT = True

# 512, not the medication evaluation's 1024. Two reasons, both measured:
#
# 1. Expected output is far smaller. That task wanted six kinds of medication
#    detail (~30 entities/chunk); this one wants billing evidence terms, and a
#    note carries a median of 4-6 gold terms in total. 512 is ~10x headroom.
# 2. The medication run hit its cap on the SAME 5 of 21 chunks at both 1024 and
#    1536. A cap that binds identically at two very different limits is a
#    repetition loop, not a length shortfall; a lower cap ends the loop sooner
#    and costs less GPU time.
#
# The runner counts cap hits (n_cap_hits). If the smoke run shows any, raise
# this back to 1024 rather than assuming truncation is harmless.
GEN_CONFIG = {
    "max_new_tokens": int(os.environ.get("MDACE_MAX_NEW_TOKENS", "512")),
    "do_sample": False,
}

# A reply within this many tokens of the cap is treated as truncated.
CAP_MARGIN = 8

# --- Scoring ----------------------------------------------------------------

# The gold code systems, and which of them are plain diseases. 84% of gold rows
# are non-Z, non-injury ICD-10-CM (i.e. actual conditions); the other 16% are
# procedures (CPT, ICD-10-PCS), injuries (S/T), and status/history (Z). A prompt
# asking only for "diagnoses" therefore caps recall at 0.84 before the model
# does anything, which is why the prompt asks for all of it and the report
# breaks out the conditions-only slice instead.
CONDITION_SLICE = "conditions"


def code_bucket(code_system, code):
    """Coarse bucket for a gold code, used for the conditions-only slice.

    Returns one of: "conditions", "status_history", "injury", "procedure".
    """
    code = str(code or "").strip().upper()
    if code_system == "ICD-10-CM":
        if code.startswith("Z"):
            return "status_history"
        if code[:1] in ("S", "T"):
            return "injury"
        return "conditions"
    if code_system in ("CPT", "ICD-10-PCS"):
        return "procedure"
    return "conditions"          # ICD-9-CM (2 rows) — treat as a condition
