"""Configuration for the MIMIC-IV medication-NER evaluation.

Separate from src/config.py because the label space is disjoint: the NCBI/BC5CDR
work scores Disease/Chemical, this scores the six i2b2-2009-style medication
types. Everything else (prompt parsing, span alignment, seqeval scoring) is the
shared machinery in src/prompt.py, src/align.py and src/scoring.py.

PATHS BELOW POINT OUTSIDE THE REPO AND MUST STAY THAT WAY. The credentialed
MIMIC data is never copied in; build_mimic_sample.py reads from here and writes
a small gitignored sample file.
"""

import os

# --- Source data (outside the repo; local machine only) ---------------------

MIMIC_BASE = os.environ.get("MIMIC_BASE", "/home/shifat/zeda/zeda_ml_dataset")

DISCHARGE_CSV = os.environ.get(
    "MIMIC_DISCHARGE_CSV",
    os.path.join(
        MIMIC_BASE,
        "mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz",
    ),
)

LABEL_DIR = os.environ.get(
    "MIMIC_LABEL_DIR",
    os.path.join(
        MIMIC_BASE,
        "medication-extraction-labels-for-mimic-iv-note-clinical-database-1.0.0/"
        "mimic-iv-note-labels",
    ),
)

# --- S3 source (alternative to the local paths above) -----------------------
#
# The bucket mirrors the PhysioNet layout but with shorter prefix names, so the
# S3 paths are NOT derivable from MIMIC_BASE — they are spelled out here.
#
#   s3://zeda-mimic-dataset/mimic-iv-note/note/discharge.csv.gz          1,139 MB
#   s3://zeda-mimic-dataset/medication-extraction-labels/
#       mimic-iv-note-labels/                                    600 CSVs, 1.8 MB
#
# Select S3 with `build_mimic_sample --source s3`, or point the two env vars at
# s3:// URIs directly (any path starting with s3:// is handled transparently by
# src/s3_io.py). Credentials come from the normal boto3 chain; --aws-profile or
# MIMIC_AWS_PROFILE pins a profile.
#
# ONLY the sample build reads from S3. The evaluation always reads the small
# local sample file, so a long GPU run never depends on network or credentials.
S3_BUCKET = os.environ.get("MIMIC_S3_BUCKET", "zeda-mimic-dataset")
S3_DISCHARGE_CSV = f"s3://{S3_BUCKET}/mimic-iv-note/note/discharge.csv.gz"
S3_LABEL_DIR = f"s3://{S3_BUCKET}/medication-extraction-labels/mimic-iv-note-labels"

# --- Repo-internal, gitignored ---------------------------------------------

# The small extracted sample (notes + gold labels) that gets uploaded to Colab.
SAMPLE_FILE = os.environ.get("MIMIC_SAMPLE_FILE", "data/samples/mimic_med_sample.jsonl")

# Per-note run state, resume files, and per-example error dumps. Error dumps
# QUOTE NOTE TEXT — this directory is gitignored and must stay that way.
OUTPUT_DIR = os.environ.get("MIMIC_OUTPUT_DIR", "outputs/mimic")

# Aggregate metrics + markdown report. PHI-free, safe to commit.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# --- Label space -----------------------------------------------------------

# 1:1 pass-through of the six gold annotation types. Capitalized because
# prompt.parse_entities normalizes a model-emitted type with .capitalize(), and
# every type here is a single word so that round-trips exactly.
GOLD_TYPE_MAP = {
    "MEDICATION": "Medication",
    "DOSE": "Dose",
    "MODE": "Mode",
    "FREQUENCY": "Frequency",
    "DURATION": "Duration",
    "REASON": "Reason",
}

ENTITY_TYPES = list(GOLD_TYPE_MAP.values())

# Tie-break for the ~58 gold spans where one (start, end) carries two different
# annotation types. Lower index wins. Also the deterministic order in which
# gold spans are painted onto tokens.
TYPE_PRIORITY = ["Medication", "Dose", "Mode", "Frequency", "Duration", "Reason"]

BIO_LABELS = ["O"] + [
    f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")
]

# Label files with ~no annotations. Median is 97 annotations/note, so 0 and 1
# are almost certainly gold-labeling failures rather than medication-free
# discharge summaries. Excluded from the sampling pool; documented in the report.
EXCLUDED_NOTE_IDS = {
    "19933834-DS-2": "0 annotations in label file (median 97) — likely labeling failure",
    "12619324-DS-22": "1 annotation in label file (median 97) — likely labeling failure",
}

# --- Sampling --------------------------------------------------------------

SEED = int(os.environ.get("MIMIC_SEED", "13"))

# The 50-note set is the first 50 of the same seeded 100-note draw, so it is a
# strict subset and the two runs are directly comparable.
SAMPLE_SIZES = [50, 100]
MAX_SAMPLE = max(SAMPLE_SIZES)

# --- Chunking --------------------------------------------------------------

# Discharge summaries run 181-5,425 whitespace tokens (median 1,521), and a note
# can carry up to 381 gold entities. Chunking bounds the GENERATION per call
# (~100 entities would blow past any sane max_new_tokens) as much as the input.
CHUNK_WORDS = int(os.environ.get("MIMIC_CHUNK_WORDS", "400"))
OVERLAP_WORDS = int(os.environ.get("MIMIC_OVERLAP_WORDS", "80"))

# How a predicted string is located in the text — the biggest lever on the
# precision ceiling. Compare candidates with `--oracle` (no GPU, ~7 s each).
#   all-per-chunk   every occurrence within the producing chunk
#   first-per-chunk first occurrence within each chunk
#   first-note      first occurrence in the note, predictions pooled
#
# Default chosen by measurement, not intuition. Oracle ceiling over the 100-note
# sample (micro P / R / F1):
#   first-per-chunk  0.8583 / 0.8817 / 0.8698   <- default
#   all-per-chunk    0.6976 / 0.9505 / 0.8046
#   first-note       0.7917 / 0.5524 / 0.6507
# first-per-chunk wins micro F1 by +6.5 points and wins on all six types
# individually: trading 6.9 points of recall for 18.6 of precision. See
# results/mimic_ner_align_mode_comparison.md.
DEFAULT_ALIGN_MODE = "first-per-chunk"
ALIGN_MODE = os.environ.get("MIMIC_ALIGN_MODE", DEFAULT_ALIGN_MODE)

# --- Model -----------------------------------------------------------------

MODEL_ID = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
MODEL_NAME = os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-4b-it")

LOAD_IN_4BIT = True

# Greedy decoding for reproducibility. max_new_tokens is well above the 512 used
# for single sentences: a 400-word chunk of a discharge summary can legitimately
# contain 30+ medication entities.
GEN_CONFIG = {
    "max_new_tokens": int(os.environ.get("MIMIC_MAX_NEW_TOKENS", "1024")),
    "do_sample": False,
}

# A reply within this many tokens of the cap is treated as truncated.
CAP_MARGIN = 8
