.PHONY: setup test smoke eval lint clean \
        mimic-sample mimic-smoke mimic-50 mimic-100 mimic-oracle mimic-check \
        mdace-sample mdace-oracle mdace-smoke mdace-50 mdace-run \
        recall-oracle recall-smoke recall-ab recall-compare recall-chunks \
        recall-run \
        recall-rescore recall-judge \
        recall-questions clean-recall-runs \
        billing-sample billing-show billing-oracle billing-check \
        billing-run billing-ab billing-rescore billing-ceiling billing-extract \
        clean-billing-runs

# Install dependencies via uv
setup:
	uv sync --extra test

# Run the CPU-only unit tests (no GPU, no model download)
test:
	uv run pytest -q

# Smoke test: evaluate the first 10 test examples (needs a GPU + HF login)
smoke:
	uv run python -m src.evaluate --limit 10

# Full evaluation -> results/comparison.csv (needs a GPU + HF login)
eval:
	uv run python -m src.evaluate

# --- MIMIC-IV medication NER -------------------------------------------------
# REAL PATIENT DATA. mimic-sample reads the credentialed source data from outside
# the repo and writes a small gitignored file; run it on the machine that holds
# the data. Everything else reads only that sample file.

# Extract the 100-note sample -> data/samples/ (gitignored). LOCAL ONLY.
mimic-sample:
	uv run python -m src.build_mimic_sample

# Smoke test: 5 notes (needs a GPU + HF login)
mimic-smoke:
	uv run python -m src.evaluate_mimic --limit 5

# n=50 -> results/mimic_ner_50.csv (+ report). Resumable.
mimic-50:
	uv run python -m src.evaluate_mimic --n 50

# n=100 -> results/mimic_ner_100.csv (+ report). Reuses the n=50 work.
mimic-100:
	uv run python -m src.evaluate_mimic --n 100

# Harness structural ceiling — no model, no GPU, runs in seconds.
mimic-oracle:
	uv run python -m src.evaluate_mimic --oracle --n 100

# Confirm no data file is tracked or stageable. Run before every commit.
mimic-check:
	@echo "--- tracked files matching data patterns (must be empty) ---"
	@git ls-files | grep -Ei 'discharge|noteevents|\.csv\.gz$$|^data/|^samples/|^outputs/' \
		&& (echo "FAIL: patient-data file is TRACKED" && exit 1) || echo "ok: none tracked"
	@echo "--- git status (no data paths should appear) ---"
	@git status --short
	@echo "--- ignore rules covering the sample + outputs ---"
	@git check-ignore -v data/samples/mimic_med_sample.jsonl outputs/ 2>/dev/null \
		|| echo "WARNING: expected ignore rules not matched"

lint:
	uv run ruff check src tests

clean:
	rm -rf results/*.json results/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +

# Wipes cached per-note run state. The next run starts from scratch.
clean-mimic-runs:
	rm -rf outputs/mimic

# --- MDACE term-level billing NER (MIMIC-III) ---------------------------------
# REAL PATIENT DATA. mdace-sample reads the credentialed source files from
# outside the repo and writes a small gitignored file; run it on the machine
# that holds the data. Everything else reads only that sample file.

# Extract the 73-note union sample -> data/samples/ (gitignored). LOCAL ONLY.
mdace-sample:
	uv run python -m src.build_mdace_sample

# Harness ceiling — no model, no GPU, ~10s. B1/B2 must score 1.0000.
mdace-oracle:
	uv run python -m src.evaluate_mdace --oracle

# Smoke test: 5 notes spanning short/long and both chart types, keeping the raw
# replies. 15 chunks, ~12 min (needs a GPU + HF login).
mdace-smoke:
	uv run python -m src.evaluate_mdace --smoke 5 --dump-replies

# Phase 1: the stratified 50 notes / 122 chunks, ~1.6h. The headline result.
mdace-50:
	uv run python -m src.evaluate_mdace --limit 50

# Phase 2: the remaining 23 notes / 80 chunks, ~1.1h. Reuses phase 1's cache and
# adds views A1 and B1. Resumable.
mdace-run:
	uv run python -m src.evaluate_mdace

# --- MDACE recall benchmark (MIMIC-III) ---------------------------------------
# REAL PATIENT DATA. Reads ONE file, 8-07-mdace-ner-eval_sample_100-LOCAL.jsonl,
# which embeds note text — there is no join and no sample-building step. Point
# RECALL_SAMPLE_FILE at your copy if it is not in the default location.

# Harness ceiling — no model, no GPU, ~10s. Every source must read 1.0000 at L1
# and the combined matching must show zero false positives. Run this BEFORE
# spending GPU time; it caught real bugs twice on the previous branch.
recall-oracle:
	uv run python -m src.evaluate_recall --oracle

# Smoke test: the 3 longest notes, keeping the raw replies. The multi-chunk
# path, which is where truncation and OOM live (needs a GPU + HF login).
recall-smoke:
	uv run python -m src.evaluate_recall --smoke 3 --dump-replies

# The prompt A/B, ~25 min each. `scoped` names the categories to exclude and
# grows a rule every time a run finds a new leak; `billable` replaces all of
# them with the criterion that defines gold. They hash differently, so the two
# runs cannot mix or replay each other.
#
# Read the volume rows before the recall row: a prompt that extracts more scores
# higher recall almost regardless of quality.
recall-ab:
	uv run python -m src.evaluate_recall --smoke 3 --dump-replies --prompt scoped
	uv run python -m src.evaluate_recall --smoke 3 --dump-replies --prompt billable
	uv run python -m src.recall_compare

# Compare the two most recent finished runs. No GPU, no re-scoring.
recall-compare:
	uv run python -m src.recall_compare

# The chunk-size experiment. Neither prompt fixed the volume problem -- both
# extract 15-17x the gold -- and the prompt was never going to. Shorter windows
# give the model less to describe per call, which attacks the volume AND the
# repetition loop, since the loop is degeneration on a long list.
#
# 17 chunks at 400 words against 27 at 250, on the same 2 notes. More calls, but
# each writes less, so the wall clock is roughly a wash.
recall-chunks:
	uv run python -m src.evaluate_recall --smoke 2 --dump-replies --chunk-words 400 --overlap-words 80
	uv run python -m src.evaluate_recall --smoke 2 --dump-replies --chunk-words 250 --overlap-words 50
	uv run python -m src.recall_compare

# The benchmark: 24 notes / 82 chunks, ~1-1.5h on a T4. Resumable.
recall-run:
	uv run python -m src.evaluate_recall

# Re-score a finished run with different thresholds. No model call, no GPU.
recall-rescore:
	uv run python -m src.evaluate_recall --score-only

# L5: adjudicate the pairs L2-L4 newly accepted. `--judge medgemma` has the
# model under test grade its own matches and says so; `--judge none` writes the
# questions out for a human or an external model.
recall-judge:
	uv run python -m src.recall_judge --judge medgemma

recall-questions:
	uv run python -m src.recall_judge --judge none

# Wipes cached per-note run state for the benchmark.
clean-recall-runs:
	rm -rf outputs/mdace_recall

# --- Pediatric billing ICD-code evaluation ------------------------------------
# REAL PATIENT DATA, AND NOT DE-IDENTIFIED. Ehtesham Bhai's four encounter PDFs
# carry patient names, dates of birth, a rendering provider and a license
# number. They live in ../ai-medical-billing/ and must stay outside the repo.
# billing-sample parses them here; the GPU box only ever sees the built sample.

# Parse the four PDFs -> data/samples/ (gitignored). LOCAL ONLY. Needs
# poppler-utils (pdftotext). Read what it prints — the leak counts are checks
# with a right answer: full 16, assessment_cut 2, leakage_cut 0.
billing-sample:
	uv run python -m src.build_billing_sample

# Dump one note's three input variants, to see exactly what the model is shown.
# QUOTES NOTE TEXT to the terminal. e.g. make billing-show NOTE=26819
billing-show:
	uv run python -m src.build_billing_sample --show $(NOTE)

# Harness ceiling — no model, no GPU, ~1s. Every variant must read 1.0000 on
# both precision and recall. Run this BEFORE spending GPU time.
billing-oracle:
	uv run python -m src.evaluate_billing --oracle

# The harness check WITH the model: the note with the DX lines left in. Not a
# result — a low number here means the prompt or the parser is wrong, not the
# model. 4 calls (needs a GPU + HF login).
billing-check:
	uv run python -m src.evaluate_billing --variant full --dump-replies

# The run: 4 notes x 3 variants = 12 calls. Resumable.
billing-run:
	uv run python -m src.evaluate_billing --dump-replies

# The repetition-penalty A/B. `1.0` is off and REPLAYS the earlier no-penalty
# run from cache rather than re-running it, so this costs 12 calls, not 24.
#
# Read precision and recall separately. A generation loop produces false
# positives, not misses, so the penalty is expected to move precision and leave
# recall where it was. If recall moves, the loop was eating real answers too.
billing-ab:
	uv run python -m src.evaluate_billing --repetition-penalty 1.0
	uv run python -m src.evaluate_billing --dump-replies

# Rescore a finished run. No model call, no GPU.
billing-rescore:
	uv run python -m src.evaluate_billing --score-only

# --- Did it FIND the conditions it failed to code? -----------------------------
# A separate question from the code evaluation above, and a separate answer key
# (src/billing_evidence.py, hand-built from the notes). 0 of 16 codes cannot
# tell "never found the influenza" apart from "found it, coded it J11.9", and
# those point at different work.

# The structural ceiling: how many gold codes are evidenced in each variant's
# input at all. No GPU, ~1s. Reads 16 / 16 / 12 -- four codes lose their only
# evidence with the Problem List, so leakage_cut recall was quoted against the
# wrong denominator.
billing-ceiling:
	uv run python -m src.evaluate_billing_extract --ceiling

# Runs the existing `billable` extraction prompt (78/100 on MDACE) over the same
# notes and reports, per gold code, whether the condition was surfaced.
# 12 calls (needs a GPU + HF login).
billing-extract:
	uv run python -m src.evaluate_billing_extract --dump-phrases

# Wipes cached per-note run state.
clean-billing-runs:
	rm -rf outputs/billing_icd
