.PHONY: setup test smoke eval lint clean \
        mimic-sample mimic-smoke mimic-50 mimic-100 mimic-oracle mimic-check \
        mdace-sample mdace-oracle mdace-smoke mdace-50 mdace-run

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
