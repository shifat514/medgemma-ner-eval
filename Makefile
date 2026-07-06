.PHONY: setup test smoke eval haiku-smoke haiku-eval lint clean

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

# Claude Haiku baseline (needs ANTHROPIC_API_KEY in .env; no GPU).
# Smoke test does a 1-sentence API check first, then the first 10 examples.
haiku-smoke:
	uv run python -m src.evaluate_haiku --limit 10

# Full Haiku eval -> results/haiku_comparison.csv (does NOT touch comparison.csv).
haiku-eval:
	uv run python -m src.evaluate_haiku

lint:
	uv run ruff check src tests

clean:
	rm -rf results/*.json results/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
