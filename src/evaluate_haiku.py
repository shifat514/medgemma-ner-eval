"""Claude Haiku zero-shot clinical-NER evaluation — API sibling of evaluate.py.

Same pipeline as the MedGemma run (same combined NCBI + BC5CDR test set, same
prompt, same JSON->entities parsing, same span->BIO alignment, same seqeval
scorer), so the Haiku number is directly comparable to MedGemma's 0.58 and to the
sibling BERT table. The ONLY difference is the model: each sentence is sent to the
Anthropic Messages API (claude-haiku-4-5) instead of a local MedGemma pipeline.

Results are written to SEPARATE files so the MedGemma output is never touched:
    results/haiku_comparison.csv    (columns: model, entity, precision, recall, f1, support)
    results/haiku_full_report.json

``--limit N`` first does a single 1-sentence API call to confirm the key + model
work before looping the whole set. Estimated token usage and a rough cost are
printed at the end of every run.
"""

import argparse
import json
import os

from . import config as _config
from .evaluate import predict_example  # reused: parse -> align, with graceful degradation
from .haiku_model import load_haiku_client, run_haiku
from .prompt import parse_entities
from .scoring import _NpEncoder, report_to_rows, score


def _new_usage():
    return {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def _estimate_cost(usage):
    return (
        usage["input_tokens"] / 1e6 * _config.HAIKU_PRICE_INPUT_PER_MTOK
        + usage["output_tokens"] / 1e6 * _config.HAIKU_PRICE_OUTPUT_PER_MTOK
    )


def preflight_check(client, model_id):
    """One 1-sentence call to confirm the key + model before the full loop."""
    sentence = "The patient was treated with aspirin for chest pain."
    print(f"Preflight: sending one sentence to {model_id} to verify the key + model...")
    reply = run_haiku(client, sentence, model_id=model_id)
    ents = parse_entities(reply)
    print(f"  OK — model replied and parsed {len(ents)} entity(ies): {ents}")
    return ents


def _write_haiku_results(report, model_name, results_dir=None):
    """Write haiku_full_report.json + haiku_comparison.csv (same schema/scorer,
    separate filenames). Returns (DataFrame, csv_path)."""
    import pandas as pd

    results_dir = results_dir or _config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "haiku_full_report.json")
    with open(json_path, "w") as f:
        json.dump({model_name: report}, f, indent=2, cls=_NpEncoder)

    df = pd.DataFrame(report_to_rows(report, model_name))
    csv_path = os.path.join(results_dir, "haiku_comparison.csv")
    df.to_csv(csv_path, index=False)
    return df, csv_path


def _print_usage(usage, model_name):
    cost = _estimate_cost(usage)
    print(
        f"\nAPI usage ({model_name}): {usage['calls']} calls, "
        f"{usage['input_tokens']:,} input + {usage['output_tokens']:,} output tokens.\n"
        f"Estimated cost: ${cost:.4f} "
        f"(@ ${_config.HAIKU_PRICE_INPUT_PER_MTOK:.2f}/1M in, "
        f"${_config.HAIKU_PRICE_OUTPUT_PER_MTOK:.2f}/1M out). Rough estimate — "
        "check your Anthropic console for actual billing."
    )


def run_eval(limit=None, model_id=None, model_name=None, skip_preflight=False):
    model_id = model_id or _config.HAIKU_MODEL
    model_name = model_name or _config.HAIKU_MODEL_NAME

    from .datasets import build_test_set
    ds, _ = build_test_set()
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"Loaded {len(ds)} test examples (model={model_id})")

    client = load_haiku_client()

    if not skip_preflight:
        preflight_check(client, model_id)

    usage = _new_usage()

    def run_fn(c, sentence):
        return run_haiku(c, sentence, usage_acc=usage, model_id=model_id)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(ds, total=len(ds), desc="Haiku NER")
    except ImportError:
        iterator = ds

    gold_bio, pred_bio = [], []
    for ex in iterator:
        gold_bio.append(ex["bio"])
        pred_bio.append(predict_example(client, ex["tokens"], run_fn=run_fn))

    report = score(gold_bio, pred_bio)
    df, csv_path = _write_haiku_results(report, model_name)
    print("\n" + df.to_string(index=False))
    print(f"\nWrote {csv_path}")
    _print_usage(usage, model_name)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Claude Haiku zero-shot clinical NER evaluation"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="evaluate only the first N test examples (smoke test)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Anthropic model id (default claude-haiku-4-5)",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="row label in haiku_comparison.csv (default claude-haiku-4-5)",
    )
    parser.add_argument(
        "--no-preflight", action="store_true",
        help="skip the 1-sentence API check before the loop",
    )
    args = parser.parse_args()
    run_eval(
        limit=args.limit,
        model_id=args.model,
        model_name=args.model_name,
        skip_preflight=args.no_preflight,
    )


if __name__ == "__main__":
    main()
