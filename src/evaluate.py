"""MedGemma zero-shot clinical-NER evaluation harness (Phase 1).

Per test example:
    tokens -> sentence (" ".join)  -> prompt MedGemma -> parse JSON entities
    -> align spans onto the SAME tokens (predicted BIO) -> collect gold + pred
    -> seqeval entity-level report.

Writes results/comparison.csv with the same columns and scorer as
clinical-ner-eval, and model name "medgemma-4b-it", so combining the two = just
concatenating the CSVs.

Phase 2 (fine-tuning) will add a trained-model path; the dataset/scoring/align
plumbing here is reused unchanged.
"""

import argparse

from . import config as _config
from .align import align_entities_to_bio
from .prompt import parse_entities
from .scoring import score, write_results


def predict_example(pipe, tokens, run_fn=None):
    """Return predicted BIO tags for one tokenized example.

    `run_fn(pipe, sentence) -> reply_str` is injectable for testing (defaults to
    the real MedGemma call). Any inference/parse error degrades to no entities.
    """
    if run_fn is None:
        from .model import run_medgemma
        run_fn = run_medgemma

    sentence = " ".join(tokens)
    try:
        entities = parse_entities(run_fn(pipe, sentence))
    except Exception as e:  # noqa: BLE001 - never let one bad example kill the run
        print(f"[warn] inference/parse failed on one example: {e}")
        entities = []
    return align_entities_to_bio(tokens, entities)


def run_eval(limit=None, model_id=None, model_name=None):
    model_id = model_id or _config.MODEL_ID
    model_name = model_name or _config.MODEL_NAME

    from .datasets import build_test_set
    ds, _ = build_test_set()
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"Loaded {len(ds)} test examples (model={model_id})")

    from .model import load_medgemma
    pipe = load_medgemma(model_id)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(ds, total=len(ds), desc="MedGemma NER")
    except ImportError:
        iterator = ds

    gold_bio, pred_bio = [], []
    for ex in iterator:
        gold_bio.append(ex["bio"])
        pred_bio.append(predict_example(pipe, ex["tokens"]))

    report = score(gold_bio, pred_bio)
    df, csv_path = write_results(report, model_name)
    print("\n" + df.to_string(index=False))
    print(f"\nWrote {csv_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma zero-shot clinical NER evaluation"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="evaluate only the first N test examples (smoke test)",
    )
    parser.add_argument(
        "--model", default=None,
        help="HF model id (default google/medgemma-4b-it)",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="row label in comparison.csv (default medgemma-4b-it)",
    )
    args = parser.parse_args()
    run_eval(limit=args.limit, model_id=args.model, model_name=args.model_name)


if __name__ == "__main__":
    main()
