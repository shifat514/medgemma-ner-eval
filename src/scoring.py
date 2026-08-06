"""seqeval scoring + comparison.csv writer.

Uses the EXACT same scorer and output columns as clinical-ner-eval/src/evaluate.py
(seqeval entity-level ``classification_report``; columns model, entity, precision,
recall, f1, support), so the two comparison.csv files concatenate into one table.
Imports only seqeval/pandas — no torch — so it is importable in CPU-only tests.
"""

import json
import os

import numpy as np
import pandas as pd
from seqeval.metrics import classification_report

from . import config as _config


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def score(gold_bio, pred_bio):
    """Entity-level seqeval report as a dict (same call as the sibling repo)."""
    return classification_report(
        gold_bio, pred_bio, output_dict=True, zero_division=0
    )


def report_to_rows(report, model_name):
    """Flatten a seqeval report dict into comparison.csv rows."""
    rows = []
    for entity, scores in report.items():
        if not isinstance(scores, dict):
            continue
        rows.append({
            "model": model_name,
            "entity": entity,
            "precision": round(scores["precision"], 4),
            "recall": round(scores["recall"], 4),
            "f1": round(scores["f1-score"], 4),
            "support": scores["support"],
        })
    return rows


def write_results(report, model_name, results_dir=None,
                  csv_name="comparison.csv", json_name="full_report.json"):
    """Write the report JSON + metrics CSV; return (DataFrame, csv_path).

    `csv_name`/`json_name` are parameterized so the MIMIC evaluation can write
    mimic_ner_50.csv / mimic_ner_100.csv through this same writer. Defaults
    preserve the original NCBI/BC5CDR behavior.

    Output is aggregate metrics only — no example-level content — so it is safe
    to commit.
    """
    results_dir = results_dir or _config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, json_name), "w") as f:
        json.dump({model_name: report}, f, indent=2, cls=_NpEncoder)

    df = pd.DataFrame(report_to_rows(report, model_name))
    csv_path = os.path.join(results_dir, csv_name)
    df.to_csv(csv_path, index=False)
    return df, csv_path
