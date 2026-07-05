"""seqeval integration tests — CPU-only, no model download, no GPU.

Covers the scorer directly and the full parse -> align -> score chain via
predict_example with an injected fake MedGemma reply.
"""

import math

from src.evaluate import predict_example
from src.scoring import report_to_rows, score


def _approx(a, b, tol=1e-3):
    return math.isclose(a, b, abs_tol=tol)


def test_perfect_prediction():
    gold = [["B-Disease", "I-Disease", "O", "B-Chemical"]]
    pred = [["B-Disease", "I-Disease", "O", "B-Chemical"]]
    rep = score(gold, pred)
    assert _approx(rep["Disease"]["precision"], 1.0)
    assert _approx(rep["Disease"]["recall"], 1.0)
    assert _approx(rep["Disease"]["f1-score"], 1.0)
    assert rep["Disease"]["support"] == 1
    assert rep["Chemical"]["support"] == 1
    assert _approx(rep["micro avg"]["f1-score"], 1.0)


def test_one_missed_chemical():
    # Disease correct; Chemical missed entirely.
    gold = [["B-Disease", "I-Disease", "O", "B-Chemical"]]
    pred = [["B-Disease", "I-Disease", "O", "O"]]
    rep = score(gold, pred)
    assert _approx(rep["Disease"]["f1-score"], 1.0)
    assert rep["Disease"]["support"] == 1
    assert _approx(rep["Chemical"]["precision"], 0.0)
    assert _approx(rep["Chemical"]["recall"], 0.0)
    assert rep["Chemical"]["support"] == 1
    # micro: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=0.667
    assert _approx(rep["micro avg"]["precision"], 1.0)
    assert _approx(rep["micro avg"]["recall"], 0.5)
    assert _approx(rep["micro avg"]["f1-score"], 2 / 3)
    assert rep["micro avg"]["support"] == 2


def test_boundary_error_counts_as_wrong():
    # Gold is a 2-token disease; pred only tags the first token -> span mismatch.
    gold = [["B-Disease", "I-Disease"]]
    pred = [["B-Disease", "O"]]
    rep = score(gold, pred)
    assert _approx(rep["Disease"]["f1-score"], 0.0)


def test_report_to_rows_columns_match_sibling():
    gold = [["B-Disease", "I-Disease", "O", "B-Chemical"]]
    pred = [["B-Disease", "I-Disease", "O", "B-Chemical"]]
    rows = report_to_rows(score(gold, pred), "medgemma-4b-it")
    assert rows, "expected at least one row"
    expected_cols = {"model", "entity", "precision", "recall", "f1", "support"}
    for row in rows:
        assert set(row.keys()) == expected_cols
        assert row["model"] == "medgemma-4b-it"
    entities = {r["entity"] for r in rows}
    assert {"Disease", "Chemical", "micro avg", "macro avg", "weighted avg"} <= entities


def test_end_to_end_parse_align_score():
    """Fake reply -> predict_example -> perfect BIO -> perfect seqeval scores."""
    tokens = ["The", "patient", "has", "lung", "cancer", "treated", "with", "aspirin", "."]
    gold_bio = ["O", "O", "O", "B-Disease", "I-Disease", "O", "O", "B-Chemical", "O"]

    fake_reply = (
        '{"entities": [{"text": "lung cancer", "type": "Disease"}, '
        '{"text": "aspirin", "type": "Chemical"}]}'
    )
    pred_bio = predict_example(None, tokens, run_fn=lambda pipe, sentence: fake_reply)
    assert pred_bio == gold_bio

    rep = score([gold_bio], [pred_bio])
    assert _approx(rep["micro avg"]["f1-score"], 1.0)
    assert rep["micro avg"]["support"] == 2


def test_end_to_end_bad_reply_degrades_to_all_O():
    tokens = ["Patient", "has", "flu"]
    pred_bio = predict_example(None, tokens, run_fn=lambda pipe, sentence: "garbage")
    assert pred_bio == ["O", "O", "O"]


def test_predict_example_swallows_inference_error():
    tokens = ["Patient", "has", "flu"]

    def boom(pipe, sentence):
        raise RuntimeError("simulated inference failure")

    # Must not raise; degrades to all-O.
    assert predict_example(None, tokens, run_fn=boom) == ["O", "O", "O"]
