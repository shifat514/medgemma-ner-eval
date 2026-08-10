"""L5 adjudication: verdict parsing, the pair dump round-trip, and the summary.

The judge is the step that decides whether the ladder's gain above L1 was real,
so the one thing it must never do is reject a pair because it failed to answer.
"""

import json

import pytest

from src.recall_judge import (
    build_messages,
    judge_pairs,
    load_pairs,
    load_verdicts,
    pair_key,
    parse_verdict,
    run_judge,
    summarize,
    write_questions,
)


def _pair(span="diabetes", gold="diabetes insipidus", level="L2",
          rule="contains", note_id=1):
    return {"note_id": note_id, "level": level, "rule": rule, "score": 0.5,
            "span": span, "name": span, "gold_form": gold,
            "gold_sources": ["evidence"], "gold_codes": ["ICD-10-CM|E23.2"]}


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reply,verdict", [
    ("YES", True),
    ("yes", True),
    ("NO", False),
    ("no.", False),
    ("Yes, these refer to the same finding.", True),
    ("No — diabetes insipidus is a different disease.", False),
])
def test_parse_verdict_reads_the_answer(reply, verdict):
    assert parse_verdict(reply) is verdict


def test_the_first_word_wins_when_the_judge_argues_with_itself():
    assert parse_verdict("NO. There is no way these are the same.") is False
    assert parse_verdict("YES, though no clinician would say it that way") is True


@pytest.mark.parametrize("reply", ["", "   ", None, "I am not sure", 42])
def test_an_unreadable_verdict_is_none(reply):
    assert parse_verdict(reply) is None


def test_an_unreadable_verdict_keeps_the_pair():
    """The judge failing to answer is not evidence that the match was wrong.

    Dropping those pairs would understate recall for a reason that has nothing
    to do with the model under test.
    """
    judged = judge_pairs([_pair()], run_fn=lambda messages: "who can say")
    counts = summarize({"L2": judged})["L2"]
    assert (counts["kept"], counts["rejected"], counts["unreadable"]) == (1, 0, 1)


def test_a_judge_exception_does_not_kill_the_pass():
    def explode(messages):
        raise RuntimeError("CUDA OOM")

    judged = judge_pairs([_pair(), _pair(span="sepsis")], run_fn=explode)
    assert len(judged) == 2
    assert all(p["verdict"] is None for p in judged)


def test_rejected_pairs_are_counted_and_rated():
    judged = judge_pairs(
        [_pair(), _pair(span="HTN", gold="hypertension")],
        run_fn=lambda m: "NO" if "MODEL: diabetes" in str(m) else "YES")
    counts = summarize({"L2": judged})["L2"]
    assert (counts["kept"], counts["rejected"]) == (1, 1)
    assert counts["reject_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The question put to the judge
# --------------------------------------------------------------------------

def test_the_question_names_both_phrases():
    text = build_messages(_pair())[1]["content"][0]["text"]
    assert "diabetes" in text
    assert "diabetes insipidus" in text


def test_the_question_spells_out_the_two_shapes_l2_admits():
    """The negation and the different-disease cases are exactly what L2 lets
    through, so the judge is told about them by name."""
    text = build_messages(_pair())[1]["content"][0]["text"]
    assert "no evidence of sepsis" in text
    assert "diabetes vs diabetes insipidus" in text
    assert "HTN vs hypertension" in text


def test_a_finding_with_no_span_falls_back_to_its_name():
    pair = {**_pair(), "span": "", "name": "hypertension"}
    assert "hypertension" in build_messages(pair)[1]["content"][0]["text"]


# --------------------------------------------------------------------------
# Round trip through the run directory
# --------------------------------------------------------------------------

def _dump(tmp_path, level, pairs):
    path = tmp_path / f"new_pairs_{level}.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")


def test_load_pairs_reads_only_the_requested_levels(tmp_path):
    _dump(tmp_path, "L1", [_pair(level="L1")])
    _dump(tmp_path, "L2", [_pair()])
    _dump(tmp_path, "L3", [_pair(level="L3")])
    loaded = load_pairs(str(tmp_path), levels=("L2", "L3", "L4"))
    assert set(loaded) == {"L2", "L3"}


def test_l1_is_never_adjudicated(tmp_path):
    """L1 is exact string equality; there is nothing for a judge to decide."""
    _dump(tmp_path, "L1", [_pair(level="L1")])
    assert load_pairs(str(tmp_path)) == {}


def test_empty_levels_are_omitted(tmp_path):
    (tmp_path / "new_pairs_L2.jsonl").write_text("", encoding="utf-8")
    assert load_pairs(str(tmp_path)) == {}


def test_questions_round_trip_through_a_filled_in_file(tmp_path):
    """The --judge none path: write the questions out, get answers back."""
    path = str(tmp_path / "questions.jsonl")
    write_questions({"L2": [_pair(), _pair(span="HTN", gold="hypertension")]},
                    path)

    answered = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            answered.append({**json.loads(line), "verdict": i == 1})
    (tmp_path / "answered.jsonl").write_text(
        "\n".join(json.dumps(r) for r in answered), encoding="utf-8")

    verdicts = load_verdicts(str(tmp_path / "answered.jsonl"))
    assert verdicts[pair_key(answered[0])] is False
    assert verdicts[pair_key(answered[1])] is True


def test_unanswered_questions_are_left_unanswered(tmp_path):
    """A `verdict` still null means nobody decided, not "rejected"."""
    path = str(tmp_path / "questions.jsonl")
    write_questions({"L2": [_pair()]}, path)
    assert load_verdicts(path) == {}


def test_supplied_verdicts_are_used_instead_of_calling_the_judge():
    pair = _pair()
    judged = judge_pairs([pair], run_fn=lambda m: pytest.fail("judge was called"),
                         resume={pair_key(pair): False})
    assert judged[0]["verdict"] is False


def test_pair_key_separates_pairs_that_differ_only_in_gold_form():
    assert pair_key(_pair(gold="a")) != pair_key(_pair(gold="b"))


# --------------------------------------------------------------------------
# The CLI path
# --------------------------------------------------------------------------

def test_run_judge_none_writes_the_questions_out(tmp_path):
    _dump(tmp_path, "L2", [_pair()])
    counts = run_judge(run_dir=str(tmp_path), judge="none")
    assert counts["L2"]["unreadable"] == 1
    assert (tmp_path / "l5_questions.jsonl").exists()
    assert (tmp_path / "verdicts_L2.jsonl").exists()
    assert json.loads((tmp_path / "l5_summary.json").read_text())["judge"] == "none"


def test_run_judge_says_nothing_to_do_rather_than_crashing(tmp_path):
    assert run_judge(run_dir=str(tmp_path), judge="none") == {}


def test_no_run_is_reported_differently_from_nothing_to_adjudicate(tmp_path, capsys):
    """Confusing the two wastes an afternoon."""
    run_judge(run_dir=str(tmp_path), judge="none")
    assert "Run `python -m src.evaluate_recall` first" in capsys.readouterr().out

    (tmp_path / "new_pairs_L2.jsonl").write_text("", encoding="utf-8")
    run_judge(run_dir=str(tmp_path), judge="none")
    assert "Every match was exact" in capsys.readouterr().out


def test_an_unknown_judge_is_refused(tmp_path):
    _dump(tmp_path, "L2", [_pair()])
    with pytest.raises(ValueError, match="unknown judge"):
        run_judge(run_dir=str(tmp_path), judge="gpt-9")
