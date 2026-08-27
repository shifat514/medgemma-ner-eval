"""Reply parsing and exact-code-match scoring.

The scoring here is deliberately simple — set intersection on normalized codes —
so most of these tests are about the two places simple scoring can still be
wrong: what the parser accepts from a messy reply, and what it must refuse to
throw away.

THE LOAD-BEARING ONE IS ``test_malformed_code_is_kept_as_a_false_positive``. A
parser that drops a returned "99213" because it is not shaped like ICD-10 is
deleting a false positive, which raises precision for free. That is the one bug
in this file that would produce a better-looking number rather than a crash.
"""

from src.evaluate_billing import aggregate, make_oracle_reply, score_note
from src.prompt_billing import build_messages, parse_codes, prompt_fingerprint

# --- reply parsing ----------------------------------------------------------


def test_parses_the_requested_shape():
    reply = '{"codes": [{"code": "B08.5", "description": "Enteroviral pharyngitis"}]}'
    assert parse_codes(reply) == [
        {"code": "B08.5", "description": "Enteroviral pharyngitis",
         "well_formed": True},
    ]


def test_tolerates_markdown_fences_and_prose():
    reply = (
        "Here are the codes:\n"
        "```json\n"
        '{"codes": [{"code": "J11.1", "description": "Influenza"}]}\n'
        "```\n"
        "Let me know if you need more."
    )
    assert [c["code"] for c in parse_codes(reply)] == ["J11.1"]


def test_accepts_alternate_container_and_field_names():
    reply = '{"diagnoses": [{"icd10": "R63.6", "diagnosis": "Underweight"}]}'
    assert [c["code"] for c in parse_codes(reply)] == ["R63.6"]


def test_accepts_a_bare_list_of_strings():
    reply = '["B97.89 Other viral agents", "- R63.6: Underweight"]'
    got = parse_codes(reply)
    assert [c["code"] for c in got] == ["B97.89", "R63.6"]
    assert got[1]["description"] == "Underweight"


def test_salvages_a_code_and_description_crammed_into_one_field():
    reply = '{"codes": [{"code": "S52.501A Unsp fracture of right radius"}]}'
    got = parse_codes(reply)
    assert got[0]["code"] == "S52.501A"
    assert "fracture" in got[0]["description"]


def test_deduplicates_within_one_reply():
    """A model that says J30.2 twice has made one claim, not two."""
    reply = '{"codes": [{"code": "J30.2"}, {"code": "J30.2"}]}'
    assert len(parse_codes(reply)) == 1


def test_unparseable_reply_yields_no_codes_rather_than_raising():
    for reply in ("", "   ", "I cannot code this note.", "{broken json"):
        assert parse_codes(reply) == []


def test_malformed_code_is_kept_as_a_false_positive():
    """CPT codes and invented strings are counted against the model, not dropped.

    Dropping them would quietly delete false positives and inflate precision.
    """
    reply = '{"codes": [{"code": "99213"}, {"code": "B08.5"}]}'
    got = parse_codes(reply)
    assert [c["code"] for c in got] == ["99213", "B08.5"]
    assert [c["well_formed"] for c in got] == [False, True]


# --- scoring ----------------------------------------------------------------


def _pred(*codes):
    return [{"code": c, "description": "", "well_formed": True} for c in codes]


def test_perfect_prediction():
    s = score_note(["B08.5", "D18.00"], _pred("B08.5", "D18.00"))
    assert (s["n_tp"], s["n_fp"], s["n_fn"]) == (2, 0, 0)


def test_partial_prediction_splits_into_tp_fp_fn():
    s = score_note(["B08.5", "D18.00"], _pred("B08.5", "J06.9"))
    assert s["tp"] == ["B08.5"]
    assert s["fp"] == ["J06.9"]
    assert s["fn"] == ["D18.00"]


def test_match_is_case_and_whitespace_insensitive():
    s = score_note(["S52.501A"], _pred(" s52.501a "))
    assert s["n_tp"] == 1


def test_a_less_specific_code_is_not_a_match():
    """B08 is a different claim from B08.5 and must not score as a hit."""
    s = score_note(["B08.5"], _pred("B08"))
    assert (s["n_tp"], s["n_fp"], s["n_fn"]) == (0, 1, 1)


def test_duplicate_gold_is_collapsed_before_scoring():
    """Note 96176 lists Z68.51 twice; gold for it is 3 codes, not 4."""
    s = score_note(["B97.89", "R63.6", "Z68.51", "Z68.51"], _pred("Z68.51"))
    assert s["n_gold"] == 3
    assert s["n_tp"] == 1
    assert s["n_fn"] == 2


def test_repeating_a_code_cannot_buy_recall():
    s = score_note(["J30.2", "L20.9"], _pred("J30.2", "J30.2", "J30.2"))
    assert s["n_pred"] == 1
    assert s["n_tp"] == 1
    assert s["n_fn"] == 1


def test_empty_prediction_is_zero_recall_not_a_crash():
    s = score_note(["B08.5"], [])
    assert (s["n_tp"], s["n_fp"], s["n_fn"]) == (0, 0, 1)


def test_malformed_predictions_are_counted():
    s = score_note(["B08.5"], parse_codes('{"codes": [{"code": "99213"}]}'))
    assert s["n_malformed"] == 1
    assert s["n_fp"] == 1


# --- aggregate --------------------------------------------------------------


def test_aggregate_is_micro_not_macro():
    """Pool the counts, then divide — a 6-code note must outweigh a 2-code note.

    Macro-averaging these two notes gives (1.0 + 0.0)/2 = 0.5 recall. Micro gives
    2/8 = 0.25, which is the answer to the question actually asked: what share of
    billed codes came back.
    """
    rows = [
        score_note(["A00.0", "A00.1"], _pred("A00.0", "A00.1")),
        score_note(["B00.0", "B00.1", "B00.2", "B00.3", "B00.4", "B00.5"], []),
    ]
    agg = aggregate(rows)
    assert agg["n_gold"] == 8
    assert agg["n_tp"] == 2
    assert abs(agg["recall"] - 0.25) < 1e-9


def test_aggregate_f1_is_zero_when_nothing_matches():
    agg = aggregate([score_note(["A00.0"], _pred("Z99.9"))])
    assert agg["precision"] == 0.0
    assert agg["recall"] == 0.0
    assert agg["f1"] == 0.0


# --- oracle -----------------------------------------------------------------


def test_oracle_reply_round_trips_through_the_real_parser():
    """The oracle must exercise the shipped parser, or it checks nothing."""
    record = {"gold_codes": ["B08.5", "D18.00"]}
    s = score_note(record["gold_codes"], parse_codes(make_oracle_reply(record)))
    assert s["n_tp"] == 2
    assert s["n_fp"] == 0
    assert s["n_fn"] == 0


# --- prompt -----------------------------------------------------------------


def test_prompt_fingerprint_is_stable_and_short():
    assert prompt_fingerprint() == prompt_fingerprint()
    assert len(prompt_fingerprint()) == 8


def test_messages_carry_the_note_and_ask_for_icd_only():
    msgs = build_messages("Assessment redacted. Patient has a cough.")
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"][0]["text"]
    assert "Patient has a cough." in user
    assert "ICD-10-CM" in user
    assert "99213" in user          # the negative example naming CPT explicitly
