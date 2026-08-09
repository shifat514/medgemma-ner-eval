"""Reply parsing for the MDACE prompt.

The medication evaluation learned this the expensive way: its first parser
accepted exactly one reply shape, silently returned [] for 17 of 22 realistic
shapes, and scored F1=0.136 against an 0.870 ceiling. These tests enumerate the
shapes a 4B instruct model actually produces so that failure cannot repeat.

The key difference here is that TYPE LABELS ARE NOT SCORED — only term strings
are — so this parser keeps anything carrying text, including bare strings and
unrecognized type names. In the medication parser those had to be dropped,
because a guessed type was both a false positive and a still-missed false
negative. Here there is no such penalty.
"""

import json

import pytest

from src.prompt_mdace import build_messages, build_prompt, parse_terms, parse_terms_diag


def _texts(reply):
    return [t for t, _ in parse_terms(reply)]


# --------------------------------------------------------------------------
# Shapes that must all work
# --------------------------------------------------------------------------

def test_canonical_shape():
    reply = json.dumps({"terms": [
        {"text": "depression", "type": "Condition"},
        {"text": "HTN", "type": "Condition"},
    ]})
    assert _texts(reply) == ["depression", "HTN"]


def test_entities_key_instead_of_terms():
    reply = json.dumps({"entities": [{"text": "asthma", "type": "Condition"}]})
    assert _texts(reply) == ["asthma"]


def test_bare_top_level_array():
    reply = json.dumps([{"text": "sepsis", "type": "Condition"}])
    assert _texts(reply) == ["sepsis"]


def test_bare_list_of_strings():
    """A plain list of terms is a complete answer when types are not scored."""
    reply = json.dumps(["depression", "HTN", "asthma"])
    assert _texts(reply) == ["depression", "HTN", "asthma"]


def test_unrecognized_type_is_kept():
    """`type` carries no scoring weight, so an odd label must not lose the term."""
    reply = json.dumps({"terms": [{"text": "chest pain", "type": "Symptom"},
                                  {"text": "CABG", "type": "surgical history"}]})
    assert _texts(reply) == ["chest pain", "CABG"]


def test_missing_type_is_kept():
    reply = json.dumps({"terms": [{"text": "asystole"}]})
    assert _texts(reply) == ["asystole"]


def test_alternate_text_keys():
    for key in ("term", "span", "entity", "value", "mention", "phrase", "name"):
        reply = json.dumps({"terms": [{key: "anemia", "type": "Condition"}]})
        assert _texts(reply) == ["anemia"], key


def test_markdown_fenced_json():
    reply = "```json\n" + json.dumps({"terms": ["COPD"]}) + "\n```"
    assert _texts(reply) == ["COPD"]


def test_json_with_prose_around_it():
    reply = ('Here are the findings I identified:\n'
             '{"terms": [{"text": "atrial fibrillation", "type": "Condition"}]}\n'
             'Let me know if you need more.')
    assert _texts(reply) == ["atrial fibrillation"]


def test_jsonl_one_object_per_line():
    reply = ('{"text": "hypertension", "type": "Condition"}\n'
             '{"text": "diabetes", "type": "Condition"}')
    assert _texts(reply) == ["hypertension", "diabetes"]


def test_nested_container_keys():
    for key in ("findings", "conditions", "diagnoses", "results", "items"):
        reply = json.dumps({key: [{"text": "pneumonia"}]})
        assert _texts(reply) == ["pneumonia"], key


def test_single_bare_object():
    reply = json.dumps({"text": "cirrhosis", "type": "Condition"})
    assert _texts(reply) == ["cirrhosis"]


# --------------------------------------------------------------------------
# Failure modes must be reported, never silent
# --------------------------------------------------------------------------

def test_empty_reply_is_flagged():
    terms, diag = parse_terms_diag("")
    assert terms == []
    assert diag["shape"] == "empty-reply"


def test_non_json_prose_is_flagged():
    terms, diag = parse_terms_diag("I could not find any conditions in this text.")
    assert terms == []
    assert diag["shape"] in ("no-json", "no-item-list")


def test_explicit_empty_list_is_distinguished_from_a_parse_failure():
    """"Nothing here" and "I broke" must never look the same in diagnostics."""
    terms, diag = parse_terms_diag(json.dumps({"terms": []}))
    assert terms == []
    assert diag["empty_list"] is True
    assert diag["shape"] == "json"


def test_truncated_json_does_not_raise():
    reply = '{"terms": [{"text": "sepsis", "type": "Condi'
    terms, diag = parse_terms_diag(reply)
    assert isinstance(terms, list)
    assert isinstance(diag["shape"], str)


def test_diag_counts_items_and_keeps():
    reply = json.dumps({"terms": [
        {"text": "sepsis", "type": "Condition"},
        {"type": "Condition"},                      # no text at all
        {"text": "  ", "type": "Condition"},        # whitespace only
    ]})
    terms, diag = parse_terms_diag(reply)
    assert len(terms) == 1
    assert diag["n_items"] == 3
    assert diag["n_kept"] == 1
    assert diag["n_no_text"] == 2


def test_types_are_recorded_for_diagnostics():
    reply = json.dumps({"terms": [{"text": "a", "type": "Condition"},
                                  {"text": "b", "type": "Procedure"},
                                  {"text": "c", "type": "Condition"}]})
    _terms, diag = parse_terms_diag(reply)
    assert diag["types"] == {"Condition": 2, "Procedure": 1}


@pytest.mark.parametrize("reply", [None, 123, [], {}])
def test_non_string_replies_do_not_raise(reply):
    terms, diag = parse_terms_diag(reply)
    assert terms == []
    assert diag["shape"] == "empty-reply"


# --------------------------------------------------------------------------
# Prompt content
# --------------------------------------------------------------------------

def test_prompt_asks_for_conditions_procedures_and_injuries():
    """A conditions-only prompt caps recall at 0.84; these three reach 94.5%."""
    prompt = build_prompt("some note text")
    for category in ("Condition", "Procedure", "Injury"):
        assert category in prompt


def test_prompt_excludes_medications():
    """Asking for medication/status items returned 138 of 421 extractions on a
    smoke run (33%) for a category worth 5.5% of gold, and the volume truncated
    12 of 15 replies. The category is out, and saying so explicitly matters more
    than omitting it."""
    prompt = build_prompt("x")
    assert "Do NOT extract medications" in prompt
    assert '"Status"' not in prompt


def test_prompt_example_output_omits_the_medication_in_its_input():
    """The negative example is the load-bearing part: aspirin and the tobacco
    line appear in the example INPUT and must not appear in the example OUTPUT.

    Asserted against the example block itself rather than "anything after the
    word Example output:", because the prompt goes on to point the omission out
    in prose — and that prose naturally mentions aspirin.
    """
    from src.prompt_mdace import _EXAMPLE_INPUT, _EXAMPLE_OUTPUT

    assert "aspirin 81mg daily" in _EXAMPLE_INPUT
    assert "Tobacco history" in _EXAMPLE_INPUT
    assert "aspirin" not in _EXAMPLE_OUTPUT.lower()
    assert "tobacco" not in _EXAMPLE_OUTPUT.lower()
    # ...and the prompt says so out loud, so the omission cannot read as an
    # oversight to the model.
    assert "neither appears in the example output" in build_prompt("x")


def test_prompt_demands_verbatim_copying():
    prompt = build_prompt("x")
    assert "VERBATIM" in prompt
    assert "HTN" in prompt          # the do-not-expand-abbreviations example


def test_prompt_tells_the_model_to_skip_redaction_markers():
    """MDACE notes carry a median of 16 `[**...**]` placeholders each."""
    prompt = build_prompt("x")
    assert "[**" in prompt


def test_prompt_embeds_the_chunk_text():
    assert "PATIENT CHUNK MARKER" in build_prompt("PATIENT CHUNK MARKER")


def test_messages_structure():
    msgs = build_messages("note")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"][0]["type"] == "text"
    assert "note" in msgs[1]["content"][0]["text"]


def test_prompt_example_is_synthetic():
    """No real note content may live in the prompt module."""
    prompt = build_prompt("x")
    assert "Known lastname" in prompt      # the placeholder form, not real data
    assert "___" not in prompt             # MIMIC-IV style, wrong corpus
