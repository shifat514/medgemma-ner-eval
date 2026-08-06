"""Tolerant medication-reply parsing — CPU-only, synthetic replies only.

NO REAL NOTE TEXT. These fixtures are the reply shapes that caused a smoke run to
score micro F1=0.136 against an 0.870 oracle ceiling: the original parser
silently returned [] for 17 of 22 realistic shapes, with Duration and Reason at
exactly 0.0000 because their type strings were dropped without a log.
"""

import pytest

from src.prompt_mimic import (
    build_messages,
    normalize_type,
    parse_entities,
    parse_entities_diag,
)


# --- type normalization: the exact-zero bug -------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Medication", "Medication"), ("medication", "Medication"),
    ("MEDICATION", "Medication"), ("  Dose  ", "Dose"),
    ("Drug", "Medication"), ("drugs", "Medication"), ("med", "Medication"),
    ("Dosage", "Dose"), ("amount", "Dose"), ("strength", "Dose"),
    ("Route", "Mode"), ("route of administration", "Mode"),
    ("mode of administration", "Mode"), ("administration", "Mode"),
    ("Indication", "Reason"), ("reason for use", "Reason"),
    ("Reason for administration", "Reason"), ("condition", "Reason"),
    ("Duration of therapy", "Duration"), ("how long", "Duration"),
    ("freq", "Frequency"), ("schedule", "Frequency"),
    (["Medication"], "Medication"),
])
def test_type_synonyms_resolve(raw, expected):
    assert normalize_type(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "Person", "Allergy", "xyzzy"])
def test_unrelated_types_still_rejected(raw):
    assert normalize_type(raw) is None


def test_duration_and_reason_survive_verbose_names():
    """The two types that scored exactly 0.0000 in the smoke run."""
    reply = ('{"entities": [{"text": "for 7 days", "type": "Duration of therapy"},'
             ' {"text": "CHF", "type": "Reason for use"}]}')
    assert parse_entities(reply) == [("for 7 days", "Duration"), ("CHF", "Reason")]


# --- reply shapes ---------------------------------------------------------

def test_canonical_shape():
    reply = '{"entities": [{"text": "Lasix", "type": "Medication"}]}'
    assert parse_entities(reply) == [("Lasix", "Medication")]


def test_bare_top_level_array():
    reply = '[{"text": "Lasix", "type": "Medication"}, {"text": "IV", "type": "Mode"}]'
    assert parse_entities(reply) == [("Lasix", "Medication"), ("IV", "Mode")]


def test_jsonl_one_object_per_line():
    reply = ('{"text": "Lasix", "type": "Medication"}\n'
             '{"text": "40mg", "type": "Dose"}')
    assert parse_entities(reply) == [("Lasix", "Medication"), ("40mg", "Dose")]


def test_grouped_field_keyed_object():
    reply = ('{"entities": [{"medication": "Lasix", "dose": "40mg", '
             '"mode": "IV", "frequency": "daily"}]}')
    got = dict((t, ty) for t, ty in parse_entities(reply))
    assert got == {"Lasix": "Medication", "40mg": "Dose", "IV": "Mode",
                   "daily": "Frequency"}


def test_grouped_under_medications_key_with_name_and_route():
    reply = '{"medications": [{"name": "Lasix", "dose": "40mg", "route": "IV"}]}'
    got = dict((t, ty) for t, ty in parse_entities(reply))
    assert got == {"Lasix": "Medication", "40mg": "Dose", "IV": "Mode"}


def test_nested_attributes_are_recovered():
    reply = ('{"entities": [{"text": "Lasix", "type": "Medication", '
             '"attributes": {"dose": "40mg", "duration": "for 7 days"}}]}')
    got = parse_entities(reply)
    assert ("Lasix", "Medication") in got


@pytest.mark.parametrize("reply", [
    '{"entities": [{"span": "Lasix", "label": "Medication"}]}',
    '{"entities": [{"entity": "Lasix", "category": "Medication"}]}',
    '{"entities": [{"value": "Lasix", "entity_type": "Medication"}]}',
    '{"entities": [{"mention": "Lasix", "tag": "Medication"}]}',
])
def test_alternative_key_names(reply):
    assert parse_entities(reply) == [("Lasix", "Medication")]


def test_markdown_fence_and_prose():
    reply = ('Sure!\n```json\n{"entities": [{"text": "Lasix", '
             '"type": "Medication"}]}\n```\nLet me know.')
    assert parse_entities(reply) == [("Lasix", "Medication")]


def test_ncbi_bc5cdr_parser_is_untouched():
    """The shared parser must not have gained MIMIC tolerance."""
    from src.prompt import parse_entities as shared
    assert shared('[{"text": "flu", "type": "Disease"}]') == []
    assert shared('{"entities": [{"text": "flu", "type": "Disease"}]}') == \
        [("flu", "Disease")]


# --- diagnostics ----------------------------------------------------------

def test_rejected_types_are_reported_not_silent():
    reply = ('{"entities": [{"text": "x", "type": "Allergy"}, '
             '{"text": "y", "type": "Allergy"}, '
             '{"text": "Lasix", "type": "Medication"}]}')
    ents, diag = parse_entities_diag(reply)
    assert ents == [("Lasix", "Medication")]
    assert diag["rejected_types"] == {"Allergy": 2}
    assert diag["n_items"] == 3 and diag["n_kept"] == 1


def test_aliased_types_are_reported():
    reply = '{"entities": [{"text": "IV", "type": "Route"}]}'
    ents, diag = parse_entities_diag(reply)
    assert ents == [("IV", "Mode")]
    assert diag["aliased_types"] == {"Route": 1}


def test_no_json_is_flagged():
    _, diag = parse_entities_diag("I cannot help with that.")
    assert diag["shape"] in ("no-json", "no-entity-list")


def test_empty_reply_flagged():
    _, diag = parse_entities_diag("")
    assert diag["shape"] == "empty-reply"


def test_explicit_empty_list_is_distinguished_from_failure():
    ents, diag = parse_entities_diag('{"entities": []}')
    assert ents == []
    assert diag["empty_list"] is True
    assert diag["shape"] == "json"


def test_malformed_json_does_not_raise():
    assert parse_entities('{"entities": [{"text": "a", ') == []
    assert parse_entities(None) == []


# --- prompt ---------------------------------------------------------------

def test_prompt_names_all_six_types_and_forbids_synonyms():
    text = build_messages("chunk")[1]["content"][0]["text"]
    for t in ("Medication", "Dose", "Mode", "Frequency", "Duration", "Reason"):
        assert f'"{t}"' in text
    assert "Drug" in text and "Dosage" in text and "Route" in text  # forbidden list
    assert "flat" in text.lower()


def test_prompt_contains_a_worked_example_with_synthetic_drugs_only():
    text = build_messages("chunk")[1]["content"][0]["text"]
    assert "Drugzol" in text and "Example output" in text
    assert '"type": "Duration"' in text and '"type": "Reason"' in text


def test_prompt_example_output_round_trips_through_the_parser():
    """The few-shot example must itself parse — otherwise we teach a bad shape."""
    text = build_messages("chunk")[1]["content"][0]["text"]
    start = text.index('{"entities"')
    end = text.index("\n", start)
    ents = parse_entities(text[start:end])
    assert len(ents) == 15
    assert {t for _, t in ents} == {"Medication", "Dose", "Mode", "Frequency",
                                    "Duration", "Reason"}


def test_container_object_is_not_double_counted_as_a_bare_item():
    """{"medications": [...]} must not also be parsed as a grouped item.

    'medications' is itself a grouped key, so the container was being processed
    twice: kept exceeded emitted, and phantom no-text drops appeared.
    """
    reply = '{"medications": [{"name": "Lasix", "dose": "40mg", "route": "IV"}]}'
    ents, diag = parse_entities_diag(reply)
    assert len(ents) == 3
    assert diag["n_items"] == 1
    assert diag["n_kept"] == 3
    assert diag["n_no_text"] == 0


def test_entities_container_is_not_double_counted():
    reply = '{"entities": [{"text": "Lasix", "type": "Medication"}]}'
    ents, diag = parse_entities_diag(reply)
    assert ents == [("Lasix", "Medication")]
    assert diag["n_items"] == 1 and diag["n_no_text"] == 0
