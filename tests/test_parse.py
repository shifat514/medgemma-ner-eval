"""Parsing tests — CPU-only, no model download, no GPU."""

from src.prompt import build_messages, parse_entities


def test_plain_json():
    reply = '{"entities": [{"text": "lung cancer", "type": "Disease"}]}'
    assert parse_entities(reply) == [("lung cancer", "Disease")]


def test_multiple_entities_both_types():
    reply = (
        '{"entities": ['
        '{"text": "asthma", "type": "Disease"}, '
        '{"text": "aspirin", "type": "Chemical"}]}'
    )
    assert parse_entities(reply) == [("asthma", "Disease"), ("aspirin", "Chemical")]


def test_markdown_fenced_json():
    reply = 'Here you go:\n```json\n{"entities": [{"text": "flu", "type": "Disease"}]}\n```'
    assert parse_entities(reply) == [("flu", "Disease")]


def test_prose_surrounding_object():
    reply = 'Sure! {"entities": [{"text": "diabetes", "type": "Disease"}]} Hope that helps.'
    assert parse_entities(reply) == [("diabetes", "Disease")]


def test_empty_entities():
    assert parse_entities('{"entities": []}') == []


def test_case_insensitive_type_normalization():
    reply = (
        '{"entities": [{"text": "x", "type": "disease"}, '
        '{"text": "y", "type": "CHEMICAL"}]}'
    )
    assert parse_entities(reply) == [("x", "Disease"), ("y", "Chemical")]


def test_invalid_type_dropped():
    reply = (
        '{"entities": [{"text": "John", "type": "Person"}, '
        '{"text": "flu", "type": "Disease"}]}'
    )
    assert parse_entities(reply) == [("flu", "Disease")]


def test_missing_or_blank_fields_dropped():
    reply = (
        '{"entities": [{"type": "Disease"}, {"text": "  ", "type": "Disease"}, '
        '{"text": "gout", "type": "Disease"}]}'
    )
    assert parse_entities(reply) == [("gout", "Disease")]


def test_malformed_json_returns_empty():
    assert parse_entities("not json at all") == []
    assert parse_entities('{"entities": [broken') == []


def test_non_string_and_empty_inputs():
    assert parse_entities(None) == []
    assert parse_entities("") == []
    assert parse_entities("   ") == []


def test_entities_not_a_list():
    assert parse_entities('{"entities": "flu"}') == []


def test_braces_inside_string_value():
    reply = '{"entities": [{"text": "type {A} fracture", "type": "Disease"}]}'
    assert parse_entities(reply) == [("type {A} fracture", "Disease")]


def test_build_messages_is_text_only():
    msgs = build_messages("Patient has flu.")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    for m in msgs:
        assert isinstance(m["content"], list)
        assert m["content"][0]["type"] == "text"
        assert "image" not in {c["type"] for c in m["content"]}
    assert "Patient has flu." in msgs[1]["content"][0]["text"]
