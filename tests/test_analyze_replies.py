"""Reply-repetition analysis — CPU-only, synthetic replies. NO REAL NOTE TEXT."""

import json

from src.analyze_replies import _looks_truncated, _max_run, analyze_file


def _write(tmp_path, replies, name="raw.jsonl"):
    p = tmp_path / name
    with open(p, "w") as f:
        for i, r in enumerate(replies):
            f.write(json.dumps({"note_id": f"n{i}-DS-1", "chunk": [0, 400],
                                "shape": "json", "n_kept": 1, "reply": r}) + "\n")
    return str(p)


def test_max_run():
    assert _max_run([1, 1, 1, 2]) == 3
    assert _max_run([1, 2, 1, 2]) == 1
    assert _max_run([]) == 0


def test_truncation_detection():
    assert _looks_truncated('{"entities": [{"text": "a", "type": "Dose"')
    assert not _looks_truncated('{"entities": []}')
    assert not _looks_truncated('```json\n{"entities": []}\n```')
    assert not _looks_truncated("")


def test_repetition_loop_is_flagged(tmp_path):
    item = '{"text": "Drugzol", "type": "Medication"}'
    looping = '{"entities": [' + ", ".join([item] * 8)      # truncated mid-list
    path = _write(tmp_path, [looping])
    per_chunk, loops = analyze_file(path)
    assert len(loops) == 1
    assert per_chunk[0]["items"] == 8
    assert per_chunk[0]["unique_items"] == 1
    assert per_chunk[0]["max_repeat_run"] == 8
    assert per_chunk[0]["truncated"] is True


def test_healthy_reply_is_not_flagged(tmp_path):
    reply = json.dumps({"entities": [
        {"text": "Drugzol", "type": "Medication"},
        {"text": "250 mg", "type": "Dose"},
        {"text": "PO", "type": "Mode"},
    ]})
    path = _write(tmp_path, [reply])
    per_chunk, loops = analyze_file(path)
    assert loops == []
    assert per_chunk[0]["dup_items"] == 0
    assert per_chunk[0]["truncated"] is False


def test_low_uniqueness_flags_even_without_a_consecutive_run(tmp_path):
    a = '{"text": "Drugzol", "type": "Medication"}'
    b = '{"text": "250 mg", "type": "Dose"}'
    reply = '{"entities": [' + ", ".join([a, b] * 4) + "]}"
    path = _write(tmp_path, [reply])
    _, loops = analyze_file(path)
    assert len(loops) == 1   # 8 items, 2 unique


def test_empty_file(tmp_path):
    path = _write(tmp_path, [])
    per_chunk, loops = analyze_file(path)
    assert per_chunk == [] and loops == []


def test_analysis_never_returns_reply_text(tmp_path):
    """The whole point: structure out, content stays in."""
    reply = json.dumps({"entities": [{"text": "SECRETDRUG", "type": "Medication"}]})
    path = _write(tmp_path, [reply])
    per_chunk, _ = analyze_file(path)
    assert "SECRETDRUG" not in json.dumps(per_chunk)


def test_truncated_array_ending_in_a_closing_brace_is_detected():
    """The bug this check exists for: a cut-off list ends with '}'.

    Checking the last character says "clean close" and reports 0 truncations,
    which is exactly wrong for a run that hit max_new_tokens.
    """
    item = '{"text": "Drugzol", "type": "Medication"}'
    assert _looks_truncated('{"entities": [' + ", ".join([item] * 8))
    assert _looks_truncated('{"entities": [' + item)


def test_complete_variants_are_not_flagged():
    assert not _looks_truncated('{"entities": [{"text": "a", "type": "Dose"}]}')
    assert not _looks_truncated('[{"text": "a", "type": "Dose"}]')
    # JSONL: several complete objects in sequence
    assert not _looks_truncated('{"text": "a", "type": "Dose"}\n'
                                '{"text": "b", "type": "Mode"}')
    assert not _looks_truncated('```json\n{"entities": []}\n```')
