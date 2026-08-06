"""MIMIC label-parsing and gold-BIO tests — CPU-only, synthetic data only.

NO REAL NOTE TEXT AND NO REAL LABEL FILES IN THIS FILE. Filenames are synthetic
but follow the real naming convention; note text is invented. Nothing here reads
the credentialed dataset, so the suite runs anywhere.
"""

import pytest

from src.datasets.mimic_meds import (
    build_gold,
    extract_note_id,
    read_label_rows,
    resolve_label_spans,
    sample_note_ids,
)
from src.mimic_config import TYPE_PRIORITY
from src.prompt_mimic import build_messages, parse_entities


# --- filename -> note_id ---------------------------------------------------

def test_extract_note_id_real_filename_shape():
    fn = ("Labels_PromptName-direct-group-yaml_NoteID-10026165-DS-14_"
          "HadmID-20319648_NoteType-DS_NoteSeq-14.csv")
    assert extract_note_id(fn) == "10026165-DS-14"


def test_extract_note_id_stops_at_hadmid_not_greedy_past_underscore():
    fn = "Labels_x_NoteID-19933834-DS-2_HadmID-22352379_NoteType-DS_NoteSeq-2.csv"
    assert extract_note_id(fn) == "19933834-DS-2"


def test_extract_note_id_ignores_leading_directories():
    path = "/some/dir/Labels_y_NoteID-12345678-DS-9_HadmID-1_NoteType-DS.csv"
    assert extract_note_id(path) == "12345678-DS-9"


def test_extract_note_id_raises_without_marker():
    with pytest.raises(ValueError):
        extract_note_id("Labels_no_note_id_here.csv")


# --- label CSV reading -----------------------------------------------------

def _write_labels(tmp_path, body, name="Labels_x_NoteID-1-DS-1_HadmID-2.csv"):
    p = tmp_path / name
    p.write_text("Start Position,End Position,Annotation,Group\n" + body)
    return str(p)


def test_read_label_rows_maps_all_six_gold_types(tmp_path):
    path = _write_labels(tmp_path, (
        "0,5,MEDICATION,1_AAA111\n"
        "6,10,DOSE,1_AAA111\n"
        "11,13,MODE,1_AAA111\n"
        "14,19,FREQUENCY,1_AAA111\n"
        "20,26,DURATION,1_AAA111\n"
        "27,39,REASON,1_AAA111\n"
    ))
    rows, skipped = read_label_rows(path)
    assert [r[2] for r in rows] == [
        "Medication", "Dose", "Mode", "Frequency", "Duration", "Reason"
    ]
    assert skipped == []


def test_read_label_rows_skips_unknown_type(tmp_path):
    path = _write_labels(tmp_path, "0,5,MEDICATION,1_A\n6,9,ALLERGY,1_A\n")
    rows, skipped = read_label_rows(path)
    assert [r[2] for r in rows] == ["Medication"]
    assert len(skipped) == 1


def test_read_label_rows_skips_malformed_and_degenerate_spans(tmp_path):
    path = _write_labels(tmp_path, (
        "0,5,MEDICATION,1_A\n"
        "9,9,DOSE,1_A\n"        # start == end
        "12,8,DOSE,1_A\n"       # end < start
        "-1,4,DOSE,1_A\n"       # negative start
        "abc,5,DOSE,1_A\n"      # unparseable
    ))
    rows, skipped = read_label_rows(path)
    assert rows == [(0, 5, "Medication")]
    assert len(skipped) == 4


def test_read_label_rows_empty_file_is_not_an_error(tmp_path):
    path = _write_labels(tmp_path, "")
    assert read_label_rows(path) == ([], [])


def test_read_label_rows_type_is_case_insensitive(tmp_path):
    path = _write_labels(tmp_path, "0,5,medication,1_A\n6,9, Dose ,1_A\n")
    rows, _ = read_label_rows(path)
    assert [r[2] for r in rows] == ["Medication", "Dose"]


# --- dedupe + tie-break ----------------------------------------------------

def test_resolve_dedupes_identical_span_and_type_across_groups():
    # The real corpus has 4,408 of these: one shared span reused by several
    # medication Groups.
    rows = [
        (10, 13, "Mode"),
        (10, 13, "Mode"),
        (10, 13, "Mode"),
    ]
    spans, conflicts = resolve_label_spans(rows)
    assert spans == [(10, 13, "Mode")]
    assert conflicts == []


def test_resolve_tie_break_follows_configured_priority():
    rows = [(5, 8, "Reason"), (5, 8, "Medication")]
    spans, conflicts = resolve_label_spans(rows)
    assert spans == [(5, 8, "Medication")]
    assert conflicts == [(5, 8, "Medication", ["Reason"])]


def test_resolve_tie_break_full_priority_order():
    # Every type competing for one span -> Medication wins outright.
    rows = [(0, 4, t) for t in reversed(TYPE_PRIORITY)]
    spans, conflicts = resolve_label_spans(rows)
    assert spans == [(0, 4, "Medication")]
    assert conflicts[0][2] == "Medication"
    assert conflicts[0][3] == TYPE_PRIORITY[1:]


def test_resolve_tie_break_is_order_independent():
    a, _ = resolve_label_spans([(1, 2, "Dose"), (1, 2, "Mode")])
    b, _ = resolve_label_spans([(1, 2, "Mode"), (1, 2, "Dose")])
    assert a == b == [(1, 2, "Dose")]


def test_resolve_keeps_distinct_spans_and_sorts():
    rows = [(20, 25, "Dose"), (0, 5, "Medication"), (10, 12, "Mode")]
    spans, conflicts = resolve_label_spans(rows)
    assert spans == [(0, 5, "Medication"), (10, 12, "Mode"), (20, 25, "Dose")]
    assert conflicts == []


def test_resolve_overlapping_but_distinct_spans_both_kept():
    # Overlap is resolved later, when painting onto tokens — not here.
    rows = [(0, 10, "Reason"), (4, 8, "Medication")]
    spans, conflicts = resolve_label_spans(rows)
    assert len(spans) == 2
    assert conflicts == []


def test_resolve_empty():
    assert resolve_label_spans([]) == ([], [])


# --- deterministic sampling ------------------------------------------------

def test_sample_is_deterministic_for_a_seed():
    pool = [f"{i}-DS-1" for i in range(600)]
    assert sample_note_ids(pool, 50, seed=13) == sample_note_ids(pool, 50, seed=13)


def test_hundred_draw_has_no_duplicates():
    pool = [f"{i}-DS-1" for i in range(600)]
    hundred = sample_note_ids(pool, 100, seed=13)
    assert len(hundred) == 100
    assert len(set(hundred)) == 100


def test_sample_50_is_NOT_a_subset_of_sample_100():
    """Guards the reason the pipeline draws 100 once and slices.

    ``random.sample(pool, 50)`` is not a prefix or even a subset of
    ``random.sample(pool, 100)`` for the same seed — they consume the RNG
    differently. If anyone "simplifies" build_sample to draw each size
    separately, the 50-note set stops being a subset of the 100-note set and the
    two runs stop being comparable. This test documents that trap.
    """
    pool = [f"{i}-DS-1" for i in range(600)]
    hundred = sample_note_ids(pool, 100, seed=13)
    fifty = sample_note_ids(pool, 50, seed=13)
    assert not set(fifty) <= set(hundred)


def test_slicing_one_draw_is_what_gives_the_subset_property():
    """The invariant the two runs actually rely on: slice a single draw."""
    pool = [f"{i}-DS-1" for i in range(600)]
    hundred = sample_note_ids(pool, 100, seed=13)
    fifty = hundred[:50]
    assert len(fifty) == 50
    assert set(fifty) < set(hundred)
    # ...and the same prefix every time, since the draw is seeded.
    assert sample_note_ids(pool, 100, seed=13)[:50] == fifty


def test_sample_independent_of_input_ordering():
    pool = [f"{i}-DS-1" for i in range(200)]
    a = sample_note_ids(pool, 20, seed=7)
    b = sample_note_ids(list(reversed(pool)), 20, seed=7)
    assert a == b


def test_different_seeds_give_different_samples():
    pool = [f"{i}-DS-1" for i in range(600)]
    assert sample_note_ids(pool, 50, seed=1) != sample_note_ids(pool, 50, seed=2)


def test_sample_larger_than_pool_returns_whole_pool():
    pool = ["a-DS-1", "b-DS-1"]
    assert sorted(sample_note_ids(pool, 10, seed=3)) == pool


# --- gold BIO construction -------------------------------------------------

def test_build_gold_from_char_offsets():
    text = "Give aspirin 325 mg PO daily for pain"
    #       0123456789...
    record = {
        "note_id": "synthetic-1",
        "text": text,
        "spans": [
            [5, 12, "Medication"],   # aspirin
            [13, 19, "Dose"],        # 325 mg
            [20, 22, "Mode"],        # PO
            [23, 28, "Frequency"],   # daily
            [33, 37, "Reason"],      # pain
        ],
    }
    gold = build_gold(record)
    assert gold["tokens"] == ["Give", "aspirin", "325", "mg", "PO", "daily",
                              "for", "pain"]
    assert gold["bio"] == [
        "O", "B-Medication", "B-Dose", "I-Dose", "B-Mode", "B-Frequency",
        "O", "B-Reason",
    ]
    assert gold["n_gold_spans"] == 5
    assert gold["n_gold_out_of_bounds"] == 0
    assert gold["n_gold_dropped_overlap"] == 0


def test_build_gold_span_containing_newline():
    # The real corpus has spans that straddle a line break.
    text = "hold it prior to\ngoing to bed"
    record = {"note_id": "s", "text": text, "spans": [[8, 28, "Duration"]]}
    gold = build_gold(record)
    assert gold["bio"] == ["O", "O", "B-Duration", "I-Duration", "I-Duration",
                           "I-Duration", "I-Duration"]


def test_build_gold_out_of_bounds_span_is_counted_not_crashed():
    record = {"note_id": "s", "text": "short text", "spans": [[0, 5000, "Dose"]]}
    gold = build_gold(record)
    assert gold["n_gold_out_of_bounds"] == 1
    assert gold["bio"] == ["O", "O"]


def test_build_gold_nested_overlap_outer_wins_and_is_counted():
    text = "for severe chest pain today"
    record = {
        "note_id": "s",
        "text": text,
        "spans": [[4, 21, "Reason"], [11, 21, "Reason"]],
    }
    gold = build_gold(record)
    assert gold["bio"] == ["O", "B-Reason", "I-Reason", "I-Reason", "O"]
    assert gold["n_gold_dropped_overlap"] == 1


def test_build_gold_empty_spans_all_O():
    record = {"note_id": "s", "text": "no meds here", "spans": []}
    gold = build_gold(record)
    assert gold["bio"] == ["O", "O", "O"]
    assert gold["n_gold_spans"] == 0


def test_build_gold_counts_are_consistent():
    record = {
        "note_id": "s",
        "text": "Lasix 40mg IV",
        "spans": [[0, 5, "Medication"], [6, 10, "Dose"], [11, 13, "Mode"]],
        "n_type_conflicts": 2,
    }
    gold = build_gold(record)
    assert gold["n_gold_raw"] == 3
    assert gold["n_tokens"] == 3
    assert gold["n_chars"] == len("Lasix 40mg IV")
    assert gold["n_type_conflicts"] == 2


# --- medication prompt -----------------------------------------------------

def test_prompt_parses_all_six_medication_types():
    reply = (
        '{"entities": ['
        '{"text": "aspirin", "type": "Medication"}, '
        '{"text": "325 mg", "type": "Dose"}, '
        '{"text": "PO", "type": "Mode"}, '
        '{"text": "daily", "type": "Frequency"}, '
        '{"text": "for 7 days", "type": "Duration"}, '
        '{"text": "pain", "type": "Reason"}]}'
    )
    assert parse_entities(reply) == [
        ("aspirin", "Medication"), ("325 mg", "Dose"), ("PO", "Mode"),
        ("daily", "Frequency"), ("for 7 days", "Duration"), ("pain", "Reason"),
    ]


def test_prompt_normalizes_uppercase_gold_style_types():
    reply = '{"entities": [{"text": "Lasix", "type": "MEDICATION"}]}'
    assert parse_entities(reply) == [("Lasix", "Medication")]


def test_prompt_drops_disease_chemical_types_from_the_other_task():
    reply = (
        '{"entities": [{"text": "asthma", "type": "Disease"}, '
        '{"text": "Lasix", "type": "Medication"}]}'
    )
    assert parse_entities(reply) == [("Lasix", "Medication")]


def test_prompt_malformed_reply_yields_no_entities():
    assert parse_entities("the model refused") == []
    assert parse_entities('{"entities": [truncated mid-jso') == []


def test_prompt_messages_are_text_only_and_contain_the_chunk():
    msgs = build_messages("Discharge Medications: Lasix 40mg")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    for m in msgs:
        assert m["content"][0]["type"] == "text"
        assert "image" not in {c["type"] for c in m["content"]}
    assert "Lasix 40mg" in msgs[1]["content"][0]["text"]


def test_prompt_lists_every_type_it_must_emit():
    text = build_messages("x")[1]["content"][0]["text"]
    for t in ("Medication", "Dose", "Mode", "Frequency", "Duration", "Reason"):
        assert f'"{t}"' in text
