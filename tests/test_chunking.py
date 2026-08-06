"""Chunk-window and chunk-merge tests — CPU-only, synthetic text only.

NO REAL NOTE TEXT IN THIS FILE. Every fixture is invented.
"""

from src.chunking import (
    bio_to_spans,
    char_spans_to_token_spans,
    chunk_windows,
    merge_chunk_spans,
    spans_to_bio,
    token_spans_to_char,
    tokenize_with_spans,
)


# --- tokenization ----------------------------------------------------------

def test_tokenize_with_spans_offsets_are_exact():
    text = "Give Drugzol 250 mg PO daily."
    tokens, spans = tokenize_with_spans(text)
    assert tokens == ["Give", "Drugzol", "250", "mg", "PO", "daily."]
    for tok, (s, e) in zip(tokens, spans):
        assert text[s:e] == tok


def test_tokenize_handles_newlines_and_runs_of_space():
    text = "Lasix 40mg\n\n  IV   BID"
    tokens, spans = tokenize_with_spans(text)
    assert tokens == ["Lasix", "40mg", "IV", "BID"]
    for tok, (s, e) in zip(tokens, spans):
        assert text[s:e] == tok


def test_tokenize_empty_text():
    assert tokenize_with_spans("") == ([], [])


# --- chunk windows ---------------------------------------------------------

def test_windows_cover_every_token():
    for n in (1, 5, 99, 100, 101, 400, 401, 1000, 1521):
        windows = chunk_windows(n, 400, 80)
        covered = set()
        for a, b in windows:
            covered.update(range(a, b))
        assert covered == set(range(n)), f"gap or overrun at n={n}"


def test_windows_have_expected_stride_and_overlap():
    windows = chunk_windows(1000, 400, 80)
    assert windows[0] == (0, 400)
    assert windows[1] == (320, 720)   # stride 400-80
    assert windows[2] == (640, 1000)  # clamped to the end (would be 1040)
    assert windows[-1][1] == 1000


def test_short_note_is_a_single_window():
    assert chunk_windows(100, 400, 80) == [(0, 100)]
    assert chunk_windows(400, 400, 80) == [(0, 400)]


def test_windows_empty_for_empty_note():
    assert chunk_windows(0, 400, 80) == []


def test_overlap_clamped_below_chunk_so_stride_is_positive():
    # overlap >= chunk would mean stride <= 0 and an infinite loop.
    windows = chunk_windows(50, 10, 10)
    assert windows[0] == (0, 10)
    assert windows[1] == (1, 11)          # stride clamped to 1
    assert len(windows) == 41             # starts 0..40, last window ends at 50
    assert windows[-1] == (40, 50)


def test_zero_overlap_is_disjoint_tiling():
    windows = chunk_windows(25, 10, 0)
    assert windows == [(0, 10), (10, 20), (20, 25)]


# --- BIO <-> spans ---------------------------------------------------------

def test_bio_to_spans_basic():
    bio = ["O", "B-Medication", "O", "B-Dose", "I-Dose", "O"]
    assert bio_to_spans(bio) == [(1, 2, "Medication"), (3, 5, "Dose")]


def test_bio_to_spans_adjacent_same_type_are_separate_spans():
    bio = ["B-Medication", "B-Medication"]
    assert bio_to_spans(bio) == [(0, 1, "Medication"), (1, 2, "Medication")]


def test_bio_to_spans_i_of_other_type_does_not_extend():
    bio = ["B-Dose", "I-Medication"]
    assert bio_to_spans(bio) == [(0, 1, "Dose")]


def test_bio_to_spans_ignores_orphan_i_tag():
    assert bio_to_spans(["I-Dose", "O"]) == []


def test_bio_to_spans_empty():
    assert bio_to_spans([]) == []
    assert bio_to_spans(["O", "O"]) == []


def test_spans_to_bio_roundtrip():
    spans = [(1, 2, "Medication"), (3, 5, "Dose")]
    bio, dropped = spans_to_bio(6, spans)
    assert bio == ["O", "B-Medication", "O", "B-Dose", "I-Dose", "O"]
    assert dropped == []
    assert bio_to_spans(bio) == spans


def test_spans_to_bio_all_or_nothing_on_partial_overlap():
    # (2,5) is painted first (earlier start); (4,7) partially overlaps -> dropped
    # whole rather than producing a corrupt I--led span.
    bio, dropped = spans_to_bio(8, [(2, 5, "Reason"), (4, 7, "Dose")])
    assert bio == ["O", "O", "B-Reason", "I-Reason", "I-Reason", "O", "O", "O"]
    assert dropped == [(4, 7, "Dose")]


def test_spans_to_bio_longest_wins_at_same_start():
    # Nested span: the outer span claims the tokens, the inner one is dropped.
    bio, dropped = spans_to_bio(4, [(0, 1, "Dose"), (0, 3, "Reason")])
    assert bio == ["B-Reason", "I-Reason", "I-Reason", "O"]
    assert dropped == [(0, 1, "Dose")]


def test_spans_to_bio_priority_breaks_identical_span_tie():
    priority = ["Medication", "Dose", "Mode", "Frequency", "Duration", "Reason"]
    bio, dropped = spans_to_bio(2, [(0, 1, "Reason"), (0, 1, "Medication")],
                                priority=priority)
    assert bio == ["B-Medication", "O"]
    assert dropped == [(0, 1, "Reason")]


def test_spans_to_bio_is_order_independent():
    spans = [(4, 7, "Dose"), (2, 5, "Reason"), (0, 1, "Medication")]
    a, _ = spans_to_bio(8, spans)
    b, _ = spans_to_bio(8, list(reversed(spans)))
    assert a == b


def test_spans_to_bio_clamps_and_drops_out_of_range():
    bio, dropped = spans_to_bio(3, [(1, 99, "Dose")])
    assert bio == ["O", "B-Dose", "I-Dose"]
    bio, dropped = spans_to_bio(3, [(5, 6, "Dose")])
    assert bio == ["O", "O", "O"]
    assert len(dropped) == 1


# --- merge / dedupe --------------------------------------------------------

def test_merge_shifts_chunk_local_indices_into_note_space():
    # window at token 100; chunk-local token 2 is note-level token 102.
    chunk_results = [(100, ["O", "O", "B-Medication", "O"])]
    spans, dupes = merge_chunk_spans(chunk_results)
    assert spans == [(102, 103, "Medication")]
    assert dupes == 0


def test_merge_dedupes_entity_predicted_twice_in_overlap():
    # Windows (0,10) and (8,18) overlap on tokens 8-9. The same entity at token 8
    # is predicted by both chunks and must be counted once.
    chunk_a = ["O"] * 8 + ["B-Medication", "I-Medication"]
    chunk_b = ["B-Medication", "I-Medication"] + ["O"] * 8
    spans, dupes = merge_chunk_spans([(0, chunk_a), (8, chunk_b)])
    assert spans == [(8, 10, "Medication")]
    assert dupes == 1


def test_merge_keeps_distinct_spans_from_different_chunks():
    spans, dupes = merge_chunk_spans([
        (0, ["B-Medication", "O"]),
        (2, ["B-Dose", "O"]),
    ])
    assert spans == [(0, 1, "Medication"), (2, 3, "Dose")]
    assert dupes == 0


def test_merge_same_span_different_type_is_not_a_duplicate():
    spans, dupes = merge_chunk_spans([
        (0, ["B-Medication"]),
        (0, ["B-Dose"]),
    ])
    assert spans == [(0, 1, "Dose"), (0, 1, "Medication")]
    assert dupes == 0


def test_merge_counts_triple_prediction_as_two_duplicates():
    chunk = ["B-Dose"]
    _, dupes = merge_chunk_spans([(5, chunk), (5, chunk), (5, chunk)])
    assert dupes == 2


def test_merge_empty_input():
    assert merge_chunk_spans([]) == ([], 0)
    assert merge_chunk_spans([(0, ["O", "O"])]) == ([], 0)


def test_merge_output_is_sorted():
    spans, _ = merge_chunk_spans([
        (50, ["B-Dose"]),
        (10, ["B-Medication"]),
        (30, ["B-Mode"]),
    ])
    assert spans == sorted(spans)


# --- char <-> token span conversion ---------------------------------------

def test_char_to_token_span_conversion_exact_boundaries():
    text = "Give Drugzol 250 mg PO daily."
    tokens, char_spans = tokenize_with_spans(text)
    # "Drugzol" is chars 5-12; "250 mg" is chars 13-19
    got = char_spans_to_token_spans(
        [(5, 12, "Medication"), (13, 19, "Dose")], char_spans
    )
    assert got == [(1, 2, "Medication"), (2, 4, "Dose")]


def test_token_to_char_roundtrip():
    text = "Lasix 40mg IV BID for 3 days"
    tokens, char_spans = tokenize_with_spans(text)
    original = [(0, 5, "Medication"), (6, 10, "Dose"), (18, 28, "Duration")]
    tok_spans = char_spans_to_token_spans(original, char_spans)
    back = token_spans_to_char(tok_spans, char_spans)
    assert back == original
    for s, e, _ in back:
        assert text[s:e].strip() == text[s:e]


def test_char_span_covering_no_token_is_skipped():
    text = "a  b"
    _, char_spans = tokenize_with_spans(text)
    # chars 1-2 are whitespace only
    assert char_spans_to_token_spans([(1, 2, "Dose")], char_spans) == []


def test_char_span_spanning_whitespace_covers_both_tokens():
    text = "250 mg"
    _, char_spans = tokenize_with_spans(text)
    assert char_spans_to_token_spans([(0, 6, "Dose")], char_spans) == [(0, 2, "Dose")]


def test_token_spans_to_char_skips_out_of_range():
    _, char_spans = tokenize_with_spans("a b")
    assert token_spans_to_char([(0, 99, "Dose")], char_spans) == []
    assert token_spans_to_char([(1, 1, "Dose")], char_spans) == []
