"""Span -> BIO alignment tests — CPU-only, no model download, no GPU."""

from src.align import align_entities_to_bio, normalize_token


def test_normalize_strips_punctuation_and_case():
    assert normalize_token("Cancer.") == "cancer"
    assert normalize_token("(hepatitis)") == "hepatitis"
    assert normalize_token("ASPIRIN") == "aspirin"
    assert normalize_token("5-FU") == "5-fu"
    assert normalize_token("...") == ""


def test_single_token_span():
    tokens = ["Patient", "has", "asthma", "."]
    bio = align_entities_to_bio(tokens, [("asthma", "Disease")])
    assert bio == ["O", "O", "B-Disease", "O"]


def test_multi_word_span_contiguous():
    tokens = ["The", "patient", "has", "lung", "cancer", "today"]
    bio = align_entities_to_bio(tokens, [("lung cancer", "Disease")])
    assert bio == ["O", "O", "O", "B-Disease", "I-Disease", "O"]


def test_case_and_punctuation_insensitive_match():
    tokens = ["Treated", "with", "Aspirin."]
    bio = align_entities_to_bio(tokens, [("aspirin", "Chemical")])
    assert bio == ["O", "O", "B-Chemical"]


def test_multiple_non_overlapping_occurrences_all_tagged():
    tokens = ["cancer", "and", "more", "cancer", "here"]
    bio = align_entities_to_bio(tokens, [("cancer", "Disease")])
    assert bio == ["B-Disease", "O", "O", "B-Disease", "O"]


def test_span_not_present_leaves_all_O():
    tokens = ["Patient", "is", "healthy"]
    bio = align_entities_to_bio(tokens, [("lung cancer", "Disease")])
    assert bio == ["O", "O", "O"]


def test_two_types_in_one_sentence():
    tokens = ["asthma", "treated", "with", "aspirin"]
    ents = [("asthma", "Disease"), ("aspirin", "Chemical")]
    bio = align_entities_to_bio(tokens, ents)
    assert bio == ["B-Disease", "O", "O", "B-Chemical"]


def test_no_overwrite_first_come_first_served():
    # "lung cancer" claims tokens 0-1 first; a later "cancer" span must not
    # overwrite the already-tagged token.
    tokens = ["lung", "cancer"]
    ents = [("lung cancer", "Disease"), ("cancer", "Chemical")]
    bio = align_entities_to_bio(tokens, ents)
    assert bio == ["B-Disease", "I-Disease"]


def test_empty_entities_all_O():
    tokens = ["a", "b", "c"]
    assert align_entities_to_bio(tokens, []) == ["O", "O", "O"]


def test_punctuation_only_span_ignored():
    tokens = ["a", "b"]
    assert align_entities_to_bio(tokens, [("...", "Disease")]) == ["O", "O"]


# --- first_only (added for the MIMIC --align-mode work) --------------------

def test_first_only_tags_just_the_first_occurrence():
    tokens = ["cancer", "and", "more", "cancer", "here"]
    bio = align_entities_to_bio(tokens, [("cancer", "Disease")], first_only=True)
    assert bio == ["B-Disease", "O", "O", "O", "O"]


def test_first_only_default_is_off_so_existing_behavior_is_unchanged():
    tokens = ["cancer", "and", "more", "cancer"]
    assert align_entities_to_bio(tokens, [("cancer", "Disease")]) == \
        ["B-Disease", "O", "O", "B-Disease"]


def test_first_only_multi_word_span():
    tokens = ["lung", "cancer", "then", "lung", "cancer"]
    bio = align_entities_to_bio(tokens, [("lung cancer", "Disease")], first_only=True)
    assert bio == ["B-Disease", "I-Disease", "O", "O", "O"]


def test_first_only_still_tags_each_distinct_entity():
    tokens = ["aspirin", "aspirin", "asthma", "asthma"]
    ents = [("aspirin", "Chemical"), ("asthma", "Disease")]
    bio = align_entities_to_bio(tokens, ents, first_only=True)
    assert bio == ["B-Chemical", "O", "B-Disease", "O"]


def test_first_only_skips_blocked_tokens_to_find_a_free_occurrence():
    # "lung cancer" claims tokens 0-1; a later first_only "cancer" must land on
    # token 3, not give up because token 1 was taken.
    tokens = ["lung", "cancer", "and", "cancer"]
    ents = [("lung cancer", "Disease"), ("cancer", "Chemical")]
    bio = align_entities_to_bio(tokens, ents, first_only=True)
    assert bio == ["B-Disease", "I-Disease", "O", "B-Chemical"]


def test_first_only_absent_span_leaves_all_O():
    assert align_entities_to_bio(["a", "b"], [("zzz", "Disease")], first_only=True) \
        == ["O", "O"]
