"""The matching ladder: string rules, the 1:1 constraint, and monotonicity."""

import pytest

from src.recall_matching import (
    MEASURED_COSINE,
    assign,
    build_edges,
    candidate_strings,
    char_ratio,
    dice,
    match,
    no_threshold_separates,
    string_rule,
    token_contains,
)

LEVELS = ("L1", "L2", "L3", "L4")


# --------------------------------------------------------------------------
# The measured pair table
#
# These eight pairs are the argument for the ladder existing at all, so they are
# pinned rather than described. If a rule change moves any of them, the report's
# threshold justification has stopped being true and must be rewritten.
# --------------------------------------------------------------------------

MEASURED = [
    # model,               gold,                       contains, dice, char
    ("hyperlipidema",      "hyperlipidemia",           False, 0.00, 0.96),
    ("back pain",          "chronic back pain",        True,  0.80, 0.69),
    ("acute kidney injury", "kidney injury acute",     False, 1.00, 0.68),
    ("htn",                "hypertension",             False, 0.00, 0.40),
    ("chf",                "congestive heart failure", False, 0.00, 0.22),
    ("acute renal failure", "chronic renal failure",   False, 0.67, 0.75),
    ("sepsis",             "no evidence of sepsis",    True,  0.40, 0.44),
    ("diabetes",           "diabetes insipidus",       True,  0.67, 0.62),
]


@pytest.mark.parametrize("a,b,contains,d,c", MEASURED)
def test_measured_pairs_still_measure_the_same(a, b, contains, d, c):
    assert token_contains(a, b) is contains
    assert dice(a, b) == pytest.approx(d, abs=0.01)
    assert char_ratio(a, b) == pytest.approx(c, abs=0.01)


def test_no_string_threshold_separates_the_good_from_the_bad():
    """The whole reason L4 exists, asserted rather than asserted in prose.

    CHF/congestive heart failure is a pair we WANT and scores 0.22 on chars;
    acute/chronic renal failure is a pair we do NOT want and scores 0.75. Any
    character threshold catching the first admits the second.
    """
    want = char_ratio("chf", "congestive heart failure")
    unwanted = char_ratio("acute renal failure", "chronic renal failure")
    assert want < unwanted


def test_no_cosine_threshold_separates_them_either():
    """The finding that makes L5 load-bearing rather than optional.

    L4 was expected to be the level that finally tells a real synonym from a
    near-miss, since it is the only one that reaches abbreviations at all. It
    reaches them. It does not separate them: on the default biomedical encoder
    `acute renal failure` against `chronic renal failure` scores 0.833, above
    eight of the ten pairs L4 exists to catch.

    If an encoder change ever makes this fail, the report's L4 caveat has
    stopped being true and must be rewritten — which is why the assertion is on
    the property and not on the numbers.
    """
    assert no_threshold_separates()

    want = [s for _a, _b, keep, s in MEASURED_COSINE if keep]
    worst = max(s for _a, _b, keep, s in MEASURED_COSINE if not keep)
    assert sum(1 for s in want if s < worst) == 8


def test_the_default_cosine_reaches_every_abbreviation_measured():
    """The threshold is a floor chosen to reach L4's purpose, not a cutoff."""
    from src.recall_config import COSINE_MIN

    missed = [(a, b) for a, b, keep, s in MEASURED_COSINE
              if keep and s < COSINE_MIN]
    assert missed == []


@pytest.mark.parametrize("a,b,rule", [
    ("sepsis", "sepsis", "exact"),
    ("back pain", "chronic back pain", "contains"),
    ("acute kidney injury", "kidney injury acute", "dice"),
    ("hyperlipidema", "hyperlipidemia", "ratio"),
    ("acute renal failure", "chronic renal failure", None),
    ("htn", "hypertension", None),
    ("chf", "congestive heart failure", None),
])
def test_string_rule_at_default_thresholds(a, b, rule):
    got = string_rule(a, b)
    assert (got[0] if got else None) == rule


def test_containment_is_whole_token():
    """`ca` must not match inside `cabg` — the padding is load-bearing."""
    assert not token_contains("ca", "cabg")
    assert token_contains("ca", "ca lung")


def test_empty_strings_never_match():
    assert string_rule("", "sepsis") is None
    assert string_rule("sepsis", "") is None
    assert dice("", "sepsis") == 0.0


# --------------------------------------------------------------------------
# The 1:1 constraint
# --------------------------------------------------------------------------

def test_one_prediction_cannot_satisfy_two_gold_forms():
    """A vague prediction contained in two gold phrases is worth ONE match."""
    ladder = match([("pain",)], {"chronic back pain", "chest pain"},
                   levels=LEVELS)
    assert len(ladder["L2"]["matched_forms"]) == 1
    assert len(ladder["L2"]["pairs"]) == 1


def test_two_predictions_can_satisfy_two_gold_forms():
    ladder = match([("chronic back pain",), ("chest pain",)],
                   {"chronic back pain", "chest pain"}, levels=LEVELS)
    assert ladder["L1"]["matched_forms"] == {"chronic back pain", "chest pain"}


def test_a_gold_form_is_claimed_once():
    """Two identical predictions against one gold form: one hit, one spare."""
    ladder = match([("sepsis",), ("sepsis",)], {"sepsis"}, levels=LEVELS)
    assert len(ladder["L1"]["pairs"]) == 1


def test_augmenting_beats_plain_greedy():
    """The case that made pure greedy unusable.

    Finding 0 can reach both forms; finding 1 can reach only `hypertension`. A
    greedy pass that hands `hypertension` to finding 0 strands finding 1 and
    reports recall 1/2. Augmenting recovers the perfect matching.
    """
    findings = [("hypertension", "sepsis"), ("hypertension",)]
    ladder = match(findings, {"hypertension", "sepsis"}, levels=LEVELS)
    assert ladder["L1"]["matched_forms"] == {"hypertension", "sepsis"}
    assert len(ladder["L1"]["pairs"]) == 2


def test_assign_returns_nothing_when_nodes_are_taken():
    edges = [("exact", 1.0, 0, "sepsis")]
    assert assign(edges, free_findings=set(), free_forms={"sepsis"}) == {}
    assert assign(edges, free_findings={0}, free_forms=set()) == {}


# --------------------------------------------------------------------------
# Ladder behaviour
# --------------------------------------------------------------------------

def test_recall_is_monotonically_non_decreasing():
    findings = [("back pain",), ("hyperlipidema",), ("acute kidney injury",)]
    gold = {"chronic back pain", "hyperlipidemia", "kidney injury acute"}
    ladder = match(findings, gold, levels=LEVELS)
    counts = [len(ladder[lv]["matched_forms"]) for lv in LEVELS]
    assert counts == sorted(counts)
    assert counts[0] == 0            # nothing is an exact match
    assert counts[-1] == 3           # all three arrive by L3


def test_each_level_reports_only_what_it_added():
    ladder = match([("sepsis",), ("back pain",)],
                   {"sepsis", "chronic back pain"}, levels=LEVELS)
    assert [p[1] for p in ladder["L1"]["new"]] == ["sepsis"]
    assert [p[1] for p in ladder["L2"]["new"]] == ["chronic back pain"]
    assert ladder["L3"]["new"] == []


def test_lower_level_assignments_are_frozen():
    """A pair credited to L1 keeps its rule at every level above it."""
    ladder = match([("sepsis",)], {"sepsis"}, levels=LEVELS)
    for level in LEVELS:
        assert ladder[level]["pairs"][0][1] == "exact"


def test_a_pair_reachable_by_two_rules_is_credited_to_the_stricter():
    """`back pain` is both contained in and Dice-close to `chronic back pain`."""
    ladder = match([("back pain",)], {"chronic back pain"}, levels=LEVELS)
    assert ladder["L2"]["pairs"][0][1] == "contains"


def test_l4_edges_need_an_embedder():
    """Without a backend the ladder simply stops short — never a silent zero."""
    ladder = match([("chf",)], {"congestive heart failure"}, levels=LEVELS)
    assert ladder["L4"]["matched_forms"] == set()


def test_l4_edges_are_added_when_an_embedder_is_supplied():
    class FakeEmbedder:
        """Stands in for the biomedical encoder: knows one abbreviation."""

        def similarity(self, left, right):
            return {(l, r): (0.95 if {l, r} == {"chf", "congestive heart failure"}
                             else 0.1)
                    for l in left for r in right}

    ladder = match([("chf",)], {"congestive heart failure"},
                   embedder=FakeEmbedder(), levels=LEVELS)
    assert ladder["L3"]["matched_forms"] == set()
    assert ladder["L4"]["matched_forms"] == {"congestive heart failure"}
    assert ladder["L4"]["pairs"][0][1] == "cosine"


def test_levels_can_be_truncated_without_blowing_up():
    ladder = match([("back pain",)], {"chronic back pain"}, levels=("L1", "L2"))
    assert set(ladder) == {"L1", "L2"}


# --------------------------------------------------------------------------
# Candidate strings
# --------------------------------------------------------------------------

def test_both_fields_are_offered_to_the_matcher():
    assert candidate_strings({"span": "HTN", "name": "hypertension"}) == \
        ("htn", "hypertension")


def test_a_repeated_field_is_offered_once():
    assert candidate_strings({"span": "Sepsis", "name": "sepsis"}) == ("sepsis",)


def test_a_missing_field_is_dropped_not_blanked():
    assert candidate_strings({"span": "sepsis", "name": ""}) == ("sepsis",)
    assert candidate_strings({"span": "", "name": ""}) == ()


def test_name_alone_can_match_the_catalogue_wording():
    """The point of the two-field prompt: HTN reaches `hypertension`."""
    edges = build_edges([("htn", "hypertension")], ["hypertension"])
    assert [e[0] for e in edges] == ["exact"]
