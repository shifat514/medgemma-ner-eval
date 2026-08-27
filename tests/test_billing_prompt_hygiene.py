"""The prompt's example must not resemble the notes it is evaluated on.

WHY THIS FILE EXISTS. The first version of the example was an otitis media visit
coded H66.001. Note 112976 opens with the chief complaint "Ear infection, fever,
ear pulling", and H66.001 came back as a false positive on that note in every
variant and every generation config tried — despite that note's exam recording
normal canals and TMs, which rules otitis media out on the page.

The code's *shape* leaked too. Under the repetition-penalty run, 10 of 16 false
positives carried three digits after the decimal point (H66.001, Z00.001,
B95.001, R17.001, F20.901) against 2 of 16 gold codes shaped that way.

None of that was caught by a test, because every test asked whether the parser
and the scorer worked. They did. The prompt was the thing that was wrong, and a
prompt has no return value to assert on.

So these tests assert the one property the example must have: it must share no
clinical content and no code with the corpus it is scored against.

THE CORPUS-DEPENDENT TESTS SKIP WITHOUT THE BUILT SAMPLE, which is gitignored
and absent on a fresh clone or in CI. The rest run everywhere.
"""

import json
import os
import re

import pytest

from src.billing_config import SAMPLE_FILE
from src.prompt_billing import _EXAMPLE_INPUT, _EXAMPLE_OUTPUT, instruction

# Clinical content words in the example. If any of these appears in a note, the
# model can score by echoing the prompt rather than by reading the chart.
_EXAMPLE_TERMS = (
    "impetigo", "tinea", "mupirocin", "clotrimazole", "lactose",
    "honey-crusted", "annular",
)

_EXAMPLE_CODES = ("L01.00", "B35.4")

_CODE_RE = re.compile(r"\b[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?\b")


def _corpus():
    if not os.path.exists(SAMPLE_FILE):
        pytest.skip(f"{SAMPLE_FILE} not built (gitignored); run make billing-sample")
    return [json.loads(line) for line in open(SAMPLE_FILE, encoding="utf-8")
            if line.strip()]


# --- no shared content ------------------------------------------------------


def test_example_terms_appear_in_no_note():
    """The failure this file was written for, stated directly."""
    records = _corpus()
    text = " ".join(r["variants"]["full"] for r in records).lower()
    hits = [t for t in _EXAMPLE_TERMS if t in text]
    assert hits == [], (
        f"example terms {hits} appear in the notes — the model can echo the "
        "prompt instead of reading the chart"
    )


def test_example_codes_are_not_gold_anywhere():
    records = _corpus()
    gold = {c for r in records for c in r["gold_codes"]}
    assert gold.isdisjoint(_EXAMPLE_CODES)


def test_example_codes_share_no_category_with_gold():
    """Not just the code — the three-character category.

    H66.001 was never gold, and that was not enough. A category collision lets
    the model land in the right neighbourhood for the wrong reason.
    """
    records = _corpus()
    gold_cats = {c.split(".")[0] for r in records for c in r["gold_codes"]}
    example_cats = {c.split(".")[0] for c in _EXAMPLE_CODES}
    overlap = gold_cats & example_cats
    assert overlap == set(), f"example shares ICD category {overlap} with gold"


# --- shape ------------------------------------------------------------------


def test_example_does_not_teach_one_code_shape():
    """A single shape gets copied. Two different ones cannot both be the template."""
    tails = {c.split(".")[1] if "." in c else "" for c in _EXAMPLE_CODES}
    lengths = {len(t) for t in tails}
    assert len(lengths) > 1, (
        f"every example code has a {lengths} digit tail; vary them, or the "
        "model learns the format as if it were the answer"
    )


def test_example_shows_more_than_one_code():
    """Gold runs 2-6 codes per note. A one-code example under-anchors the count."""
    codes = json.loads(_EXAMPLE_OUTPUT)["codes"]
    assert len(codes) >= 2


def test_example_output_codes_match_the_declared_set():
    """Keeps _EXAMPLE_CODES honest when someone edits the JSON and not the tuple."""
    codes = [c["code"] for c in json.loads(_EXAMPLE_OUTPUT)["codes"]]
    assert sorted(codes) == sorted(_EXAMPLE_CODES)


# --- the example still teaches what it is for -------------------------------


def test_example_input_carries_something_deliberately_not_coded():
    """The negative half. Without it the example only demonstrates extraction."""
    assert "lactose intolerance" in _EXAMPLE_INPUT.lower()
    assert "lactose" not in _EXAMPLE_OUTPUT.lower()


def test_instruction_explains_why_the_uncoded_thing_is_uncoded():
    text = instruction().lower()
    assert "lactose intolerance" in text
    assert "past history" in text


def test_every_code_in_the_example_output_appears_in_the_instruction():
    """The example is embedded in the instruction, not merely defined beside it."""
    text = instruction()
    for code in _EXAMPLE_CODES:
        assert code in text


def test_instruction_still_rules_out_cpt():
    assert "99213" in instruction()
