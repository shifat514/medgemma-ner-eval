"""The two-field prompt and its deliberately permissive reply parser.

Nothing here is scored except strings, so the only failure mode is losing usable
text. Every test below is a shape a 4B model actually produces or plausibly
will; a parser that rejects one of them turns a real extraction into a silent
false negative.
"""

import json

import pytest

from src.prompt_recall import (
    build_messages,
    build_prompt,
    parse_findings,
    parse_findings_diag,
    prompt_fingerprint,
)


def _spans(findings):
    return [f["span"] for f in findings]


def _names(findings):
    return [f["name"] for f in findings]


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_parses_the_shape_the_prompt_asks_for():
    reply = json.dumps({"findings": [
        {"span": "HTN", "name": "hypertension"},
        {"span": "CHF", "name": "congestive heart failure"},
    ]})
    findings = parse_findings(reply)
    assert _spans(findings) == ["HTN", "CHF"]
    assert _names(findings) == ["hypertension", "congestive heart failure"]


def test_both_fields_survive_independently():
    """Either may match the accept-set, so neither may be dropped."""
    findings = parse_findings(
        json.dumps({"findings": [{"span": "HTN", "name": "hypertension"}]}))
    assert findings[0]["span"] == "HTN"
    assert findings[0]["name"] == "hypertension"


def test_an_explicitly_empty_list_is_recorded_not_treated_as_a_failure():
    findings, diag = parse_findings_diag(json.dumps({"findings": []}))
    assert findings == []
    assert diag["empty_list"] is True
    assert diag["shape"] == "json"


# --------------------------------------------------------------------------
# Shapes the model produces anyway
# --------------------------------------------------------------------------

def test_markdown_fences_are_stripped():
    reply = '```json\n{"findings": [{"span": "sepsis", "name": "sepsis"}]}\n```'
    assert _spans(parse_findings(reply)) == ["sepsis"]


def test_prose_around_the_json_is_tolerated():
    reply = ('Here are the findings I identified:\n'
             '{"findings": [{"span": "sepsis", "name": "sepsis"}]}\n'
             'Let me know if you need more.')
    assert _spans(parse_findings(reply)) == ["sepsis"]


def test_a_bare_list_with_no_wrapper_key():
    reply = json.dumps([{"span": "sepsis", "name": "sepsis"}])
    assert _spans(parse_findings(reply)) == ["sepsis"]


def test_a_bare_string_is_a_span_with_no_name():
    """Still perfectly usable — a span alone matches the evidence-text source."""
    findings, diag = parse_findings_diag(json.dumps({"findings": ["sepsis"]}))
    assert findings == [{"span": "sepsis", "name": ""}]
    assert diag["n_bare_string"] == 1
    assert diag["n_span_only"] == 1


@pytest.mark.parametrize("key", ["text", "phrase", "mention", "evidence"])
def test_alternative_span_keys_are_accepted(key):
    reply = json.dumps({"findings": [{key: "sepsis", "name": "sepsis"}]})
    assert _spans(parse_findings(reply)) == ["sepsis"]


@pytest.mark.parametrize("key", ["name", "term", "concept", "standard_name"])
def test_alternative_name_keys_are_accepted(key):
    reply = json.dumps({"findings": [{"span": "HTN", key: "hypertension"}]})
    assert _names(parse_findings(reply)) == ["hypertension"]


@pytest.mark.parametrize("container", ["terms", "entities", "conditions",
                                       "items", "results"])
def test_alternative_container_keys_are_accepted(container):
    reply = json.dumps({container: [{"span": "sepsis", "name": "sepsis"}]})
    assert _spans(parse_findings(reply)) == ["sepsis"]


def test_a_type_label_is_recorded_and_never_becomes_a_finding():
    """Without the guard, {"type": "Condition"} manufactures a false positive
    out of a schema word."""
    reply = json.dumps({"findings": [
        {"span": "sepsis", "name": "sepsis", "type": "Condition"}]})
    findings, diag = parse_findings_diag(reply)
    assert _spans(findings) == ["sepsis"]
    assert diag["types"] == {"Condition": 1}


def test_an_object_with_only_a_type_yields_nothing():
    findings, diag = parse_findings_diag(
        json.dumps({"findings": [{"type": "Condition"}]}))
    assert findings == []
    assert diag["n_no_text"] == 1


def test_an_unrecognized_key_still_donates_its_string():
    """Better a finding under a strange key than a silently lost extraction."""
    reply = json.dumps({"findings": [{"whatever": "sepsis"}]})
    assert _spans(parse_findings(reply)) == ["sepsis"]


def test_a_name_with_no_span_is_kept_and_counted():
    """It cannot be checked against the note, so the count is reported."""
    findings, diag = parse_findings_diag(
        json.dumps({"findings": [{"name": "hypertension"}]}))
    assert findings == [{"span": "", "name": "hypertension"}]
    assert diag["n_name_only"] == 1


# --------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------

def test_a_reply_cut_mid_array_keeps_its_complete_objects():
    """The object form degrades better than the flat-string form did: each
    finding is its own balanced object, so the scanner recovers the prefix."""
    reply = ('{"findings": [{"span": "sepsis", "name": "sepsis"}, '
             '{"span": "HTN", "name": "hypertension"}, {"span": "acute kidney inj')
    findings = parse_findings(reply)
    assert _spans(findings) == ["sepsis", "HTN"]


def test_a_reply_cut_before_any_object_closed_falls_back_to_salvage():
    reply = '{"findings": ["sepsis", "HTN", "acute kidney inj'
    findings, diag = parse_findings_diag(reply)
    assert _spans(findings) == ["sepsis", "HTN"]
    assert diag["shape"] == "salvaged-truncated"
    assert diag["n_salvaged"] == 2


def test_salvage_does_not_donate_schema_words_as_findings():
    reply = '{"findings": [{"span": "sepsis", "name'
    findings = parse_findings(reply)
    assert "name" not in _spans(findings)
    assert "span" not in _spans(findings)


def test_salvage_only_runs_when_nothing_else_worked():
    reply = json.dumps({"findings": [{"span": "sepsis", "name": "sepsis"}]})
    _findings, diag = parse_findings_diag(reply)
    assert diag["shape"] == "json"
    assert diag["n_salvaged"] == 0


# --------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reply,shape", [
    ("", "empty-reply"),
    ("   ", "empty-reply"),
    (None, "empty-reply"),
    ("I could not find any findings.", "no-json"),
])
def test_degenerate_replies_report_a_reason_rather_than_a_silent_zero(reply, shape):
    findings, diag = parse_findings_diag(reply)
    assert findings == []
    assert diag["shape"] == shape


def test_a_json_object_with_no_list_is_named_as_such():
    _findings, diag = parse_findings_diag(json.dumps({"note": "nothing here"}))
    assert diag["shape"] == "no-item-list"


# --------------------------------------------------------------------------
# The prompt itself
# --------------------------------------------------------------------------

def test_prompt_asks_for_both_fields():
    prompt = build_prompt("")
    assert '"span"' in prompt
    assert '"name"' in prompt
    assert "HTN" in prompt and "hypertension" in prompt


def test_prompt_still_excludes_medications():
    """Carried forward by measurement: medications produced 33% of extraction
    for 5.5% of gold and truncated 12 of 15 chunks."""
    prompt = build_prompt("")
    assert "Do NOT extract medications" in prompt
    assert "aspirin" in prompt          # the negative example


def test_prompt_still_names_the_redaction_markers():
    assert "[**Known lastname 1234**]" in build_prompt("")


def test_prompt_asks_for_one_entry_per_distinct_finding():
    """This is what pays for the second field's token cost."""
    assert "ONE entry per distinct finding" in build_prompt("")


def test_the_example_output_is_valid_json_in_the_requested_shape():
    prompt = build_prompt("")
    start = prompt.index('{"findings"')
    end = prompt.index("\n", start)
    parsed = json.loads(prompt[start:end])
    assert all(set(f) == {"span", "name"} for f in parsed["findings"])
    assert {"span": "HTN", "name": "hypertension"} in parsed["findings"]


def test_the_example_leaves_the_medication_unextracted():
    prompt = build_prompt("")
    start = prompt.index('{"findings"')
    parsed = json.loads(prompt[start:prompt.index("\n", start)])
    spans = " ".join(f["span"] for f in parsed["findings"]).lower()
    assert "aspirin" not in spans
    assert "tobacco" not in spans


def test_the_chunk_text_is_appended_verbatim():
    assert build_prompt("NOTE BODY").endswith("NOTE BODY")


def test_fingerprint_is_stable_and_short():
    assert prompt_fingerprint() == prompt_fingerprint()
    assert len(prompt_fingerprint()) == 8


def test_fingerprint_differs_from_the_term_ner_prompt():
    """The run directory is keyed on this, so a shared hash would replay the
    old prompt's cached results with no error."""
    from src.prompt_mdace import prompt_fingerprint as term_ner_fingerprint

    assert prompt_fingerprint() != term_ner_fingerprint()


def test_messages_are_text_only_in_medgemmas_structure():
    messages = build_messages("text")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(c["type"] == "text" for m in messages for c in m["content"])


# --------------------------------------------------------------------------
# Scope leaks the first smoke run found
#
# Rules 1 and 2 already forbade all of these in the abstract and the model
# extracted them anyway: SBPs as "systolic blood pressure", pRBCs as "packed red
# blood cells", the bowel prep GoLYTEly, and a bare "R sided kidney". Abstract
# prohibitions do not work on a 4B model; named ones and demonstrated ones do.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("leak", ["GoLYTELY", "pRBCs", "blood products",
                                  "bowel preparations"])
def test_prompt_names_the_substances_it_leaked(leak):
    assert leak in build_prompt("")


@pytest.mark.parametrize("leak", ["SBP", "hematocrit", "heart rate"])
def test_prompt_names_the_measurements_it_leaked(leak):
    assert leak in build_prompt("")


def test_prompt_distinguishes_a_measurement_from_the_diagnosis_it_supports():
    """Excluding labs must not exclude the condition they evidence."""
    prompt = build_prompt("")
    assert 'extract "anemia" if the note says anemia' in prompt


def test_prompt_rejects_a_bare_body_part():
    prompt = build_prompt("")
    assert '"right kidney" is not a finding' in prompt


def test_the_example_demonstrates_the_measurement_exclusions(tmp_path=None):
    """A negative example, not an omitted category — the technique that finally
    worked for medications, applied to vitals and labs."""
    prompt = build_prompt("")
    example_input = prompt.split("Example input:\n")[1].split("\n\nExample output")[0]
    assert "SBPs" in example_input
    assert "hematocrit" in example_input

    start = prompt.index('{"findings"')
    parsed = json.loads(prompt[start:prompt.index("\n", start)])
    emitted = " ".join(f["span"] + " " + f["name"] for f in parsed["findings"]).lower()
    for leaked in ("sbp", "hematocrit", "aspirin", "tobacco"):
        assert leaked not in emitted


def test_prompt_tells_the_model_to_stop_rather_than_restart():
    """The observed failure was a verbatim replay of items 2-9 after item 16 --
    greedy degeneration on a long list, which ran into the token cap."""
    assert "Do not start the list again from the beginning" in build_prompt("")
