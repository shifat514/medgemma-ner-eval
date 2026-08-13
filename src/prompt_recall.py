"""Two-field extraction prompt and reply parsing for the recall benchmark.

WHY THE PROMPT CHANGED. The term-NER prompt (`prompt_mdace`, hash `090a0072`)
says *copy VERBATIM, do not expand abbreviations, keep it 1-3 words*. That was
correct when gold was the note's own wording. It is wrong here: the accept-set
now includes the ICD catalogue wording and SNOMED terms, and `HTN` can never
string-match `Essential (primary) hypertension` no matter how good the matcher
is.

So every finding carries two fields:

    span   the phrase as written in the note. Needed for the not-in-note check,
           and it is what matches the evidence-text source.
    name   the standard clinical name. What a real code lookup would consume,
           and what matches the description and SNOMED sources.

Either may match the accept-set. MedGemma already knows the expansions; using
its own expansion is better than making the matcher infer one, and it is what
separates a genuine miss from a vocabulary mismatch.

THE COST, AND WHAT PAYS FOR IT. Two fields is ~2x tokens per finding — exactly
what the flat-string change bought back on the term-NER branch, where the typed
form truncated 7 of 15 smoke chunks at 1024. It is paid back by asking for one
entry per distinct finding with no repeats, which also serves the
false-positive concern: repeats were pure volume.

CARRIED FORWARD UNCHANGED, all of it measured on the previous branch: the
redaction-marker instruction stays, `max_new_tokens` stays at 1024, and the
truncation salvage stays.

MEDICATIONS ARE VARIANT-SPECIFIC, AND THIS PARAGRAPH USED TO GET IT WRONG. The
measured exclusion — asking for medications produced 33% of extraction for 5.5%
of gold and truncated 12 of 15 chunks — is carried forward by `_SCOPED` ONLY.
`_BILLABLE`, which is the default, has no medication rule at all; whether the
model still leaves them alone follows from asking only for codeable findings and
is unmeasured. So `scoped` caps recall at about 0.945 by choice, and `billable`
has an unmeasured ceiling that may be higher.

This docstring asserted the exclusion as a property of the benchmark for one
commit after the default changed, which is exactly the failure the prompt hash
exists to prevent one level down: prose describing a configuration that is no
longer running. `tests/test_recall_prompt` now pins each claim to the variant
that actually makes it.

THE PROMPT HASH IS PART OF THE RUN-DIRECTORY NAME, so this rewrite starts a
fresh cache automatically and cannot replay the term-NER prompt's results.

PARSING IS DELIBERATELY PERMISSIVE. Nothing here is scored except strings, so
the only failure mode is losing usable text. The tolerant scanner from
`prompt_mdace` is reused rather than reimplemented — it already handles JSONL,
prose around the JSON, markdown fences, and replies cut off at the token cap.
The object form actually degrades better than the flat-string form did: a reply
truncated mid-array still contains complete `{"span": ..., "name": ...}` objects,
and the scanner recovers them.
"""

import hashlib
import json
import os

from .prompt_mdace import _item_lists, _iter_json_values, _norm_key

# Keys that may carry the note-verbatim phrase, in priority order.
_SPAN_KEYS = ("span", "text", "phrase", "mention", "evidence", "quote",
              "verbatim", "snippet", "surface")

# Keys that may carry the standard clinical name.
_NAME_KEYS = ("name", "standard_name", "standard", "concept", "canonical",
              "normalized", "term", "expansion", "full_name")

# Recorded only, never scored — the same rule as the term-NER parser.
_TYPE_KEYS = ("type", "label", "category", "entity_type", "class", "kind")


def _first_str(item, keys):
    wanted = [_norm_key(k) for k in keys]
    lowered = {_norm_key(k): v for k, v in item.items()}
    for want in wanted:
        val = lowered.get(want)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def parse_findings_diag(reply):
    """Parse a reply into ``(findings, diag)``.

    A finding is ``{"span": str, "name": str}``; either may be empty, and a
    finding with both empty is dropped. Duplicates are kept — the caller pools
    and dedupes across a note's chunks.

    `diag` records what happened, so an under-extracting run reports a reason
    rather than a silent zero:

      shape        which strategy matched, or "no-json"/"no-item-list"/
                   "empty-reply"/"salvaged-truncated"
      n_items      raw items seen
      n_kept       items yielding usable text
      n_no_text    items with no usable string
      n_span_only  findings the model gave no standard name for
      n_name_only  findings the model gave no note-verbatim span for
      empty_list   the model explicitly returned an empty list
      types        {type string: count}, diagnostics only
    """
    diag = {"shape": "no-json", "n_items": 0, "n_kept": 0, "n_no_text": 0,
            "n_span_only": 0, "n_name_only": 0, "n_bare_string": 0,
            "empty_list": False, "types": {}, "n_salvaged": 0}

    if not isinstance(reply, str) or not reply.strip():
        diag["shape"] = "empty-reply"
        return [], diag

    out, saw_list = [], False
    for value in _iter_json_values(reply):
        diag["shape"] = "json"
        for items in _item_lists(value):
            saw_list = True
            if not items:
                diag["empty_list"] = True
            for item in items:
                diag["n_items"] += 1

                # A bare string is still usable: it is a span with no name, and
                # a span alone matches the evidence-text source perfectly well.
                if isinstance(item, str):
                    if item.strip():
                        out.append({"span": item.strip(), "name": ""})
                        diag["n_kept"] += 1
                        diag["n_bare_string"] += 1
                        diag["n_span_only"] += 1
                    else:
                        diag["n_no_text"] += 1
                    continue

                if not isinstance(item, dict):
                    diag["n_no_text"] += 1
                    continue

                span = _first_str(item, _SPAN_KEYS)
                name = _first_str(item, _NAME_KEYS)
                if span is None and name is None:
                    # Field-keyed object using none of the recognized keys: fall
                    # back to any other string value rather than discard a real
                    # extraction. TYPE keys are excluded — without that guard
                    # {"type": "Condition"} yields the finding "Condition",
                    # a false positive manufactured out of a schema word.
                    type_keys = {_norm_key(k) for k in _TYPE_KEYS}
                    for key, val in item.items():
                        if _norm_key(key) in type_keys:
                            continue
                        if isinstance(val, str) and val.strip():
                            span = val.strip()
                            break
                if span is None and name is None:
                    diag["n_no_text"] += 1
                    continue

                etype = _first_str(item, _TYPE_KEYS)
                if etype:
                    diag["types"][etype[:60]] = diag["types"].get(etype[:60], 0) + 1
                if span is None:
                    diag["n_name_only"] += 1
                elif name is None:
                    diag["n_span_only"] += 1
                out.append({"span": span or "", "name": name or ""})
                diag["n_kept"] += 1

    if diag["shape"] == "json" and not saw_list:
        diag["shape"] = "no-item-list"

    # Last resort, and only when nothing at all was recovered. The object form
    # rarely needs this — the scanner above already recovers complete objects
    # from a truncated array — but a reply that never opened an object still
    # carries usable quoted strings before the cut.
    if not out:
        from .prompt_mdace import _salvage_truncated_strings

        salvaged = [s for s in _salvage_truncated_strings(reply)
                    if _norm_key(s) not in {_norm_key(k) for k in
                                            _SPAN_KEYS + _NAME_KEYS}]
        if salvaged:
            out = [{"span": s, "name": ""} for s in salvaged]
            diag["shape"] = "salvaged-truncated"
            diag["n_items"] += len(salvaged)
            diag["n_kept"] += len(salvaged)
            diag["n_span_only"] += len(salvaged)
            diag["n_salvaged"] = len(salvaged)

    return out, diag


def parse_findings(reply):
    """Parse a reply into a list of findings (diagnostics discarded)."""
    return parse_findings_diag(reply)[0]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise clinical information-extraction system. You read "
    "hospital notes and return structured JSON. You always return valid JSON "
    "and never add commentary."
)

# SYNTHETIC example — no MIMIC note content appears in this file. It shows all
# three requested categories, a de-identification placeholder being ignored, and
# the span/name split doing the one job it exists for: HTN is copied as written
# AND expanded, so the same finding can match either the note's wording or the
# billing catalogue's.
_EXAMPLE_INPUT = (
    "Mr. [**Known lastname 1234**] is a [**Age over 90 **] year old man with "
    "HTN and type 2 diabetes who presented with chest pain. He underwent "
    "cardiac catheterization on [**2145-6-7**]. He sustained a fall at home "
    "with a fracture of the left wrist. Tobacco history: quit smoking 20 years "
    "ago. Continued on aspirin 81mg daily for secondary prevention. His SBPs "
    "ran 90-110 and his hematocrit was 24."
)

# Five things in the example input are deliberately absent from this output:
# "aspirin 81mg daily", the tobacco history, "SBPs" and "hematocrit". A negative
# example is the strongest available signal that a category is out of scope — a
# version that merely omitted medications still drew 33% medication items — and
# the first smoke run showed the same leak for the rest: it extracted `SBPs` as
# "systolic blood pressure", `pRBCs` as "packed red blood cells" and the bowel
# prep `GoLYTEly`, all of which rules 1 and 2 already forbade in the abstract.
# Abstract prohibitions do not work on a 4B model; demonstrated ones do.
_EXAMPLE_OUTPUT = json.dumps({"findings": [
    {"span": "HTN", "name": "hypertension"},
    {"span": "type 2 diabetes", "name": "type 2 diabetes mellitus"},
    {"span": "chest pain", "name": "chest pain"},
    {"span": "cardiac catheterization", "name": "cardiac catheterization"},
    {"span": "fall", "name": "fall"},
    {"span": "fracture of the left wrist", "name": "fracture of left wrist"},
]}, indent=None)

_SCOPED = (
    "Extract every DIAGNOSIS, PROCEDURE and INJURY from the clinical text "
    "below.\n"
    "\n"
    "Extract all three of these:\n"
    "  - a disease, diagnosis, symptom, or clinical finding "
    "(e.g. sepsis, atrial fibrillation, chest pain, acute kidney injury)\n"
    "  - a procedure performed on the patient "
    "(e.g. colonoscopy, intubation, CABG, cardiac catheterization)\n"
    "  - an injury, fracture, wound, burn, poisoning, or overdose "
    "(e.g. fall, hip fracture, laceration)\n"
    "\n"
    "Rules:\n"
    "0. Return a JSON object with ONE key, \"findings\", whose value is a list "
    "of objects. Each object has EXACTLY two keys:\n"
    "     \"span\" - the phrase copied from the text, character for character, "
    "exactly as written. Do not reword it, do not expand it, do not correct "
    "its spelling.\n"
    "     \"name\" - the standard clinical name for that same finding, spelled "
    "out in full. Expand abbreviations here. If the text says \"HTN\", the "
    "span is \"HTN\" and the name is \"hypertension\".\n"
    "   If the phrase in the text is already the standard name, repeat it in "
    "both fields.\n"
    "1. Do NOT extract medications, drug names, doses, IV fluids, bowel "
    "preparations, or blood products. A medication list is not a finding. If "
    "the note says \"Continued on aspirin 81mg daily\", extract nothing from "
    "it. GoLYTELY and pRBCs are substances given to the patient, not "
    "findings.\n"
    "2. Do NOT extract vital signs or lab values. SBP, blood pressure, heart "
    "rate, temperature, hematocrit, creatinine and white count are measurements, "
    "not findings. Extract the diagnosis they support only if the note names it "
    "— extract \"anemia\" if the note says anemia, but never \"hematocrit\".\n"
    "3. Do NOT extract social history, family history, or allergies.\n"
    "4. Do NOT extract a body part or a location on its own. \"right kidney\" "
    "is not a finding; \"right renal mass\" is.\n"
    "5. Keep the span SHORT — the specific words naming the finding, not the "
    "whole sentence around it. Most spans are one to three words.\n"
    "6. Text in square brackets like [**Known lastname 1234**] or "
    "[**2145-6-7**] is removed patient information. Never extract it and never "
    "include it inside a span.\n"
    "7. Extract from the WHOLE text, including narrative prose and past "
    "medical history, not only from headed lists.\n"
    "8. ONE entry per distinct finding. Do not repeat a finding you have "
    "already listed, even if the text mentions it again.\n"
    "9. When you have listed every distinct finding once, close the JSON and "
    "STOP. Do not start the list again from the beginning.\n"
    "10. Return ONLY the JSON object — no prose, no markdown fences, no "
    "explanation.\n"
    "\n"
    "Example input:\n"
    f"{_EXAMPLE_INPUT}\n"
    "\n"
    "Example output:\n"
    f"{_EXAMPLE_OUTPUT}\n"
    "\n"
    "Note that the example input mentions aspirin, a tobacco history, SBPs and "
    "a hematocrit, and none of them appears in the example output. That is "
    "correct — they are not diagnoses, procedures or injuries. Note also that "
    "the output lists each finding exactly once and then stops.\n"
    "\n"
    'If there are genuinely no findings, return {"findings": []}.\n'
    "\n"
    "Now do the same for this text.\n"
    "Text:\n"
)


# --------------------------------------------------------------------------
# Variant B — one positive criterion instead of four exclusions
#
# The scoped variant above grows by one rule every time a run finds a new leak:
# medications, then vital signs, then lab values, then blood products, then bare
# anatomy. That is a bug-fix log, not a specification, and there is no principle
# in it that says when to stop — the list of things a clinical note contains and
# a coder does not bill is effectively unbounded.
#
# This variant tests the alternative: replace all four exclusions with the
# criterion that actually defines gold. MDACE evidence is the phrase a coder
# highlighted to justify a code they submitted, so "would a coder assign a code
# to this?" is not a proxy for the target, it IS the target. A vital sign has no
# code. A lab value has no code. A drug's name has no code.
#
# The risk runs the wrong way, which is why this is measured rather than
# assumed: recall is the metric, and a model with a weak grasp of billability
# under-extracts. That is the one failure this benchmark punishes hardest. It is
# worth testing now precisely because the scoped variant over-extracts by ~17x,
# so there is room to lose volume before scarcity binds.
#
# The worked example stays in both variants and is NOT the thing under test. An
# example costs O(1) — it demonstrates a boundary without enumerating it — while
# rules cost O(leaks found). That distinction is the whole point of the
# comparison.
_BILLABLE = (
    "You are reading this note the way a medical coder reads it.\n"
    "\n"
    "Extract every finding in the clinical text below that a medical coder "
    "would assign a billing code to — an ICD-10 diagnosis code, or a CPT or "
    "ICD-10-PCS procedure code.\n"
    "\n"
    "A finding qualifies if a coder could look it up in a code book:\n"
    "  - a disease, diagnosis, symptom, or clinical finding "
    "(e.g. sepsis, atrial fibrillation, chest pain, acute kidney injury)\n"
    "  - a procedure performed on the patient "
    "(e.g. colonoscopy, intubation, CABG, cardiac catheterization)\n"
    "  - an injury, fracture, wound, burn, poisoning, or overdose "
    "(e.g. fall, hip fracture, laceration)\n"
    "\n"
    "**If you cannot name the code a coder would assign to it, it is not a "
    "finding.** Apply that test to every candidate before you write it down.\n"
    "\n"
    "Rules:\n"
    "0. Return a JSON object with ONE key, \"findings\", whose value is a list "
    "of objects. Each object has EXACTLY two keys:\n"
    "     \"span\" - the phrase copied from the text, character for character, "
    "exactly as written. Do not reword it, do not expand it, do not correct "
    "its spelling.\n"
    "     \"name\" - the standard clinical name for that same finding, spelled "
    "out in full. Expand abbreviations here. If the text says \"HTN\", the "
    "span is \"HTN\" and the name is \"hypertension\".\n"
    "   If the phrase in the text is already the standard name, repeat it in "
    "both fields.\n"
    "1. Keep the span SHORT — the specific words naming the finding, not the "
    "whole sentence around it. Most spans are one to three words.\n"
    "2. Text in square brackets like [**Known lastname 1234**] or "
    "[**2145-6-7**] is removed patient information. Never extract it and never "
    "include it inside a span.\n"
    "3. Extract from the WHOLE text, including narrative prose and past "
    "medical history, not only from headed lists.\n"
    "4. ONE entry per distinct finding. Do not repeat a finding you have "
    "already listed, even if the text mentions it again.\n"
    "5. When you have listed every distinct finding once, close the JSON and "
    "STOP. Do not start the list again from the beginning.\n"
    "6. Return ONLY the JSON object — no prose, no markdown fences, no "
    "explanation.\n"
    "\n"
    "Example input:\n"
    f"{_EXAMPLE_INPUT}\n"
    "\n"
    "Example output:\n"
    f"{_EXAMPLE_OUTPUT}\n"
    "\n"
    "The example input mentions aspirin, a tobacco history, SBPs and a "
    "hematocrit. None of them appears in the output, because none of them is "
    "something a coder assigns a code to. Note also that the output lists each "
    "finding exactly once and then stops.\n"
    "\n"
    'If there are genuinely no findings, return {"findings": []}.\n'
    "\n"
    "Now do the same for this text.\n"
    "Text:\n"
)

# The two variants under comparison. `scoped` is what the first smoke run used;
# `billable` is the experiment. The run tag carries the prompt hash, so the two
# land in separate run directories and neither can replay the other's numbers.
# `billable` plus a hard limit on how many findings one call may return. The
# untried lever from the plan, and the direct attack on the repetition loop:
# 413 of 1,706 emitted items on the 24-note run were repeats WITHIN a single
# reply, and that looping is what put 21 of 82 chunks into the token cap.
#
# 25 is chosen against the measured distribution, not picked: the median chunk
# returned 17 findings and the maximum was 84. A cap of 25 leaves the median
# untouched and cuts only the tail, which is where the loop lives.
_BILLABLE_CAPPED = _BILLABLE.replace(
    "4. ONE entry per distinct finding. Do not repeat a finding you have "
    "already listed, even if the text mentions it again.\n",
    "4. ONE entry per distinct finding. Do not repeat a finding you have "
    "already listed, even if the text mentions it again.\n"
    "4b. Return AT MOST 25 findings. If the text contains more, return the 25 "
    "most specific ones and stop.\n",
)

VARIANTS = {"scoped": _SCOPED, "billable": _BILLABLE,
            "billable_capped": _BILLABLE_CAPPED}

# `billable` is the default on measurement, not taste. Head to head on the
# same 17 chunks it cut hallucinated spans from 9.22% to 1.02% (95% CIs
# 6.0-13.8 and 0.3-3.6, non-overlapping) at identical row recall, with
# slightly lower volume and slightly fewer false positives. `scoped` is kept
# runnable so the comparison can be repeated rather than taken on trust.
DEFAULT_VARIANT = os.environ.get("RECALL_PROMPT", "billable")


def instruction(variant=None):
    """The instruction text for `variant`, defaulting to RECALL_PROMPT."""
    name = variant or DEFAULT_VARIANT
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown prompt variant {name!r}; expected one of "
            f"{sorted(VARIANTS)}"
        ) from None


def prompt_fingerprint(variant=None):
    """Short hash of the prompt, for the run-directory name.

    The resume cache is keyed on the run directory, and every other input that
    changes the model's output — model id, chunk geometry, token cap — is
    already in that name. On the previous branch the prompt was not, so editing
    it and re-running silently replayed the old prompt's results: "cached 24, to
    run 0", numbers from the old prompt, no model call made, no error.
    """
    body = _SYSTEM + instruction(variant)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:8]


def build_prompt(chunk_text, variant=None):
    """The user-turn text for one note chunk."""
    return instruction(variant) + chunk_text


def build_messages(chunk_text, variant=None):
    """Text-only chat messages in MedGemma's expected structure."""
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user",
         "content": [{"type": "text", "text": build_prompt(chunk_text, variant)}]},
    ]
