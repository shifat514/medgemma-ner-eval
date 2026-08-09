"""Billing-evidence extraction prompt and reply parsing for MDACE note chunks.

TYPE LABELS ARE NOT SCORED. Scoring compares normalized term strings only, so a
term labelled "Procedure" that gold treats as a condition still counts as a hit.
That makes this parser deliberately far more permissive than
``prompt_mimic.parse_entities_diag``, which had to reject items with an
unrecognized type because a wrong type there was both a false positive and a
still-missed false negative. Here the only failure mode is losing usable text,
so anything carrying a string is kept. Types are recorded for diagnostics and
never consulted by the scorer.

WHY THE PROMPT ASKS FOR MORE THAN DIAGNOSES — AND WHERE IT STOPS. Only 84% of
gold rows are plain conditions; the rest are procedures (CPT 3.6%, ICD-10-PCS
2.9%), injuries and poisonings (S/T codes, 4.0%), and status or history (Z
codes, 5.5%). Asking only for diagnoses would cap recall at 0.84 before the
model did anything, and that cap would read as a model weakness in the report.

Status/history is deliberately NOT requested, and that is a measured decision
rather than an oversight. An earlier version of this prompt asked for it, with
"a medication the patient is maintained on" as an example, to reach the Z79.82
long-term-drug-therapy codes. On a 15-chunk smoke run the model returned 138
Status items — 33% of everything it extracted — for a category worth 5.5% of
gold, because a discharge summary lists every drug the patient is on. The
resulting flood truncated 12 of 15 replies against a 512-token cap and pushed
extraction to 68.6 distinct terms per note against ~10 gold.

So the prompt asks for Conditions, Procedures and Injuries, which reach 94.5% of
gold, and explicitly tells the model NOT to extract medications. The example
output demonstrates that by leaving the medication and tobacco lines in the
example input unextracted — a negative example teaches this far better than an
omitted category does. The 5.5% ceiling is documented in the report.
"""

import json
import re

from .prompt import _find_json_object

# Keys that may carry the span text, in priority order.
_TEXT_KEYS = ("text", "term", "span", "entity", "value", "mention", "phrase",
              "string", "name", "evidence", "keyword")

# Keys that may carry a type. Recorded only; never scored.
_TYPE_KEYS = ("type", "label", "category", "entity_type", "entity type", "tag",
              "class", "kind")

_CONTAINER_KEYS = ("terms", "entities", "findings", "conditions", "results",
                   "items", "extractions", "output", "data", "annotations",
                   "diagnoses", "spans")

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm_key(raw):
    return _NORM_RE.sub(" ", str(raw).strip().lower()).strip()


def _strip_fence(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _iter_json_values(reply):
    """Every top-level JSON value in `reply`, tolerating JSONL and prose."""
    text = reply.strip()

    for cand in (text, _strip_fence(text)):
        if not cand:
            continue
        try:
            yield json.loads(cand)
            return
        except (json.JSONDecodeError, ValueError):
            pass

    found = False
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        yield value
        found = True
        i = end

    if not found:
        obj = _find_json_object(reply)
        if obj:
            try:
                yield json.loads(obj)
            except (json.JSONDecodeError, ValueError):
                pass


def _item_lists(value):
    """Pull candidate item lists out of a parsed JSON value."""
    if isinstance(value, list):
        yield value
        return
    if not isinstance(value, dict):
        return

    found = False
    for key in _CONTAINER_KEYS:
        got = value.get(key)
        if isinstance(got, list):
            found = True
            yield got
    if found:
        return

    # A single bare item: {"text": "...", "type": "..."}
    if any(_norm_key(k) in {_norm_key(t) for t in _TEXT_KEYS} for k in value):
        yield [value]


def _first_str(item, keys):
    wanted = [_norm_key(k) for k in keys]
    for want in wanted:
        for actual, val in item.items():
            if _norm_key(actual) == want and isinstance(val, str) and val.strip():
                return val.strip()
    return None


def parse_terms_diag(reply):
    """Parse a reply into ``(terms, diag)``.

    `terms` is a list of ``(text, type_or_None)`` in emission order, duplicates
    kept — the caller normalizes and dedupes. `diag` records what happened so an
    under-extracting run reports a reason rather than a silent zero:

      shape       which strategy matched, or "no-json"/"no-item-list"/"empty-reply"
      n_items     raw items seen
      n_kept      items yielding usable text
      n_no_text   items with no usable string
      empty_list  the model explicitly returned an empty list
      types       {type string: count}, diagnostics only
    """
    diag = {"shape": "no-json", "n_items": 0, "n_kept": 0, "n_no_text": 0,
            "empty_list": False, "types": {}}

    if not isinstance(reply, str) or not reply.strip():
        diag["shape"] = "empty-reply"
        return [], diag

    out = []
    saw_list = False
    for value in _iter_json_values(reply):
        diag["shape"] = "json"
        for items in _item_lists(value):
            saw_list = True
            if not items:
                diag["empty_list"] = True
            for item in items:
                diag["n_items"] += 1

                # A bare string in the list is perfectly usable here: the type
                # carries no scoring weight, so "depression" alone is a complete
                # prediction. The medication parser had to drop these.
                if isinstance(item, str):
                    if item.strip():
                        out.append((item.strip(), None))
                        diag["n_kept"] += 1
                    else:
                        diag["n_no_text"] += 1
                    continue

                if not isinstance(item, dict):
                    diag["n_no_text"] += 1
                    continue

                text = _first_str(item, _TEXT_KEYS)
                if text is None:
                    # Field-keyed object with no recognized text key: fall back
                    # to any other string value rather than discard a real
                    # extraction. TYPE keys are excluded — without that guard
                    # {"type": "Condition"} yields the term "Condition", which
                    # is a guaranteed false positive manufactured out of a
                    # schema word.
                    type_keys = {_norm_key(k) for k in _TYPE_KEYS}
                    for key, val in item.items():
                        if _norm_key(key) in type_keys:
                            continue
                        if isinstance(val, str) and val.strip():
                            text = val.strip()
                            break
                if text is None:
                    diag["n_no_text"] += 1
                    continue

                etype = _first_str(item, _TYPE_KEYS)
                if etype:
                    diag["types"][etype[:60]] = diag["types"].get(etype[:60], 0) + 1
                out.append((text, etype))
                diag["n_kept"] += 1

    if diag["shape"] == "json" and not saw_list:
        diag["shape"] = "no-item-list"
    return out, diag


def parse_terms(reply):
    """Parse a reply into a list of ``(text, type)`` (diagnostics discarded)."""
    return parse_terms_diag(reply)[0]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise clinical information-extraction system. You read "
    "hospital notes and return structured JSON. You always return valid JSON "
    "and never add commentary."
)

# SYNTHETIC example — no MIMIC note content appears in this file. It shows all
# four categories at once, a de-identification placeholder being ignored, and an
# abbreviation copied as written rather than expanded.
_EXAMPLE_INPUT = (
    "Mr. [**Known lastname 1234**] is a [**Age over 90 **] year old man with "
    "HTN and type 2 diabetes who presented with chest pain. He underwent "
    "cardiac catheterization on [**2145-6-7**]. He sustained a fall at home "
    "with a fracture of the left wrist. Tobacco history: quit smoking 20 years "
    "ago. Continued on aspirin 81mg daily for secondary prevention."
)

# Note what is ABSENT: "aspirin 81mg daily" and the tobacco line appear in the
# example input and are deliberately NOT in this output. A negative example is
# the strongest signal available that medication lists are out of scope — an
# earlier version that merely omitted the category still saw the model return
# 33% medication items.
_EXAMPLE_OUTPUT = json.dumps({"terms": [
    {"text": "HTN", "type": "Condition"},
    {"text": "type 2 diabetes", "type": "Condition"},
    {"text": "chest pain", "type": "Condition"},
    {"text": "cardiac catheterization", "type": "Procedure"},
    {"text": "fall", "type": "Injury"},
    {"text": "fracture of the left wrist", "type": "Injury"},
]}, indent=None)

_INSTRUCTION = (
    "Extract every DIAGNOSIS, PROCEDURE and INJURY from the clinical text "
    "below.\n"
    "\n"
    "Use exactly these three categories:\n"
    '  "Condition" - a disease, diagnosis, symptom, or clinical finding '
    "(e.g. sepsis, atrial fibrillation, chest pain, acute kidney injury)\n"
    '  "Procedure" - something that was performed on the patient '
    "(e.g. colonoscopy, intubation, CABG, cardiac catheterization)\n"
    '  "Injury"    - an injury, fracture, wound, burn, poisoning, or overdose '
    "(e.g. fall, hip fracture, laceration)\n"
    "\n"
    "Rules:\n"
    "1. Do NOT extract medications, drug names, doses, or IV fluids. A "
    "medication list is not a finding. If the note says \"Continued on aspirin "
    "81mg daily\", extract nothing from it.\n"
    "2. Do NOT extract social history, family history, vital signs, lab values, "
    "or allergies.\n"
    "3. Copy each term VERBATIM from the text, character for character. Do not "
    "reword, do not expand abbreviations, do not correct spelling. If the text "
    "says \"HTN\", return \"HTN\", not \"hypertension\". If the text misspells "
    "a word, copy the misspelling.\n"
    "4. Keep each term SHORT — the specific words naming the finding, not the "
    "whole sentence around it. Most terms are one to three words.\n"
    "5. Text in square brackets like [**Known lastname 1234**] or "
    "[**2145-6-7**] is removed patient information. Never extract it and never "
    "include it inside a term.\n"
    "6. Extract from the WHOLE text, including narrative prose and past "
    "medical history, not only from headed lists.\n"
    "7. Return ONE FLAT list, and do not repeat the same term twice.\n"
    "8. Return ONLY the JSON object — no prose, no markdown fences, no "
    "explanation.\n"
    "\n"
    "Example input:\n"
    f"{_EXAMPLE_INPUT}\n"
    "\n"
    "Example output:\n"
    f"{_EXAMPLE_OUTPUT}\n"
    "\n"
    "Note that the example input mentions aspirin and a tobacco history, and "
    "neither appears in the example output. That is correct — they are not "
    "diagnoses, procedures or injuries.\n"
    "\n"
    'If there are genuinely no findings, return {"terms": []}.\n'
    "\n"
    "Now do the same for this text.\n"
    "Text:\n"
)


def build_prompt(chunk_text):
    """The user-turn text for one note chunk."""
    return _INSTRUCTION + chunk_text


def build_messages(chunk_text):
    """Text-only chat messages in MedGemma's expected structure."""
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": build_prompt(chunk_text)}],
        },
    ]
