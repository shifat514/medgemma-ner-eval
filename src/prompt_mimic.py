"""Medication-NER prompt and tolerant reply parsing for MIMIC-IV note chunks.

The first version of this module reused ``prompt.parse_entities`` directly and
accepted exactly one reply shape: ``{"entities":[{"text":..,"type":..}]}`` with
the type spelled exactly as one of the six labels. A smoke run scored micro
F1=0.136 against an 0.870 oracle ceiling, with `Duration` and `Reason` at exactly
0.0000 — the signature of a categorical silent drop, not a hard type.

Probing the old parser with 22 reply shapes a 4B instruct model realistically
produces showed **17 yielded zero entities**, including:

  - ``"type": "Reason for use"`` / ``"Duration of therapy"`` / ``"Indication"``
    / ``"Route"``            -> dropped, no log
  - a bare top-level array   -> dropped wholesale (data was a list, not a dict)
  - one JSON object per line -> dropped wholesale (no "entities" key)
  - grouped ``{"medication":..,"dose":..}`` objects -> dropped wholesale
  - ``span``/``label``, ``entity``/``category``, ``value``/``entity_type`` keys
                             -> dropped

So this module now owns medication-specific parsing: it accepts those shapes,
maps common type synonyms onto the six labels, and — critically — **reports what
it rejected** instead of silently returning []. ``prompt.parse_entities`` is left
untouched so the NCBI/BC5CDR evaluation cannot regress.

Tolerance here is normalization, not scoring leniency: a span still has to match
the gold span token-for-token to count. Accepting ``"Route"`` as ``Mode`` only
stops us from throwing away an answer the model got right.
"""

import json
import re

from .mimic_config import ENTITY_TYPES
from .prompt import _find_json_object

_VALID_TYPES = set(ENTITY_TYPES)

# Synonyms a model reaches for instead of the six exact labels. Keys are
# normalized (lowercase, non-alphanumerics collapsed to single spaces).
_TYPE_ALIASES = {
    # Medication
    "drug": "Medication", "drugs": "Medication", "medicine": "Medication",
    "med": "Medication", "meds": "Medication", "medications": "Medication",
    "drug name": "Medication", "medication name": "Medication",
    "fluid": "Medication", "infusion": "Medication", "name": "Medication",
    # Dose
    "dosage": "Dose", "amount": "Dose", "strength": "Dose", "quantity": "Dose",
    "dose amount": "Dose", "doses": "Dose",
    # Mode
    "route": "Mode", "administration": "Mode", "route of administration": "Mode",
    "mode of administration": "Mode", "method": "Mode", "delivery": "Mode",
    "administration route": "Mode",
    # Frequency
    "freq": "Frequency", "schedule": "Frequency", "interval": "Frequency",
    "how often": "Frequency", "timing": "Frequency",
    # Duration
    "duration of therapy": "Duration", "duration of treatment": "Duration",
    "length": "Duration", "how long": "Duration", "period": "Duration",
    # Reason
    "indication": "Reason", "reason for use": "Reason", "condition": "Reason",
    "diagnosis": "Reason", "indication for use": "Reason", "purpose": "Reason",
    "reason for administration": "Reason", "why": "Reason",
}

# Keys that may hold the span text / the type, in priority order.
_TEXT_KEYS = ("text", "span", "entity", "value", "mention", "string", "name")
_TYPE_KEYS = ("type", "label", "category", "entity_type", "entity type",
              "tag", "class")

# Field-keyed ("grouped") replies: {"medication": "Lasix", "dose": "40mg", ...}
_GROUPED_KEYS = {}
for _t in ENTITY_TYPES:
    _GROUPED_KEYS[_t.lower()] = _t
for _alias, _canon in _TYPE_ALIASES.items():
    _GROUPED_KEYS.setdefault(_alias, _canon)

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm_type(raw):
    return _NORM_RE.sub(" ", str(raw).strip().lower()).strip()


def normalize_type(raw):
    """Map a model-emitted type string onto one of the six labels, or None.

    Tries, in order: exact label, alias table, then a containment check so
    ``"reason for administration (indication)"`` still resolves to ``Reason``.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):           # {"type": ["Medication"]}
        raw = raw[0] if raw else None
        if raw is None:
            return None
    norm = _norm_type(raw)
    if not norm:
        return None

    for label in ENTITY_TYPES:                   # exact, case-insensitive
        if norm == label.lower():
            return label
    if norm in _TYPE_ALIASES:
        return _TYPE_ALIASES[norm]

    words = set(norm.split())
    for label in ENTITY_TYPES:                   # "reason for use" -> Reason
        if label.lower() in words:
            return label
    for alias, canon in _TYPE_ALIASES.items():
        if " " not in alias and alias in words:
            return canon
    return None


def _iter_json_objects(reply):
    """Every top-level JSON value in `reply`, tolerating JSONL and prose.

    Yields parsed values. Handles: a single object, a bare array, several objects
    concatenated or newline-separated, and a fenced block inside prose.
    """
    text = reply.strip()

    for cand in (text, _strip_fence(text)):
        if not cand:
            continue
        try:
            yield json.loads(cand)
            return
        except (json.JSONDecodeError, ValueError):
            pass

    # Scan for consecutive balanced objects/arrays (covers JSONL and prose).
    found = False
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        ch = text[i]
        if ch not in "{[":
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


def _strip_fence(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else ""


_CONTAINER_KEYS = ("entities", "medications", "results", "items", "extractions",
                   "output", "data", "annotations")


def _entity_lists(value):
    """Pull candidate entity-item lists out of a parsed JSON value.

    A value that carries a recognized container list is NOT also treated as a
    bare item. Doing both double-counted grouped replies like
    ``{"medications": [...]}`` — "medications" is itself a grouped key — which
    inflated the kept count above the emitted count and invented phantom
    "no usable span text" drops.
    """
    if isinstance(value, list):
        yield value
        return
    if not isinstance(value, dict):
        return

    found_container = False
    for key in _CONTAINER_KEYS:
        got = value.get(key)
        if isinstance(got, list):
            found_container = True
            yield got
    if found_container:
        return

    # A single bare item: {"text": .., "type": ..} or a grouped object.
    if any(k in value for k in _TEXT_KEYS) or any(
        _norm_type(k) in _GROUPED_KEYS for k in value
    ):
        yield [value]


def _flat_pairs(item, _depth=0):
    """``(key, value)`` pairs, descending one level into nested dicts.

    Covers ``{"text":.., "type":.., "attributes": {"dose": "40mg"}}``, where the
    attributes would otherwise be invisible.
    """
    for key, val in item.items():
        if isinstance(val, dict) and _depth < 2:
            yield from _flat_pairs(val, _depth + 1)
        else:
            yield key, val


def _first_str(item, keys):
    for key in keys:
        for actual in item:
            if _norm_type(actual) == _norm_type(key):
                val = item[actual]
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return None


def parse_entities_diag(reply):
    """Parse a reply into ``(entities, diag)``.

    `entities` is a list of ``(text, type)``. `diag` records what happened so a
    run can report *why* it under-extracted instead of silently scoring zero:

      shape           which strategy matched, or "no-json"/"no-entity-list"
      n_items         raw items seen
      n_kept          items kept
      rejected_types  {type string: count} for items dropped on type
      aliased_types   {type string: count} for items rescued by normalization
      n_no_text       items with no usable span text
      empty_list      the model explicitly returned an empty list
    """
    diag = {"shape": "no-json", "n_items": 0, "n_kept": 0,
            "rejected_types": {}, "aliased_types": {}, "n_no_text": 0,
            "empty_list": False}

    if not isinstance(reply, str) or not reply.strip():
        diag["shape"] = "empty-reply"
        return [], diag

    out = []
    saw_list = False
    for value in _iter_json_objects(reply):
        diag["shape"] = "json"
        for items in _entity_lists(value):
            saw_list = True
            if not items:
                diag["empty_list"] = True
            for item in items:
                if not isinstance(item, dict):
                    # A bare string in the list: no type, unusable.
                    if isinstance(item, str) and item.strip():
                        diag["n_items"] += 1
                        diag["n_no_text"] += 1
                    continue
                diag["n_items"] += 1

                raw_text = _first_str(item, _TEXT_KEYS)
                raw_type = None
                for key in _TYPE_KEYS:
                    for actual in item:
                        if _norm_type(actual) == _norm_type(key):
                            raw_type = item[actual]
                            break
                    if raw_type is not None:
                        break

                if raw_text is not None and raw_type is not None:
                    etype = normalize_type(raw_type)
                    if etype is None:
                        key = str(raw_type)[:80]
                        diag["rejected_types"][key] = \
                            diag["rejected_types"].get(key, 0) + 1
                        continue
                    if _norm_type(raw_type) != etype.lower():
                        key = str(raw_type)[:80]
                        diag["aliased_types"][key] = \
                            diag["aliased_types"].get(key, 0) + 1
                    out.append((raw_text, etype))
                    diag["n_kept"] += 1
                    continue

                # Grouped/field-keyed object: {"medication": "Lasix", ...}, and
                # the nested variant {"text":.., "type":.., "attributes":{..}}.
                got_any = False
                for actual, val in _flat_pairs(item):
                    label = _GROUPED_KEYS.get(_norm_type(actual))
                    if not label:
                        continue
                    values = val if isinstance(val, list) else [val]
                    for sub in values:
                        if isinstance(sub, str) and sub.strip():
                            out.append((sub.strip(), label))
                            diag["n_kept"] += 1
                            got_any = True
                if got_any:
                    diag["shape"] = "json-grouped"
                    continue

                if raw_text is None:
                    diag["n_no_text"] += 1
                elif raw_type is None:
                    diag["rejected_types"]["<missing type>"] = \
                        diag["rejected_types"].get("<missing type>", 0) + 1

    if diag["shape"] == "json" and not saw_list:
        diag["shape"] = "no-entity-list"
    return out, diag


def parse_entities(reply):
    """Parse a reply against the six medication types (diagnostics discarded)."""
    return parse_entities_diag(reply)[0]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise clinical information-extraction system. You read "
    "hospital discharge summaries and return structured JSON. You always return "
    "valid JSON and never add commentary."
)

# A worked example matters more than prose for a 4B model: it pins the exact
# output shape and shows all six types at once. The example text is SYNTHETIC —
# no MIMIC note content appears in this file.
_EXAMPLE_INPUT = (
    "Discharge Medications:\n"
    "1. Drugzol 250 mg PO BID for 10 days for wound infection\n"
    "2. Placebolol 5 mg IV once daily\n"
    "He was given Fictamol 650 mg by mouth q6h PRN for pain."
)

_EXAMPLE_OUTPUT = json.dumps({"entities": [
    {"text": "Drugzol", "type": "Medication"},
    {"text": "250 mg", "type": "Dose"},
    {"text": "PO", "type": "Mode"},
    {"text": "BID", "type": "Frequency"},
    {"text": "for 10 days", "type": "Duration"},
    {"text": "wound infection", "type": "Reason"},
    {"text": "Placebolol", "type": "Medication"},
    {"text": "5 mg", "type": "Dose"},
    {"text": "IV", "type": "Mode"},
    {"text": "once daily", "type": "Frequency"},
    {"text": "Fictamol", "type": "Medication"},
    {"text": "650 mg", "type": "Dose"},
    {"text": "by mouth", "type": "Mode"},
    {"text": "q6h PRN", "type": "Frequency"},
    {"text": "pain", "type": "Reason"},
]}, indent=None)

_INSTRUCTION = (
    "Extract EVERY medication mention and every medication detail from the "
    "clinical text.\n"
    "\n"
    "Use exactly these six type names, spelled exactly like this:\n"
    '  "Medication" - a drug, medication, IV fluid, or infusion '
    "(e.g. aspirin, Lasix, NS, insulin, vancomycin)\n"
    '  "Dose"       - how much (e.g. 325 mg, 10mg, 1000 mL, 2 tabs, 5 units, 3L)\n'
    '  "Mode"       - the route (e.g. IV, PO, IM, SC, by mouth, topical, '
    "inhaled, per rectum)\n"
    '  "Frequency"  - how often (e.g. daily, BID, TID, q6h, PRN, once, '
    "every morning, at bedtime)\n"
    '  "Duration"   - how long (e.g. for 7 days, x2 weeks, overnight, '
    "for 3 more days)\n"
    '  "Reason"     - the condition or indication it is given for '
    "(e.g. hypertension, for pain, CHF, aspiration pneumonia)\n"
    "\n"
    "Do NOT invent other type names. Do NOT use Drug, Dosage, Route, "
    "Indication, or Frequency-of-use — use only the six names above.\n"
    "\n"
    "Rules:\n"
    "1. Copy each span VERBATIM from the text, character for character. Do not "
    "reword, expand abbreviations, or fix spelling.\n"
    "2. Return ONE FLAT list. Do not group a drug with its dose — emit the drug "
    "and the dose as separate entries.\n"
    "3. Extract from the WHOLE text, including narrative sentences and the "
    "hospital-course prose, not only numbered medication lists.\n"
    "4. Be exhaustive. A section of this length usually contains 10 to 40 "
    "entities. Listing too few is a failure.\n"
    "5. Return ONLY the JSON object — no prose, no markdown fences, no "
    "explanation.\n"
    "\n"
    "Example input:\n"
    f"{_EXAMPLE_INPUT}\n"
    "\n"
    "Example output:\n"
    f"{_EXAMPLE_OUTPUT}\n"
    "\n"
    'If there are genuinely no medications, return {"entities": []}.\n'
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
