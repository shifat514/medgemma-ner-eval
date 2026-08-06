"""Medication-NER prompt for MIMIC-IV discharge-summary chunks.

Only the instruction text and the valid type set are new — the JSON extraction
and tolerant parsing come from src/prompt.py unchanged, via
``parse_entities(reply, valid_types=...)``.

The prompt describes the six i2b2-2009-style types the gold labels actually use
(MEDICATION/DOSE/MODE/FREQUENCY/DURATION/REASON), asks for verbatim spans, and
asks for a flat entity list rather than the grouped YAML the gold labels were
produced with — grouping is irrelevant to token-level scoring.
"""

from .mimic_config import ENTITY_TYPES
from .prompt import parse_entities as _parse_entities

_VALID_TYPES = set(ENTITY_TYPES)

_SYSTEM = (
    "You are a precise clinical named-entity recognition system that extracts "
    "medication information from hospital discharge summaries."
)

_INSTRUCTION = (
    "Extract every medication mention and its associated details from the "
    "clinical text below.\n"
    '- "Medication": the name of a drug, medication, infusion, or fluid.\n'
    '- "Dose": the amount given (e.g. "325 mg", "10mg", "1000 mL", "2 tabs").\n'
    '- "Mode": the route or method of administration (e.g. "IV", "PO", '
    '"subcutaneous", "by mouth").\n'
    '- "Frequency": how often it is given (e.g. "daily", "q6h", "twice a day", '
    '"PRN").\n'
    '- "Duration": how long it is given for (e.g. "for 7 days", "x2 weeks", '
    '"overnight").\n'
    '- "Reason": the condition or indication the medication is given for '
    '(e.g. "hypertension", "for pain").\n'
    "Return ONLY a JSON object, with no prose and no markdown fences, in exactly "
    "this form:\n"
    '{"entities": [{"text": "<exact span copied from the text>", "type": '
    '"Medication"}]}\n'
    'The "type" must be exactly one of "Medication", "Dose", "Mode", '
    '"Frequency", "Duration", "Reason". Copy each span verbatim from the text, '
    "character for character. Do not group entities and do not add commentary. "
    'If there are no entities, return {"entities": []}.\n\n'
    "Clinical text:\n"
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


def parse_entities(reply):
    """Parse a reply against the six medication types."""
    return _parse_entities(reply, valid_types=_VALID_TYPES)
