"""Prompt construction and robust JSON-reply parsing for zero-shot MedGemma NER.

MedGemma is generative, so we ask it to emit a strict JSON object and parse that
back into (text, type) spans. Parsing is deliberately forgiving: a malformed
reply yields no entities (equivalent to all-O) rather than crashing the run.
"""

import json
import re

from .config import ENTITY_TYPES

_VALID_TYPES = set(ENTITY_TYPES)

_SYSTEM = (
    "You are a precise clinical named-entity recognition system that extracts "
    "disease and chemical/drug mentions from biomedical text."
)

_INSTRUCTION = (
    "Extract every disease and chemical (drug) mention from the sentence below.\n"
    '- "Disease": diseases, disorders, syndromes, symptoms, and abnormal '
    "conditions.\n"
    '- "Chemical": drugs, chemicals, and other chemical substances.\n'
    "Return ONLY a JSON object, with no prose and no markdown fences, in exactly "
    "this form:\n"
    '{"entities": [{"text": "<exact span copied from the sentence>", "type": '
    '"Disease"}]}\n'
    'The "type" must be exactly "Disease" or "Chemical". Copy each span verbatim '
    'from the sentence. If there are no entities, return {"entities": []}.\n\n'
    "Sentence: "
)


def build_prompt(sentence):
    """The user-turn text for one sentence."""
    return _INSTRUCTION + sentence


def build_messages(sentence):
    """Text-only chat messages in MedGemma's expected structure."""
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": build_prompt(sentence)}]},
    ]


def _find_json_object(text):
    """Return the first balanced ``{...}`` substring in `text`, or None.

    String-aware so braces inside quoted values don't confuse the depth count.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_entities(reply):
    """Parse a MedGemma reply into a list of ``(text, type)`` tuples.

    Tolerates markdown fences and surrounding prose. Returns ``[]`` on any
    failure or malformed content. ``type`` is normalized to "Disease"/"Chemical";
    entries with any other type are dropped.
    """
    if not isinstance(reply, str) or not reply.strip():
        return []

    candidates = [reply.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)```", reply, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    obj = _find_json_object(reply)
    if obj:
        candidates.append(obj)

    data = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            break
        except (json.JSONDecodeError, ValueError):
            continue
    if not isinstance(data, dict):
        return []

    ents = data.get("entities")
    if not isinstance(ents, list):
        return []

    out = []
    for item in ents:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        etype = item.get("type")
        if not isinstance(text, str) or not isinstance(etype, str):
            continue
        text = text.strip()
        etype = etype.strip().capitalize()
        if text and etype in _VALID_TYPES:
            out.append((text, etype))
    return out
