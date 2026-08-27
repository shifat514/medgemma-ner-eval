"""ICD-10 code-assignment prompt and reply parsing for pediatric encounter notes.

THE TASK IS CODE ASSIGNMENT, NOT EXTRACTION, AND THE PROMPT SAYS SO. Every other
prompt in this repo asks the model to copy phrases out of a note; the downstream
phrase->code lookup was always someone else's step and was never built. This one
skips the phrase entirely and asks for the code, because that is the question
Ehtesham Bhai sent and because gold here is a code set.

WHY THE PROMPT ASKS FOR "ASSESSED AND ADDRESSED AT THIS VISIT". Gold is the
clinician's Assessment block: 2 to 6 codes per note, averaging 4. A note mentions
far more codeable things than that — the past medical history in note 26819
alone names asthma, a heart murmur, constipation and enuresis, none of which is
billed for this visit. A prompt that says "extract every codeable finding" would
be answering a different question and would score precision near zero for a
reason that is the prompt's fault, not the model's. So the prompt names the
actual criterion: what the clinician assessed and managed today.

That framing is a description of the task, not a hint at the answer. It says
nothing about which conditions, how many, or where in the note to look.

WHY THE DESCRIPTION IS REQUESTED BUT NOT SCORED. Scoring is exact code match and
nothing else — see billing_config. The description field exists for two other
reasons: it makes a wrong reply readable (a code with the description
"Influenza" next to it tells you the model found the condition and missed the
digit, which is a completely different failure from inventing a code), and
naming the thing before coding it is a small amount of free reasoning for a 4B
model. It costs ~10 tokens per code against a 512-token budget.

NO PROMPT VARIANTS, DELIBERATELY. The recall branch ran an A/B because the
question there was which of two framings extracted better. Here the experiment
is on the INPUT — three variants, one prompt — so that every difference between
the three numbers is attributable to the text that was removed. Adding a second
prompt would confound the two.
"""

import hashlib
import json
import re

from .prompt import _find_json_object

# Keys a reply might carry the code under, in priority order.
_CODE_KEYS = ("code", "icd10", "icd_10", "icd", "icd10_code", "icd_code",
              "diagnosis_code", "dx", "value")

_DESC_KEYS = ("description", "desc", "name", "diagnosis", "label", "text",
              "title", "condition")

_CONTAINER_KEYS = ("codes", "icd10_codes", "icd_codes", "diagnoses",
                   "diagnosis_codes", "assessment", "findings", "results",
                   "items", "output", "data")

# ICD-10-CM shape. Used to classify a returned code as well-formed or not; a
# malformed code is still counted as a false positive, never silently dropped.
_ICD10_RE = re.compile(r"^[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$")

# Pulls a code out of a bare string like "B08.5 Enteroviral vesicular pharyngitis"
# or "- J30.2: Other seasonal allergic rhinitis".
_LEADING_CODE_RE = re.compile(
    r"^[\s\-*•]*([A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?)\b\s*[:\-–]?\s*(.*)$",
    re.IGNORECASE,
)


_SYSTEM = (
    "You are an experienced medical coder. You read clinical notes and assign "
    "ICD-10-CM diagnosis codes. You always return valid JSON and never add "
    "commentary."
)

# SYNTHETIC. No content from the supplied notes appears here. It is a pediatric
# sick visit because the corpus is pediatric outpatient, and it demonstrates the
# one distinction the prompt most needs to land: the otitis media being treated
# today is coded, the eczema mentioned in passing in the history is not.
_EXAMPLE_INPUT = (
    "Visit Information\n"
    "Appointment type: SICK VISIT, EST\n"
    "\n"
    "CC/HPI\n"
    "3 days of fever and right ear pain. Pulling at the right ear.\n"
    "\n"
    "Patient History\n"
    "Past Medical History: eczema, well controlled, no flare in over a year.\n"
    "\n"
    "Vital Signs\n"
    "Temp: 101.4F\n"
    "\n"
    "Exam Findings\n"
    "Ears: ABNORMAL right TM erythematous and bulging. Left TM normal.\n"
    "\n"
    "Plan\n"
    "Amoxicillin 400mg/5mL. Recheck in 2 weeks if not improved."
)

_EXAMPLE_OUTPUT = json.dumps({"codes": [
    {"code": "H66.001",
     "description": "Acute suppurative otitis media without spontaneous "
                    "rupture of ear drum, right ear"},
]}, indent=None)


_INSTRUCTION = (
    "Read the clinical note below and assign the ICD-10-CM diagnosis codes for "
    "this encounter.\n"
    "\n"
    "Code what the clinician ASSESSED AND ADDRESSED AT THIS VISIT — the "
    "problems this encounter was about, the ones evaluated, treated, or "
    "managed today.\n"
    "\n"
    "Do NOT code:\n"
    "  - conditions named only as past history, family history, or a resolved "
    "problem that was not addressed today\n"
    "  - normal findings, negative test results, or symptoms the patient "
    "denies\n"
    "  - procedures or visit levels — no CPT codes, no E&M codes "
    "(99213, 99214 and the like). ICD-10-CM diagnosis codes only.\n"
    "\n"
    "Rules:\n"
    "1. Return a JSON object with ONE key, \"codes\", whose value is a list of "
    "objects. Each object has EXACTLY two keys:\n"
    "     \"code\"        - the ICD-10-CM code, with its decimal point, "
    "e.g. \"J06.9\" or \"S52.501A\".\n"
    "     \"description\" - the official description of that code.\n"
    "2. Code to the highest level of detail the note supports. If the note "
    "says which side, which episode, or which organism, the code must reflect "
    "it.\n"
    "3. ONE entry per code. Do not list the same code twice.\n"
    "4. Return ONLY the codes this encounter would be billed with. Most visits "
    "have between one and six.\n"
    "5. Return ONLY the JSON object — no prose, no markdown fences, no "
    "explanation.\n"
    "\n"
    "Example input:\n"
    f"{_EXAMPLE_INPUT}\n"
    "\n"
    "Example output:\n"
    f"{_EXAMPLE_OUTPUT}\n"
    "\n"
    "The example note mentions eczema, but it is past history that was not "
    "addressed at this visit, so it is not coded. The fever and the ear pain "
    "are symptoms of the otitis media, so they are not coded separately "
    "either.\n"
    "\n"
    'If there are genuinely no codeable diagnoses, return {"codes": []}.\n'
    "\n"
    "Now do the same for this note.\n"
    "Note:\n"
)


def instruction():
    """The instruction text. One prompt only — see the module docstring."""
    return _INSTRUCTION


def prompt_fingerprint():
    """Short hash of system + instruction, for the run-directory name.

    The resume cache is keyed on the run directory. On the recall branch the
    prompt was NOT in that name, so editing the prompt and re-running silently
    replayed the old prompt's results with no model call and no error. Same
    mistake, same cost, so the same guard.
    """
    return hashlib.sha256((_SYSTEM + _INSTRUCTION).encode("utf-8")).hexdigest()[:8]


def build_prompt(note_text):
    """The user-turn text for one note."""
    return _INSTRUCTION + note_text


def build_messages(note_text):
    """Text-only chat messages in MedGemma's expected structure."""
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user",
         "content": [{"type": "text", "text": build_prompt(note_text)}]},
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _first_str(item, keys):
    for key in keys:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, (int, float)):
            return str(val)
    return None


def _from_bare_string(text):
    """``"B08.5 Enteroviral vesicular pharyngitis"`` -> ``("B08.5", "Enteroviral ...")``."""
    m = _LEADING_CODE_RE.match(text.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return text.strip(), ""


def parse_codes(reply):
    """Parse a MedGemma reply into ``[{"code":..., "description":..., "well_formed":...}]``.

    DELIBERATELY PERMISSIVE ABOUT SHAPE, DELIBERATELY STRICT ABOUT CONTENT. Any
    JSON shape that carries a code string is accepted — bare strings, the wrong
    container key, the wrong field name — because losing a real prediction to a
    formatting quirk would understate precision AND recall at once.

    But a returned code that is not shaped like ICD-10-CM (a CPT code such as
    "99213", a description with no code, an invented string) is KEPT and flagged
    ``well_formed: False``, never dropped. Dropping it would quietly delete a
    false positive and inflate precision. The evaluator counts it against the
    model and reports the malformed count separately.

    Returns ``[]`` on a reply that yields no parseable JSON at all.
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

    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in _CONTAINER_KEYS:
            val = data.get(key)
            if isinstance(val, list):
                items = val
                break
        # A single object with a code field, returned unwrapped.
        if items is None and _first_str(data, _CODE_KEYS):
            items = [data]
    if items is None:
        return []

    out, seen = [], set()
    for item in items:
        if isinstance(item, str):
            code, desc = _from_bare_string(item)
        elif isinstance(item, dict):
            code = _first_str(item, _CODE_KEYS)
            desc = _first_str(item, _DESC_KEYS) or ""
            if code is None:
                continue
            # "code": "B08.5 Enteroviral vesicular pharyngitis" happens too.
            if not _ICD10_RE.match(code.strip().upper()):
                salvaged, tail = _from_bare_string(code)
                if _ICD10_RE.match(salvaged.upper()):
                    code, desc = salvaged, desc or tail
        else:
            continue

        code = code.strip().upper().rstrip(".,;")
        if not code:
            continue
        if code in seen:
            continue
        seen.add(code)
        out.append({
            "code": code,
            "description": desc,
            "well_formed": bool(_ICD10_RE.match(code)),
        })
    return out
