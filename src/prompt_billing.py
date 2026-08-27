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

# SYNTHETIC. No content from the supplied notes appears here, and that is
# enforced by more than good intentions — see the note on contamination below.
#
# It is a pediatric sick visit because the corpus is pediatric outpatient, and
# it demonstrates the two distinctions the prompt most needs to land: the
# conditions treated today are coded, the lactose intolerance sitting in the
# history is not, and the itching is a symptom of what was already coded rather
# than a finding of its own.
#
# THE FIRST VERSION OF THIS EXAMPLE CONTAMINATED THE RESULTS. It was an otitis
# media visit coded H66.001. One of the four notes opens with the chief
# complaint "Ear infection, fever, ear pulling" — and H66.001 came back as a
# false positive on that note in every variant and every generation config,
# despite its exam recording normal canals and TMs.
#
# The shape leaked as well as the code. Under the repetition-penalty run, 10 of
# the 16 false positives carried three digits after the decimal point — H66.001,
# Z00.001, B95.001, R17.001, F20.901 — against only 2 of 16 gold codes shaped
# that way. A single example teaches its format, not only its content.
#
# So: two conditions rather than one, closer to the 2-6 codes these notes
# actually carry; two different code shapes (".00" and ".4") so neither is the
# obvious template; and clinical content chosen to be absent from all four
# notes. `tests/test_billing_prompt_hygiene.py` re-checks that absence against
# the built sample, so the next person to edit this cannot reintroduce the
# problem silently.
_EXAMPLE_INPUT = (
    "Visit Information\n"
    "Appointment type: SICK VISIT, EST\n"
    "\n"
    "CC/HPI\n"
    "Crusted sores around the chin for four days, and a scaly ring on the left "
    "forearm for two weeks. Both itchy. No fever.\n"
    "\n"
    "Patient History\n"
    "Past Medical History: lactose intolerance, diet-controlled.\n"
    "\n"
    "Vital Signs\n"
    "Temp: 98.4F\n"
    "\n"
    "Exam Findings\n"
    "Skin: ABNORMAL honey-crusted lesions over the chin. ABNORMAL annular "
    "scaly patch with central clearing on the left forearm.\n"
    "\n"
    "Plan\n"
    "Mupirocin ointment to the face three times daily for seven days.\n"
    "Topical clotrimazole to the forearm twice daily for three weeks."
)

_EXAMPLE_OUTPUT = json.dumps({"codes": [
    {"code": "L01.00", "description": "Impetigo, unspecified"},
    {"code": "B35.4", "description": "Tinea corporis"},
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
    "The example note mentions lactose intolerance, but it is past history that "
    "was not addressed at this visit, so it is not coded. The itching is a "
    "symptom of the two skin conditions, so it is not coded separately "
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


def _iter_json_objects(text):
    """Yield every balanced ``{...}`` substring in `text`, left to right.

    ``prompt._find_json_object`` returns only the first one. That is the right
    behaviour for a complete reply and the wrong behaviour for a truncated one —
    see ``_salvage_truncated``.
    """
    i, n = 0, len(text)
    while True:
        start = text.find("{", i)
        if start == -1:
            return

        depth, in_str, esc, end = 0, False, False, -1
        for j in range(start, n):
            c = text[j]
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
                    end = j
                    break

        if end == -1:
            # Unbalanced from here — in a truncated reply this is the OUTER
            # object, whose closing brace was never written. Advance one
            # character and try the next brace rather than giving up, which is
            # what reaches the complete code objects nested inside it.
            i = start + 1
        else:
            yield text[start:end + 1]
            i = end + 1


def _salvage_truncated(reply):
    """Recover every COMPLETE code object from a reply that was cut off mid-array.

    THIS IS WHERE THE FIRST RUN LOST ITS NUMBERS, so it is worth being explicit
    about the failure. A reply that hits the token cap looks like:

        {"codes": [{"code": "J11.1", ...}, {"code": "R06.2", ...}, {"code": "S52.

    The outer object never closes. ``_find_json_object`` scans for the first
    BALANCED ``{...}``, fails on the outer one, moves to the next ``{`` — and
    returns the first *code* object, which parses cleanly as a dict with a code
    field. So the old path did not error and did not return empty: it returned
    exactly one code and silently discarded every other complete one.

    On the 2026-08-27 run that hit 6 of 12 replies, and every one of them scored
    as a single prediction. ``leakage_cut`` read 0.0000 across the board from
    four notes whose answers were never actually read.

    A truncated reply is still a truncated reply — ``truncated`` stays flagged
    and the run reports it — but the codes the model did finish writing are now
    counted, in both directions: recall for the ones that are right, precision
    for the ones that are wrong.
    """
    out = []
    for chunk in _iter_json_objects(reply):
        try:
            obj = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and _first_str(obj, _CODE_KEYS):
            out.append(obj)
    return out


def parse_codes(reply):
    """Parse a MedGemma reply into ``[{"code":..., "description":..., "well_formed":...}]``.

    DELIBERATELY PERMISSIVE ABOUT SHAPE, DELIBERATELY STRICT ABOUT CONTENT. Any
    JSON shape that carries a code string is accepted — bare strings, the wrong
    container key, the wrong field name, a reply truncated mid-array — because
    losing a real prediction to a formatting quirk would understate precision
    AND recall at once.

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

    # A CONTAINER IS PROOF THE REPLY CLOSED; A BARE CODE OBJECT IS NOT. Reaching
    # here with no container means either the model returned one unwrapped code,
    # or the reply was cut off mid-array and the first code object is all that
    # balanced. Those are indistinguishable at this point and the second is the
    # common case, so salvage the whole reply rather than trusting the first hit.
    if items is None:
        salvaged = _salvage_truncated(reply)
        if salvaged:
            items = salvaged
        elif isinstance(data, dict) and _first_str(data, _CODE_KEYS):
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
