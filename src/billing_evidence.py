"""What each gold code is evidenced by, in the note text. Hand-built answer key.

WHY THIS EXISTS. The code evaluation says MedGemma recovered 0 of 16 billed
codes. That number alone cannot tell "it never found the condition" apart from
"it found the condition and could not code it", and those two point at
completely different fixes — a different model versus a code-book lookup. This
file is the key that separates them.

BUILT FROM THE NOTES, BEFORE ANY EXTRACTION OUTPUT WAS LOOKED AT. That ordering
is the only thing keeping it from being fitted to whatever the model happened to
say. Every entry cites the line it came from.

WHAT AN ENTRY MEANS. `terms` are strings whose presence in the input means a
reader could plausibly reach that diagnosis. They are NOT the code's official
description — "Enteroviral vesicular pharyngitis" never appears in note 112976;
"coxsackie" does. Matching the catalogue wording would measure whether the model
memorised ICD, which is the thing already known to fail.

THE FINDING THAT FELL OUT OF BUILDING IT. Four of the sixteen gold codes have no
supporting text at all once the Problem List is removed:

    D18.00  hemangioma        note 112976 — Problem List and Assessment only
    L20.9   atopic dermatitis note 26819  — Problem List and Assessment only;
                              the word "eczema" appears nowhere in the corpus
    J30.2   allergic rhinitis note 55688  — Problem List only (note 26819 also
                              carries it in Past Medical History, so that one
                              survives; 55688 does not)
    R06.2   wheezing          note 55688  — Problem List only. The one other
                              mention is the ROS line "Denies wheezing or
                              difficulty breathing", which is evidence AGAINST.

So `leakage_cut` had a structural ceiling of 12 of 16, not 16 of 16, and the
reported 0.0000 recall was measured against a target four codes of which were
unreachable by any reader. That does not rescue the result — the model also
missed all twelve that WERE supported — but the ceiling belongs in the report,
and `evaluate_billing`'s numbers were published without it.

These four are chronic problems carried forward on the chart. A real coder bills
them from the Problem List, which is exactly why decision 3 refused to call that
section pure contamination. Removing it removed real clinical evidence along
with the leaked code strings, and this file is what measures that cost.
"""

import re

# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------
#
# (note_id, code) -> {condition, terms, source, note}
#
#   condition  what a reader would call it, in plain words
#   terms      strings that evidence it; matched case-insensitively against
#              whitespace-normalized text, because the PDF wraps lines mid-phrase
#              ("Allergic\n rhinitis or other allergy" in note 26819)
#   source     where in the note the evidence lives
#   ceiling    False when the ONLY source is the Problem List or the Assessment,
#              i.e. the code is unreachable under leakage_cut

EVIDENCE = {
    # --- 112976: sick visit, coxsackie -------------------------------------
    ("112976", "B08.5"): {
        "condition": "coxsackie / non-polio enterovirus",
        "terms": ("coxsackie", "enterovirus", "hand-foot-mouth",
                  "posterior pharyngeal erythema", "perioral rash"),
        "source": "Plan ('Discussed natural hx of coxsackie illness'), "
                  "Patient Instructions ('a Coxsackie viral infection'), "
                  "Exam ('mild posterior pharyngeal erythema', "
                  "'early rash soles/perioral rash')",
        "ceiling": True,
    },
    ("112976", "D18.00"): {
        "condition": "hemangioma",
        "terms": ("hemangioma",),
        "source": "Problem List ('- HEMANGIOMA', 'Hemangioma on left arm') "
                  "and the Assessment. NOWHERE ELSE.",
        "ceiling": False,
    },

    # --- 26819: well visit --------------------------------------------------
    ("26819", "Z00.121"): {
        "condition": "well-child visit with abnormal findings",
        "terms": ("well visit", "13-14 year old well visit", "preventive exam"),
        "source": "Visit Information ('Appointment type: WELL VISIT, EST'), "
                  "Interval History, Plan ('Well 13-14 year old')",
        "ceiling": True,
    },
    ("26819", "Z68.52"): {
        "condition": "BMI 5th to <85th percentile",
        "terms": ("bmi", "17.8", "24 %ile"),
        "source": "Vital Signs ('BMI: 17.8 (24 %ile)')",
        "ceiling": True,
    },
    ("26819", "J30.2"): {
        "condition": "seasonal allergic rhinitis",
        "terms": ("allergic rhinitis", "rhinitis or other allergy",
                  "nasal allergies"),
        "source": "Past Medical History ('Allergic rhinitis or other allergy' "
                  "— line-wrapped in the PDF) as well as the Problem List",
        "ceiling": True,
    },
    ("26819", "L20.9"): {
        "condition": "atopic dermatitis / eczema",
        "terms": ("atopic dermatitis", "eczema"),
        "source": "Problem List ('- L20.9 ATOPIC DERMATITIS, UNSPECIFIED') and "
                  "the Assessment. 'eczema' appears NOWHERE in the corpus.",
        "ceiling": False,
    },
    ("26819", "Z55.3"): {
        "condition": "underachievement in school",
        "terms": ("iep", "underachievement", "doing well in school",
                  "school performance"),
        "source": "Past Medical History ('IEP for reading, writing, math'), "
                  "ROS ('Denies: doing well in school')",
        "ceiling": True,
    },
    ("26819", "R41.840"): {
        "condition": "attention and concentration deficit / ADHD",
        "terms": ("adhd", "add/adhd", "attention", "concentration",
                  "vanderbilt"),
        "source": "Interval History ('Parental concerns: concern for ADHD'), "
                  "Past Medical History ('ADHD, IEP for math, reading, "
                  "writing')",
        "ceiling": True,
    },

    # --- 55688: sick visit, influenza + wrist -------------------------------
    ("55688", "J11.1"): {
        "condition": "influenza",
        "terms": ("influenza", "flu b+", "flu-like illness", "flu illness"),
        "source": "Diagnostic Tests ('INFLUENZA B: POSITIVE'), Plan "
                  "('Flu B+'), Patient Instructions ('FLU ILLNESS')",
        "ceiling": True,
    },
    ("55688", "Z68.52"): {
        "condition": "BMI 5th to <85th percentile",
        "terms": ("bmi", "19.2", "84 %ile"),
        "source": "Vital Signs ('BMI: 19.2 (84 %ile)')",
        "ceiling": True,
    },
    ("55688", "J30.2"): {
        "condition": "seasonal allergic rhinitis",
        "terms": ("allergic rhinitis", "seasonal allergic"),
        "source": "Problem List ('- SEASONAL ALLERGIC RHINITIS') ONLY. The ROS "
                  "reports nasal congestion, which is a symptom of the acute "
                  "flu here rather than evidence of a chronic allergy.",
        "ceiling": False,
    },
    ("55688", "R06.2"): {
        "condition": "wheezing",
        "terms": ("wheezing", "albuterol"),
        "source": "Problem List ('- WHEEZING', 'Albuterol PRN') ONLY. The one "
                  "other mention is ROS 'Denies wheezing or difficulty "
                  "breathing' — evidence AGAINST, not for.",
        "ceiling": False,
    },
    ("55688", "S52.501A"): {
        "condition": "distal radius (wrist) fracture, initial encounter",
        "terms": ("buckle fracture", "fracture", "wrist injury",
                  "wrist in brace", "orthopedic"),
        "source": "CC/HPI ('found to have a buckle fracture'), Exam ('right "
                  "wrist in brace'), Orders ('Right hand buckle fracture')",
        "ceiling": True,
    },

    # --- 96176: sick visit, viral + underweight -----------------------------
    ("96176", "B97.89"): {
        "condition": "viral illness of unspecified agent",
        "terms": ("viral illness", "viral syndrome", "virus",
                  "no evidence of bacterial illness"),
        "source": "Patient Instructions ('VIRAL SYNDROME', 'seen today for a "
                  "viral illness'), Plan ('No evidence of bacterial illness')",
        "ceiling": True,
    },
    ("96176", "R63.6"): {
        "condition": "underweight / poor weight gain",
        "terms": ("weight", "underweight", "pediasure",
                  "not gain adequate weight", "high fat high quality food"),
        "source": "CC/HPI ('follow up on his weight'), Plan ('Discussed "
                  "increasing high fat high quality food', 'if continues to "
                  "not gain adequate weight')",
        "ceiling": True,
    },
    ("96176", "Z68.51"): {
        "condition": "BMI below 5th percentile",
        "terms": ("bmi", "13.8", "(low)"),
        "source": "Vital Signs ('BMI: 13.8 (Low)')",
        "ceiling": True,
    },
}

_WS_RE = re.compile(r"\s+")

# A term inside the scope of one of these is evidence AGAINST the diagnosis, and
# counting it would overstate what a reader could reach.
#
# THIS IS NOT DEFENSIVE CODING, IT CHANGED A NUMBER. Note 55688's Problem List
# carries "- WHEEZING"; strip it for `leakage_cut` and the only surviving
# mention is the ROS line "Denies wheezing or difficulty breathing". Without
# negation handling the ceiling read 13 of 16 and R06.2 looked reachable from a
# sentence saying the patient does not have it.
#
# These notes are ROS-heavy and most of the ROS is negative, so this is the
# common case here rather than an edge one.
_NEGATION_CUES = (
    "denies", "denied", "negative for", "no evidence of", "without",
    "not ", " no ",
)

# Negation reaches to the end of its clause: "Denies nausea, vomiting,
# diarrhea." negates all three. So scope runs back to the previous clause break.
#
# NEWLINES ARE NOT AVAILABLE HERE, WHICH COST A DEBUGGING ROUND. `normalize`
# collapses all whitespace so that the PDF's mid-phrase line wraps stop breaking
# term matches — which means by the time this runs there are no newlines left to
# split on. Scoping on "[.;\\n]" therefore ran back through the entire Problem
# List into the ROS, found "denies", and reported note 55688's J30.2 and R06.2
# as unreachable in `assessment_cut` when both are printed in its Problem List.
#
# What survives normalization is the list bullet, "\\n  - SEASONAL" becoming
# " - seasonal". That plus sentence and label punctuation is the break set, and
# the lookback is bounded so a clause can never span half a note.
_CLAUSE_BREAK_RE = re.compile(r"[.;:]|\s-\s")

_NEGATION_WINDOW = 150


def _negated(haystack, term):
    """True when EVERY occurrence of `term` in `haystack` sits under a negation.

    One positive mention is enough to make a diagnosis reachable, so a term is
    only discounted when it never appears outside a negated clause.
    """
    start = 0
    while True:
        idx = haystack.find(term, start)
        if idx == -1:
            return True
        window_start = max(0, idx - _NEGATION_WINDOW)
        breaks = [m.end() for m in
                  _CLAUSE_BREAK_RE.finditer(haystack, window_start, idx)]
        clause = haystack[(breaks[-1] if breaks else window_start):idx]
        if not any(cue in clause for cue in _NEGATION_CUES):
            return False          # an un-negated mention: real evidence
        start = idx + 1


def normalize(text):
    """Lowercase and collapse whitespace.

    The collapse is load-bearing, not cosmetic: the PDF wraps phrases across
    lines, so note 26819's Past Medical History reads "Allergic\\n rhinitis or
    other allergy". Matching "allergic rhinitis" without this finds nothing and
    would report a code as unreachable when its evidence is right there.
    """
    return _WS_RE.sub(" ", str(text).lower())


def evidence_for(note_id, code):
    """The key entry, or None if this code was never catalogued."""
    return EVIDENCE.get((str(note_id), str(code)))


def present_in(text, note_id, code):
    """Which of the code's evidence terms appear in `text`, un-negated.

    A term mentioned only inside "Denies ..." does not count — see `_negated`.
    """
    entry = evidence_for(note_id, code)
    if entry is None:
        return []
    haystack = normalize(text)
    return [t for t in entry["terms"]
            if normalize(t) in haystack and not _negated(haystack, normalize(t))]


def reachable(text, note_id, code):
    """Could a reader reach this diagnosis from `text` at all?"""
    return bool(present_in(text, note_id, code))


def ceiling(records, variant):
    """How many gold codes are evidenced in `variant`'s input, per note and total.

    This is the number `evaluate_billing` should have been quoting recall
    against. Under `leakage_cut` it is 12 of 16, not 16 of 16 — see the module
    docstring.
    """
    per_note, total, unreachable = {}, 0, []
    for rec in records:
        text = rec["variants"][variant]
        hits = []
        for code in rec["gold_codes"]:
            if reachable(text, rec["note_id"], code):
                hits.append(code)
            else:
                unreachable.append((rec["note_id"], code))
        per_note[rec["note_id"]] = hits
        total += len(hits)
    return {"per_note": per_note, "n_reachable": total,
            "unreachable": unreachable}


def matched_by(phrases, note_id, code):
    """Which extracted `phrases` evidence this code.

    A phrase counts if an evidence term is inside it or it is inside an evidence
    term — "influenza" against the extracted "Influenza B positive" has to match,
    and so does the extracted "flu" against "flu b+". Substring both ways is
    crude and it is the right amount of crude: the question is whether the model
    surfaced the condition, not whether it worded it well.
    """
    entry = evidence_for(note_id, code)
    if entry is None:
        return []
    terms = [normalize(t) for t in entry["terms"]]
    out = []
    for phrase in phrases:
        p = normalize(phrase)
        if not p:
            continue
        if any(t in p or (len(p) > 3 and p in t) for t in terms):
            out.append(phrase)
    return out
