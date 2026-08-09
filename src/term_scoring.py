"""Term-set scoring for the MDACE evaluation: micro P/R/F1, FP buckets, code recall.

No positions, no BIO, no seqeval. Per note, gold and prediction are each a SET of
normalized terms; TP/FP/FN are set operations. This is the right unit because
4,770 of 9,499 gold terms recur inside their own note — under position scoring,
"aspirin" said once by the model against 15 occurrences in the note needs an
arbitrary policy, and under set scoring the question does not arise.

Micro averaging pools every note's counts before dividing, so a note with 30 gold
terms carries more weight than one with 2. Macro (score each note, average the
scores) would weight them equally and is not what the spec asks for.

THE PRECISION CAVEAT, MADE NUMERIC. MDACE annotates evidence for codes that were
actually BILLED, not for every condition documented. An Inpatient note runs ~1,113
words against ~6 gold terms, so a model that correctly lists 40 real conditions
scores precision 6/40 = 0.15 — the score a *perfect* extractor gets. Two devices
keep that legible instead of looking like model failure:

  precision_ceiling  the best micro precision achievable given how many terms
                     were predicted: sum(min(|gold|,|pred|)) / sum(|pred|).
  fp_buckets         every false positive split three ways (see classify_fp).

Neither is applied to the headline score. Dropping "correct but unbilled" false
positives before dividing would raise precision by construction and measure
nothing, so they stay counted and are reported alongside.
"""

import math

from .datasets.mdace import normalize_term

# ---------------------------------------------------------------------------
# Gold selectors — a "view" is a set of notes plus one of these.
# ---------------------------------------------------------------------------

GOLD_KEYS = ("full", "shipped", "descr", "conditions")


def note_gold(record, gold_key="full"):
    """The gold term set for one note under `gold_key`.

    full        every evidence phrase annotated on this note (the correct key)
    shipped     only the phrases Ehtesham Bhai's 100-row cut happens to carry —
                99 of the 195 those 24 notes really have
    descr       the ICD catalogue wording from his shipped rows, NOT what the
                note says. Used by view A1 only, to measure what that choice
                costs: catalogue and note wording agree in 4.5% of rows.
    conditions  the 84% slice that is a plain disease — no procedures, injuries,
                or status/history codes
    """
    if gold_key == "full":
        return set(record.get("gold_terms") or ())
    if gold_key == "shipped":
        return set(record.get("sample100_terms") or ())
    if gold_key == "descr":
        return set(record.get("sample100_descrs") or ())
    if gold_key == "conditions":
        return {g["norm"] for g in record.get("gold", ())
                if g.get("bucket") == "conditions"}
    raise ValueError(f"gold_key must be one of {GOLD_KEYS}, got {gold_key!r}")


# ---------------------------------------------------------------------------
# Core set scoring
# ---------------------------------------------------------------------------

def score_note(gold, pred):
    """``(tp, fp, fn)`` as sets, for one note."""
    gold, pred = set(gold), set(pred)
    return gold & pred, pred - gold, gold - pred


def micro_prf(tp, fp, fn):
    """Micro precision, recall, F1 from pooled counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def wilson_ci(successes, total, z=1.96):
    """95% Wilson score interval for a rate.

    Reported next to every rate because the samples are small: recall over 110
    gold terms carries roughly +/-9 points, over 214 roughly +/-7. Without this,
    a 4-point gap between strata reads as a finding when it is noise. Wilson
    rather than the textbook normal interval because it stays inside [0, 1] and
    behaves at rates near 0 or 1, which is exactly where precision will land.
    """
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# False-positive classification
# ---------------------------------------------------------------------------

def classify_fp(term, note_norm, admission_gold, note_gold_terms):
    """Bucket one false positive. Returns a bucket name.

    billed_elsewhere  the term is gold evidence somewhere else in the SAME
                      admission. An ICD code belongs to an admission but MDACE
                      marks its evidence on whichever note the coder read, so
                      this is a correct extraction of a genuinely billed finding
                      that was annotated on a different note. The strongest
                      evidence that a false positive is not a model error.
    in_note_unbilled  the term really is in this note's text but nothing in the
                      admission was billed against it. A correct extraction of
                      something nobody billed.
    not_in_note       the term is not in the note at all: the model paraphrased,
                      expanded an abbreviation, or invented it. A real error,
                      and the only bucket that is a genuine model-quality signal.
    """
    if term in admission_gold and term not in note_gold_terms:
        return "billed_elsewhere"
    if term and f" {term} " in note_norm:
        return "in_note_unbilled"
    return "not_in_note"


def padded_note_norm(text):
    """Normalized note text, space-padded so substring tests hit word edges.

    Normalization collapses every run of non-alphanumerics to a single space, so
    a space-padded ``in`` test is a whole-token containment check: "ca" will not
    match inside "cabg".
    """
    return f" {normalize_term(text)} "


# ---------------------------------------------------------------------------
# Code-level recall
# ---------------------------------------------------------------------------

def code_coverage(record, pred):
    """``(covered, total)`` billed codes whose evidence was extracted.

    A code counts as covered if ANY of the phrases annotated for it on this note
    was predicted — one code is often evidenced by two phrasings ("third degree
    heart block" and "Complete Heart Block" both carry I44.2), and finding
    either one recovers the code.

    This is the ceiling on the downstream term -> ICD lookup: it can never
    retrieve a code whose evidence phrase was never extracted. Measured on the
    same notes as the NER, so unlike an end-to-end code-prediction run it
    separates the extraction step from the lookup step.
    """
    by_code = {}
    for g in record.get("gold", ()):
        by_code.setdefault((g["code_system"], g["code"]), set()).add(g["norm"])
    covered = sum(1 for terms in by_code.values() if terms & pred)
    return covered, len(by_code)


# ---------------------------------------------------------------------------
# View scoring
# ---------------------------------------------------------------------------

def score_view(pairs, gold_key="full", with_fp_buckets=True):
    """Score one view. `pairs` is an iterable of ``(record, predicted_terms)``.

    `predicted_terms` is a set of normalized strings. Returns a metrics dict;
    everything in it is an integer count or a derived rate, so it is safe to
    write to the committed report.
    """
    tp = fp = fn = 0
    n_gold = n_pred = 0
    best_possible_tp = 0
    codes_covered = codes_total = 0
    buckets = {"billed_elsewhere": 0, "in_note_unbilled": 0, "not_in_note": 0}
    per_note = []

    for record, pred in pairs:
        gold = note_gold(record, gold_key)
        pred = set(pred)
        hit, miss_fp, miss_fn = score_note(gold, pred)

        tp += len(hit)
        fp += len(miss_fp)
        fn += len(miss_fn)
        n_gold += len(gold)
        n_pred += len(pred)
        best_possible_tp += min(len(gold), len(pred))

        if with_fp_buckets and miss_fp:
            note_norm = padded_note_norm(record.get("text", ""))
            adm = set(record.get("admission_gold_terms") or ())
            for term in miss_fp:
                buckets[classify_fp(term, note_norm, adm, gold)] += 1

        cov, tot = code_coverage(record, pred)
        codes_covered += cov
        codes_total += tot

        per_note.append({
            "note_id": record["note_id"],
            "chart_type": record.get("chart_type"),
            "n_gold": len(gold),
            "n_pred": len(pred),
            "tp": len(hit),
            "fp": len(miss_fp),
            "fn": len(miss_fn),
        })

    precision, recall, f1 = micro_prf(tp, fp, fn)
    p_lo, p_hi = wilson_ci(tp, tp + fp)
    r_lo, r_hi = wilson_ci(tp, tp + fn)

    return {
        "n_notes": len(per_note),
        "n_gold": n_gold,
        "n_pred": n_pred,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "precision_ci": [p_lo, p_hi],
        "recall_ci": [r_lo, r_hi],
        # Best micro precision reachable given how many terms were predicted.
        # If this is 0.15, no extractor of any quality scores above 0.15 here.
        "precision_ceiling": (best_possible_tp / n_pred) if n_pred else 0.0,
        # Terms predicted per gold term. 6.7 means the model listed nearly seven
        # times what the answer key holds — the shape of the annotation-scope
        # problem, visible as a ratio.
        "extraction_ratio": (n_pred / n_gold) if n_gold else 0.0,
        "pred_per_note": (n_pred / len(per_note)) if per_note else 0.0,
        "gold_per_note": (n_gold / len(per_note)) if per_note else 0.0,
        "fp_buckets": buckets,
        "codes_covered": codes_covered,
        "codes_total": codes_total,
        "code_recall": (codes_covered / codes_total) if codes_total else 0.0,
        "code_recall_ci": list(wilson_ci(codes_covered, codes_total)),
        "per_note": per_note,
    }


def score_by_chart_type(pairs, gold_key="full"):
    """``{chart_type: metrics}`` — Profee and Inpatient scored separately.

    Never pooled. Precision means different things in the two strata: Profee
    notes are short and densely billed, Inpatient notes are long and sparsely
    billed, so a pooled precision is an average of two incomparable quantities.
    """
    pairs = list(pairs)
    out = {}
    for chart_type in ("Profee", "Inpatient"):
        subset = [(r, p) for r, p in pairs if r.get("chart_type") == chart_type]
        if subset:
            out[chart_type] = score_view(subset, gold_key)
    return out
