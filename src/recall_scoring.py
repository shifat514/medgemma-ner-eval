"""Aggregate the ladder's per-note matchings into the benchmark's numbers.

RECALL IS NEVER QUOTED BARE. With loose matching and no volume control, a model
that lists every phrase in the note scores near 1.00, so a bare recall figure
cannot rank MedGemma against a larger model later — which is the entire purpose
of the benchmark. Every recall number here travels with the volume it was bought
with: false positives as a first-class count and rate, findings per note, and
the not-in-note (hallucination) rate.

THREE UNITS, ALL REPORTED. The file has 100 annotation rows, 91 distinct
`(note, code_system, code)` triples, and a different number of accepted forms per
source. They answer slightly different questions and will land close together;
reporting all three removes an argument rather than postponing one.

    forms   of that source's accepted phrasings, how many were matched
    rows    of the 100 billed rows, how many had at least one form matched
    codes   of the 91 distinct billed codes, likewise

ROWS AND CODES ARE THE QUOTABLE UNITS; FORMS IS A DIAGNOSTIC. A row is recalled
by matching any ONE of its four-odd accepted forms, and matching is 1:1, so a
model that recalls every row while offering one phrasing each leaves most forms
unmatched by construction. Combined form recall therefore has a ceiling set by
how many phrasings the model emits rather than by how much it found. It is
carried because the per-source breakdown is measured in forms and the two have
to reconcile.

PER-SOURCE FP IS EASY TO MISREAD, and the report says so in as many words. Each
source is scored by its own independent matching, so a prediction matching only
the catalogue wording is a hit on the description line and a false positive on
the evidence-text line — per-source FP means "matched nothing *in that source*".
Every individual line therefore reads high, and only the combined line counts
predictions that matched nothing anywhere.

DENOMINATORS DIFFER PER SOURCE AND ARE ALWAYS PRINTED. SNOMED ships terms for 53
of 100 rows; scoring SNOMED recall out of 100 would measure the file's coverage
and call it model performance.
"""

from .datasets.mdace_recall import (
    normalize_term,
    padded_note_norm,
    reachable_codes,
    reachable_rows,
    source_forms,
)
from .recall_config import COMBINED, COSINE_MIN, DICE_MIN, LEVELS, RATIO_MIN, SOURCES
from .recall_matching import candidate_strings, field_candidates, match

# One 10-line statistical function, unchanged from the term-NER branch. Imported
# rather than copied so the two sets of intervals cannot drift apart.
from .term_scoring import wilson_ci


def dedupe_findings(findings):
    """Pool a note's findings across chunks, keeping one per distinct pair.

    Deduped on ``(normalized span, normalized name)`` rather than on span alone:
    the oracle emits the same span under several standard names on purpose, and
    a model that genuinely reports two different findings sharing a span should
    not have one of them silently dropped. Overlapping windows re-read the same
    text, so without this the overlap would inflate volume.
    """
    seen, out = set(), []
    for finding in findings:
        cands = candidate_strings(finding)
        if not cands:
            continue
        span, name = finding.get("span") or "", finding.get("name") or ""
        key = (normalize_term(span), normalize_term(name))
        if key in seen:
            continue
        seen.add(key)
        out.append({"span": span, "name": name, "cands": cands})
    return out


def not_in_note(findings, note_text):
    """``(n_checked, n_missing)`` for the hallucination check.

    Only the `span` field is checkable — it is the one the prompt asks to be
    copied character for character. A finding the model gave no span for cannot
    be checked and is excluded from the denominator rather than counted as
    clean.
    """
    note_norm = padded_note_norm(note_text)
    checked = missing = 0
    for finding in findings:
        span = normalize_term(finding.get("span"))
        if not span:
            continue
        checked += 1
        if f" {span} " not in note_norm:
            missing += 1
    return checked, missing


def match_notes(records, preds, source=None, embedder=None, dice_min=DICE_MIN,
                ratio_min=RATIO_MIN, cosine_min=COSINE_MIN, levels=LEVELS,
                rejected=None, field="both"):
    """Run the ladder on every scored note. Returns ``{note_id: ladder}``.

    `preds` maps note_id to a list of deduped findings. `field` restricts which
    of a finding's two strings may match — see `field_candidates`.
    """
    out = {}
    for record in records:
        findings = preds.get(record["note_id"])
        if findings is None:
            continue
        # A rejected pair is removed as an EDGE, not as a finished match, so
        # the matcher can re-solve: a finding whose pairing the judge threw out
        # may legitimately match a different form, and deleting the assignment
        # instead of the edge would lose that.
        blocked = None
        if rejected:
            blocked = {
                (i, form)
                for i, f in enumerate(findings)
                for form in source_forms(record, source)
                if (record["note_id"], f["span"], f["name"], form) in rejected
            }
        cands = [f["cands"] if field == "both"
                 else field_candidates(f, field) for f in findings]
        out[record["note_id"]] = match(
            cands, source_forms(record, source),
            embedder=embedder, dice_min=dice_min, ratio_min=ratio_min,
            cosine_min=cosine_min, levels=levels, blocked=blocked,
        )
    return out


def _rate(hit, total):
    return (hit / total) if total else 0.0


def score_source(records, preds, ladders, source=None, levels=LEVELS):
    """``{level: metrics}`` for one gold source (``None`` = combined)."""
    scored = [r for r in records if r["note_id"] in ladders]
    out, previous = {}, None

    for level in levels:
        forms_total = forms_hit = 0
        rows_total = rows_hit = 0
        codes_total = codes_hit = 0
        n_pred = fp = 0
        fp_buckets = {"in_note_unbilled": 0, "not_in_note": 0, "no_span": 0}

        for record in scored:
            matched = ladders[record["note_id"]][level]["matched_forms"]
            pairs = ladders[record["note_id"]][level]["pairs"]

            forms_total += len(source_forms(record, source))
            forms_hit += len(matched)

            wanted_rows = reachable_rows(record, source)
            rows_total += len(wanted_rows)
            hit_codes = set()
            for entry in record["rows"]:
                if entry["row_id"] not in wanted_rows:
                    continue
                accept = {norm for norm, srcs in entry["accept"].items()
                          if source is None or source in srcs}
                if accept & matched:
                    rows_hit += 1
                    hit_codes.add(entry["code_key"])
            codes_total += len(reachable_codes(record, source))
            codes_hit += len(hit_codes)

            note_findings = preds[record["note_id"]]
            n_pred += len(note_findings)
            fp += len(note_findings) - len(pairs)

            # WHAT THE FALSE POSITIVES ARE, not just how many. MDACE marks
            # evidence only for codes that were actually BILLED, so a note is
            # full of real findings nobody billed. Counting those as model error
            # measures the annotation scope instead of the model. The split:
            #
            #   in_note_unbilled  the phrase really is in the note; nothing was
            #                     billed against it. A correct extraction.
            #   not_in_note       the phrase is not in the note at all. The
            #                     model invented or paraphrased it. REAL ERROR,
            #                     and the only bucket that judges the model.
            #   no_span           the model gave no verbatim span, so the claim
            #                     cannot be checked either way.
            note_norm = padded_note_norm(record.get("text", ""))
            for index, finding in enumerate(note_findings):
                if index in pairs:
                    continue
                span = normalize_term(finding.get("span"))
                if not span:
                    fp_buckets["no_span"] += 1
                elif f" {span} " in note_norm:
                    fp_buckets["in_note_unbilled"] += 1
                else:
                    fp_buckets["not_in_note"] += 1

        metrics = {
            "level": level,
            "n_notes": len(scored),
            "forms_total": forms_total, "forms_matched": forms_hit,
            "form_recall": _rate(forms_hit, forms_total),
            "form_recall_ci": list(wilson_ci(forms_hit, forms_total)),
            "rows_total": rows_total, "rows_matched": rows_hit,
            "row_recall": _rate(rows_hit, rows_total),
            "row_recall_ci": list(wilson_ci(rows_hit, rows_total)),
            "codes_total": codes_total, "codes_matched": codes_hit,
            "code_recall": _rate(codes_hit, codes_total),
            "code_recall_ci": list(wilson_ci(codes_hit, codes_total)),
            "n_pred": n_pred, "fp": fp, "fp_rate": _rate(fp, n_pred),
            "fp_buckets": fp_buckets,
            # The hallucination rate among false positives specifically. The
            # headline not-in-note rate is over ALL findings; this one is over
            # the ones that missed, which is what "are the false positives the
            # model's fault" actually asks.
            "fp_not_in_note_rate": _rate(fp_buckets["not_in_note"], fp),
            "gain_rows": rows_hit - (previous["rows_matched"] if previous else 0),
            "gain_forms": forms_hit - (previous["forms_matched"] if previous else 0),
        }
        out[level] = metrics
        previous = metrics
    return out


def new_pairs(records, preds, ladders, levels=LEVELS):
    """``{level: [pair records]}`` — what each level newly accepted.

    CONTAINS NOTE-DERIVED TEXT: these are the model's phrases beside the gold
    phrases they were matched to. They stay in the gitignored run dir, and they
    are what L5 adjudicates — running the judge on these rather than on
    everything is what keeps its cost bounded.
    """
    by_id = {r["note_id"]: r for r in records}
    out = {level: [] for level in levels}
    for note_id, ladder in ladders.items():
        record = by_id.get(note_id)
        if record is None:
            continue
        findings = preds[note_id]
        for level in levels:
            for index, form, rule, score in ladder[level]["new"]:
                finding = findings[index]
                slot = record["forms"].get(form, {})
                out[level].append({
                    "note_id": note_id,
                    "level": level,
                    "rule": rule,
                    "score": round(float(score), 4),
                    "span": finding.get("span", ""),
                    "name": finding.get("name", ""),
                    "gold_form": form,
                    "gold_sources": slot.get("sources", []),
                    "gold_codes": slot.get("codes", []),
                })
    for pairs in out.values():
        pairs.sort(key=lambda p: (p["note_id"], -p["score"], p["gold_form"]))
    return out


def volume(records, preds):
    """Findings per note, and the not-in-note rate. Recall's companion numbers."""
    scored = [r for r in records if r["note_id"] in preds]
    n_pred = sum(len(preds[r["note_id"]]) for r in scored)
    checked = missing = no_span = 0
    for record in scored:
        c, m = not_in_note(preds[record["note_id"]], record["text"])
        checked += c
        missing += m
        no_span += len(preds[record["note_id"]]) - c
    return {
        "n_notes": len(scored),
        "n_pred": n_pred,
        "pred_per_note": _rate(n_pred, len(scored)),
        "n_span_checked": checked,
        "n_not_in_note": missing,
        "not_in_note_rate": _rate(missing, checked),
        "not_in_note_ci": list(wilson_ci(missing, checked)),
        "n_no_span": no_span,
    }


def score_run(records, preds, embedder=None, dice_min=DICE_MIN,
              ratio_min=RATIO_MIN, cosine_min=COSINE_MIN, levels=LEVELS,
              rejected=None):
    """The whole benchmark result.

    Each source is scored by its OWN independent matching, which is what makes
    "recall per source" mean "of that source's entries, how many were matched"
    and "FP per source" mean "matched nothing in that source". The combined
    matching is the headline, and the only one whose FP column counts
    predictions that matched nothing anywhere.

    Returns ``{"by_source", "volume", "new_pairs", "levels"}``.
    """
    by_source, combined_ladders = {}, None
    for source in (COMBINED,) + tuple(SOURCES):
        key = None if source == COMBINED else source
        ladders = match_notes(records, preds, source=key, embedder=embedder,
                              dice_min=dice_min, ratio_min=ratio_min,
                              cosine_min=cosine_min, levels=levels,
                              rejected=rejected)
        by_source[source] = score_source(records, preds, ladders, source=key,
                                         levels=levels)
        if source == COMBINED:
            combined_ladders = ladders

    # The span/name split, on the combined accept-set only. Four sources times
    # three fields would be twelve matchings for a question that only needs
    # three: how much of the result is faithful copying, how much is medical
    # vocabulary, and how much is either.
    by_field = {}
    for field in ("span", "name", "both"):
        ladders = match_notes(records, preds, source=None, embedder=embedder,
                              dice_min=dice_min, ratio_min=ratio_min,
                              cosine_min=cosine_min, levels=levels,
                              rejected=rejected, field=field)
        by_field[field] = score_source(records, preds, ladders, source=None,
                                       levels=levels)

    return {
        "levels": list(levels),
        "by_source": by_source,
        "by_field": by_field,
        "volume": volume(records, preds),
        # Only the combined matching's new pairs are dumped. It is the headline
        # matching and the one L5 adjudicates; four near-identical dumps would
        # cost review attention and buy nothing.
        "new_pairs": new_pairs(records, preds, combined_ladders, levels),
    }


def trailing_repeat_len(findings):
    """How many findings at the END of a list repeat one seen earlier in it.

    The signal that separates two very different truncations. A reply cut while
    the model was still producing new findings has genuinely lost content, and
    recall is understated. A reply cut while the model was replaying its own
    list -- the observed failure was a verbatim replay of items 2-9 after item
    16 -- has lost nothing, because everything past the cut was a duplicate the
    pooling step would have collapsed anyway.

    Without this the cap-hit count says "recall is understated" for both cases,
    which is pessimistic in a way that would send someone hunting for findings
    that were never missing.
    """
    seen, is_new = set(), []
    for finding in findings:
        key = (normalize_term(finding.get("span")),
               normalize_term(finding.get("name")))
        is_new.append(key not in seen)
        seen.add(key)

    trailing = 0
    for new in reversed(is_new):
        if new:
            break
        trailing += 1
    return trailing
