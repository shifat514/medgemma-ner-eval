"""Term-set scoring: micro P/R/F1, the precision ceiling, FP buckets, code recall."""

import pytest

from src.report_mdace import build_report
from src.term_scoring import (
    classify_fp,
    code_coverage,
    micro_prf,
    note_gold,
    padded_note_norm,
    score_by_chart_type,
    score_note,
    score_view,
    wilson_ci,
)


def _record(note_id=1, chart_type="Profee", terms=(), text="", gold_rows=None,
            admission=(), shipped=(), descrs=()):
    gold = list(gold_rows or [
        {"term": t, "norm": t, "code": f"C{i}", "code_system": "ICD-10-CM",
         "descr": "", "descr_norm": "", "bucket": "conditions"}
        for i, t in enumerate(terms)
    ])
    return {
        "note_id": note_id,
        "chart_type": chart_type,
        "text": text,
        "gold": gold,
        "gold_terms": sorted({g["norm"] for g in gold}),
        "admission_gold_terms": sorted(admission),
        "sample100_terms": sorted(shipped),
        "sample100_descrs": sorted(descrs),
    }


# --------------------------------------------------------------------------
# Set arithmetic
# --------------------------------------------------------------------------

def test_score_note_partitions_cleanly():
    tp, fp, fn = score_note({"a", "b", "c"}, {"b", "c", "d"})
    assert tp == {"b", "c"}
    assert fp == {"d"}
    assert fn == {"a"}


def test_repeated_prediction_counts_once():
    """The whole reason for scoring sets: 4,770 gold terms recur in their own
    note, so occurrence counting would need an arbitrary policy."""
    tp, fp, fn = score_note({"aspirin"}, {"aspirin"})
    assert (len(tp), len(fp), len(fn)) == (1, 0, 0)


@pytest.mark.parametrize("tp,fp,fn,p,r", [
    (0, 0, 0, 0.0, 0.0),
    (0, 5, 5, 0.0, 0.0),
    (5, 0, 0, 1.0, 1.0),
    (1, 1, 1, 0.5, 0.5),
    (6, 34, 0, 0.15, 1.0),        # the perfect-extractor Inpatient case
])
def test_micro_prf(tp, fp, fn, p, r):
    precision, recall, _f1 = micro_prf(tp, fp, fn)
    assert precision == pytest.approx(p)
    assert recall == pytest.approx(r)


def test_f1_sits_nearer_the_worse_of_the_two():
    _p, _r, f1 = micro_prf(tp=6, fp=34, fn=0)   # P=0.15, R=1.00
    assert f1 < 0.5


def test_micro_weights_notes_by_size():
    """Micro pools counts; a 30-gold note must outweigh a 2-gold note.

    Under macro averaging these two would count equally and the result would be
    0.5 rather than 30/32.
    """
    big = _record(1, terms=[f"t{i}" for i in range(30)])
    small = _record(2, terms=["x", "y"])
    m = score_view([(big, set(big["gold_terms"])), (small, set())],
                   with_fp_buckets=False)
    assert m["recall"] == pytest.approx(30 / 32)


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------

def test_wilson_contains_the_point_estimate():
    lo, hi = wilson_ci(60, 110)
    assert lo < 60 / 110 < hi


def test_wilson_stays_in_bounds_at_the_extremes():
    for lo, hi in (wilson_ci(0, 20), wilson_ci(20, 20)):
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_narrows_as_the_sample_grows():
    small = wilson_ci(55, 110)
    large = wilson_ci(550, 1100)
    assert (small[1] - small[0]) > (large[1] - large[0])


def test_wilson_width_matches_the_reported_rule_of_thumb():
    """The report claims ~+/-9 points at n=110 and ~+/-7 at n=214."""
    lo, hi = wilson_ci(55, 110)
    assert 0.16 < (hi - lo) < 0.20
    lo, hi = wilson_ci(107, 214)
    assert 0.12 < (hi - lo) < 0.15


def test_wilson_of_empty_sample():
    assert wilson_ci(0, 0) == (0.0, 0.0)


# --------------------------------------------------------------------------
# Precision ceiling and extraction ratio
# --------------------------------------------------------------------------

def test_precision_ceiling_bounds_a_perfect_extractor():
    """6 gold, 40 predicted including all 6: the best possible precision is 0.15,
    and the actual precision must not exceed it."""
    rec = _record(1, terms=[f"g{i}" for i in range(6)], text="")
    pred = set(rec["gold_terms"]) | {f"extra{i}" for i in range(34)}
    m = score_view([(rec, pred)], with_fp_buckets=False)
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(6 / 40)
    assert m["precision_ceiling"] == pytest.approx(6 / 40)
    assert m["precision"] <= m["precision_ceiling"] + 1e-12


def test_extraction_ratio_is_pred_over_gold():
    rec = _record(1, terms=["a", "b"])
    m = score_view([(rec, {"a", "b", "c", "d", "e", "f"})], with_fp_buckets=False)
    assert m["extraction_ratio"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# False-positive buckets
# --------------------------------------------------------------------------

def test_padded_note_norm_prevents_substring_false_matches():
    note = padded_note_norm("Patient underwent CABG yesterday")
    assert " cabg " in note
    assert " ca " not in note          # must not match inside "cabg"


def test_fp_bucket_billed_elsewhere_in_the_admission():
    """An ICD code belongs to an admission but its evidence is marked on one
    note; a correct extraction on a sibling note must not read as an error."""
    bucket = classify_fp("sepsis", padded_note_norm("patient with sepsis"),
                         admission_gold={"sepsis"}, note_gold_terms=set())
    assert bucket == "billed_elsewhere"


def test_fp_bucket_in_note_but_unbilled():
    bucket = classify_fp("obesity", padded_note_norm("history of obesity"),
                         admission_gold=set(), note_gold_terms=set())
    assert bucket == "in_note_unbilled"


def test_fp_bucket_not_in_note_is_hallucination():
    bucket = classify_fp("lupus", padded_note_norm("history of obesity"),
                         admission_gold=set(), note_gold_terms=set())
    assert bucket == "not_in_note"


def test_fp_buckets_sum_to_the_false_positive_count():
    rec = _record(1, terms=["sepsis"], text="patient has sepsis and obesity",
                  admission={"sepsis", "anemia"})
    pred = {"sepsis", "obesity", "anemia", "lupus"}
    m = score_view([(rec, pred)])
    assert m["fp"] == 3
    assert sum(m["fp_buckets"].values()) == m["fp"]
    assert m["fp_buckets"] == {"billed_elsewhere": 1, "in_note_unbilled": 1,
                               "not_in_note": 1}


def test_buckets_are_not_subtracted_from_precision():
    """Removing 'correct but unbilled' FPs before dividing would inflate
    precision by construction. The headline must stay honest."""
    rec = _record(1, terms=["sepsis"], text="sepsis obesity",
                  admission={"sepsis"})
    m = score_view([(rec, {"sepsis", "obesity"})])
    assert m["precision"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Code recall
# --------------------------------------------------------------------------

def test_code_covered_by_either_of_two_phrasings():
    """I44.2 is evidenced by both 'third degree heart block' and 'Complete Heart
    Block' in one real note; finding either recovers the code."""
    gold_rows = [
        {"term": "third degree heart block", "norm": "third degree heart block",
         "code": "I44.2", "code_system": "ICD-10-CM", "descr": "",
         "descr_norm": "", "bucket": "conditions"},
        {"term": "Complete Heart Block", "norm": "complete heart block",
         "code": "I44.2", "code_system": "ICD-10-CM", "descr": "",
         "descr_norm": "", "bucket": "conditions"},
    ]
    rec = _record(1, gold_rows=gold_rows)
    assert code_coverage(rec, {"complete heart block"}) == (1, 1)
    assert code_coverage(rec, {"third degree heart block"}) == (1, 1)
    assert code_coverage(rec, set()) == (0, 1)


def test_one_phrase_can_cover_several_codes():
    gold_rows = [
        {"term": "hypertension", "norm": "hypertension", "code": "I10",
         "code_system": "ICD-10-CM", "descr": "", "descr_norm": "",
         "bucket": "conditions"},
        {"term": "hypertension", "norm": "hypertension", "code": "I12.9",
         "code_system": "ICD-10-CM", "descr": "", "descr_norm": "",
         "bucket": "conditions"},
    ]
    rec = _record(1, gold_rows=gold_rows)
    assert code_coverage(rec, {"hypertension"}) == (2, 2)


# --------------------------------------------------------------------------
# Gold selectors and views
# --------------------------------------------------------------------------

def test_gold_key_descr_is_not_the_note_wording():
    """View A1's whole point: the catalogue wording and the note wording are
    different strings, so a correct extraction scores zero against `descr`."""
    rec = _record(1, terms=["depression"],
                  descrs=["major depressive disorder single episode unspecified"])
    assert note_gold(rec, "full") == {"depression"}
    assert note_gold(rec, "descr") != note_gold(rec, "full")

    m = score_view([(rec, {"depression"})], gold_key="descr",
                   with_fp_buckets=False)
    assert m["recall"] == 0.0


def test_gold_key_conditions_excludes_procedures_and_status():
    gold_rows = [
        {"term": "sepsis", "norm": "sepsis", "code": "A41.9",
         "code_system": "ICD-10-CM", "descr": "", "descr_norm": "",
         "bucket": "conditions"},
        {"term": "colonoscopy", "norm": "colonoscopy", "code": "45378",
         "code_system": "CPT", "descr": "", "descr_norm": "",
         "bucket": "procedure"},
        {"term": "tobacco history", "norm": "tobacco history", "code": "Z87.891",
         "code_system": "ICD-10-CM", "descr": "", "descr_norm": "",
         "bucket": "status_history"},
    ]
    rec = _record(1, gold_rows=gold_rows)
    assert note_gold(rec, "conditions") == {"sepsis"}
    assert len(note_gold(rec, "full")) == 3


def test_gold_key_rejects_unknown():
    with pytest.raises(ValueError, match="gold_key"):
        note_gold(_record(), "nonsense")


def test_score_by_chart_type_keeps_strata_apart():
    profee = _record(1, "Profee", terms=["a", "b"])
    inpatient = _record(2, "Inpatient", terms=["c", "d"])
    out = score_by_chart_type([(profee, {"a", "b"}), (inpatient, set())])
    assert out["Profee"]["recall"] == pytest.approx(1.0)
    assert out["Inpatient"]["recall"] == pytest.approx(0.0)
    assert out["Profee"]["n_notes"] == 1


def test_score_view_handles_empty_input():
    m = score_view([])
    assert m["n_notes"] == 0
    assert m["precision"] == m["recall"] == m["f1"] == 0.0


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------

def _views():
    profee = _record(1, "Profee", terms=["a", "b"], text="a b")
    inpatient = _record(2, "Inpatient", terms=["c"], text="c d")
    pairs = [(profee, {"a", "b"}), (inpatient, {"c", "d"})]
    return {
        "B2": {
            "title": "Stratified 50 notes — THE HEADLINE",
            "detail": "25 Profee + 25 Inpatient",
            "metrics": score_view(pairs),
            "by_chart_type": score_by_chart_type(pairs),
        }
    }


def _meta(**over):
    meta = {"model_name": "medgemma-4b-it", "model_id": "google/medgemma-4b-it",
            "max_new_tokens": 512, "chunk_words": 400, "overlap_words": 80,
            "seed": 13, "n_notes_scored": 2, "n_cap_hits": 0,
            "n_mixed_chart_type": 0, "oracle": False}
    meta.update(over)
    return meta


def test_report_renders_the_caveats_that_matter():
    md = build_report(_views(), _meta())
    assert "billed" in md
    assert "best precision possible" in md
    assert "not a model failure" in md.lower() or "not a model failure" in md
    assert "Profee" in md and "Inpatient" in md


def test_report_flags_an_oracle_run():
    md = build_report(_views(), _meta(oracle=True))
    assert "ORACLE" in md


def test_report_contains_no_note_text_or_terms():
    """The report is committed; only counts and rates may reach it."""
    md = build_report(_views(), _meta())
    for leaked in ("note text", '"a"', '"b"', '"c"', '"d"'):
        assert leaked not in md


def test_write_report_round_trips(tmp_path):
    import json

    from src.report_mdace import write_report

    md_path, json_path = write_report(_views(), _meta(),
                                      results_dir=str(tmp_path), label="t")
    assert md_path.endswith(".md")
    data = json.loads(open(json_path, encoding="utf-8").read())
    # per_note carries note_ids, so it must not reach the committed artifact.
    assert "per_note" not in data["views"]["B2"]["metrics"]
    assert data["run"]["seed"] == 13


# --------------------------------------------------------------------------
# Partial-run view gating
# --------------------------------------------------------------------------

def _sample(n_strat=3, n_ship=2, overlap=1):
    """Records shaped like the real sample: stratified first, then the cut.

    Notes in the cut carry two gold terms and ship only one, mirroring the real
    file: 99 of the 195 phrases those notes really hold. Without that gap the
    A2 -> B1 step has nothing to measure.
    """
    recs = []
    for i in range(n_strat):
        in_cut = i < overlap
        terms = [f"s{i}", f"s{i}x"] if in_cut else [f"s{i}"]
        r = _record(100 + i, "Profee", terms=terms, text=" ".join(terms),
                    shipped=[f"s{i}"] if in_cut else ())
        r["in_stratified"], r["in_sample100"] = True, in_cut
        recs.append(r)
    for i in range(n_ship - overlap):
        terms = [f"p{i}", f"p{i}x"]
        r = _record(200 + i, "Inpatient", terms=terms, text=" ".join(terms),
                    shipped=[f"p{i}"])
        r["in_stratified"], r["in_sample100"] = False, True
        recs.append(r)
    return recs


def test_phase1_emits_only_the_headline_view():
    """--limit 50 runs the stratified draw and one shared note.

    A 1-note A1 rendered beside a 50-note B2 invites exactly the comparison the
    ladder exists to prevent, so the incomplete views must be withheld.
    """
    from src.evaluate_mdace import build_views

    recs = _sample()
    strat = [r for r in recs if r["in_stratified"]]
    preds = {r["note_id"]: set(r["gold_terms"]) for r in strat}

    views = build_views(recs, preds)
    assert set(views) == {"B2"}
    assert views["B2"]["metrics"]["n_notes"] == len(strat)


def test_phase2_completes_every_view():
    from src.evaluate_mdace import build_views

    recs = _sample()
    preds = {r["note_id"]: set(r["gold_terms"]) for r in recs}
    views = build_views(recs, preds)
    assert set(views) == {"A1", "A2", "B1", "B2"}


def test_a2_scores_against_the_phrases_the_cut_actually_ships():
    """A2 exists to be comparable with a number computed from that file alone.

    It must use `sample100_terms`, not the full gold — otherwise the A2 -> B1
    jump collapses to zero and the cost of the truncated key is invisible.
    """
    from src.evaluate_mdace import build_views

    recs = _sample()
    preds = {r["note_id"]: set(r["gold_terms"]) for r in recs}
    views = build_views(recs, preds)

    shipped_gold = sum(len(r["sample100_terms"])
                       for r in recs if r["in_sample100"])
    full_gold = sum(len(r["gold_terms"]) for r in recs if r["in_sample100"])

    assert views["A2"]["metrics"]["n_gold"] == shipped_gold
    assert views["B1"]["metrics"]["n_gold"] == full_gold


def test_no_views_when_nothing_has_run():
    from src.evaluate_mdace import build_views

    assert build_views(_sample(), {}) == {}


# --------------------------------------------------------------------------
# Smoke-note selection
# --------------------------------------------------------------------------

def _sized(note_id, chart_type, n_chunks):
    r = _record(note_id, chart_type, terms=["x"], text="x")
    r["n_chunks"] = n_chunks
    return r


def test_smoke_selection_includes_the_longest_note():
    """File order is stratified-first and starts with short Profee notes, so a
    head-of-file smoke test never exercises the long-note path where OOM and
    truncation live."""
    from src.evaluate_mdace import select_smoke_notes

    recs = ([_sized(i, "Profee", 1) for i in range(10)]
            + [_sized(50, "Inpatient", 10)])
    picked = select_smoke_notes(recs, 5)
    assert max(r["n_chunks"] for r in picked) == 10


def test_smoke_selection_covers_both_chart_types():
    from src.evaluate_mdace import select_smoke_notes

    recs = ([_sized(i, "Profee", 1) for i in range(10)]
            + [_sized(50, "Inpatient", 10)])
    picked = select_smoke_notes(recs, 5)
    assert {r["chart_type"] for r in picked} == {"Profee", "Inpatient"}
    assert len(picked) == 5
    assert len({r["note_id"] for r in picked}) == 5


def test_smoke_selection_returns_everything_when_n_exceeds_the_sample():
    from src.evaluate_mdace import select_smoke_notes

    recs = [_sized(i, "Profee", 1) for i in range(3)]
    assert len(select_smoke_notes(recs, 10)) == 3
