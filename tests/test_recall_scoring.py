"""Aggregation: recall in three units, false positives per source, volume.

The per-source arithmetic is the part most likely to be misread, so it is the
part pinned hardest here.
"""

import json

import pytest

from src.datasets.mdace_recall import build_notes
from src.recall_scoring import (
    dedupe_findings,
    not_in_note,
    score_run,
    volume,
)
from src.report_recall import build_report

LEVELS = ("L1", "L2", "L3")


def _row(note_id=1, code="I10", system="ICD-10-CM", evidence="HTN",
         descr="Essential (primary) hypertension", snomed=(), text=None):
    return {
        "note_id": note_id,
        "hadm_id": 100,
        "chart_type": "Inpatient",
        "code_system": system,
        "gold_code": code,
        "gold_code_description": descr,
        "mdace_gold_evidence_text": evidence,
        "gold_snomed_concepts": [{"concept_id": str(i), "term": t}
                                 for i, t in enumerate(snomed)],
        "gold_snomed_concept_count": len(snomed),
        "note_text": text if text is not None else f"The patient has {evidence}.",
    }


def _notes(tmp_path, rows):
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return build_notes(str(path))


def _preds(note_id, findings):
    return {note_id: dedupe_findings(findings)}


# --------------------------------------------------------------------------
# Deduping
# --------------------------------------------------------------------------

def test_the_same_finding_seen_in_two_windows_counts_once():
    """Windows overlap by 80 words; without this the overlap inflates volume."""
    findings = dedupe_findings([{"span": "HTN", "name": "hypertension"}] * 3)
    assert len(findings) == 1


def test_one_span_under_two_names_is_two_findings():
    """The oracle relies on this, and a model reporting two findings that share
    a span should not have one silently dropped."""
    findings = dedupe_findings([
        {"span": "HTN", "name": "hypertension"},
        {"span": "HTN", "name": "hypertensive disorder"},
    ])
    assert len(findings) == 2


def test_findings_with_no_usable_string_are_dropped():
    assert dedupe_findings([{"span": "", "name": ""}, {"span": "  ", "name": None}]) == []


def test_dedupe_is_case_and_punctuation_insensitive():
    findings = dedupe_findings([
        {"span": "Chest pain.", "name": "chest pain"},
        {"span": "chest  pain", "name": "Chest Pain"},
    ])
    assert len(findings) == 1


# --------------------------------------------------------------------------
# Recall in three units
# --------------------------------------------------------------------------

def test_a_hit_on_any_accepted_form_recalls_the_row(tmp_path):
    """The point of the accept-set: catalogue wording counts."""
    records, _ = _notes(tmp_path, [_row()])
    preds = _preds(1, [{"span": "", "name": "essential primary hypertension"}])
    result = score_run(records, preds, levels=LEVELS)
    assert result["by_source"]["combined"]["L1"]["row_recall"] == 1.0


def test_rows_and_codes_are_reported_separately(tmp_path):
    """Nine codes in the real file are evidenced twice, so the two units differ.

    Recalling one of a code's two rows recalls the code but only half its rows.
    """
    records, _ = _notes(tmp_path, [
        _row(code="A1", evidence="chest pain", descr="Angina", text="chest pain here"),
        _row(code="A1", evidence="angina", descr="Angina", text="chest pain here"),
    ])
    preds = _preds(1, [{"span": "chest pain", "name": ""}])
    m = score_run(records, preds, levels=LEVELS)["by_source"]["combined"]["L1"]
    assert (m["rows_matched"], m["rows_total"]) == (1, 2)
    assert (m["codes_matched"], m["codes_total"]) == (1, 1)


def test_recall_never_falls_as_the_ladder_loosens(tmp_path):
    records, _ = _notes(tmp_path, [
        _row(evidence="chronic back pain", descr="Back pain, unspecified",
             text="chronic back pain noted")])
    preds = _preds(1, [{"span": "back pain", "name": "back pain"}])
    result = score_run(records, preds, levels=LEVELS)
    recalls = [result["by_source"]["combined"][lv]["form_recall"]
               for lv in LEVELS]
    assert recalls == sorted(recalls)
    assert recalls[0] == 0.0             # not an exact match
    assert recalls[1] > 0.0              # containment reaches it


def test_each_level_reports_the_gain_it_is_responsible_for(tmp_path):
    records, _ = _notes(tmp_path, [
        _row(evidence="chronic back pain", descr="Dorsalgia, unspecified",
             text="chronic back pain noted")])
    preds = _preds(1, [{"span": "back pain", "name": "back pain"}])
    result = score_run(records, preds, levels=LEVELS)
    assert result["by_source"]["combined"]["L2"]["gain_rows"] == 1
    assert result["by_source"]["combined"]["L3"]["gain_rows"] == 0


# --------------------------------------------------------------------------
# Per source — the part that is easy to misread
# --------------------------------------------------------------------------

def test_recall_per_source_says_whose_wording_the_model_produced(tmp_path):
    records, _ = _notes(tmp_path, [_row(snomed=["Hypertensive disorder"])])
    preds = _preds(1, [{"span": "HTN", "name": ""}])
    by_source = score_run(records, preds, levels=LEVELS)["by_source"]
    assert by_source["evidence"]["L1"]["form_recall"] == 1.0
    assert by_source["description"]["L1"]["form_recall"] == 0.0
    assert by_source["snomed"]["L1"]["form_recall"] == 0.0


def test_a_catalogue_match_is_an_fp_on_the_evidence_line(tmp_path):
    """The documented trap: per-source FP means 'matched nothing IN THAT
    SOURCE', so every individual line reads high and only the combined line
    counts predictions that matched nothing anywhere."""
    records, _ = _notes(tmp_path, [_row()])
    preds = _preds(1, [{"span": "", "name": "essential primary hypertension"}])
    by_source = score_run(records, preds, levels=LEVELS)["by_source"]
    assert by_source["description"]["L1"]["fp"] == 0
    assert by_source["evidence"]["L1"]["fp"] == 1
    assert by_source["combined"]["L1"]["fp"] == 0


def test_denominators_differ_per_source(tmp_path):
    records, _ = _notes(tmp_path, [
        _row(code="A1", snomed=["Hypertensive disorder"]),
        _row(code="A2", evidence="sepsis", descr="Sepsis, unspecified",
             text="HTN and sepsis"),
    ])
    # A note that produced no findings still contributes its denominators; a
    # note that was never run contributes nothing and is warned about instead.
    by_source = score_run(records, {1: []}, levels=LEVELS)["by_source"]
    assert by_source["snomed"]["L1"]["rows_total"] == 1
    assert by_source["evidence"]["L1"]["rows_total"] == 2
    assert by_source["combined"]["L1"]["rows_total"] == 2


# --------------------------------------------------------------------------
# Volume — recall is never quoted bare
# --------------------------------------------------------------------------

def test_false_positives_are_counted_not_implied_by_precision(tmp_path):
    records, _ = _notes(tmp_path, [_row()])
    preds = _preds(1, [{"span": "HTN", "name": ""},
                       {"span": "pneumonia", "name": "pneumonia"}])
    m = score_run(records, preds, levels=LEVELS)["by_source"]["combined"]["L1"]
    assert (m["n_pred"], m["fp"]) == (2, 1)
    assert m["fp_rate"] == pytest.approx(0.5)


def test_not_in_note_is_the_hallucination_signal():
    findings = dedupe_findings([
        {"span": "sepsis", "name": "sepsis"},
        {"span": "pneumonia", "name": "pneumonia"},
    ])
    assert not_in_note(findings, "Patient has sepsis.") == (2, 1)


def test_not_in_note_is_whole_token():
    findings = dedupe_findings([{"span": "ca", "name": ""}])
    assert not_in_note(findings, "Patient underwent CABG.") == (1, 1)


def test_a_finding_with_no_span_is_excluded_from_the_check_not_called_clean():
    findings = dedupe_findings([{"span": "", "name": "hypertension"}])
    checked, missing = not_in_note(findings, "Patient has HTN.")
    assert (checked, missing) == (0, 0)


def test_volume_reports_the_unspanned_findings_separately(tmp_path):
    records, _ = _notes(tmp_path, [_row()])
    preds = _preds(1, [{"span": "", "name": "hypertension"},
                       {"span": "HTN", "name": ""}])
    v = volume(records, preds)
    assert v["n_no_span"] == 1
    assert v["n_span_checked"] == 1
    assert v["pred_per_note"] == 2.0


# --------------------------------------------------------------------------
# The audit trail
# --------------------------------------------------------------------------

def test_each_level_dumps_the_pairs_it_newly_accepted(tmp_path):
    records, _ = _notes(tmp_path, [
        _row(evidence="chronic back pain", descr="Dorsalgia, unspecified",
             text="chronic back pain noted")])
    preds = _preds(1, [{"span": "back pain", "name": "back pain"}])
    pairs = score_run(records, preds, levels=LEVELS)["new_pairs"]
    assert pairs["L1"] == []
    assert len(pairs["L2"]) == 1
    assert pairs["L2"][0]["rule"] == "contains"
    assert pairs["L2"][0]["gold_form"] == "chronic back pain"
    assert pairs["L2"][0]["gold_sources"] == ["evidence"]


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def _report(tmp_path, oracle=False, cap_hits=0):
    records, stats = _notes(tmp_path, [_row(snomed=["Hypertensive disorder"])])
    preds = _preds(1, [{"span": "HTN", "name": "hypertension"}])
    result = score_run(records, preds, levels=LEVELS)
    meta = {
        "model_name": "medgemma-4b-it", "model_id": "google/medgemma-4b-it",
        "max_new_tokens": 1024, "chunk_words": 400, "overlap_words": 80,
        "n_notes_scored": 1, "n_chunks": 1, "n_cap_hits": cap_hits,
        "n_chunks_salvaged": 0, "prompt_id": "abc12345", "run_tag": "medgemma_x",
        "oracle": oracle, "levels": list(LEVELS),
        "thresholds": {"dice_min": 0.8, "ratio_min": 0.9, "cosine_min": 0.8,
                       "embed_model": None},
    }
    return build_report(result, meta, stats)


def test_report_prints_the_thresholds_it_used(tmp_path):
    text = _report(tmp_path)
    assert "0.8" in text and "0.9" in text
    assert "never silently chosen" in text


def test_report_says_l4_did_not_run_when_it_did_not(tmp_path):
    assert "**L4 did not run in this report.**" in _report(tmp_path)


def test_report_carries_the_prompt_hash_and_run_tag(tmp_path):
    """Without these, a replayed run is indistinguishable from a fresh one."""
    text = _report(tmp_path)
    assert "abc12345" in text
    assert "medgemma_x" in text


def test_report_states_the_per_source_fp_trap(tmp_path):
    text = _report(tmp_path)
    assert "matched nothing in that source" in text
    assert "only the combined line counts predictions that matched nothing" in text


def test_report_refuses_to_let_recall_stand_alone(tmp_path):
    text = _report(tmp_path)
    assert "Recall is never quotable on its own" in text
    assert "findings per note" in text


def test_report_flags_an_oracle_run_at_the_top(tmp_path):
    assert "ORACLE RUN" in _report(tmp_path, oracle=True)


def test_report_only_warns_about_truncation_when_it_happened(tmp_path):
    assert "floor rather than an estimate" not in _report(tmp_path)
    assert "floor rather than an estimate" in _report(tmp_path, cap_hits=3)


def test_report_carries_no_note_text_or_phrases(tmp_path):
    """The report is committed. Everything in it must be an integer or a rate."""
    text = _report(tmp_path)
    assert "The patient has" not in text        # note text
    assert "Hypertensive disorder" not in text  # a gold phrase


def test_committed_metrics_carry_no_absolute_paths(tmp_path):
    """The artifact is public and the input lives outside the repo.

    The filename identifies which input produced the numbers; the directory it
    happened to sit in on one laptop is nobody's business.
    """
    from src.evaluate_recall import run_eval

    records, _ = _notes(tmp_path, [_row()])
    del records
    run_eval(sample_file=str(tmp_path / "sample.jsonl"), oracle=True,
             results_dir=str(tmp_path / "results"),
             output_dir=str(tmp_path / "out"), embed=False)

    written = json.loads(
        next((tmp_path / "results").glob("*_metrics.json")).read_text())

    def strings(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from strings(v)
        elif isinstance(obj, str):
            yield obj

    assert [s for s in strings(written) if s.startswith("/")] == []
    assert written["run"]["input_file"] == "sample.jsonl"


# --------------------------------------------------------------------------
# Diagnostics that have to distinguish a repetition loop from window overlap
# --------------------------------------------------------------------------

def test_repeats_within_one_reply_are_counted_separately_from_overlap():
    """A looping model and an overlapping window look identical after pooling.

    Repeats ACROSS a note's chunks are expected — windows overlap by design — so
    only the within-reply count is evidence of a repetition loop, which is the
    thing that decides whether the fix is the prompt or the chunk size.
    """
    from src.evaluate_recall import predict_note

    reply = json.dumps({"findings": [
        {"span": "sepsis", "name": "sepsis"},
        {"span": "Sepsis.", "name": "sepsis"},      # same finding, said twice
        {"span": "HTN", "name": "hypertension"},
    ]})
    record = {"note_id": 1, "text": "sepsis and HTN", "rows": [], "forms": {}}
    _findings, st = predict_note(
        None, record, run_fn=lambda p, c: reply, count_fn=lambda p, r: None)

    assert st["n_items_dup_in_chunk"] == 1
    assert st["n_findings"] == 2


def test_a_truncated_chunk_that_parsed_cleanly_is_still_flagged_as_lost():
    """The object format recovers the prefix, so `shape` stays "json" and the
    salvage counter never fires — which made the diagnostics read as though a
    cut-off chunk was fine. Everything past the cut is still gone."""
    from src.evaluate_recall import predict_note

    reply = '{"findings": [{"span": "sepsis", "name": "sepsis"}, {"span": "acute kid'
    record = {"note_id": 1, "text": "sepsis", "rows": [], "forms": {}}
    _findings, st = predict_note(
        None, record, run_fn=lambda p, c: reply,
        count_fn=lambda p, r: 1024, gen_config={"max_new_tokens": 1024})

    assert st["n_cap_hits"] == 1
    assert st["n_chunks_salvaged"] == 0        # nothing needed salvaging
    assert st["n_chunks_cut_but_parsed"] == 1  # ...but content was still lost


def test_a_cut_during_a_replay_is_not_counted_as_lost_content():
    """The observed failure: sixteen genuine findings, then a verbatim replay of
    items 2-9, cut mid-replay. Nothing was lost -- pooling collapses those
    duplicates anyway -- so calling it "recall is understated" sends someone
    hunting for findings that were never missing."""
    from src.evaluate_recall import predict_note
    from src.recall_scoring import trailing_repeat_len

    genuine = [{"span": f"finding {i}", "name": f"name {i}"} for i in range(16)]
    assert trailing_repeat_len(genuine) == 0
    assert trailing_repeat_len(genuine + genuine[1:9]) == 8

    looping = json.dumps({"findings": genuine + genuine[1:9]})
    record = {"note_id": 1, "text": "x", "rows": [], "forms": {}}
    _f, st = predict_note(None, record, run_fn=lambda p, c: looping,
                          count_fn=lambda p, r: 1024,
                          gen_config={"max_new_tokens": 1024})
    assert st["n_cap_hits"] == 1
    assert st["n_cap_hits_while_repeating"] == 1

    still_producing = json.dumps({"findings": genuine})
    _f, st = predict_note(None, record, run_fn=lambda p, c: still_producing,
                          count_fn=lambda p, r: 1024,
                          gen_config={"max_new_tokens": 1024})
    assert st["n_cap_hits"] == 1
    assert st["n_cap_hits_while_repeating"] == 0


def test_each_prompt_variant_gets_its_own_results_filename(tmp_path):
    """Both arms of an A/B wrote the same file, and the second silently
    overwrote the first -- which then made the comparison tool pick up whatever
    unrelated run happened to be next-most-recent. An oracle run, as it turned
    out, which reads as a spectacular win for whichever arm survived."""
    from src.evaluate_recall import run_eval

    _notes(tmp_path, [_row()])
    results = tmp_path / "results"
    for variant in ("scoped", "billable"):
        run_eval(sample_file=str(tmp_path / "sample.jsonl"), oracle=True,
                 results_dir=str(results), output_dir=str(tmp_path / "out"),
                 embed=False, prompt_variant=variant)

    written = sorted(p.name for p in results.glob("*_metrics.json"))
    assert len(written) == 2, written
    assert any("_scoped_" in name for name in written)
    assert any("_billable_" in name for name in written)


def test_comparing_runs_over_different_notes_is_flagged_not_rendered_quietly():
    from src.recall_compare import render

    def _row_for(prompt, notes, label="smoke_2"):
        return {"label": label, "prompt": prompt, "hash": "abc", "notes": notes,
                "chunks": notes * 8, "per_note": 90.0, "not_in_note": 0.01,
                "cap_hits": 5, "cap_looping": 3, "dup_in_chunk": 100,
                "row_recall_l1": 0.8, "row_recall_top": 1.0,
                "code_recall_top": 1.0, "fp_top": 100, "fp_rate_top": 0.9,
                "top": "L4"}

    mismatched = render([_row_for("billable", 2), _row_for("scoped", 24)])
    assert "different notes" in mismatched

    with_oracle = render([_row_for("billable", 2),
                          _row_for("scoped", 2, label="oracle_24")])
    assert "ORACLE run" in with_oracle

    clean = render([_row_for("billable", 2), _row_for("scoped", 2)])
    assert "⚠️" not in clean


def test_chunk_geometry_also_gets_its_own_results_filename(tmp_path):
    """The variant fix closed one axis and left every other one open. A
    chunk-size A/B would have destroyed an arm exactly the same way."""
    from src.evaluate_recall import run_eval

    _notes(tmp_path, [_row(text=" ".join(["word"] * 900) + " HTN")])
    results = tmp_path / "results"
    for chunk_words in (400, 250):
        run_eval(sample_file=str(tmp_path / "sample.jsonl"), oracle=True,
                 results_dir=str(results), output_dir=str(tmp_path / "out"),
                 embed=False, chunk_words=chunk_words)

    written = sorted(p.name for p in results.glob("*_metrics.json"))
    assert len(written) == 2, written
    assert any("cw400" in name for name in written)
    assert any("cw250" in name for name in written)


# --------------------------------------------------------------------------
# L5 adjudication feeding back into the ladder
# --------------------------------------------------------------------------

def test_a_rejected_pair_is_removed_as_an_edge_not_as_a_finished_match(tmp_path):
    """The judge rules that `back pain` is not `chronic back pain`. The finding
    must then be free to match something else, which deleting the assignment
    instead of the edge would prevent."""
    records, _ = _notes(tmp_path, [
        _row(code="A1", evidence="chronic back pain", descr="Dorsalgia",
             text="chronic back pain and sepsis"),
        _row(code="A2", evidence="sepsis", descr="Sepsis, unspecified",
             text="chronic back pain and sepsis"),
    ])
    preds = _preds(1, [{"span": "back pain", "name": "sepsis"}])

    raw = score_run(records, preds, levels=LEVELS)["by_source"]["combined"]
    assert raw["L1"]["rows_matched"] == 1        # exact on "sepsis"

    rejected = {(1, "back pain", "sepsis", "chronic back pain")}
    adj = score_run(records, preds, levels=LEVELS,
                    rejected=rejected)["by_source"]["combined"]
    assert adj["L2"]["rows_matched"] <= raw["L2"]["rows_matched"]
    assert adj["L1"]["rows_matched"] == 1        # the exact match is untouched


def test_rejecting_a_pair_demotes_a_row_only_if_it_had_no_other_support(tmp_path):
    """Rows fall by less than pairs do, which is why both get quoted."""
    records, _ = _notes(tmp_path, [
        _row(evidence="chronic back pain", descr="Dorsalgia, unspecified",
             text="chronic back pain noted")])
    preds = _preds(1, [{"span": "back pain", "name": "back pain"}])

    raw = score_run(records, preds, levels=LEVELS)["by_source"]["combined"]
    assert raw["L2"]["rows_matched"] == 1

    rejected = {(1, "back pain", "back pain", "chronic back pain")}
    adj = score_run(records, preds, levels=LEVELS,
                    rejected=rejected)["by_source"]["combined"]
    assert adj["L2"]["rows_matched"] == 0        # that was its only support


def test_unreadable_verdicts_are_not_treated_as_rejections(tmp_path):
    """A judge that failed to answer is not evidence a match was wrong."""
    from src.recall_judge import load_rejected

    (tmp_path / "verdicts_L4.jsonl").write_text("\n".join([
        json.dumps({"note_id": 1, "span": "a", "name": "a",
                    "gold_form": "x", "verdict": False}),
        json.dumps({"note_id": 1, "span": "b", "name": "b",
                    "gold_form": "y", "verdict": None}),
        json.dumps({"note_id": 1, "span": "c", "name": "c",
                    "gold_form": "z", "verdict": True}),
    ]), encoding="utf-8")

    rejected = load_rejected(str(tmp_path))
    assert rejected == {(1, "a", "a", "x")}


def test_the_report_states_the_ceiling_that_matches_the_prompt_that_ran(tmp_path):
    """scoped excludes medications and therefore caps near 0.945; billable does
    not exclude them and its ceiling is unmeasured. A committed report asserting
    the wrong one is a claim somebody would reasonably rely on."""
    records, stats = _notes(tmp_path, [_row()])
    preds = _preds(1, [{"span": "HTN", "name": "hypertension"}])
    result = score_run(records, preds, levels=LEVELS)

    def report_for(variant):
        meta = {
            "model_name": "m", "model_id": "m", "max_new_tokens": 1024,
            "chunk_words": 400, "overlap_words": 80, "n_notes_scored": 1,
            "n_chunks": 1, "n_cap_hits": 0, "n_chunks_salvaged": 0,
            "prompt_id": "abc12345", "run_tag": "t", "oracle": False,
            "levels": list(LEVELS), "prompt_variant": variant,
            "thresholds": {"dice_min": 0.8, "ratio_min": 0.9,
                           "cosine_min": 0.6, "embed_model": None},
        }
        return build_report(result, meta, stats)

    scoped = report_for("scoped")
    assert "ceiling here is about **0.98**" in scoped

    billable = report_for("billable")
    assert "ceiling is unmeasured" in billable
    assert "ceiling here is about **0.98**" not in billable


def test_the_report_never_calls_the_run_zero_shot(tmp_path):
    """Both variants carry a worked example. The example is doing real work, so
    the honest label is one-shot."""
    records, stats = _notes(tmp_path, [_row()])
    result = score_run(records, _preds(1, [{"span": "HTN", "name": "h"}]),
                       levels=LEVELS)
    meta = {"model_name": "m", "model_id": "m", "max_new_tokens": 1024,
            "chunk_words": 400, "overlap_words": 80, "n_notes_scored": 1,
            "n_chunks": 1, "n_cap_hits": 0, "n_chunks_salvaged": 0,
            "prompt_id": "a", "run_tag": "t", "oracle": False,
            "levels": list(LEVELS), "prompt_variant": "billable",
            "thresholds": {"dice_min": 0.8, "ratio_min": 0.9,
                           "cosine_min": 0.6, "embed_model": None}}
    text = build_report(result, meta, stats)
    assert "One-shot, not zero-shot" in text
    # The word may appear only inside the sentence that corrects it, never as a
    # description of the run.
    assert "zero-shot recall benchmark" not in text
    assert "zero-shot" not in text.split("One-shot, not zero-shot")[0]


# --------------------------------------------------------------------------
# The span/name split
# --------------------------------------------------------------------------

def test_span_and_name_recall_are_scored_separately(tmp_path):
    """L1 pooled two different abilities. A model that only ever expands
    abbreviations scores on `name` and zero on `span`, and the old single number
    could not tell that apart from faithful copying."""
    records, _ = _notes(tmp_path, [_row(evidence="HTN")])
    # span matches the note wording; name matches the catalogue wording.
    preds = _preds(1, [{"span": "HTN", "name": "essential primary hypertension"}])
    by_field = score_run(records, preds, levels=LEVELS)["by_field"]

    assert by_field["span"]["L1"]["forms_matched"] == 1     # the note phrase
    assert by_field["name"]["L1"]["forms_matched"] == 1     # the catalogue phrase
    # NOT 2. One prediction claims at most one gold form, so a single finding
    # whose two fields each reach a different form still scores one match. The
    # union is never the sum, and the report says so.
    assert by_field["both"]["L1"]["forms_matched"] == 1


def test_the_union_is_bounded_below_by_each_field_and_above_by_the_11_rule(tmp_path):
    """`both` offers a superset of edges, so it can never score less than either
    field alone -- but the 1:1 constraint keeps it under their sum."""
    records, _ = _notes(tmp_path, [
        _row(code="A1", evidence="HTN"),
        _row(code="A2", evidence="sepsis", descr="Sepsis, unspecified",
             text="The patient has HTN and sepsis."),
    ])
    preds = _preds(1, [{"span": "HTN", "name": "sepsis"}])
    by_field = score_run(records, preds, levels=LEVELS)["by_field"]

    span = by_field["span"]["L1"]["forms_matched"]
    name = by_field["name"]["L1"]["forms_matched"]
    both = by_field["both"]["L1"]["forms_matched"]
    assert both >= max(span, name)
    assert both <= span + name
    assert both == 1          # one finding, one form, even reaching two


def test_a_name_only_model_scores_nothing_on_span(tmp_path):
    records, _ = _notes(tmp_path, [_row(evidence="HTN")])
    preds = _preds(1, [{"span": "", "name": "essential primary hypertension"}])
    by_field = score_run(records, preds, levels=LEVELS)["by_field"]

    assert by_field["span"]["L1"]["row_recall"] == 0.0
    assert by_field["name"]["L1"]["row_recall"] == 1.0


def test_span_recall_never_exceeds_the_union(tmp_path):
    records, _ = _notes(tmp_path, [
        _row(code="A1", evidence="HTN"),
        _row(code="A2", evidence="sepsis", descr="Sepsis, unspecified",
             text="The patient has HTN and sepsis."),
    ])
    preds = _preds(1, [{"span": "HTN", "name": "hypertension"},
                       {"span": "sepsis", "name": "sepsis"}])
    by_field = score_run(records, preds, levels=LEVELS)["by_field"]
    for level in LEVELS:
        for field in ("span", "name"):
            assert (by_field[field][level]["row_recall"]
                    <= by_field["both"][level]["row_recall"])


def test_the_report_shows_the_field_split_and_names_the_conservative_row(tmp_path):
    text = _report(tmp_path)
    assert "Which of the model's two fields did the work?" in text
    assert "Read the `span` row as the conservative result" in text
    assert "closest thing to the old" in text


# --------------------------------------------------------------------------
# What the false positives ARE — the half a raw count cannot give
# --------------------------------------------------------------------------

def test_false_positives_are_split_into_error_and_unbilled(tmp_path):
    """MDACE marks only codes that were BILLED, so a note is full of genuine
    findings nobody billed. Counting those as model error measures the dataset."""
    records, _ = _notes(tmp_path, [
        _row(evidence="HTN", text="The patient has HTN, sepsis and cough.")])
    preds = _preds(1, [
        {"span": "HTN", "name": "hypertension"},        # a hit
        {"span": "sepsis", "name": "sepsis"},           # in the note, unbilled
        {"span": "cough", "name": "cough"},             # in the note, unbilled
        {"span": "pneumonia", "name": "pneumonia"},     # NOT in the note
        {"span": "", "name": "diabetes"},               # no span to check
    ])
    m = score_run(records, preds, levels=LEVELS)["by_source"]["combined"]["L1"]

    assert m["fp_buckets"]["in_note_unbilled"] == 2
    assert m["fp_buckets"]["not_in_note"] == 1
    assert m["fp_buckets"]["no_span"] == 1
    assert sum(m["fp_buckets"].values()) == m["fp"]


def test_the_fp_hallucination_rate_is_over_misses_not_over_everything(tmp_path):
    """'Are the false positives the model's fault' is a question about the
    findings that missed, not about all findings."""
    records, _ = _notes(tmp_path, [
        _row(evidence="HTN", text="The patient has HTN and sepsis.")])
    preds = _preds(1, [{"span": "HTN", "name": "hypertension"},
                       {"span": "pneumonia", "name": "pneumonia"}])
    m = score_run(records, preds, levels=LEVELS)["by_source"]["combined"]["L1"]
    assert m["fp"] == 1
    assert m["fp_not_in_note_rate"] == 1.0        # the single FP was invented


def test_the_report_says_which_fp_bucket_is_model_error(tmp_path):
    text = _report(tmp_path)
    assert "What the L3 false positives actually are" in text
    assert "Only the middle row is model error" in text
    assert "billed elsewhere" in text or "billed elsewhere in" in text


def test_the_report_cross_checks_against_the_right_column(tmp_path):
    """The previous branch matched copied phrasing against note wording only.
    That is span-only at L1, not the combined row -- quoting the combined row as
    the comparison overstates the gain."""
    text = _report(tmp_path)
    assert "0.5278" in text
    assert "not the combined row" in text
