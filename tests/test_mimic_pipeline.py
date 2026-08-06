"""End-to-end MIMIC note pipeline with a fake model — CPU-only, synthetic text.

NO REAL NOTE TEXT IN THIS FILE. The "notes" are generated filler sentences.
Covers the chunk -> predict -> per-chunk align -> merge -> dedupe -> BIO path and
the statistics the report depends on.
"""

import json

from src.datasets.mimic_meds import build_gold
from src.evaluate_mimic import predict_note, run_eval
from src.scoring import score


def _fake_run(reply):
    return lambda pipe, chunk_text: reply


def _no_count(pipe, reply):
    return None


# --- single-chunk behavior --------------------------------------------------

def test_short_note_single_chunk_perfect_prediction():
    text = "Give Drugzol 250 mg PO daily for pain"
    record = {
        "note_id": "s1", "text": text,
        "spans": [[5, 12, "Medication"], [13, 19, "Dose"], [20, 22, "Mode"],
                  [23, 28, "Frequency"], [33, 37, "Reason"]],
    }
    gold = build_gold(record)
    reply = json.dumps({"entities": [
        {"text": "Drugzol", "type": "Medication"},
        {"text": "250 mg", "type": "Dose"},
        {"text": "PO", "type": "Mode"},
        {"text": "daily", "type": "Frequency"},
        {"text": "pain", "type": "Reason"},
    ]})

    pred_bio, st, pred_char = predict_note(
        None, gold, text, chunk_words=400, overlap_words=80,
        run_fn=_fake_run(reply), count_fn=_no_count,
    )
    assert pred_bio == gold["bio"]
    assert st["n_chunks"] == 1
    assert st["n_pred_spans"] == 5
    assert st["n_overlap_duplicates"] == 0
    assert st["n_unmatched"] == 0
    assert st["tokens_covered"] == gold["n_tokens"]

    rep = score([gold["bio"]], [pred_bio])
    assert rep["micro avg"]["f1-score"] == 1.0

    # Char offsets come back correct, which is what the error dump relies on.
    assert (5, 12, "Medication") in [tuple(s) for s in pred_char]


def test_bad_reply_degrades_to_all_O_not_a_crash():
    text = "Lasix 40mg IV"
    gold = build_gold({"note_id": "s", "text": text, "spans": [[0, 5, "Medication"]]})
    pred_bio, st, _ = predict_note(
        None, gold, text, run_fn=_fake_run("the model apologizes"),
        count_fn=_no_count,
    )
    assert pred_bio == ["O", "O", "O"]
    assert st["n_pred_spans"] == 0


def test_chunk_inference_exception_is_contained():
    text = " ".join(["word"] * 50)
    gold = build_gold({"note_id": "s", "text": text, "spans": []})

    def boom(pipe, chunk_text):
        raise RuntimeError("simulated OOM")

    pred_bio, st, _ = predict_note(
        None, gold, text, chunk_words=10, overlap_words=2,
        run_fn=boom, count_fn=_no_count,
    )
    assert set(pred_bio) == {"O"}
    assert len(pred_bio) == 50
    assert st["n_chunk_failures"] == st["n_chunks"] > 1


# --- multi-chunk: coverage, dedupe, expansion ------------------------------

def test_every_token_is_sent_to_the_model():
    text = " ".join(f"w{i}" for i in range(1000))
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    _, st, _ = predict_note(
        None, gold, text, chunk_words=400, overlap_words=80,
        run_fn=_fake_run('{"entities": []}'), count_fn=_no_count,
    )
    # No content dropped — the report states this as 0 tokens truncated.
    assert st["tokens_covered"] == gold["n_tokens"] == 1000
    assert st["n_chunks"] == 3


def test_entity_in_overlap_region_is_deduped():
    # "Lasix" sits at token 350, inside the (320,400) overlap of windows
    # (0,400) and (320,720), so both chunks predict it.
    words = [f"w{i}" for i in range(500)]
    words[350] = "Lasix"
    text = " ".join(words)
    start = text.index("Lasix")
    gold = build_gold({
        "note_id": "s", "text": text,
        "spans": [[start, start + 5, "Medication"]],
    })
    reply = json.dumps({"entities": [{"text": "Lasix", "type": "Medication"}]})

    pred_bio, st, _ = predict_note(
        None, gold, text, chunk_words=400, overlap_words=80,
        run_fn=_fake_run(reply), count_fn=_no_count,
    )
    assert st["n_overlap_duplicates"] == 1
    assert st["n_pred_spans"] == 1          # counted once, not twice
    assert pred_bio.count("B-Medication") == 1
    assert pred_bio == gold["bio"]
    rep = score([gold["bio"]], [pred_bio])
    assert rep["micro avg"]["precision"] == 1.0


def test_multi_occurrence_expansion_is_measured():
    # "Drugzol" appears 4 times inside one chunk; the model names it once, and
    # all-per-chunk alignment tags all 4. The stats must expose that — this
    # expansion is exactly why first-per-chunk became the default.
    words = [f"w{i}" for i in range(40)]
    for i in (3, 11, 22, 33):
        words[i] = "Drugzol"
    text = " ".join(words)
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    reply = json.dumps({"entities": [{"text": "Drugzol", "type": "Medication"}]})

    _, st, _ = predict_note(
        None, gold, text, chunk_words=400, overlap_words=80,
        run_fn=_fake_run(reply), count_fn=_no_count,
        align_mode="all-per-chunk",
    )
    assert st["n_pred_mentions"] == 1
    assert st["n_pred_unique"] == 1
    assert st["n_pred_unique_chunksum"] == 1
    assert st["n_aligned_spans"] == 4        # the expansion the report names
    assert st["n_pred_spans"] == 4


def test_hallucinated_span_counted_as_unmatched():
    text = "Lasix 40mg IV"
    gold = build_gold({"note_id": "s", "text": text, "spans": [[0, 5, "Medication"]]})
    reply = json.dumps({"entities": [
        {"text": "Lasix", "type": "Medication"},
        {"text": "metoprolol", "type": "Medication"},   # not in the text
    ]})
    _, st, _ = predict_note(
        None, gold, text, run_fn=_fake_run(reply), count_fn=_no_count,
    )
    assert st["n_pred_unique"] == 2
    assert st["n_unmatched"] == 1
    assert st["n_pred_spans"] == 1


def test_generation_cap_hit_is_logged():
    text = "Lasix 40mg IV"
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    reply = '{"entities": [{"text": "Lasix", "type": "Medication"}'  # truncated

    # count_fn reports a length at the configured cap.
    _, st, _ = predict_note(
        None, gold, text, run_fn=_fake_run(reply),
        count_fn=lambda pipe, r: 1024, gen_config={"max_new_tokens": 1024},
    )
    assert st["n_cap_hits"] == 1


def test_generation_below_cap_is_not_flagged():
    text = "Lasix 40mg IV"
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    _, st, _ = predict_note(
        None, gold, text, run_fn=_fake_run('{"entities": []}'),
        count_fn=lambda pipe, r: 12, gen_config={"max_new_tokens": 1024},
    )
    assert st["n_cap_hits"] == 0


def test_unknown_token_count_is_not_treated_as_truncation():
    text = "Lasix 40mg IV"
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    _, st, _ = predict_note(
        None, gold, text, run_fn=_fake_run('{"entities": []}'), count_fn=_no_count,
    )
    assert st["n_cap_hits"] == 0


# --- incremental save + resume --------------------------------------------

def _write_sample(tmp_path, n):
    """A synthetic sample file in the shape build_mimic_sample.py produces."""
    path = tmp_path / "sample.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            text = f"Note{i} patient received Lasix 40mg IV daily"
            start = text.index("Lasix")
            f.write(json.dumps({
                "note_id": f"n{i}-DS-1",
                "text": text,
                "spans": [[start, start + 5, "Medication"]],
                "n_type_conflicts": 0,
            }) + "\n")
    return str(path)


def _run(tmp_path, sample, **kw):
    reply = json.dumps({"entities": [{"text": "Lasix", "type": "Medication"}]})
    import src.evaluate_mimic as ev
    orig = ev.predict_note

    def patched(pipe, gold, text, **inner):
        inner.pop("run_fn", None)
        inner.pop("count_fn", None)
        return orig(pipe, gold, text, run_fn=_fake_run(reply),
                    count_fn=_no_count, **inner)

    ev.predict_note = patched
    try:
        return ev.run_eval(
            sample_file=sample,
            results_dir=str(tmp_path / "results"),
            output_dir=str(tmp_path / "outputs"),
            load_model=False, **kw
        )
    finally:
        ev.predict_note = orig


def test_per_note_results_are_written_incrementally(tmp_path):
    sample = _write_sample(tmp_path, 4)
    _run(tmp_path, sample, n=4)

    per_note = list((tmp_path / "outputs").rglob("per_note.jsonl"))
    assert len(per_note) == 1
    lines = [json.loads(x) for x in per_note[0].read_text().splitlines() if x.strip()]
    assert len(lines) == 4
    assert {l["note_id"] for l in lines} == {f"n{i}-DS-1" for i in range(4)}


def test_per_note_file_contains_no_note_text(tmp_path):
    """The resume file must stay PHI-free: BIO tags and integer counts only."""
    sample = _write_sample(tmp_path, 3)
    _run(tmp_path, sample, n=3)
    raw = list((tmp_path / "outputs").rglob("per_note.jsonl"))[0].read_text()
    assert "Lasix" not in raw
    assert "patient received" not in raw
    for rec in (json.loads(x) for x in raw.splitlines() if x.strip()):
        for key, value in rec.items():
            assert isinstance(value, (int, list, str))
            if isinstance(value, list):
                assert all(v == "O" or v[:2] in ("B-", "I-") for v in value)


def test_resume_skips_already_finished_notes(tmp_path):
    sample = _write_sample(tmp_path, 6)
    _run(tmp_path, sample, n=3)
    per_note = list((tmp_path / "outputs").rglob("per_note.jsonl"))[0]
    assert len(per_note.read_text().strip().splitlines()) == 3

    # Second pass over 6 notes must only add the 3 that were missing.
    _run(tmp_path, sample, n=6)
    lines = [json.loads(x) for x in per_note.read_text().splitlines() if x.strip()]
    assert len(lines) == 6
    assert len({l["note_id"] for l in lines}) == 6


def test_no_resume_reruns_everything(tmp_path):
    sample = _write_sample(tmp_path, 3)
    _run(tmp_path, sample, n=3)
    _run(tmp_path, sample, n=3, resume=False)
    per_note = list((tmp_path / "outputs").rglob("per_note.jsonl"))[0]
    # Appended again, so scoring must dedupe by note_id rather than trust length.
    assert len(per_note.read_text().strip().splitlines()) == 6


def test_partial_final_line_from_a_killed_run_is_tolerated(tmp_path):
    sample = _write_sample(tmp_path, 3)
    _run(tmp_path, sample, n=2)
    per_note = list((tmp_path / "outputs").rglob("per_note.jsonl"))[0]
    with open(per_note, "a") as f:
        f.write('{"note_id": "n2-DS-1", "gold_bio": ["O"')  # truncated mid-write

    report = _run(tmp_path, sample, n=3)
    assert report["micro avg"]["support"] == 3


def test_smoke_limit_writes_its_own_results_file(tmp_path):
    sample = _write_sample(tmp_path, 10)
    _run(tmp_path, sample, limit=3)
    results = {p.name for p in (tmp_path / "results").iterdir()}
    assert "mimic_ner_smoke_3.csv" in results
    assert "mimic_ner_smoke_3_report.md" in results


def test_results_csv_schema_matches_the_existing_comparison_csv(tmp_path):
    import pandas as pd
    sample = _write_sample(tmp_path, 4)
    _run(tmp_path, sample, n=4)
    df = pd.read_csv(tmp_path / "results" / "mimic_ner_4.csv")
    assert list(df.columns) == ["model", "entity", "precision", "recall", "f1",
                                "support"]
    assert "micro avg" in set(df.entity)


def test_report_is_written_and_has_no_note_text(tmp_path):
    sample = _write_sample(tmp_path, 4)
    _run(tmp_path, sample, n=4)
    md = (tmp_path / "results" / "mimic_ner_4_report.md").read_text()
    assert "Lasix" not in md
    assert "patient received" not in md
    for section in ("## Sample", "## Model and configuration", "## Results",
                    "## Gold type", "## Chunking", "## Caveats"):
        assert section in md
    assert "LLM-generated" in md


# --- alignment modes ------------------------------------------------------

def _repeated_note(n_repeats=4, n_tokens=40):
    """A note where one drug name recurs inside a single chunk."""
    words = [f"w{i}" for i in range(n_tokens)]
    for i in range(3, 3 + n_repeats * 8, 8):
        words[i] = "Drugzol"
    return " ".join(words)


def test_align_mode_default_is_first_per_chunk():
    from src.mimic_config import ALIGN_MODE, DEFAULT_ALIGN_MODE
    assert DEFAULT_ALIGN_MODE == "first-per-chunk"
    assert ALIGN_MODE == "first-per-chunk"


def test_all_per_chunk_tags_every_occurrence():
    text = _repeated_note()
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    reply = json.dumps({"entities": [{"text": "Drugzol", "type": "Medication"}]})
    _, st, _ = predict_note(None, gold, text, run_fn=_fake_run(reply),
                            count_fn=_no_count, align_mode="all-per-chunk")
    assert st["n_pred_spans"] == 4


def test_first_per_chunk_tags_one_per_chunk():
    text = _repeated_note()
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    reply = json.dumps({"entities": [{"text": "Drugzol", "type": "Medication"}]})
    _, st, _ = predict_note(None, gold, text, chunk_words=400, overlap_words=80,
                            run_fn=_fake_run(reply), count_fn=_no_count,
                            align_mode="first-per-chunk")
    assert st["n_chunks"] == 1
    assert st["n_pred_spans"] == 1


def test_first_note_tags_one_per_note_across_chunks():
    # 500 tokens -> 2 chunks; the drug recurs in both.
    words = [f"w{i}" for i in range(500)]
    for i in (10, 100, 350, 450):
        words[i] = "Drugzol"
    text = " ".join(words)
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    reply = json.dumps({"entities": [{"text": "Drugzol", "type": "Medication"}]})

    _, st_note, _ = predict_note(None, gold, text, chunk_words=400,
                                 overlap_words=80, run_fn=_fake_run(reply),
                                 count_fn=_no_count, align_mode="first-note")
    assert st_note["n_chunks"] > 1
    assert st_note["n_pred_spans"] == 1          # one span for the whole note
    assert st_note["n_overlap_duplicates"] == 0  # nothing stitched, so none
    # n_aligned_spans must describe what was actually scored, not discarded
    # per-chunk work.
    assert st_note["n_aligned_spans"] == 1


def test_modes_are_ordered_by_predicted_span_count():
    words = [f"w{i}" for i in range(500)]
    for i in (10, 100, 350, 450):
        words[i] = "Drugzol"
    text = " ".join(words)
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    reply = json.dumps({"entities": [{"text": "Drugzol", "type": "Medication"}]})
    counts = {}
    for mode in ("all-per-chunk", "first-per-chunk", "first-note"):
        _, st, _ = predict_note(None, gold, text, chunk_words=400,
                                overlap_words=80, run_fn=_fake_run(reply),
                                count_fn=_no_count, align_mode=mode)
        counts[mode] = st["n_pred_spans"]
    assert counts["all-per-chunk"] > counts["first-per-chunk"] >= counts["first-note"]


def test_unknown_align_mode_raises():
    import pytest
    text = "Lasix 40mg IV"
    gold = build_gold({"note_id": "s", "text": text, "spans": []})
    with pytest.raises(ValueError, match="align_mode"):
        predict_note(None, gold, text, run_fn=_fake_run('{"entities": []}'),
                     count_fn=_no_count, align_mode="nonsense")


def test_run_tag_separates_align_modes():
    from src.evaluate_mimic import run_tag
    a = run_tag(13, 400, 80, "m", "all-per-chunk")
    b = run_tag(13, 400, 80, "m", "first-per-chunk")
    assert a != b, "resume caches from different modes must not collide"


def test_default_mode_keeps_plain_result_filenames(tmp_path):
    sample = _write_sample(tmp_path, 3)
    _run(tmp_path, sample, n=3)
    names = {p.name for p in (tmp_path / "results").iterdir()}
    assert "mimic_ner_3.csv" in names
    assert not any("first-per-chunk" in n for n in names)


def test_non_default_mode_suffixes_result_filenames(tmp_path):
    sample = _write_sample(tmp_path, 3)
    _run(tmp_path, sample, n=3, align_mode="all-per-chunk")
    names = {p.name for p in (tmp_path / "results").iterdir()}
    assert "mimic_ner_3_all-per-chunk.csv" in names
