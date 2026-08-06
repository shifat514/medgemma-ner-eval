"""Report rendering — CPU-only, synthetic counts only. NO REAL NOTE TEXT."""

from src.report_mimic import build_report

BASE_NOTE = {
    "n_chunks": 21, "n_cap_hits": 0, "n_items_kept": 582, "n_items_seen": 582,
    "n_tokens": 8000, "tokens_covered": 8000, "n_pred_spans": 330,
    "n_chunks_no_json": 0, "n_chunks_empty_list": 0, "n_chunks_zero_entities": 0,
    "n_items_rejected_type": 0, "n_items_aliased": 0, "n_items_no_text": 0,
    "n_gold_raw": 372, "n_gold_spans": 372, "n_label_rows_raw": 380,
    "n_unique_triples": 375, "n_gold_out_of_bounds": 0,
    "n_gold_dropped_overlap": 0, "n_gold_no_token": 0,
    "n_gold_boundary_snapped": 30, "n_type_conflicts": 0,
    "n_pred_mentions": 582, "n_pred_unique": 400, "n_pred_unique_chunksum": 450,
    "n_aligned_spans": 340, "n_unmatched": 50, "n_overlap_duplicates": 10,
    "n_pred_dropped_overlap": 0, "n_chunk_failures": 0,
}
REPORT = {"micro avg": {"precision": .3242, "recall": .2876,
                        "f1-score": .3048, "support": 372}}
META = {"label": "smoke_5", "seed": 13, "model_id": "google/medgemma-4b-it",
        "gen_config": {"max_new_tokens": 1536}, "chunk_words": 400,
        "overlap_words": 80, "align_mode": "first-per-chunk", "oracle": False,
        "notes_missing": []}


def _render(**note_overrides):
    note = {**BASE_NOTE, **note_overrides}
    return build_report(REPORT, [note], META)


def test_truncation_banner_appears_when_cap_is_binding():
    md = _render(n_cap_hits=5)
    assert "Generation hit `max_new_tokens` on 5 of 21 chunks (23.8%)" in md
    # must be near the top, before the metrics, so it cannot be missed
    assert md.index("max_new_tokens") < md.index("## Results")


def test_no_truncation_banner_when_cap_never_hit():
    md = _render(n_cap_hits=0)
    assert "⚠️" not in md


def test_cap_row_is_present_in_extraction_health_either_way():
    assert "generation hit `max_new_tokens`" in _render(n_cap_hits=0).lower()


def test_over_extraction_uses_the_oracle_baseline_not_raw_gold():
    """582 items / 372 gold = 1.56x raw, but 1.23x of that is chunk overlap."""
    md = _render()
    assert "| items emitted per gold span | 1.56x |" in md
    assert "**over-extraction vs a perfect extractor** | **1.27x**" in md


def test_span_level_rate_is_reported_separately_from_item_level():
    """The model can over-emit items while under-predicting spans."""
    md = _render()
    assert "| predicted spans per gold span | 0.89x |" in md


def test_a_perfect_extractor_scores_1x_against_the_baseline():
    """Sanity: feed the oracle's own measured rate, expect ~1.00x."""
    md = _render(n_items_kept=int(372 * 1.23), n_pred_spans=372)
    assert "**over-extraction vs a perfect extractor** | **1.00x**" in md


def test_report_still_contains_no_note_text_fields():
    md = _render(n_cap_hits=5)
    for banned in ("gold_bio", "pred_bio", "text"):
        assert f'"{banned}"' not in md
