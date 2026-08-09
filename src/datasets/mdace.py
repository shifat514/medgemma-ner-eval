"""MDACE billing-evidence data: load, join, stratify, and build the sample file.

Two source files that join 1:1 on ``note_id``:

  ``*_dataset-LOCAL.jsonl``  9,499 annotation rows, no note text. One row is one
                            billed code plus the exact phrase a human coder
                            highlighted in the note as its justification.
  ``*_notes-LOCAL.jsonl``    1,074 rows of ``{note_id, note_text}``.

All 9,499 evidence offsets slice exactly — ``note_text[begin:end] ==
mdace_gold_evidence_text`` — so the phrases are guaranteed to be literal
substrings of their note. ``verify_offsets`` re-checks that rather than assuming
it, because everything downstream depends on it.

WHAT COUNTS AS GOLD. ``mdace_gold_evidence_text`` (what the note says), never
``gold_code_description`` (the ICD catalogue wording). The two match after
normalization in only 424 of 9,499 rows — the note says "depression", the
catalogue says "Major depressive disorder, single episode, unspecified". Scoring
against the description would mark a correct extraction wrong. The description
is still carried through, because view A1 scores against it deliberately to show
what that choice costs.
"""

import json
import os
import random
import re

from ..mdace_config import (
    CHART_TYPES,
    CHUNK_WORDS,
    OVERLAP_WORDS,
    SEED,
    STRATUM_N,
    STRATUM_ORDER,
    code_bucket,
)

# Lowercase, every run of non-alphanumerics (punctuation AND whitespace,
# including the newlines inside multi-line gold spans) collapsed to one space.
# Applied identically to gold and predictions. This exact normalizer collapses
# the 9,499 rows to 7,663 distinct (note_id, term) pairs, matching the corpus
# figure — a cheap check that it is the same rule used upstream.
_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_term(text):
    """Fold a term to its comparison form. Returns "" for unusable input."""
    if not isinstance(text, str):
        return ""
    return " ".join(_NORM_RE.sub(" ", text.lower()).split())


def read_jsonl(path):
    """Parse a JSONL file, skipping blank and truncated lines."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_notes(notes_file):
    """``{note_id: note_text}``."""
    return {r["note_id"]: r["note_text"] for r in read_jsonl(notes_file)}


def note_chart_type(rows):
    """Collapse a note's annotation rows to one chart type.

    chart_type is per-row, and 118 notes carry both — the same encounter billed
    as Profee (an individual clinician's professional fee) and as Inpatient (the
    facility's stay). Inpatient wins, which reproduces the corpus split of 604
    Inpatient / 470 Profee and both word/term medians. See mdace_config.
    """
    types = {r.get("chart_type") for r in rows}
    return "Inpatient" if "Inpatient" in types else "Profee"


def verify_offsets(rows, notes):
    """Count rows whose evidence offsets do not slice their own note exactly.

    Should be 0. A non-zero result means the notes file and the annotation file
    have drifted apart and no downstream number can be trusted.
    """
    bad = 0
    for r in rows:
        text = notes.get(r["note_id"])
        if text is None:
            bad += 1
            continue
        sliced = text[r["evidence_char_begin"]:r["evidence_char_end"]]
        if sliced != r["mdace_gold_evidence_text"]:
            bad += 1
    return bad


def build_corpus(dataset_file, notes_file):
    """Join the two files into per-note records. Returns ``(notes, stats)``.

    Each record carries every gold annotation for its note, deduped on
    ``(normalized term, code_system, code)`` — 9,499 raw rows collapse to fewer
    because one highlighted phrase often justifies several codes, and the same
    code is sometimes evidenced by two different phrasings in one note.
    """
    rows = read_jsonl(dataset_file)
    notes_text = load_notes(notes_file)

    by_note = {}
    for r in rows:
        by_note.setdefault(r["note_id"], []).append(r)

    # Admission-level gold: the union of every phrase highlighted anywhere in
    # the encounter. An ICD code belongs to an admission, but MDACE marks its
    # evidence on whichever note the coder was reading — so a note can mention a
    # condition that was genuinely billed and still carry no highlight itself.
    # Predictions matching here are almost certainly correct-and-billed, which
    # is the strongest of the three false-positive buckets.
    adm_terms = {}
    for r in rows:
        norm = normalize_term(r["mdace_gold_evidence_text"])
        if norm:
            adm_terms.setdefault(r["hadm_id"], set()).add(norm)

    records, missing = [], []
    for note_id, note_rows in by_note.items():
        text = notes_text.get(note_id)
        if text is None:
            missing.append(note_id)
            continue

        seen, gold = set(), []
        for r in note_rows:
            norm = normalize_term(r["mdace_gold_evidence_text"])
            if not norm:
                continue
            key = (norm, r["code_system"], r["gold_code"])
            if key in seen:
                continue
            seen.add(key)
            gold.append({
                "term": r["mdace_gold_evidence_text"],
                "norm": norm,
                "code": r["gold_code"],
                "code_system": r["code_system"],
                "descr": r.get("gold_code_description") or "",
                "descr_norm": normalize_term(r.get("gold_code_description") or ""),
                "bucket": code_bucket(r["code_system"], r["gold_code"]),
            })

        hadm = note_rows[0].get("hadm_id")
        records.append({
            "note_id": note_id,
            "hadm_id": hadm,
            "chart_type": note_chart_type(note_rows),
            "mixed_chart_type": len({r.get("chart_type") for r in note_rows}) > 1,
            "text": text,
            "gold": gold,
            "gold_terms": sorted({g["norm"] for g in gold}),
            "admission_gold_terms": sorted(adm_terms.get(hadm, set())),
            "n_rows_raw": len(note_rows),
        })

    records.sort(key=lambda r: r["note_id"])
    stats = {
        "n_rows": len(rows),
        "n_notes_dataset": len(by_note),
        "n_notes_file": len(notes_text),
        "n_notes_joined": len(records),
        "n_notes_missing_text": len(missing),
        "n_offset_mismatches": verify_offsets(rows, notes_text),
        "n_distinct_note_term_pairs": sum(len(r["gold_terms"]) for r in records),
        "n_mixed_chart_type": sum(1 for r in records if r["mixed_chart_type"]),
    }
    return records, stats


def stratified_sample(records, n_per_stratum=STRATUM_N, seed=SEED,
                      order=STRATUM_ORDER):
    """Draw `n_per_stratum` notes from each chart type. Returns note_ids.

    THE DRAW ORDER IS PART OF THE SPECIFICATION, not an implementation detail.
    One shared generator walks the strata in `order`, so the Inpatient draw
    continues from the state the Profee draw left behind. Giving each stratum
    its own Random(seed) is an equally defensible implementation and produces a
    completely different sample (167 chunks / 432 terms instead of 122 / 324).
    Pools are sorted first so the draw cannot inherit dict ordering.
    """
    pools = {t: sorted(r["note_id"] for r in records if r["chart_type"] == t)
             for t in CHART_TYPES}
    rng = random.Random(seed)
    picked = []
    for stratum in order:
        pool = pools.get(stratum, [])
        if len(pool) < n_per_stratum:
            raise ValueError(
                f"stratum {stratum!r} has {len(pool)} notes, need {n_per_stratum}"
            )
        picked.extend(rng.sample(pool, n_per_stratum))
    return picked


def load_sample_100(path):
    """Ehtesham Bhai's 100-row cut, keyed by note_id.

    Returns ``{note_id: {"terms": [...], "descrs": [...]}}`` — the normalized
    evidence phrases and code descriptions that this file actually ships.

    This is NOT a complete answer key. The cut was made at 100 annotation ROWS,
    which lands on 24 notes; those 24 notes have 195 distinct gold terms in the
    full dataset and this file carries 99 of them. Scored against it, a model
    that correctly extracts one of the missing 96 is penalized for being right.
    It is used for view A1 only, to measure exactly that effect.
    """
    out = {}
    for r in read_jsonl(path):
        slot = out.setdefault(r["note_id"], {"terms": set(), "descrs": set()})
        term = normalize_term(r.get("mdace_gold_evidence_text"))
        descr = normalize_term(r.get("gold_code_description"))
        if term:
            slot["terms"].add(term)
        if descr:
            slot["descrs"].add(descr)
    return {k: {"terms": sorted(v["terms"]), "descrs": sorted(v["descrs"])}
            for k, v in out.items()}


def n_chunks(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Model calls this note will cost."""
    from ..chunking import chunk_windows, tokenize_with_spans

    tokens, _ = tokenize_with_spans(text)
    return len(chunk_windows(len(tokens), chunk_words, overlap_words))


def build_sample(dataset_file, notes_file, sample_100_file=None,
                 n_per_stratum=STRATUM_N, seed=SEED):
    """Assemble the notes to run, tagged with which views each belongs to.

    The run set is the UNION of the stratified draw and Ehtesham Bhai's 24
    notes. They overlap by exactly one note, so the union is 73 notes / 202
    chunks against 122 for the stratified draw alone. One inference pass covers
    every view: the views differ only in which notes are counted and which
    answer key they are scored against, and neither touches the model.
    """
    records, stats = build_corpus(dataset_file, notes_file)
    by_id = {r["note_id"]: r for r in records}

    stratified = stratified_sample(records, n_per_stratum, seed)
    strat_set = set(stratified)

    shipped = {}
    if sample_100_file and os.path.exists(sample_100_file):
        shipped = load_sample_100(sample_100_file)
    ship_set = set(shipped) & set(by_id)

    selected = []
    for note_id in stratified + sorted(ship_set - strat_set):
        rec = dict(by_id[note_id])
        s = shipped.get(note_id)
        rec["in_stratified"] = note_id in strat_set
        rec["in_sample100"] = note_id in ship_set
        rec["sample100_terms"] = s["terms"] if s else []
        rec["sample100_descrs"] = s["descrs"] if s else []
        rec["n_chunks"] = n_chunks(rec["text"])
        selected.append(rec)

    strat_recs = [r for r in selected if r["in_stratified"]]
    ship_recs = [r for r in selected if r["in_sample100"]]
    stats.update({
        "seed": seed,
        "n_per_stratum": n_per_stratum,
        "stratum_order": list(STRATUM_ORDER),
        "n_selected": len(selected),
        "n_stratified": len(strat_recs),
        "n_sample100": len(ship_recs),
        "n_overlap": len(strat_set & ship_set),
        "chunks_total": sum(r["n_chunks"] for r in selected),
        "chunks_stratified": sum(r["n_chunks"] for r in strat_recs),
        "chunks_sample100": sum(r["n_chunks"] for r in ship_recs),
        "gold_terms_stratified": sum(len(r["gold_terms"]) for r in strat_recs),
        "gold_terms_sample100_full": sum(len(r["gold_terms"]) for r in ship_recs),
        "gold_terms_sample100_shipped": sum(
            len(r["sample100_terms"]) for r in ship_recs),
        "stratified_by_type": {
            t: sum(1 for r in strat_recs if r["chart_type"] == t)
            for t in CHART_TYPES},
        "sample100_by_type": {
            t: sum(1 for r in ship_recs if r["chart_type"] == t)
            for t in CHART_TYPES},
        "n_mixed_in_stratified": sum(
            1 for r in strat_recs if r["mixed_chart_type"]),
    })
    return selected, stats


def write_sample(records, stats, path):
    """Write the sample JSONL + a stats sidecar. CONTAINS NOTE TEXT."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    stats_path = path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    return path, stats_path


def load_sample(path):
    """Read the sample file back, in selection order."""
    return read_jsonl(path)
