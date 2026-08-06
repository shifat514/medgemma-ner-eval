"""MIMIC-IV medication-NER data loading — the only genuinely new logic.

Source data (both OUTSIDE the repo, credentialed PhysioNet downloads):
  - discharge.csv.gz            note_id, subject_id, hadm_id, note_type,
                                note_seq, charttime, storetime, text
  - mimic-iv-note-labels/*.csv  Start Position, End Position, Annotation, Group
                                one file per annotated note, 600 files,
                                filename encodes the note_id.

Label facts established by inspecting all 600 files (64,106 annotation rows):
  - 6 annotation types: MEDICATION, DOSE, FREQUENCY, MODE, REASON, DURATION.
  - Char offsets are half-open [start, end) into the note's raw `text`.
  - Zero malformed spans; zero spans with surrounding whitespace; zero spans
    starting mid-word -> offsets land on whitespace-token boundaries.
  - `Group` ties one medication event's spans together. Irrelevant to
    token-level NER, so it is dropped after deduping.
  - 4,408 rows are exact (start, end, type) duplicates across groups.
  - ~58 (start, end) pairs carry two different types -> resolved by
    mimic_config.TYPE_PRIORITY and logged.

Nothing here writes note text anywhere except the gitignored sample file.
"""

import csv
import io
import json
import os
import random
import re

from .. import s3_io
from ..chunking import char_spans_to_token_spans, spans_to_bio, tokenize_with_spans
from ..mimic_config import (
    DISCHARGE_CSV,
    EXCLUDED_NOTE_IDS,
    GOLD_TYPE_MAP,
    LABEL_DIR,
    MAX_SAMPLE,
    SEED,
    TYPE_PRIORITY,
)

# Same as the dataset README's regex. `_` is outside the character class, so the
# greedy match stops at `_HadmID`.
_NOTE_ID_RE = re.compile(r"NoteID-([0-9A-Za-z-]+)")


def extract_note_id(filename):
    """Pull the note_id out of a label filename.

    'Labels_..._NoteID-10026165-DS-14_HadmID-20319648_...csv' -> '10026165-DS-14'
    """
    m = _NOTE_ID_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"could not extract note_id from {filename!r}")
    return m.group(1)


def list_label_files(label_dir=None):
    """All label CSV paths, sorted for determinism.

    `label_dir` may be a local directory or an ``s3://bucket/prefix``.
    """
    return s3_io.list_csv(label_dir or LABEL_DIR)


def _parse_label_text(text):
    """Parse label-CSV text into ``(rows, skipped)``. See read_label_rows."""
    rows, skipped = [], []
    for rec in csv.DictReader(io.StringIO(text)):
        try:
            start = int(rec["Start Position"])
            end = int(rec["End Position"])
            gold_type = (rec["Annotation"] or "").strip().upper()
        except (KeyError, TypeError, ValueError):
            skipped.append(dict(rec))
            continue
        if gold_type not in GOLD_TYPE_MAP or start < 0 or end <= start:
            skipped.append(dict(rec))
            continue
        rows.append((start, end, GOLD_TYPE_MAP[gold_type]))
    return rows, skipped


# path -> (rows, skipped). eligible_note_ids() reads all ~600 label files to
# find the empty ones, then build_sample() re-reads the sampled subset. Without
# this cache that is ~700 reads: cheap locally, 700 network round trips on S3.
_LABEL_CACHE = {}


def read_label_rows(path):
    """Raw ``(start, end, gold_type)`` rows from one label CSV.

    Unknown annotation types and unparseable rows are skipped, and returned in
    the `skipped` list so nothing vanishes silently. Results are cached per path.
    """
    if path not in _LABEL_CACHE:
        _LABEL_CACHE[path] = _parse_label_text(s3_io.read_text(path))
    rows, skipped = _LABEL_CACHE[path]
    return list(rows), list(skipped)


def prefetch_label_files(paths):
    """Warm the read cache, fetching S3 objects in parallel.

    Turns ~600 sequential S3 round trips into a few parallel batches. A no-op
    for local paths (already fast) and for anything already cached.
    """
    todo = [p for p in paths if p not in _LABEL_CACHE]
    if not todo:
        return
    for path, text in s3_io.read_text_many(todo).items():
        _LABEL_CACHE[path] = _parse_label_text(text)


def clear_label_cache():
    _LABEL_CACHE.clear()


def resolve_label_spans(rows):
    """Dedupe rows and resolve type conflicts on identical spans.

    Returns ``(spans, conflicts)``:
      - `spans`: sorted, unique ``(start, end, type)``, one type per (start, end).
      - `conflicts`: ``(start, end, winner, [losers])`` for each (start, end) that
        carried more than one type, so the report can state how many there were.

    The winner is the highest-priority type per mimic_config.TYPE_PRIORITY.
    """
    priority = {t: i for i, t in enumerate(TYPE_PRIORITY)}
    fallback = len(priority)

    by_span = {}
    for start, end, etype in rows:
        by_span.setdefault((start, end), set()).add(etype)

    spans, conflicts = [], []
    for (start, end), types in sorted(by_span.items()):
        ranked = sorted(types, key=lambda t: (priority.get(t, fallback), t))
        winner = ranked[0]
        if len(ranked) > 1:
            conflicts.append((start, end, winner, ranked[1:]))
        spans.append((start, end, winner))
    return spans, conflicts


def eligible_note_ids(label_dir=None):
    """Note IDs eligible for sampling, plus what was excluded and why.

    Returns ``(note_ids, excluded)``. `note_ids` is sorted so the seeded sample
    is reproducible regardless of filesystem ordering. `excluded` maps note_id ->
    reason, covering both mimic_config.EXCLUDED_NOTE_IDS and any label file that
    turns out to have no usable annotations.
    """
    paths = list_label_files(label_dir)
    prefetch_label_files(paths)  # parallel fetch when these live in S3

    by_id, excluded = {}, {}
    for path in paths:
        note_id = extract_note_id(path)
        if note_id in EXCLUDED_NOTE_IDS:
            excluded[note_id] = EXCLUDED_NOTE_IDS[note_id]
            continue
        rows, _ = read_label_rows(path)
        if not rows:
            excluded[note_id] = "no usable annotation rows in label file"
            continue
        by_id[note_id] = path
    return sorted(by_id), excluded


def sample_note_ids(note_ids, n, seed=None):
    """Deterministic sample of `n` note IDs.

    ``random.Random(seed).sample`` over a sorted pool. Because the caller draws
    MAX_SAMPLE once and slices, the 50-note set is the first 50 of the 100-note
    draw and therefore a strict subset — the two runs are directly comparable.
    """
    seed = SEED if seed is None else seed
    pool = sorted(note_ids)
    n = min(n, len(pool))
    return random.Random(seed).sample(pool, n)


def build_sample(n=None, seed=None, label_dir=None, discharge_csv=None):
    """Read the source data and return the sample records + provenance stats.

    One streaming pass over discharge.csv.gz (~40 s for 331,793 rows) pulling
    only the sampled notes. The full 1.1GB file is never copied or loaded.

    Each record: ``{note_id, text, spans: [[start, end, type], ...]}``.
    """
    n = MAX_SAMPLE if n is None else n
    seed = SEED if seed is None else seed
    label_dir = label_dir or LABEL_DIR
    discharge_csv = discharge_csv or DISCHARGE_CSV

    pool, excluded = eligible_note_ids(label_dir)
    selected = sample_note_ids(pool, n, seed)
    wanted = set(selected)

    label_paths = {extract_note_id(p): p for p in list_label_files(label_dir)}

    labels, n_conflicts, conflicts_total, conflict_examples = {}, {}, 0, []
    n_rows_raw, n_unique_triples = {}, {}
    for note_id in selected:
        rows, _ = read_label_rows(label_paths[note_id])
        spans, conflicts = resolve_label_spans(rows)
        labels[note_id] = spans
        n_conflicts[note_id] = len(conflicts)
        # Pre-dedupe counts, recorded here because resolve_label_spans runs at
        # sample-build time — the sample file only ever sees resolved spans.
        n_rows_raw[note_id] = len(rows)
        n_unique_triples[note_id] = len(set(rows))
        conflicts_total += len(conflicts)
        for start, end, winner, losers in conflicts:
            conflict_examples.append(
                {"note_id": note_id, "start": start, "end": end,
                 "kept": winner, "dropped": losers}
            )

    # Stream discharge.csv.gz with the stdlib csv reader. The `text` field is
    # multi-line quoted CSV, which the csv module handles and a line-oriented
    # scan would not. field_size_limit is raised for the 36KB note bodies.
    #
    # s3_io streams and decompresses on the fly, so an s3:// source never lands
    # the 1.1 GB file on local disk. The loop breaks as soon as every sampled
    # note is found, which for a mid-file seed still means reading most of the
    # stream (~326k of 331k rows for seed 13).
    csv.field_size_limit(1 << 24)
    found, rows_scanned = {}, 0
    with s3_io.open_gzip_text(discharge_csv) as fh:
        for rec in csv.DictReader(fh):
            rows_scanned += 1
            note_id = rec.get("note_id")
            if note_id in wanted and note_id not in found:
                found[note_id] = rec.get("text") or ""
                if len(found) == len(wanted):
                    break

    missing = sorted(wanted - set(found))

    records = []
    for note_id in selected:
        if note_id not in found:
            continue
        records.append({
            "note_id": note_id,
            "text": found[note_id],
            "spans": [list(s) for s in labels[note_id]],
            # Carried per-note so the report can total tie-breaks and pre-dedupe
            # counts for whatever sample size is scored, not just the full draw.
            "n_type_conflicts": n_conflicts[note_id],
            "n_label_rows_raw": n_rows_raw[note_id],
            "n_unique_triples": n_unique_triples[note_id],
        })

    stats = {
        "seed": seed,
        "requested": n,
        "pool_size": len(pool),
        "label_files": len(label_paths),
        "excluded": excluded,
        "selected_order": selected,
        "notes_found": len(found),
        "notes_missing": missing,
        "discharge_rows_scanned": rows_scanned,
        "type_conflicts": conflicts_total,
        "type_conflict_detail": conflict_examples,
        "gold_type_map": dict(GOLD_TYPE_MAP),
    }
    return records, stats


def write_sample(records, stats, path):
    """Write the sample as JSONL + a sidecar stats JSON.

    CONTAINS NOTE TEXT — `path` must live under a gitignored directory.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    stats_path = os.path.splitext(path)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path, stats_path


def load_sample(path=None):
    """Read the sample JSONL back. Order is the seeded selection order."""
    from ..mimic_config import SAMPLE_FILE
    path = path or SAMPLE_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"sample file not found: {path}\n"
            "Build it locally with:  python -m src.build_mimic_sample\n"
            "(in Colab, upload the file produced by that command)"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_gold(record):
    """Tokenize one sample record and build its gold BIO sequence.

    Returns a dict with `tokens`, `char_spans`, `bio`, and the counts needed for
    honest reporting (`n_gold_spans`, `n_dropped_overlap`).
    """
    text = record["text"]
    tokens, char_spans = tokenize_with_spans(text)

    raw = [(int(s), int(e), t) for s, e, t in record["spans"]]
    out_of_bounds = [s for s in raw if s[1] > len(text)]
    in_bounds = [s for s in raw if s[1] <= len(text)]

    indexed = char_spans_to_token_spans(in_bounds, char_spans, with_index=True)
    token_spans = [(a, b, t) for a, b, t, _ in indexed]

    # How often snapping a gold span to whole tokens moves a boundary by more
    # than one character. One char is a trailing '.'/',' and is noise; more than
    # that means the gold boundary fell inside a run-together token. Reported so
    # the tokenization's cost is visible rather than assumed away.
    snapped = 0
    for a, b, _, i in indexed:
        gold_start, gold_end, _ = in_bounds[i]
        if abs(char_spans[a][0] - gold_start) > 1 or abs(char_spans[b - 1][1] - gold_end) > 1:
            snapped += 1

    bio, dropped = spans_to_bio(len(tokens), token_spans, priority=TYPE_PRIORITY)

    return {
        "note_id": record["note_id"],
        "tokens": tokens,
        "char_spans": char_spans,
        "bio": bio,
        "n_chars": len(text),
        "n_tokens": len(tokens),
        "n_gold_raw": len(raw),
        "n_gold_spans": len(token_spans),
        "n_gold_out_of_bounds": len(out_of_bounds),
        "n_gold_dropped_overlap": len(dropped),
        "n_gold_no_token": len(in_bounds) - len(token_spans),
        "n_gold_boundary_snapped": snapped,
        "n_type_conflicts": int(record.get("n_type_conflicts", 0)),
        "n_label_rows_raw": int(record.get("n_label_rows_raw", len(raw))),
        "n_unique_triples": int(record.get("n_unique_triples", len(set(
            (int(s), int(e), t) for s, e, t in record["spans"])))),
    }
