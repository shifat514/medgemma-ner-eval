"""Load the 100-row MDACE cut and build one accept-set per billed code.

ONE FILE, NO JOIN. `8-07-mdace-ner-eval_sample_100-LOCAL.jsonl` embeds
`note_text` on every row, so unlike the term-NER path there is no notes file and
no join step. 100 rows land on 24 notes and 91 distinct
`(note, code_system, code)` triples — nine codes are evidenced twice on their
own note, which is why the row count and the code count differ and why recall is
reported both ways.

THE ACCEPT-SET. Scoring a model against `mdace_gold_evidence_text` alone made
`HTN` unable to match `Essential (primary) hypertension` no matter how right the
model was. So per row the accept-set is the union of three columns:

    mdace_gold_evidence_text     what the note says
    gold_code_description        the ICD catalogue wording
    gold_snomed_concepts[].term  the SNOMED terms shipped for that code

Median 4 accepted forms per row, min 2, max 5 — against 1 before. Every form
carries the source it came from, because recall and false positives are reported
per source as well as combined.

THE TRUNCATION QUESTION IS CLOSED, NOT DEFERRED. The cut was made at 100
annotation rows, so the 24 notes it reaches carry 195 evidence phrases in the
full corpus and this file ships 99 of them. Under the term-NER scope that
inflated false positives. Under a benchmark scope gold is whatever this file
says it is, and the missing 96 are simply not part of the question.

WHAT THE SNOMED COLUMN IS AND IS NOT. It is capped at 3 entries and the
survivors are not the top-ranked ones: an HCV row ships two pregnancy-related
concepts and omits plain "hepatitis C". This is a limit on the number, not a
request to anyone to fix the file — `snomed_coverage` measures it so the report
can state it.
"""

import json
import os
import re

from ..recall_config import CHUNK_WORDS, OVERLAP_WORDS, SOURCES

# Lowercase, every run of non-alphanumerics collapsed to one space. IDENTICAL to
# datasets.mdace.normalize_term by design: the 0.53 reference number from the
# term-NER branch is only comparable if both sides fold strings the same way.
# Applied to gold and predictions alike.
_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_term(text):
    """Fold a term to its comparison form. Returns "" for unusable input."""
    if not isinstance(text, str):
        return ""
    return " ".join(_NORM_RE.sub(" ", text.lower()).split())


def padded_note_norm(text):
    """Normalized note text, space-padded so substring tests hit word edges.

    Normalization collapses every run of non-alphanumerics to a single space, so
    a space-padded ``in`` test is whole-token containment: "ca" will not match
    inside "cabg". Used for the not-in-note (hallucination) check.
    """
    return f" {normalize_term(text)} "


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


def row_accept_set(row):
    """``{normalized form: sorted sources}`` for one annotation row.

    A form reachable from two columns keeps both sources, so it is credited to
    each of them in the per-source breakdown rather than being assigned to
    whichever column happened to be read first.
    """
    forms = {}

    def add(raw, source):
        norm = normalize_term(raw)
        if norm:
            forms.setdefault(norm, set()).add(source)

    add(row.get("mdace_gold_evidence_text"), "evidence")
    add(row.get("gold_code_description"), "description")
    for concept in row.get("gold_snomed_concepts") or ():
        if isinstance(concept, dict):
            add(concept.get("term"), "snomed")

    return {norm: sorted(src) for norm, src in forms.items()}


def n_chunks(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Model calls this note will cost."""
    from ..chunking import chunk_windows, tokenize_with_spans

    tokens, _ = tokenize_with_spans(text)
    return len(chunk_windows(len(tokens), chunk_words, overlap_words))


def build_notes(path, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Group the file into per-note records. Returns ``(records, stats)``.

    Each record carries:

      rows    one entry per annotation row, in file order, with its accept-set
      forms   ``{normalized form: {"sources", "rows", "codes"}}`` — the note's
              whole accept-set, deduped across rows. Deduping here is what stops
              one prediction being charged twice: a phrase justifying two codes
              is ONE gold form that credits BOTH rows, not two forms that a
              single prediction has to satisfy separately.

    Records are ordered longest-note-first so a `--limit` run exercises the
    multi-chunk path rather than a handful of one-chunk notes.
    """
    rows = read_jsonl(path)

    by_note = {}
    for idx, row in enumerate(rows):
        by_note.setdefault(row["note_id"], []).append((idx, row))

    records = []
    for note_id, note_rows in by_note.items():
        text = note_rows[0][1].get("note_text") or ""
        entries, forms = [], {}

        for idx, row in note_rows:
            accept = row_accept_set(row)
            code_key = f"{row.get('code_system')}|{row.get('gold_code')}"
            entries.append({
                "row_id": idx,
                "code_system": row.get("code_system"),
                "code": row.get("gold_code"),
                "code_key": code_key,
                # Raw, unnormalized. Only the oracle uses it, to locate which
                # chunk a gold row belongs to; scoring never touches it.
                "evidence_text": row.get("mdace_gold_evidence_text") or "",
                "n_snomed_shipped": len(row.get("gold_snomed_concepts") or ()),
                "n_snomed_reported": row.get("gold_snomed_concept_count") or 0,
                "accept": accept,
            })
            for norm, srcs in accept.items():
                slot = forms.setdefault(
                    norm, {"sources": set(), "rows": set(), "codes": set()})
                slot["sources"].update(srcs)
                slot["rows"].add(idx)
                slot["codes"].add(code_key)

        records.append({
            "note_id": note_id,
            "hadm_id": note_rows[0][1].get("hadm_id"),
            "chart_type": note_rows[0][1].get("chart_type"),
            "text": text,
            "n_chunks": n_chunks(text, chunk_words, overlap_words),
            "rows": entries,
            "forms": {norm: {"sources": sorted(v["sources"]),
                             "rows": sorted(v["rows"]),
                             "codes": sorted(v["codes"])}
                      for norm, v in forms.items()},
        })

    records.sort(key=lambda r: (-r["n_chunks"], r["note_id"]))
    return records, corpus_stats(records, rows)


def source_forms(record, source=None):
    """The note's gold forms, optionally restricted to one source.

    ``source=None`` is the combined accept-set — every form from every column.
    """
    if source is None:
        return set(record["forms"])
    return {norm for norm, f in record["forms"].items()
            if source in f["sources"]}


def reachable_rows(record, source=None):
    """Row ids that CAN be recalled under `source`.

    SNOMED ships terms for 53 of 100 rows, so a row-recall denominator of 100 on
    the SNOMED line would be measuring the file's coverage and calling it model
    performance. Every per-source denominator is printed for this reason.
    """
    out = set()
    for entry in record["rows"]:
        if source is None:
            if entry["accept"]:
                out.add(entry["row_id"])
        elif any(source in srcs for srcs in entry["accept"].values()):
            out.add(entry["row_id"])
    return out


def reachable_codes(record, source=None):
    """``code_key`` values that can be recalled under `source`."""
    out = set()
    for entry in record["rows"]:
        if source is None:
            if entry["accept"]:
                out.add(entry["code_key"])
        elif any(source in srcs for srcs in entry["accept"].values()):
            out.add(entry["code_key"])
    return out


def corpus_stats(records, rows):
    """Counts the report prints so the numbers rest on the file, not on prose."""
    accept_sizes = sorted(len(e["accept"])
                          for r in records for e in r["rows"])
    mid = len(accept_sizes) // 2
    median = 0.0
    if accept_sizes:
        median = (accept_sizes[mid] if len(accept_sizes) % 2
                  else (accept_sizes[mid - 1] + accept_sizes[mid]) / 2)

    stats = {
        "n_rows": len(rows),
        "n_notes": len(records),
        "n_chunks": sum(r["n_chunks"] for r in records),
        "n_codes": len({(r["note_id"], e["code_key"])
                        for r in records for e in r["rows"]}),
        "accept_median": median,
        "accept_min": accept_sizes[0] if accept_sizes else 0,
        "accept_max": accept_sizes[-1] if accept_sizes else 0,
        "forms_by_source": {
            s: sum(len(source_forms(r, s)) for r in records) for s in SOURCES},
        "forms_combined": sum(len(source_forms(r)) for r in records),
        "rows_by_source": {
            s: sum(len(reachable_rows(r, s)) for r in records) for s in SOURCES},
        "codes_by_source": {
            s: sum(len(reachable_codes(r, s)) for r in records) for s in SOURCES},
        "code_systems": {},
    }
    for row in rows:
        key = row.get("code_system") or "?"
        stats["code_systems"][key] = stats["code_systems"].get(key, 0) + 1
    stats.update(snomed_coverage(records, rows))
    return stats


def snomed_coverage(records, rows):
    """How much of the SNOMED column the file actually ships.

    `gold_snomed_concept_count` reports how many concepts the upstream mapping
    found; `gold_snomed_concepts` ships at most 3 of them. The gap is the reason
    the report calls L4 an approximation of a real SNOMED lookup rather than the
    thing itself.
    """
    shipped = sum(len(r.get("gold_snomed_concepts") or ()) for r in rows)
    reported = sum(r.get("gold_snomed_concept_count") or 0 for r in rows)
    by_system = {}
    for row in rows:
        key = row.get("code_system") or "?"
        slot = by_system.setdefault(key, {"rows": 0, "with_snomed": 0})
        slot["rows"] += 1
        if row.get("gold_snomed_concepts"):
            slot["with_snomed"] += 1
    return {
        "snomed_rows_with_terms": sum(
            1 for r in rows if r.get("gold_snomed_concepts")),
        "snomed_terms_shipped": shipped,
        "snomed_terms_reported": reported,
        "snomed_shipped_frac": (shipped / reported) if reported else 0.0,
        "snomed_by_code_system": by_system,
    }


def load(path=None, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Load the benchmark input. Raises a pointed error if the file is absent."""
    from ..recall_config import S3_SAMPLE_100_FILE, SAMPLE_100_FILE

    path = path or SAMPLE_100_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"benchmark input not found: {path}\n"
            f"It is credentialed MIMIC-III-derived data and lives outside the "
            f"repo. Mirror: {S3_SAMPLE_100_FILE}\n"
            f"Set RECALL_SAMPLE_FILE to point at your copy."
        )
    return build_notes(path, chunk_words, overlap_words)
