"""Parse the pediatric encounter PDFs into notes, gold codes, and input variants.

LOCAL MACHINE ONLY — this module is the only place a PDF is opened, and the
PDFs hold real patient information. Everything downstream reads the gitignored
sample file this produces.

THE PIPELINE, IN ORDER:

    PDF -> pdftotext -layout -> strip page furniture -> split into sections
        -> gold codes from the Assessment block
        -> three input variants, differing only in which blocks are removed

WHY ``pdftotext -layout`` AND NOT A PYTHON PDF LIBRARY. These are Connexin EHR
printouts: the clinically important content is in indented, wrapped, sometimes
two-column-ish blocks, and ``-layout`` is the mode that preserves the line
structure the section parser keys on. A naive text extraction interleaves the
page header with the first body line and the "Assessment" heading stops being a
line of its own — which silently breaks both the section split and the gold
extraction. The binary is checked for at call time with a message that says what
to install.

PAGE FURNITURE IS DROPPED, THE PATIENT BANNER IS KEPT ONCE. Each of the three
pages repeats the clinic address, the report title, the patient's name/sex/DOB,
the date of visit, a "Generated ... Page N of M" line and a Connexin copyright.
The furniture is noise and is dropped outright. The banner is NOT noise — sex
and date of birth are load-bearing for pediatric coding (Z68.5x is *pediatric*
BMI-for-age, Z00.121 is a *child* health exam), so dropping it would remove
information a coder genuinely uses. It is kept on first occurrence and the two
repeats are dropped, so the model sees it exactly once.

WHAT COUNTS AS GOLD. The ``DX n:`` lines inside the Assessment block, and
nothing else. The free-text impression lines that sit above them ("Influenza",
"Viral illness", "Suspect Coxsackie or other non-polio enterovirus") are the
clinician's wording, not codes, and are not scored. The CPT/E&M codes under
Procedures (99213, 99214, 99394) are not scored either — the question asked was
about ICD.
"""

import json
import os
import re
import shutil
import subprocess

from ..billing_config import (
    ASSESSMENT_HEADING,
    PROBLEM_LIST_MARKER,
    SECTION_HEADINGS,
    STRIP_DOT,
    VARIANTS,
)

# ---------------------------------------------------------------------------
# Code normalization
# ---------------------------------------------------------------------------

# ICD-10-CM: a letter, a digit, one alphanumeric, then optionally a dot and up
# to four more. Covers every code in these four notes, from the 4-character
# B08.5 to the 7-character S52.501A.
ICD10_RE = re.compile(r"\b([A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?)\b")

_DX_LINE_RE = re.compile(
    r"^DX\s*(\d+)\s*[:.]?\s*"
    r"([A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?)\s*(.*)$",
    re.IGNORECASE,
)


def normalize_code(code):
    """Uppercase, strip whitespace, keep the decimal point.

    The dot is significant: J11.1 and J111 are the same code written two ways,
    but B08 and B08.5 are different codes and normalizing away the dot would
    not change that. STRIP_DOT exists only so a "did the dot cost us anything?"
    rescore is one env var away; it is off, and the reported numbers keep it.
    """
    if not isinstance(code, str):
        return ""
    out = code.strip().upper().replace(" ", "")
    out = out.rstrip(".,;")
    if STRIP_DOT:
        out = out.replace(".", "")
    return out


# ---------------------------------------------------------------------------
# PDF -> text
# ---------------------------------------------------------------------------


def pdf_to_text(path):
    """Extract `path` with ``pdftotext -layout``. Raises if the binary is absent."""
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext not found. Install poppler-utils:\n"
            "    sudo apt-get install -y poppler-utils"
        )
    proc = subprocess.run(
        ["pdftotext", "-layout", path, "-"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

# Dropped wherever they appear. Every one of these was read off the four
# supplied PDFs; none of them carries clinical content.
_FURNITURE_RE = (
    re.compile(r"^Generated:.*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^Copyright\s*\(c\).*Connexin", re.IGNORECASE),
    re.compile(r"^v\d+(\.\d+)+\s*$"),
    re.compile(r"^Confidential Information\s*$", re.IGNORECASE),
    re.compile(r"^Desert Valley Pediatrics.*$", re.IGNORECASE),
    re.compile(r"^\d{10}\s*$"),                      # the bare clinic phone
    re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
)

# Kept on first occurrence, dropped on the repeats. See the module docstring.
_BANNER_RE = (
    re.compile(r"^(Encounter|Preventive Exam)\s+Summary\s*$", re.IGNORECASE),
    re.compile(r"^.+\(Sex:\s*[MFU].*DOB:.*\)\s*$", re.IGNORECASE),
    re.compile(r"^Date of Visit:.*$", re.IGNORECASE),
)


def strip_page_furniture(text):
    """Drop repeated headers/footers; keep the patient banner once.

    Returns ``(clean_text, n_dropped)`` so the builder can print how much came
    off — a sudden change in that count is the first sign a new PDF has a
    different template.
    """
    seen_banner = set()
    kept = []
    dropped = 0

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            kept.append("")
            continue

        if any(rx.match(stripped) for rx in _FURNITURE_RE):
            dropped += 1
            continue

        banner = next((i for i, rx in enumerate(_BANNER_RE)
                       if rx.match(stripped)), None)
        if banner is not None:
            key = (banner, stripped)
            if key in seen_banner:
                dropped += 1
                continue
            seen_banner.add(key)

        kept.append(line)

    # Collapse the runs of blank lines the dropped furniture leaves behind.
    out, blank = [], False
    for line in kept:
        if line.strip():
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip() + "\n", dropped


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

_HEADINGS_LOWER = {h.lower(): h for h in SECTION_HEADINGS}

# A sub-block header inside Patient History: "Allergies Reviewed by ...",
# "Medication List Reviewed by ...", "Problem List Reviewed and updated by ...".
# Used only as an end boundary for the Problem List block.
_REVIEWED_RE = re.compile(r"Reviewed by|Reviewed and updated", re.IGNORECASE)


def split_sections(text):
    """Split `text` into ``[(heading_or_None, [lines]), ...]`` in document order.

    The leading block before the first recognised heading (the patient banner)
    comes back with ``heading=None`` and is always kept — see the docstring on
    why sex and DOB stay in.
    """
    sections = [(None, [])]
    for line in text.splitlines():
        canonical = _HEADINGS_LOWER.get(line.strip().lower())
        if canonical is not None:
            sections.append((canonical, []))
        else:
            sections[-1][1].append(line)
    return sections


def _drop_problem_list(lines):
    """Remove the Problem List sub-block from one section's lines.

    The block starts at the "Problem List Reviewed ..." line and ends at the
    next "... Reviewed by/and updated" sub-block — the section boundary itself
    is already handled by the caller, which only passes one section's lines.
    """
    out, in_block, removed = [], False, 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(PROBLEM_LIST_MARKER):
            in_block = True
            removed += 1
            continue
        if in_block:
            if _REVIEWED_RE.search(stripped):
                in_block = False
                out.append(line)
            else:
                removed += 1
            continue
        out.append(line)
    return out, removed


def render_variant(sections, variant):
    """Rebuild note text for one input variant. See billing_config.VARIANTS."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")

    parts, removed = [], {"assessment_lines": 0, "problem_list_lines": 0}

    for heading, lines in sections:
        if heading == ASSESSMENT_HEADING and variant != "full":
            removed["assessment_lines"] += len([ln for ln in lines if ln.strip()])
            continue

        body = lines
        if variant == "leakage_cut":
            body, n = _drop_problem_list(body)
            removed["problem_list_lines"] += n

        chunk = "\n".join(body).strip()
        if heading is None:
            if chunk:
                parts.append(chunk)
        elif chunk:
            parts.append(f"{heading}\n{chunk}")
        else:
            parts.append(heading)

    return "\n\n".join(parts).strip() + "\n", removed


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------


def extract_gold(sections):
    """Pull the ``DX n:`` codes out of the Assessment block.

    Returns ``(codes, rows, n_duplicates)`` where `codes` is the deduplicated,
    order-preserving list actually scored against, and `rows` is every DX line
    as written — kept so the report can show that note 96176 really does list
    Z68.51 twice rather than appearing to have lost a code.
    """
    rows = []
    for heading, lines in sections:
        if heading != ASSESSMENT_HEADING:
            continue
        for line in lines:
            m = _DX_LINE_RE.match(line.strip())
            if m:
                rows.append({
                    "dx": int(m.group(1)),
                    "code": normalize_code(m.group(2)),
                    "description": m.group(3).strip(),
                })

    codes, seen = [], set()
    for row in rows:
        if row["code"] not in seen:
            seen.add(row["code"])
            codes.append(row["code"])
    return codes, rows, len(rows) - len(codes)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def note_id_from_filename(filename):
    """``"112976 encounter.pdf"`` -> ``"112976"``; falls back to the stem."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^\s*(\d+)", stem)
    return m.group(1) if m else stem.replace(" ", "_")


def visit_kind_from_filename(filename):
    """``"26819 well.pdf"`` -> ``"well"``; ``"112976 encounter.pdf"`` -> ``"encounter"``."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    rest = re.sub(r"^\s*\d+\s*", "", stem).strip().lower()
    return rest or "unknown"


def build_record(pdf_path):
    """One note: gold codes, the three input variants, and the counts to print."""
    raw = pdf_to_text(pdf_path)
    clean, n_furniture = strip_page_furniture(raw)
    sections = split_sections(clean)
    codes, rows, n_dupes = extract_gold(sections)

    variants, removed = {}, {}
    for name in VARIANTS:
        text, rm = render_variant(sections, name)
        variants[name] = text
        removed[name] = rm

    headings = [h for h, _ in sections if h]

    record = {
        "note_id": note_id_from_filename(pdf_path),
        "visit_kind": visit_kind_from_filename(pdf_path),
        "source_pdf": os.path.basename(pdf_path),
        "gold_codes": codes,
        "gold_rows": rows,
        "n_gold_lines": len(rows),
        "n_gold_unique": len(codes),
        "n_gold_duplicates": n_dupes,
        "sections_found": headings,
        "variants": variants,
        "removed": removed,
        "n_furniture_lines_dropped": n_furniture,
        "n_words": {k: len(v.split()) for k, v in variants.items()},
    }
    return record


def _leaked_codes(text, gold_codes):
    """Which gold codes appear verbatim in `text`. The leak check, as a number."""
    found = {normalize_code(c) for c in ICD10_RE.findall(text.upper())}
    return sorted(set(gold_codes) & found)


def build_sample(pdf_dir):
    """Parse every PDF in `pdf_dir`. Returns ``(records, stats)``."""
    pdfs = sorted(
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdfs:
        raise RuntimeError(f"no PDFs found in {pdf_dir}")

    records = [build_record(p) for p in pdfs]

    for rec in records:
        rec["leaked_codes"] = {
            name: _leaked_codes(rec["variants"][name], rec["gold_codes"])
            for name in VARIANTS
        }

    stats = {
        "n_notes": len(records),
        "n_gold_lines": sum(r["n_gold_lines"] for r in records),
        "n_gold_unique": sum(r["n_gold_unique"] for r in records),
        "n_gold_duplicates": sum(r["n_gold_duplicates"] for r in records),
        "leaked_by_variant": {
            name: sum(len(r["leaked_codes"][name]) for r in records)
            for name in VARIANTS
        },
        "words_by_variant": {
            name: sum(r["n_words"][name] for r in records) for name in VARIANTS
        },
    }
    return records, stats


def write_sample(records, stats, path):
    """Write the JSONL sample + a stats sidecar. Both gitignored."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)

    stats_path = os.path.splitext(path)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return path, stats_path


def load_sample(path):
    """Read the JSONL sample back."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
