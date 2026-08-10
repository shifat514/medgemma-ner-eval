"""Markdown report writer for the MDACE term-level NER evaluation.

AGGREGATE METRICS ONLY. This file must never emit note text, term strings, or
patient identifiers — the report is committed. Everything here is derived from
integer counts. Per-example detail lives in the gitignored run dir
(evaluate_mdace._dump_errors).

Every table is preceded by a sentence saying what it shows and what would count
as good or bad, and every rate is printed with the count it rests on and a 95%
interval, so a 4-point gap between two small strata cannot be read as a finding.
"""

import json
import os

_SEP = "\n---\n"


def _pct(x):
    return f"{100 * x:.1f}%"


def _rate(value, ci):
    lo, hi = ci
    return f"{value:.4f} ({_pct(lo)}–{_pct(hi)})"


def _view_row(key, view):
    m = view["metrics"]
    return (f"| **{key}** | {view['title']} | {m['n_notes']} | {m['n_gold']} | "
            f"{m['n_pred']} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} |")


def _ladder_table(views):
    lines = [
        "| view | what it is | notes | gold terms | terms predicted | precision | recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("A1", "A2", "B1", "B2"):
        if key in views:
            lines.append(_view_row(key, views[key]))
    return "\n".join(lines)


def _ladder_notes(views):
    """Only explain steps whose BOTH rows are present.

    A phase-1 report holds B2 alone. Emitting the A1->A2 prose there sends the
    reader hunting for rows that are not in the table yet.
    """
    steps = [
        ("A1", "A2",
         "- **A1 → A2** is what the answer-key *column* is worth. A1 scores "
         "against `gold_code_description`, the ICD catalogue wording, exactly "
         "as the original request described. A2 scores the same notes against "
         "the note wording that same file carries. The two columns agree in "
         "only 4.5% of rows corpus-wide — the note says `depression`, the "
         "catalogue says `Major depressive disorder, single episode, "
         "unspecified` — so A1 is near-zero by construction. That is a property "
         "of the two columns, not a fault in the model or in the data as built."),
        ("A2", "B1",
         "- **A2 → B1** is what the answer-key *completeness* is worth. Both "
         "rows use note wording on the same notes; only the key changes. The "
         "100-row cut carries 99 evidence phrases where those notes actually "
         "hold 195, because the file was cut to 100 annotation rows rather than "
         "to whole notes. Everything A2 counts as a false positive and B1 counts "
         "as a hit is a phrase that really was billed. A2 is the number "
         "comparable with any figure computed from that file directly."),
        ("B1", "B2",
         "- **B1 → B2** is what balanced sampling changes. B1 runs on the 24 "
         "notes reachable from the 100-row sample cut, which are 17 Inpatient "
         "and 7 Profee. B2 runs on 25 of each, drawn with seed 13."),
    ]
    out = [text for lo, hi, text in steps if lo in views and hi in views]
    if not out:
        return ("Only one row is available so far, so there is no jump to read "
                "yet. The remaining rows arrive when the rest of the sample has "
                "been run.")
    return "\n".join(out)


def _stratum_table(by_type):
    lines = [
        "| stratum | notes | gold terms | terms predicted | terms per gold term | precision | best precision possible | recall | F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for chart_type in ("Profee", "Inpatient"):
        m = by_type.get(chart_type)
        if not m:
            continue
        lines.append(
            f"| **{chart_type}** | {m['n_notes']} | {m['n_gold']} | {m['n_pred']} | "
            f"{m['extraction_ratio']:.1f}x | {_rate(m['precision'], m['precision_ci'])} | "
            f"{m['precision_ceiling']:.4f} | {_rate(m['recall'], m['recall_ci'])} | "
            f"{m['f1']:.4f} |"
        )
    return "\n".join(lines)


def _fp_table(by_type):
    lines = [
        "| stratum | false positives | already billed, marked on another note of the same admission | in the note, nothing billed for it | not in the note at all |",
        "|---|---|---|---|---|",
    ]
    for chart_type in ("Profee", "Inpatient"):
        m = by_type.get(chart_type)
        if not m:
            continue
        b = m["fp_buckets"]
        total = max(m["fp"], 1)
        lines.append(
            f"| **{chart_type}** | {m['fp']} | {b['billed_elsewhere']} "
            f"({_pct(b['billed_elsewhere'] / total)}) | {b['in_note_unbilled']} "
            f"({_pct(b['in_note_unbilled'] / total)}) | {b['not_in_note']} "
            f"({_pct(b['not_in_note'] / total)}) |"
        )
    return "\n".join(lines)


def _code_table(by_type):
    lines = [
        "| stratum | codes billed | codes whose evidence was found | code recall |",
        "|---|---|---|---|",
    ]
    for chart_type in ("Profee", "Inpatient"):
        m = by_type.get(chart_type)
        if not m:
            continue
        lines.append(
            f"| **{chart_type}** | {m['codes_total']} | {m['codes_covered']} | "
            f"{_rate(m['code_recall'], m['code_recall_ci'])} |"
        )
    return "\n".join(lines)


def build_report(views, run_meta):
    """Assemble the full markdown report."""
    b2 = views.get("B2") or {}
    by_type = b2.get("by_chart_type") or {}
    parts = []

    parts.append(
        "# MedGemma zero-shot term-level NER — MDACE billing evidence\n\n"
        f"Model **{run_meta['model_name']}** (`{run_meta['model_id']}`), "
        f"4-bit, greedy decoding, `max_new_tokens={run_meta['max_new_tokens']}`.\n"
        f"Chunking {run_meta['chunk_words']} words / {run_meta['overlap_words']} "
        f"overlap. Seed {run_meta['seed']}. "
        f"{run_meta['n_notes_scored']} notes scored.\n"
        # Provenance, so a shared report can always be traced back to the exact
        # configuration that produced it. The prompt hash matters most: results
        # are cached per run directory, and without seeing this in the report
        # there is no way to tell a fresh run from a replay of an older prompt's
        # numbers by looking at the artifact alone.
        f"Prompt `{run_meta.get('prompt_id', 'unknown')}`, "
        f"run `{os.path.basename(run_meta.get('run_dir', '') or 'unknown')}`."
        + ("\n\n**ORACLE RUN — no model involved.** Gold terms were fed back "
           "through the pipeline to check the harness. Recall below must read "
           "1.0000; anything less is a bug in chunking, normalization or "
           "parsing, not a model result.\n" if run_meta.get("oracle") else "\n")
    )

    parts.append(
        "## What is being measured\n\n"
        "MedGemma reads a clinical note and lists the medical findings in it. "
        "That list is compared against the phrases human medical coders "
        "highlighted in the same note as justification for the billing codes "
        "they submitted.\n\n"
        "Scoring is on **term sets**, not positions: per note, gold is the set "
        "of normalized highlighted phrases and prediction is the set of "
        "normalized phrases the model produced. Normalization lowercases, drops "
        "punctuation and collapses all whitespace including newlines. "
        "**Recall** is the fraction of highlighted phrases the model found; "
        "**precision** is the fraction of the model's phrases that were "
        "highlighted; **F1** blends the two and sits nearer the worse of them. "
        "All run 0 to 1, higher is better.\n\n"
        "Matching is **strict** — after normalization the strings must be "
        "identical. A model that answers \"hypertension\" where the coder "
        "highlighted \"HTN\" scores a miss. Recall here is therefore a floor on "
        "real end-to-end usefulness, never an overstatement."
    )

    parts.append(
        "## The ladder\n\n"
        "Read this table top to bottom as an argument, not as separate "
        "results. Every row uses **the same model output** — one inference "
        "pass — and changes exactly one thing from the row above it. Higher is "
        "better throughout, but the interesting quantity is the *jump between "
        "rows*, because each jump isolates the cost of one decision.\n\n"
        + _ladder_table(views) + "\n\n"
        + _ladder_notes(views)
    )

    if by_type:
        parts.append(
            "## B2 by chart type — the headline result\n\n"
            "Profee and Inpatient are reported separately and must never be "
            "pooled: they are different measurement regimes. **Read recall as "
            "the real result. Read precision on Inpatient with the caveat "
            "below.** \"Terms per gold term\" is how many phrases the model "
            "listed for each one the coders highlighted; a large number there "
            "is what drags precision down.\n\n"
            + _stratum_table(by_type) + "\n\n"
            "### Why Inpatient precision is not a model failure\n\n"
            "MDACE annotates evidence only for codes that were **actually "
            "billed**, not for every condition documented. A Profee note is a "
            "single clinician encounter — short, and densely billed. An "
            "Inpatient note is the record of a whole hospital stay: a median of "
            "1,113 words carrying a median of 6 highlighted phrases. A model "
            "that reads such a note and correctly lists 40 real conditions has "
            "done its job, and scores precision 6/40 = 0.15 for it.\n\n"
            "The **best precision possible** column above makes that concrete: "
            "it is the highest micro precision any extractor could reach given "
            "how many terms were predicted. Where that number is low, no model "
            "of any quality scores above it, and the F1 in that row should not "
            "be read as a quality judgement."
        )

        parts.append(
            "## What the false positives actually are\n\n"
            "Every phrase the model produced that was not in the gold set, "
            "split three ways. The first two columns are extractions that are "
            "correct about the note; only the last column is model error. A "
            "healthy result is a **small last column** — that is the "
            "hallucination rate, and it is the one precision-side number the "
            "billing-scope problem does not distort.\n\n"
            + _fp_table(by_type) + "\n\n"
            "An ICD code belongs to an *admission*, but MDACE marks its "
            "evidence on whichever note the coder happened to be reading. So "
            "the first column is a phrase that demonstrably justifies a code "
            "that really was billed for this patient's stay — counted as a "
            "false positive purely because the annotator marked it elsewhere.\n\n"
            "**None of this is subtracted from the precision figures above.** "
            "Removing your own false positives before dividing raises precision "
            "by construction and measures nothing."
        )

        parts.append(
            "## Code recall — the ceiling for the term → ICD lookup\n\n"
            "Of the billing codes on these notes, the fraction that had at "
            "least one of their evidence phrases extracted. One phrase often "
            "justifies several codes, and one code is often evidenced by two "
            "phrasings, so this is not the same number as term recall.\n\n"
            "This is the **upper bound on the downstream lookup**: it can never "
            "retrieve a code whose evidence phrase was never extracted. Higher "
            "is better, and it is measured on the same notes as the NER, so "
            "unlike an end-to-end code-prediction run it separates the "
            "extraction step from the lookup step.\n\n"
            + _code_table(by_type)
        )

    parts.append(
        "## How much to trust these numbers\n\n"
        "The samples are small and the intervals in brackets are 95% Wilson "
        "intervals — the range the true rate plausibly sits in.\n\n"
        "As a rule of thumb, a rate measured on ~110 items carries roughly ±9 "
        "percentage points and on ~214 items roughly ±7. **A gap between Profee "
        "and Inpatient smaller than about 15 points is not a finding.** Quote "
        "the headline as \"around 0.6\", never as \"0.61\".\n\n"
        f"{run_meta.get('n_mixed_chart_type', 0)} of the scored notes were "
        "billed both ways (some Profee rows, some Inpatient) and are counted as "
        "Inpatient. Their gold therefore includes Profee-billed evidence, which "
        "blurs the contrast between the two strata slightly.\n\n"
        f"Generation hit the token cap on **{run_meta.get('n_cap_hits', 0)}** of "
        f"{run_meta.get('n_chunks', 0)} chunks. Those replies were cut off "
        "mid-JSON; the parser salvages the complete part, but anything after the "
        "cut is lost, so **recall above is a floor rather than an estimate**. "
        "Raising the cap does not fix this — it was already tried, and the same "
        "chunks bound at 1024 and 1536, which is a repetition loop rather than a "
        "length problem. The lever is output volume: a narrower prompt, or "
        "smaller `--chunk-words` so each call has less to describe."
    )

    parts.append(
        "## Known limits\n\n"
        "- **Strict matching only.** No fuzzy or embedding matching. 88% of "
        "gold terms are 1–3 words and behave well; the remaining 12% are "
        "clause-length snippets a model will not reproduce word-perfectly, and "
        "those score as misses even when the model was substantially right. The "
        "gitignored `errors.jsonl` holds the actual missed phrases so that "
        "decision can be made on evidence rather than guesswork.\n"
        "- **MIMIC-III only.** MDACE is built on MIMIC-III notes (integer note "
        "IDs, 6-digit admission IDs, `[**...**]` redaction). It shares no notes "
        "with the MIMIC-IV medication evaluation in this repo, so the two sets "
        "of numbers must not be pooled or compared directly.\n"
        "- **Recall is capped at 0.945, by choice.** The prompt asks for "
        "diagnoses, procedures and injuries, and explicitly excludes "
        "medications. That puts the 5.5% of gold carried by ICD-10-CM Z-codes "
        "(status and history — smoking history, long-term drug therapy) out of "
        "reach. The alternative was measured and was worse: asking for that "
        "category too made the model return 33% medication items, which "
        "truncated 12 of 15 replies and pushed extraction to 68.6 terms per "
        "note against ~10 gold. A 5.5% ceiling costs less than the damage that "
        "volume did to the other 94.5%.\n"
        "- **Type labels are not scored.** The prompt asks the model to "
        "classify each term as Condition / Procedure / Injury, but scoring "
        "compares strings only. A term labelled wrongly still counts as a hit."
    )

    return _SEP.join(parts) + "\n"


def write_report(views, run_meta, results_dir="results", label="run"):
    """Write the markdown report and a machine-readable copy of the metrics."""
    os.makedirs(results_dir, exist_ok=True)

    md_path = os.path.join(results_dir, f"mdace_ner_{label}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_report(views, run_meta))

    # Strip the per-note lists: they carry note_ids, and the committed artifact
    # should stay aggregate-only.
    slim = {}
    for key, view in views.items():
        entry = {k: v for k, v in view.items() if k not in ("metrics", "by_chart_type")}
        entry["metrics"] = {k: v for k, v in view["metrics"].items()
                            if k != "per_note"}
        if "by_chart_type" in view:
            entry["by_chart_type"] = {
                ct: {k: v for k, v in m.items() if k != "per_note"}
                for ct, m in view["by_chart_type"].items()
            }
        slim[key] = entry

    json_path = os.path.join(results_dir, f"mdace_ner_{label}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run": run_meta, "views": slim}, f, indent=2, sort_keys=True)

    return md_path, json_path
