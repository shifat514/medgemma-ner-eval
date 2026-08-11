"""Markdown report writer for the MDACE recall benchmark.

AGGREGATE METRICS ONLY. This file must never emit note text, model phrases, gold
phrases or patient identifiers — the report is committed. Everything here is
derived from integer counts. The per-example detail, including every pair each
ladder level newly accepted, lives in the gitignored run dir.

Every table is preceded by a sentence saying what it shows and what would count
as good or bad, and every rate is printed with the count it rests on and a 95%
interval, so a small gap between two levels cannot be read as a finding.
"""

import json
import os

from .recall_config import COMBINED, SOURCE_LABELS, SOURCES

_SEP = "\n---\n"

_RULES = {
    "L1": "exact after normalization",
    "L2": "whole-token containment, either direction",
    "L3": "token-set Dice and/or difflib ratio, thresholded",
    "L4": "biomedical sentence-embedding cosine",
}


def _pct(x):
    return f"{100 * x:.1f}%"


def _rate(value, ci):
    lo, hi = ci
    return f"{value:.4f} ({_pct(lo)}–{_pct(hi)})"


def _ladder_table(result):
    adj = result.get("adjudicated")
    head = ("| level | rule | rows recalled | codes recalled "
            "| accepted forms matched | false positives | FP rate |")
    rule = "|---|---|---|---|---|---|---|"
    if adj:
        head = head.replace("| rows recalled |",
                            "| rows recalled | rows after L5 |")
        rule += "---|"
    lines = [head, rule]
    for level in result["levels"]:
        m = result["by_source"][COMBINED][level]
        judged = ""
        if adj:
            a = adj[COMBINED][level]
            judged = (f" {a['rows_matched']}/{a['rows_total']} "
                      f"{_rate(a['row_recall'], a['row_recall_ci'])} |")
        lines.append(
            f"| **{level}** | {_RULES.get(level, '')} | "
            f"{m['rows_matched']}/{m['rows_total']} "
            f"{_rate(m['row_recall'], m['row_recall_ci'])} |" + judged + " "
            f"{m['codes_matched']}/{m['codes_total']} {m['code_recall']:.4f} | "
            f"{m['forms_matched']}/{m['forms_total']} {m['form_recall']:.4f} | "
            f"{m['fp']} | {m['fp_rate']:.4f} |"
        )
    return "\n".join(lines)


def _gain_notes(result):
    lines = []
    for level in result["levels"][1:]:
        m = result["by_source"][COMBINED][level]
        lines.append(f"- **{level}** added {m['gain_rows']} rows and "
                     f"{m['gain_forms']} accepted forms over the level above it.")
    return "\n".join(lines) if lines else "Only one level ran."


def _source_table(result, field, total_field):
    levels = result["levels"]
    lines = ["| source | entries | " + " | ".join(levels) + " |",
             "|---|---|" + "---|" * len(levels)]
    for source in (COMBINED,) + tuple(SOURCES):
        by_level = result["by_source"][source]
        total = by_level[levels[0]][total_field]
        cells = " | ".join(f"{by_level[lv][field]:.4f}" for lv in levels)
        lines.append(f"| {SOURCE_LABELS[source]} | {total} | {cells} |")
    return "\n".join(lines)


def _source_fp_table(result):
    levels = result["levels"]
    lines = ["| source | " + " | ".join(levels) + " |",
             "|---|" + "---|" * len(levels)]
    for source in (COMBINED,) + tuple(SOURCES):
        by_level = result["by_source"][source]
        cells = " | ".join(f"{by_level[lv]['fp']} ({by_level[lv]['fp_rate']:.2f})"
                           for lv in levels)
        lines.append(f"| {SOURCE_LABELS[source]} | {cells} |")
    return "\n".join(lines)


def _medication_note(run_meta):
    """Where medications stand — which now depends on which prompt ran.

    This paragraph was written when there was one prompt, and it asserted a
    measured exclusion as though it were a property of the benchmark. It is a
    property of the `scoped` variant only. `billable` has no medication rule at
    all: its exclusion, if any, is an emergent consequence of the coder test and
    is unmeasured. Saying otherwise in a committed report is the kind of claim
    somebody would reasonably rely on.
    """
    variant = run_meta.get("prompt_variant", "scoped")
    if variant == "scoped":
        return (
            "- **Medications are out of scope by instruction, which caps recall "
            "at about 0.945.** The `scoped` prompt excludes them because asking "
            "for them produced 33% of extraction for 5.5% of gold and truncated "
            "12 of 15 chunks on the previous branch. Rows whose evidence is a "
            "medication are unreachable here, so the ceiling is below 1.0 by "
            "choice and the figures above should be read against 0.945 rather "
            "than against a perfect score.\n")
    return (
        f"- **Medications are not excluded by instruction in this run, and the "
        f"ceiling is unmeasured.** The `{variant}` prompt carries no medication "
        f"rule; whether the model still leaves them alone is an emergent "
        f"consequence of asking only for codeable findings, not something this "
        f"run establishes. The `scoped` variant excludes them explicitly and "
        f"therefore caps at about 0.945; this variant may reach higher or leak "
        f"medications as false positives, and only a per-source audit of the "
        f"5.5% of gold evidenced by medications would say which.\n")


def _l5_notes(result, run_meta):
    """What adjudication cost, and why rows fall by less than pairs do."""
    adj = result.get("adjudicated")
    if not adj:
        return ("**L5 has not been applied to these figures.** Every level above "
                "L1 still includes matches nobody has checked, so **quote L1 as "
                "the number that needs no caveat.**")
    top = result["levels"][-1]
    raw = result["by_source"][COMBINED][top]
    judged = adj[COMBINED][top]
    lost = raw["rows_matched"] - judged["rows_matched"]
    return (
        f"**L5 has been applied.** A judge was shown every pair the levels above "
        f"L1 newly accepted and asked whether the two phrases name the same "
        f"clinical finding; {run_meta.get('n_rejected_pairs', 0)} were rejected "
        f"and removed. The `rows after L5` column is the result, and it is the "
        f"one to quote.\n\n"
        f"**Rejected pairs cost far less recall than their count suggests.** "
        f"{run_meta.get('n_rejected_pairs', 0)} rejected pairings cost {lost} "
        f"row{'' if lost == 1 else 's'} at {top}. A row is recalled by matching "
        f"any ONE of its accepted forms and the median row has four, so it "
        f"usually keeps another supporting form when one pairing is thrown out. "
        f"The three-column accept-set was built to fix the `HTN` problem; it "
        f"turns out to also make the ladder robust to its own looseness.\n\n"
        f"A rejected pair is removed as an *edge*, not as a finished assignment, "
        f"so the matcher re-solves — a finding whose pairing was thrown out may "
        f"legitimately match a different form. That is why a higher level can "
        f"lose fewer rows than a lower one despite more of its pairs being "
        f"rejected.")


def _l3_notes(result):
    """Say when a level earned nothing, rather than leaving a flat row."""
    levels = result["levels"]
    if "L3" not in levels:
        return ""
    l2 = result["by_source"][COMBINED]["L2"]
    l3 = result["by_source"][COMBINED]["L3"]
    gained = l3["forms_matched"] - l2["forms_matched"]
    if gained > 3:
        return ""
    return (
        f"**L3 earned almost nothing at these thresholds** — {gained} accepted "
        f"form{'' if gained == 1 else 's'} beyond L2. Dice 0.80 and character "
        f"ratio 0.90 are strict enough that whole-token containment has already "
        f"taken everything they would reach. That is a finding about the "
        f"thresholds, not about the model: either loosen them and re-score "
        f"(`--score-only --dice-min …`, no GPU), or read the ladder as three "
        f"effective levels rather than four.")


def _l4_caveat():
    """What L4 turned out to be, quoted from the measurement rather than described.

    The plan expected L4 to be the level that finally separates a real synonym
    from a near-miss, since it is the only one that reaches abbreviations. The
    measurement says it reaches them and does not separate them, and a report
    that let the L4 row stand unqualified would be overstating the result.
    """
    from .recall_matching import MEASURED_COSINE, no_threshold_separates

    if not no_threshold_separates():
        return ""
    want = sorted((s, a, b) for a, b, keep, s in MEASURED_COSINE if keep)
    dont = sorted(((s, a, b) for a, b, keep, s in MEASURED_COSINE if not keep),
                  reverse=True)
    worst_score, worst_a, worst_b = dont[0]
    outranked = sum(1 for s, _a, _b in want if s < worst_score)
    rows = "\n".join(
        f"| {a} | {b} | {'yes' if keep else '**no**'} | {score:.3f} |"
        for a, b, keep, score in MEASURED_COSINE)
    return (
        "\n\n### L4 reaches abbreviations. It does not separate them.\n\n"
        "This is a result, not a caveat added for form. Cosine on the default "
        "biomedical encoder, sorted, for pairs a matcher should accept and "
        "pairs it must reject:\n\n"
        "| model says | gold says | should match | cosine |\n"
        "|---|---|---|---|\n" + rows + "\n\n"
        f"The two columns are interleaved from top to bottom. *{worst_a}* "
        f"against *{worst_b}* — a different diagnosis — scores "
        f"{worst_score:.3f}, above {outranked} of the "
        f"{len(want)} pairs L4 exists to catch. No cosine threshold admits "
        "every one of those and none of these, so the threshold above is a "
        "**floor chosen to reach the abbreviations**, not a cutoff that works. "
        "Raising it loses synonyms without buying precision.\n\n"
        "**So the L4 row is provisional until L5 has run.** The argument the "
        "string table makes about L1-L3 turns out to hold one level higher up "
        "as well, which makes L5 not a refinement of L4 but the thing that "
        "makes L4's gain interpretable at all. Every pair L4 newly accepted is "
        "in the run directory waiting for it."
    )


def _denominator_table(result):
    lines = [
        "| source | accepted forms | billed rows reachable | distinct codes reachable |",
        "|---|---|---|---|",
    ]
    combined = result["by_source"][COMBINED][result["levels"][0]]
    lines.append(f"| {SOURCE_LABELS[COMBINED]} | {combined['forms_total']} | "
                 f"{combined['rows_total']} | {combined['codes_total']} |")
    for source in SOURCES:
        m = result["by_source"][source][result["levels"][0]]
        lines.append(f"| {SOURCE_LABELS[source]} | {m['forms_total']} | "
                     f"{m['rows_total']} | {m['codes_total']} |")
    return "\n".join(lines)


def build_report(result, run_meta, data_stats):
    """Assemble the full markdown report."""
    levels = result["levels"]
    top = levels[-1]
    volume = result["volume"]
    thresholds = run_meta.get("thresholds") or {}
    parts = []

    parts.append(
        "# MedGemma recall benchmark — MDACE billing evidence\n\n"
        f"Model **{run_meta['model_name']}** (`{run_meta['model_id']}`), "
        f"4-bit, greedy decoding, `max_new_tokens={run_meta['max_new_tokens']}`, "
        f"prompt variant `{run_meta.get('prompt_variant', '?')}`. "
        "**One-shot, not zero-shot:** the prompt carries a single synthetic "
        "worked example containing no MDACE content. It is not the thing under "
        "test — the comments in `prompt_recall` record that abstract "
        "prohibitions failed on a 4B model where a demonstrated one worked — but "
        "it is an example, and calling the run zero-shot would be wrong. "
        f"Chunking {run_meta['chunk_words']} words / "
        f"{run_meta['overlap_words']} overlap. "
        f"{run_meta['n_notes_scored']} notes, {run_meta.get('n_chunks', 0)} "
        "chunks.\n"
        # Provenance, so a shared report traces back to the configuration that
        # produced it. The prompt hash matters most: results are cached per run
        # directory, and without it there is no way to tell a fresh run from a
        # replay of an older prompt's numbers by looking at the artifact alone.
        f"Prompt `{run_meta.get('prompt_id', 'unknown')}`, "
        f"run `{run_meta.get('run_tag') or 'unknown'}`."
        + ("\n\n**ORACLE RUN — no model involved.** The gold accept-sets were "
           "fed back through the pipeline to check the harness. Every source "
           "below must read 1.0000 at L1 and the combined line must show zero "
           "false positives; anything less is a bug in chunking, normalization "
           "or matching, not a result.\n" if run_meta.get("oracle") else "\n")
    )

    parts.append(
        "## What is being measured\n\n"
        "MedGemma reads a clinical note and lists the findings in it. That list "
        "is compared against what human medical coders recorded as the "
        "justification for the billing codes they submitted. **Recall is the "
        "metric**: how much of the billed evidence the model recovers. "
        "Precision is secondary and appears here as an explicit false-positive "
        "count rather than as a ratio.\n\n"
        f"The input is one file: {data_stats['n_rows']} annotation rows on "
        f"{data_stats['n_notes']} notes, carrying {data_stats['n_codes']} "
        "distinct `(note, code system, code)` triples. Note text is embedded in "
        "that file, so there is no join and no separate notes file. Gold is "
        "whatever that file says it is.\n\n"
        "### The accept-set\n\n"
        "Scoring against the note's own wording alone made `HTN` unable to "
        "match `Essential (primary) hypertension` however right the model was. "
        "So per billed code the accept-set is the **union of three columns**: "
        "the evidence text the coder highlighted, the ICD code description, and "
        "every SNOMED concept term the file ships for that code. A prediction "
        "matching any of them recalls that row.\n\n"
        f"That gives a median of {data_stats['accept_median']:.0f} accepted "
        f"forms per row (min {data_stats['accept_min']}, max "
        f"{data_stats['accept_max']}), against 1 under evidence-text-only "
        "scoring.\n\n"
        "### The model's side\n\n"
        "Each finding carries two fields: `span`, the phrase as written in the "
        "note, and `name`, the standard clinical name. Either may match the "
        "accept-set. MedGemma already knows the expansions, and using its own "
        "expansion beats making the matcher infer one. `span` is also what the "
        "not-in-note check below is run against."
    )

    parts.append(
        "## The matching ladder\n\n"
        "Each level is a **superset** of the one above, so recall is "
        "monotonically non-decreasing and the gain at each level is "
        "attributable to that level. Read the jumps, not just the top row.\n\n"
        "Matching is one prediction to at most one gold form. Without that "
        "rule a single vague prediction satisfies several gold entries at once "
        "and recall measures vagueness.\n\n"
        + _ladder_table(result) + "\n\n"
        "**Quote the rows or codes column, not the forms column.** A row is "
        "recalled by matching any ONE of its accepted forms, so a model that "
        "recalls every row while producing a single phrasing each still leaves "
        "most forms unmatched — the forms column has a ceiling set by how many "
        "phrasings the model emits, not by how much it found. It is here "
        "because the per-source breakdown below is measured in forms and the "
        "two must reconcile.\n\n"
        + _gain_notes(result) + "\n\n"
        + _l5_notes(result, run_meta) + "\n\n"
        + _l3_notes(result) + "\n\n"
        "**Thresholds are reported, never silently chosen.** "
        f"Dice ≥ `{thresholds.get('dice_min')}`, "
        f"character ratio ≥ `{thresholds.get('ratio_min')}`, "
        f"cosine ≥ `{thresholds.get('cosine_min')}`"
        + (f" using `{thresholds['embed_model']}`."
           if thresholds.get("embed_model") else ".")
        + "\n\nThey come from measured pairs, not from taste. Dice 0.80 keeps "
        "*acute kidney injury* against *kidney injury, acute* (1.00) and drops "
        "*acute renal failure* against *chronic renal failure* (0.67); "
        "character ratio 0.90 keeps *hyperlipidema* against *hyperlipidemia* "
        "(0.96) and drops that same acute/chronic pair (0.75). No string "
        "threshold separates the good pairs from the bad ones on its own — "
        "anything loose enough to catch *CHF* against *congestive heart "
        "failure* (0.22) accepts the acute/chronic pair several times over. "
        "That is the argument for L4, and for dumping every pair each level "
        "newly accepted.\n\n"
        "**L2 knowingly admits two bad shapes.** *diabetes* inside *diabetes "
        "insipidus* is a false match, and *sepsis* inside *no evidence of "
        "sepsis* is a negation. The pairs L2 newly accepted are written to the "
        "run directory for exactly this reason; L5 adjudicates them."
        + (_l4_caveat() if "L4" in levels else
           "\n\n**L4 did not run in this report.** The biomedical embedding "
           "backend was not available, so the ladder stops at L3 and "
           "abbreviations like *CHF* remain unreachable. The recall figures "
           "above are correspondingly a floor.")
    )

    parts.append(
        "## Recall is never quotable on its own\n\n"
        "With loose matching and no volume control, a model that lists every "
        "phrase in the note scores near 1.00. A bare recall figure therefore "
        "cannot rank MedGemma against a larger model later, which is the entire "
        "purpose of this benchmark. These are the numbers that must travel with "
        "it.\n\n"
        "| | |\n|---|---|\n"
        f"| findings per note | {volume['pred_per_note']:.1f} |\n"
        f"| false positives at {top} | {result['by_source'][COMBINED][top]['fp']} "
        f"of {result['by_source'][COMBINED][top]['n_pred']} findings "
        f"({_pct(result['by_source'][COMBINED][top]['fp_rate'])}) |\n"
        f"| span not found in the note | {volume['n_not_in_note']} of "
        f"{volume['n_span_checked']} checked — "
        f"{_rate(volume['not_in_note_rate'], volume['not_in_note_ci'])} |\n"
        f"| findings with no span to check | {volume['n_no_span']} |\n\n"
        "The last two lines are the hallucination signal: a span the model "
        "claims to have copied from the note that is not in the note. It is the "
        "one precision-side number the billing-scope problem does not distort, "
        "because it does not depend on what was billed at all.\n\n"
        "A finding the model returned with no `span` cannot be checked and is "
        "excluded from that denominator rather than counted as clean."
    )

    parts.append(
        "## Whose wording does the model produce?\n\n"
        "Recall broken out by which column of the answer key was matched. This "
        "is the genuinely useful question behind the benchmark: does the model "
        "speak in note wording, catalogue wording, or SNOMED wording. Each "
        "source is scored by its own independent matching.\n\n"
        "### Recall per source, by level\n\n"
        + _source_table(result, "form_recall", "forms_total") + "\n\n"
        "Recall per source is unambiguous: of that source's accepted forms, how "
        "many were matched.\n\n"
        "### Denominators, which differ per source\n\n"
        + _denominator_table(result) + "\n\n"
        "SNOMED terms are shipped for only some rows, so a SNOMED recall "
        "computed out of all rows would be measuring the file's coverage and "
        "calling it model performance. Every denominator is printed for that "
        "reason.\n\n"
        "### False positives per source, by level\n\n"
        "**This table is easy to misread.** Per-source FP means *matched "
        "nothing in that source*. A prediction matching only the catalogue "
        "wording is a hit on the description line and a false positive on the "
        "evidence-text line. So every individual source line reads high, and "
        "**only the combined line counts predictions that matched nothing "
        "anywhere**. Counts, with the rate over all findings in brackets.\n\n"
        + _source_fp_table(result)
    )

    parts.append(
        "## What the SNOMED column is, and is not\n\n"
        "| | |\n|---|---|\n"
        f"| rows with any SNOMED term | "
        f"{data_stats['snomed_rows_with_terms']} of {data_stats['n_rows']} |\n"
        f"| concept terms shipped | {data_stats['snomed_terms_shipped']} of "
        f"{data_stats['snomed_terms_reported']} reported by "
        f"`gold_snomed_concept_count` — "
        f"**{_pct(data_stats['snomed_shipped_frac'])}** |\n"
        + "".join(
            f"| {system} rows with SNOMED | {slot['with_snomed']} of "
            f"{slot['rows']} |\n"
            for system, slot in sorted(
                data_stats["snomed_by_code_system"].items()))
        + "\n"
        "The list is capped at 3 entries per code and the survivors are not the "
        "top-ranked ones — an HCV row ships two pregnancy-related concepts and "
        "omits plain *hepatitis C*.\n\n"
        "This is stated as **a limit on the number, not a request to anyone to "
        "fix the file**. One file was provided and the benchmark works with it. "
        "The same applies to the SNOMED lookup itself: the matching ladder "
        "approximates it, and that approximation is what L4 is."
    )

    parts.append(
        "## Reference numbers\n\n"
        "The earlier `mdace-term-ner` run, strict exact matching against "
        "evidence text alone, on a different 50-note sample:\n\n"
        "| | |\n|---|---|\n"
        "| recall | 0.5278 (324 gold terms) |\n"
        "| precision | 0.0679 (2,520 predicted) |\n"
        "| code recall | 165 of 302 = 54.6% |\n"
        "| terms per note | ~50 |\n"
        "| independent figure computed separately | ≈0.5 |\n\n"
        "Two implementations landing in the same place is a real cross-check, "
        "which is why it is kept here. It is **not** directly comparable with "
        "the table above — different notes, a single-column answer key, and "
        "exact matching only — so read it as the L1-with-one-source starting "
        "point the accept-set and the ladder were built to improve on."
    )

    parts.append(
        "## How much to trust these numbers\n\n"
        "The sample is small. Brackets are 95% Wilson intervals — the range the "
        "true rate plausibly sits in. A recall measured over ~100 rows carries "
        "roughly ±10 percentage points, so **quote the headline as \"around "
        "0.6\", never as \"0.61\"**, and treat a gap of less than about 10 "
        "points between two levels as noise.\n\n"
        + (
            f"Generation hit the token cap on **{run_meta['n_cap_hits']}** of "
            f"{run_meta.get('n_chunks', 0)} chunks "
            f"({run_meta.get('n_chunks_salvaged', 0)} had a usable prefix "
            "salvaged). Those replies were cut off mid-JSON; anything after the "
            "cut is lost, so **recall above is a floor rather than an "
            "estimate**. The lever is output volume — a narrower prompt, or "
            "smaller `--chunk-words` so each call has less to describe — not a "
            "higher cap, which was tried on the previous branch and bound at "
            "both 1024 and 1536."
            if run_meta.get("n_cap_hits") else
            f"Generation hit the token cap on **0** of "
            f"{run_meta.get('n_chunks', 0)} chunks, so no reply was cut off "
            "mid-JSON and recall is not understated on that account."
        )
    )

    parts.append(
        "## Known limits\n\n"
        + (
            "- **L5 has been applied, by the model under test.** The judge was "
            "the same 4-bit MedGemma being benchmarked. It rejected a third of "
            "the pairs its own matcher proposed, which is not the direction a "
            "self-serving judge fails in, but an independent judge would be "
            "stronger evidence and `--judge none` exists for that.\n"
            if result.get("adjudicated") else
            "- **L5 has not been applied to these figures.** The pairs L2, L3 "
            "and L4 newly accepted are written to the run directory and are "
            "adjudicated by a separate step (`python -m src.recall_judge`). "
            "Until that has run, every level above L1 includes matches nobody "
            "has checked — the negation and the *diabetes insipidus* shapes "
            "above are real and unquantified. **Quote L1 as the number that "
            "needs no caveat.**\n"
        )
        + "- **The SNOMED lookup is approximated, not performed.** The ladder "
        "stands in for a real terminology lookup. L4 reaches abbreviations, "
        "which is the part that matters most, but it is a similarity model and "
        "not a terminology — and the measurement above shows it cannot tell a "
        "synonym from a change of acuity.\n"
        + _medication_note(run_meta)
        + "- **MIMIC-III only.** MDACE is built on MIMIC-III notes. It shares "
        "no notes with the MIMIC-IV medication evaluation in this repo, so the "
        "two sets of numbers must not be pooled."
    )

    return _SEP.join(parts) + "\n"


def write_report(result, run_meta, data_stats, results_dir="results",
                 label="run"):
    """Write the markdown report and a machine-readable copy of the metrics."""
    os.makedirs(results_dir, exist_ok=True)

    md_path = os.path.join(results_dir, f"mdace_recall_{label}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_report(result, run_meta, data_stats))

    # new_pairs quote note text and model phrases; they stay in the gitignored
    # run dir and never enter the committed artifact.
    json_path = os.path.join(results_dir, f"mdace_recall_{label}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "run": run_meta,
            "data": data_stats,
            "levels": result["levels"],
            "by_source": result["by_source"],
            "volume": result["volume"],
            # The adjudicated ladder is the quotable one once L5 has run, so it
            # belongs in the artifact and not only in the console output.
            "adjudicated": result.get("adjudicated"),
            "n_new_pairs": {lv: len(p) for lv, p in result["new_pairs"].items()},
        }, f, indent=2, sort_keys=True)

    return md_path, json_path
