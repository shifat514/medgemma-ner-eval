# Precision phase — internal plan

Follows `recall-benchmark-internal.md`. That phase asked how much billed
evidence MedGemma-4B recovers. This one asks how much of what it returns is
worth billing.

Status: plan only. Nothing built yet.

---

## What changed, and why the metric moves

Ehtesham Bhai confirmed **the output does not go to a human coder for review**.
That single answer reframes the result we shipped.

The recall benchmark reports 78/100 with 1,083 false positives, of which 1,033
are phrases genuinely present in the note that were never billed. Under human
review those are harmless — a coder drops them. With no human in the loop, each
one is a wrong charge.

So the headline metric becomes **precision**, and recall becomes the constraint
we protect rather than the thing we maximise.

Baseline to beat, from the 24-note run (`scoped`, cw400, p5046b887):

| | |
|---|---|
| recall | 0.78 (78/100 rows, adjudicated) |
| **precision** | **0.1159** (142 of 1,225 findings) |
| false positives | 1,083 — 1,033 in-note-unbilled, 50 invented |
| volume | 51.0 findings/note against ~4.2 billed |
| best precision reachable at this volume | 0.2571 |

That last row matters: at 1,225 predictions against 318 accepted phrasings, no
extractor can exceed 0.2571 no matter how good it is. **Cutting volume raises
the ceiling itself**, which is why volume is the first lever and not the third.

## What we cannot measure yet, and why it matters

The product is note → extracted terms → SNOMED lookup → adjacent ICD codes →
bill. **We only measure the first arrow.** The lookup does not exist as code
anywhere in this repo or its siblings, so the number that actually decides
whether this works — of the codes the pipeline emits, how many were really billed
— is currently unmeasurable.

Two consequences, both of which cut against over-investing at term level:

**Term-level precision probably over-counts errors.** A false positive only
becomes a wrong charge if it maps to a billable concept. "Patient tolerated the
procedure well" maps to nothing and dies in the lookup. So 0.1159 is a floor on
the real precision, possibly a long way below it, and we do not know by how much.

**Our SNOMED figure is a floor too, and by a measured amount.** This file ships
at most 3 SNOMED terms per code against a mapped count of 1,894 — 157 of them, or
**8%**. One K83.1 row maps to 64 concepts and ships 3. The real lookup has an
order of magnitude more terms to match against, so the 47% SNOMED recall we
reported understates that side of the pipeline. **SNOMED is not the weak link it
looked like.**

The cheapest way to close this is a crude lookup: match our 1,225 findings
against a SNOMED term list and count how many produce any code at all. That
needs a SNOMED release file. **Asked, and we do not have one.** So the proxy is
all we get: term-level precision is the metric we optimise, knowing it understates
the real figure by an unknown amount. Say so wherever it is quoted.

## The four actions, in order

### 1. Failure analysis — `src/recall_failures.py`

Everything below is currently inference from totals. This turns it into a list.

Reads the finished run (findings.jsonl on Drive) plus the gold file. Writes a
gitignored per-example dump and a PHI-free counts file.

**For each false positive, attribute it to a note section.** Locate the model's
span in the note text with `str.find`, walk back to the nearest section header,
and count. Expected output: a ranked table of which sections generate false
positives. If labs and medication sections dominate, action 2 is confirmed and
sized; if the false positives are spread evenly, action 2 is not the lever and
action 3 becomes primary.

**For each of the 22 missed rows, bucket the reason:**

- `not_extracted` — no model finding is even loosely similar to any accepted
  form. The model never saw it.
- `near_miss` — a finding scores above 0 but below every threshold. Matching is
  the problem, not extraction.
- `rejected_by_l5` — matched, then thrown out by the judge.
- `truncated` — the gold evidence sits in a chunk that hit the token cap.

The last bucket needs data we do not currently record. `per_note.jsonl` stores
`n_cap_hits` but not *which* chunks hit it. Add a `cap_hit_windows` list of
`[start, end]` token indices — integers, PHI-free — so the next run can attribute
it. Until then that bucket reads "unknown".

No GPU. Runs against what is already on Drive.

### 2. Section filtering

64% of the text across the 24 notes sits in sections containing zero gold
evidence. Largest by word count: Pertinent Results (966), IMPRESSION (613), Disp
(573), FINDINGS (504), Discharge Medications (492), Medications on Admission
(431), Order date (360), Followup Instructions (246).

**Do not drop all 293 never-gold sections.** That figure is measured on 24 notes
and 293 distinct headers means most appear once or twice; a section with no gold
here will have gold elsewhere. Overfitting to the sample is the obvious failure
mode and it would show up as a recall drop we could not explain.

Drop in tiers, measuring each:

- **Tier 1, safe:** medication lists, follow-up instructions, order dates,
  disposition, activity. Administrative and pharmacy content that is not a
  finding by definition.
- **Tier 2, needs checking:** Pertinent Results, FINDINGS, IMPRESSION. These are
  labs and radiology. They carry no gold in this sample, but radiology
  impressions genuinely can carry billable findings, so this tier gets measured
  separately and kept only if recall holds.

Implementation: a `RECALL_DROP_SECTIONS` list in `recall_config`, a header regex
in the loader, and text filtering **before** chunking. Fewer words means fewer
chunks, so the run tag must include a section-filter identifier or two runs will
collide — the same mistake the prompt A/B made.

Cost: a full re-run, roughly 1 hour of T4, because the model input changes.

**Success:** volume down materially, recall within a couple of points, precision
ceiling up.

### 3. Second-pass billability filter

Extraction and filtering are currently one call, and the model is demonstrably
good at the first and bad at the second. Split them.

After extraction, ask per finding: *would a medical coder assign a billing code
to this?* Drop the rejections. This reuses `recall_judge` almost unchanged — same
one-word-answer pattern, same greedy decoding, same "unreadable verdict keeps
the item" rule.

Cost: 1,225 findings at ~16 output tokens each. Cheap per call but there are a
lot of them; batching is worth it.

**Confirmed: there is no SNOMED release file available.** So the worry that this
duplicates the lookup cannot be tested and cannot be relied on either. With no
downstream filter to defer to, term level is the only place precision can be
fixed, which makes this action firmer rather than weaker.

**This is the action most likely to work**, because it is the one that
gives the model a decision it can actually make. Asking "extract only billable findings"
inside a long extraction prompt failed twice; asking "is this one billable" about
a single phrase is a far easier question.

**Success:** precision up several-fold, recall down by less than the precision
gain. Report both, and report the trade-off curve rather than a single operating
point — the right cut-off is a product decision, not ours.

### 4. Prompt tuning

Listed last on purpose. Two variants were built and compared today. `billable`
cut invented spans 9x (9.22% → 1.02%) and **did not reduce volume at all**
(98.5 vs 108.5 findings/note). Prompt changes have already been shown not to fix
this specific problem.

Untried and worth one attempt, but only alongside 2 and 3: asking for a hard
maximum number of findings per call, which also attacks the repetition loop that
caused 26% of chunks to hit the token cap.

## Deliberately not in this plan

**Fine-tuning.** Ehtesham Bhai has not asked for it. The full 1,074-note corpus
and 9,499 labelled rows exist and would suit the volume problem well, but it is
not on the list until he says so. Nothing here forecloses it.

## Open questions, being measured rather than asked

**How much recall may we trade for precision?** Rather than asking for a number
up front, actions 2 and 3 will each produce a trade-off curve so the cut-off can
be chosen from evidence.

**Do the section headers carry over to production notes?** MDACE is MIMIC-III.
If real pipeline notes use different headers, action 2's specific list will not
transfer, though the method will. Worth building the header detection generically
rather than hardcoding MIMIC names.

## Reporting change

From the next round: **precision is the headline, recall is the constraint** —
with the caveat that term-level precision is a proxy. The business metric is
precision at the *code* level, after the lookup, and that stays unmeasurable
until the lookup exists.
Every result reports both, plus the volume that produced them. The existing rule
still applies in reverse — precision is not quotable on its own either, because
a model returning one finding per note would score high and find nothing.
