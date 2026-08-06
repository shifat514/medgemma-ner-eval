# Smoke-run history (n=5) — MedGemma medication NER

Aggregate metrics only. No note text, patient data, or example snippets.

These figures are **transcribed from Colab T4 runs**, not generated on the machine
that holds this file — there is no GPU here. The oracle rows *are* generated
locally (`--oracle` needs no GPU) and are reproducible with the command shown.

Smoke set: the first 5 notes of the seeded 100-note draw (`seed=13`), 21 chunks
at 400/80, **372 gold spans**.

## Runs

| run | config | micro P | micro R | micro F1 | entities/chunk | cap hits |
|---|---|---|---|---|---|---|
| **oracle ceiling** | perfect extractor, same pipeline | 0.8987 | 0.9059 | **0.9023** | 21.8 | 0 |
| smoke 1 | original parser, `max_new_tokens=1024` | 0.323 | 0.086 | 0.136 | 3.5 | — |
| smoke 2 | parse fix + prompt rewrite, 1024 | 0.3242 | 0.2876 | 0.3048 | 27.7 | 5/21 (24%) |
| smoke 3 | + `max_new_tokens=1536` | *pending* | | | | |

Per-type F1 at smoke 2: Medication .397, Frequency .358, Dose .336, Mode .231,
Duration .160, Reason .108.

Note the ceiling for **this 5-note set is 0.9023**, not the 0.8698 measured on
n=100 — these five notes are slightly easier to align. Compare smoke runs against
0.9023.

## What each change bought

**Smoke 1 → 2: the parse fix, not the synonym table.** Recall went 0.086 → 0.2876
(3.3x). The run reported `dropped: unrecognized type = 0` and `rescued by type
normalization = 0`, which says the rewritten prompt made the model emit the six
exact type names, and the synonym table never fired. The table stays as a safety
net — it costs nothing and catches drift — but the prompt did the work.

All 21 chunks returned parseable JSON (`reply shapes: json=21`), against 17 of 22
realistic shapes silently failing before the fix.

**Smoke 2 → 3: removing a configuration artefact.** Generation hit
`max_new_tokens` on 5 of 21 chunks (24%). A capped chunk is cut off mid-list, so
entities after the cut were never emitted at all. That depresses recall for a
config reason, not a model reason. Default raised 1024 → 1536.

## Is the model over-extracting?

The obvious reading of "27.7 entities/chunk against ~13 gold" is a 2x
over-extraction. Measured against the right baseline it is **~1.27x**, and the
2x figure is wrong twice over:

| | |
|---|---|
| gold spans per chunk | **17.7** (not ~13) |
| oracle emits per chunk | **21.8** |
| oracle items per gold span | **1.23x** ← chunk overlap, not error |
| model emits per chunk | 27.7 |
| model / gold | 1.56x |
| **model / oracle** | **1.27x** ← true over-extraction |

Chunks overlap by 80 tokens, so a gold span in an overlap region is legitimately
offered to two chunks. A *perfect* extractor therefore emits 1.23 items per gold
span before making any error at all. The model's excess over that is 1.27x —
real, but modest.

**And over-extraction is not what is costing precision.** Derived from the
reported P=0.3242 / R=0.2876 against 372 gold spans: roughly 107 true positives
and **~330 predicted spans total** — i.e. the model produced *fewer* scored spans
than the 372 gold, despite emitting more raw items per chunk. (Derived from the
reported rates, not read from a per-span dump.)

So the picture is: emits ~27% more items than ideal, but those items shed heavily
during alignment — they do not match the text verbatim, or duplicate each other —
leaving it slightly *under*-predicting at the span level. Precision 0.324 is
therefore driven by **spans being wrong**, not by there being too many of them.
Tightening the prompt to make it extract less would attack the wrong quantity.

The per-type ordering supports this: `Reason` (.108) and `Duration` (.160) are
the long free-text spans where exact boundary agreement is hardest, while
`Medication` (.397) — short, well-bounded, unambiguous — is best. That is a
boundary-precision problem.

## Reproducing the oracle rows

```bash
python -m src.evaluate_mimic --oracle --limit 5 --no-resume    # 0.9023, seconds
python -m src.evaluate_mimic --oracle --n 100                  # 0.8698
```

## Runtime

~36 s/chunk on a free T4 at 4-bit. 21 chunks = 17 min for n=5.

| sample | chunks | projected |
|---|---|---|
| n=5 | 21 | 17 min |
| n=50 | 262 | ~2.6 h |
| n=100 | 561 | ~5.6 h |

n=100 is not planned on free Colab — 5.6 h across disconnects is not worth it,
even with resume. Deferred pending better compute.
