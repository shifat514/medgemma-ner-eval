"""MedGemma ICD-10 code assignment on the pediatric encounter notes.

Per note, per input variant:
    note text -> one prompt -> parse JSON -> a set of ICD-10 codes
              -> compare against the clinician's DX lines, exact code match

Usage:
    python -m src.evaluate_billing --oracle      # no GPU: harness check, ~1s
    python -m src.evaluate_billing               # all three variants, 12 calls
    python -m src.evaluate_billing --variant assessment_cut
    python -m src.evaluate_billing --score-only  # rescore a finished run

RUN THE ORACLE FIRST, AND THEN RUN ``full`` FIRST. Two separate checks, both
cheap, both of which have caught real bugs on the previous branches:

  --oracle       replays gold back through the parser and the scorer without a
                 model. Every variant must read 1.0000/1.0000. Anything else is
                 a harness bug and no GPU number would mean anything.

  variant full   the real model, on the note WITH the DX lines still in it. This
                 is not a result, it is a second harness check — the model is
                 being asked to read codes off the page. A low number here means
                 the prompt or the parser is wrong, not the model.

ONE PROMPT, THREE INPUTS. The variants differ only in how much of the note is
shown, so every difference between the three numbers is attributable to the text
that was removed rather than to a prompt change. See billing_config.VARIANTS.

WHY THE PER-CODE TABLE IS THE HEADLINE AND THE AGGREGATE IS NOT. There are 16
gold codes. One code is 6.25 recall points, which is wider than most differences
worth reporting, so a bare precision/recall pair invites more confidence than
n=16 can support. The per-code hit/miss table is printed in full because at this
size the whole answer key fits on screen and reading it is strictly more
informative than reading its mean.

Incremental and resumable: each (note, variant) result is appended as it
finishes and a rerun skips pairs already present. 12 calls is short enough that
this rarely matters locally, and long enough to matter on a Colab that
disconnects.

REAL PATIENT DATA. per_note.jsonl holds counts and codes only. replies.jsonl
holds raw model output, which can quote the note — both live under the
gitignored output dir, and only the aggregate report is safe to commit.
"""

import argparse
import hashlib
import json
import os

from .billing_config import (
    CAP_MARGIN,
    DEFAULT_VARIANT,
    GEN_CONFIG,
    LOAD_IN_4BIT,
    MODEL_ID,
    MODEL_NAME,
    OUTPUT_DIR,
    REPETITION_PENALTY,
    RESULTS_DIR,
    SAMPLE_FILE,
    VARIANTS,
    variant_label,
)
from .datasets.billing import load_sample, normalize_code
from .prompt_billing import build_messages, parse_codes, prompt_fingerprint


def gen_fingerprint(gen):
    """Short hash of the WHOLE generation config.

    THE READABLE PART OF THE TAG IS NOT ENOUGH, AND THAT COST A RUN. The name
    used to carry the token cap and the penalty, which looked like "every input
    that changes the output" until `model.run_messages` turned out to be
    dropping most of the config before it reached `generate` (see that
    function's docstring). The repetition-penalty run was therefore written into
    a directory named `rp115` while no penalty had been applied — a stale cache
    that reported the wrong thing under a name that said otherwise.

    Hashing the whole dict means any change to what is *asked for* lands in a
    new directory, including changes to keys nobody thought to put in the name.
    It cannot detect an argument being dropped downstream, but it does guarantee
    that fixing such a bug invalidates every result produced under it.
    """
    body = json.dumps(gen, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:6]


def run_tag(model_name, gen, prompt_fp):
    """Directory name carrying every input that changes the model's output.

    The token cap and the penalty stay spelled out because a directory listing
    is read by people; the hash is what actually makes the name complete.
    """
    tokens = gen.get("max_new_tokens") or gen.get("max_tokens")
    tag = f"{model_name}_tok{tokens}"
    penalty = gen.get("repetition_penalty", 1.0)
    if penalty and penalty != 1.0:
        tag += f"_rp{str(penalty).replace('.', '')}"
    return f"{tag}_g{gen_fingerprint(gen)}_p{prompt_fp}"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def make_oracle_reply(record):
    """The reply a perfect model would give: gold, in the requested format.

    Round-tripping gold through the real parser and the real scorer is what
    makes the oracle a check on the harness rather than a check on nothing.
    """
    return json.dumps({"codes": [
        {"code": c, "description": ""} for c in record["gold_codes"]
    ]})


def predict_note(pipe, note_text, gen_config=None):
    """One model call. Returns ``(codes, reply, n_tokens, truncated)``."""
    from .model import count_tokens, run_messages

    gen = dict(GEN_CONFIG)
    if gen_config:
        gen.update(gen_config)

    reply = run_messages(pipe, build_messages(note_text), default_gen=gen)
    n_tokens = count_tokens(pipe, reply)
    cap = gen.get("max_new_tokens", GEN_CONFIG["max_new_tokens"])
    truncated = n_tokens is not None and n_tokens >= cap - CAP_MARGIN
    return parse_codes(reply), reply, n_tokens, truncated


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_note(gold_codes, predicted):
    """Exact-code-match scoring for one note.

    Both sides are deduplicated before comparison — gold because note 96176
    lists Z68.51 twice, predictions because a model that says J30.2 twice has
    made one claim, not two. Counting the repeat would let a model raise recall
    by repeating itself.
    """
    gold = []
    seen = set()
    for c in gold_codes:
        n = normalize_code(c)
        if n and n not in seen:
            seen.add(n)
            gold.append(n)

    pred, seen_p = [], set()
    for item in predicted:
        n = normalize_code(item["code"])
        if n and n not in seen_p:
            seen_p.add(n)
            pred.append(n)

    gold_set, pred_set = set(gold), set(pred)
    tp = sorted(gold_set & pred_set)
    fp = [c for c in pred if c not in gold_set]
    fn = [c for c in gold if c not in pred_set]

    return {
        "gold": gold,
        "predicted": pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gold": len(gold),
        "n_pred": len(pred),
        "n_tp": len(tp),
        "n_fp": len(fp),
        "n_fn": len(fn),
        "n_malformed": sum(
            1 for item in predicted if not item.get("well_formed", True)
        ),
    }


def _prf(n_tp, n_fp, n_fn):
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate(per_note):
    """Micro-average over notes: pool the counts, then divide.

    Micro and not macro. With 2 to 6 gold codes per note a macro average would
    weight note 112976's two codes as heavily as note 26819's six, which is not
    the question — the question is what fraction of billed codes are recovered.
    """
    n_tp = sum(r["n_tp"] for r in per_note)
    n_fp = sum(r["n_fp"] for r in per_note)
    n_fn = sum(r["n_fn"] for r in per_note)
    out = _prf(n_tp, n_fp, n_fn)
    out.update({
        "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
        "n_gold": sum(r["n_gold"] for r in per_note),
        "n_pred": sum(r["n_pred"] for r in per_note),
        "n_notes": len(per_note),
        "n_malformed": sum(r["n_malformed"] for r in per_note),
        "n_truncated": sum(1 for r in per_note if r.get("truncated")),
    })
    return out


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


def _load_done(path):
    """``{(note_id, variant): record}`` already on disk."""
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["note_id"], rec["variant"])] = rec
    return done


def _append(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def run_eval(sample_file=SAMPLE_FILE, variants=None, oracle=False,
             model_id=None, model_name=None, backend="medgemma",
             max_new_tokens=None, repetition_penalty=None, resume=True,
             dump_replies=False, output_dir=OUTPUT_DIR, score_only=False):
    records = load_sample(sample_file)
    variants = list(variants or VARIANTS)

    # ONE SWITCH, ONE CALL SITE. Everything downstream of the reply --
    # parser, scorer, gold, variants -- is shared, so the only thing that
    # differs between a MedGemma run and a Claude run is which model wrote
    # the reply. That is the entire point of the comparison.
    if backend == "anthropic":
        from . import billing_anthropic as api
        model_id = model_id or api.ANTHROPIC_MODEL
        model_name = model_name or api.ANTHROPIC_MODEL_NAME
        gen = {"max_tokens": max_new_tokens or api.ANTHROPIC_MAX_TOKENS}
        usage = api.new_usage()
    else:
        api, usage = None, None
        model_id = model_id or MODEL_ID
        model_name = model_name or MODEL_NAME
        gen = dict(GEN_CONFIG)
        if max_new_tokens:
            gen["max_new_tokens"] = max_new_tokens
        if repetition_penalty is not None:
            gen["repetition_penalty"] = repetition_penalty

    tag = "oracle" if oracle else run_tag(model_name, gen, prompt_fingerprint())
    run_dir = os.path.join(output_dir, tag)
    per_note_path = os.path.join(run_dir, "per_note.jsonl")
    replies_path = os.path.join(run_dir, "replies.jsonl")

    done = _load_done(per_note_path) if resume else {}
    todo = [(rec, v) for v in variants for rec in records
            if (rec["note_id"], v) not in done]

    print(f"sample     {sample_file}  ({len(records)} notes)")
    print(f"variants   {', '.join(variants)}")
    print(f"run dir    {run_dir}")
    print(f"prompt     {prompt_fingerprint()}")
    print(f"backend    {backend}")
    print("gen        " + "  ".join(f"{k}={v}" for k, v in sorted(gen.items())))
    print(f"cached {len(done)}, to run {len(todo)}"
          f"{'  (score-only)' if score_only else ''}\n")

    pipe = None
    if todo and not oracle and not score_only:
        if backend == "anthropic":
            print(f"using the Anthropic API, model {model_id}")
            print("NOTE TEXT LEAVES THIS MACHINE.\n")
            pipe = api.load_client()
        else:
            from .model import load_medgemma
            print(f"loading {model_id} (4bit={LOAD_IN_4BIT}) ...")
            pipe = load_medgemma(model_id)
            print("loaded.\n")

    if not score_only:
        for i, (rec, variant) in enumerate(todo, 1):
            note_id = rec["note_id"]
            text = rec["variants"][variant]
            print(f"[{i}/{len(todo)}] note {note_id}  {variant}"
                  f"  ({rec['n_words'][variant]} words) ...", flush=True)

            if oracle:
                reply = make_oracle_reply(rec)
                predicted, n_tokens, truncated = parse_codes(reply), None, False
            elif backend == "anthropic":
                predicted, reply, n_tokens, truncated = api.predict_note(
                    pipe, text, model_id=model_id,
                    max_tokens=gen["max_tokens"], usage_acc=usage,
                )
            else:
                predicted, reply, n_tokens, truncated = predict_note(
                    pipe, text, gen_config=gen
                )

            scored = score_note(rec["gold_codes"], predicted)
            scored.update({
                "note_id": note_id,
                "variant": variant,
                "visit_kind": rec["visit_kind"],
                "n_words": rec["n_words"][variant],
                "n_reply_tokens": n_tokens,
                "truncated": truncated,
            })
            _append(per_note_path, scored)
            done[(note_id, variant)] = scored

            if dump_replies:
                _append(replies_path, {
                    "note_id": note_id, "variant": variant, "reply": reply,
                })

            p = _prf(scored["n_tp"], scored["n_fp"], scored["n_fn"])
            print(f"        gold {scored['n_gold']}  pred {scored['n_pred']}  "
                  f"tp {scored['n_tp']}  fp {scored['n_fp']}  fn {scored['n_fn']}"
                  f"   P {p['precision']:.4f}  R {p['recall']:.4f}")

    result = {}
    for variant in variants:
        rows = [done[(r["note_id"], variant)] for r in records
                if (r["note_id"], variant) in done]
        if rows:
            result[variant] = {"per_note": rows, "overall": aggregate(rows)}

    run_meta = {
        "model": model_name if not oracle else "oracle",
        "model_id": model_id if not oracle else None,
        "backend": backend,
        "prompt_fingerprint": prompt_fingerprint(),
        "max_new_tokens": gen["max_new_tokens"],
        "repetition_penalty": gen.get("repetition_penalty", 1.0),
        "gen_fingerprint": gen_fingerprint(gen),
        "run_dir": run_dir,
        "n_notes": len(records),
        "oracle": oracle,
    }
    if usage and usage["calls"]:
        run_meta["usage"] = dict(usage)
        run_meta["cost_usd"] = round(api.estimate_cost(usage), 4)
        print(f"\n{usage['calls']} API calls   "
              f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out"
              f"   ~${run_meta['cost_usd']:.4f}")

    return result, run_meta, records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(result, run_meta, records):
    gold_by_note = {r["note_id"]: r for r in records}

    print("\n" + "=" * 78)
    print("RESULTS — exact ICD-10 code match")
    print("=" * 78)
    print(f"model   {run_meta['model']}   ({run_meta.get('backend', 'medgemma')})")
    print(f"prompt  {run_meta['prompt_fingerprint']}")
    print(f"gen     max_new_tokens={run_meta['max_new_tokens']}  "
          f"repetition_penalty={run_meta['repetition_penalty']}")
    if run_meta.get("cost_usd") is not None:
        print(f"cost    ~${run_meta['cost_usd']:.4f}")
    print(f"notes   {run_meta['n_notes']}\n")

    print(f"{'variant':<18} {'gold':>5} {'pred':>5} {'tp':>4} {'fp':>4} "
          f"{'fn':>4}  {'precision':>9} {'recall':>8} {'f1':>8}")
    print("-" * 78)
    for variant, block in result.items():
        o = block["overall"]
        print(f"{variant:<18} {o['n_gold']:>5} {o['n_pred']:>5} {o['n_tp']:>4} "
              f"{o['n_fp']:>4} {o['n_fn']:>4}  {o['precision']:>9.4f} "
              f"{o['recall']:>8.4f} {o['f1']:>8.4f}")
    print()
    for variant in result:
        print(f"  {variant:<18} {variant_label(variant)}")
    print()

    for variant, block in result.items():
        o = block["overall"]
        if o["n_malformed"] or o["n_truncated"]:
            print(f"  {variant}: {o['n_malformed']} malformed code(s), "
                  f"{o['n_truncated']} truncated repl(y/ies)")

    print("\n" + "-" * 78)
    print("PER NOTE")
    print("-" * 78)
    for variant, block in result.items():
        print(f"\n{variant}  —  {variant_label(variant)}")
        print(f"  {'note':<9} {'visit':<10} {'gold':>4} {'pred':>4} {'tp':>3} "
              f"{'fp':>3} {'fn':>3}   {'P':>6} {'R':>6}")
        for row in block["per_note"]:
            p = _prf(row["n_tp"], row["n_fp"], row["n_fn"])
            print(f"  {row['note_id']:<9} {row['visit_kind']:<10} "
                  f"{row['n_gold']:>4} {row['n_pred']:>4} {row['n_tp']:>3} "
                  f"{row['n_fp']:>3} {row['n_fn']:>3}   "
                  f"{p['precision']:>6.3f} {p['recall']:>6.3f}")

    print("\n" + "-" * 78)
    print("EVERY GOLD CODE, HIT OR MISS")
    print("-" * 78)
    print("16 codes is small enough to read in full, and reading it tells you "
          "more\nthan the mean does. A miss on Z68.5x is a BMI-percentile "
          "lookup, not a\nmissed diagnosis — those are listed last.\n")

    variants = list(result)
    width = max(len(v) for v in variants) if variants else 10
    print(f"  {'note':<9} {'code':<10} " +
          "  ".join(f"{v:<{width}}" for v in variants) + "   description")
    for rec in records:
        desc_by_code = {row["code"]: row["description"]
                        for row in rec["gold_rows"]}
        for code in rec["gold_codes"]:
            marks = []
            for variant in variants:
                row = next((r for r in result[variant]["per_note"]
                            if r["note_id"] == rec["note_id"]), None)
                if row is None:
                    marks.append("-")
                else:
                    marks.append("HIT" if code in row["tp"] else "miss")
            print(f"  {rec['note_id']:<9} {code:<10} " +
                  "  ".join(f"{m:<{width}}" for m in marks) +
                  f"   {desc_by_code.get(code, '')[:40]}")

    print("\n" + "-" * 78)
    print("FALSE POSITIVES (codes the model returned that were not billed)")
    print("-" * 78)
    for variant, block in result.items():
        fps = [(r["note_id"], c) for r in block["per_note"] for c in r["fp"]]
        print(f"\n{variant}: {len(fps)}")
        for note_id, code in fps:
            print(f"  {note_id:<9} {code}")

    print()
    _ = gold_by_note


def _check_oracle(result):
    bad = [(v, b["overall"]) for v, b in result.items()
           if b["overall"]["precision"] < 1.0 or b["overall"]["recall"] < 1.0]
    if bad:
        for variant, o in bad:
            print(f"ORACLE FAIL {variant}: P {o['precision']:.4f} "
                  f"R {o['recall']:.4f}  (both must be 1.0000)")
        raise SystemExit(
            "ABORT: the harness cannot reproduce gold from gold. Fix the "
            "parser or the scorer before spending GPU time."
        )
    print("oracle OK — every variant reads 1.0000/1.0000.")


def write_results(result, run_meta, results_dir=RESULTS_DIR):
    """Aggregate metrics only — counts and rates, no note text. Safe to commit."""
    os.makedirs(results_dir, exist_ok=True)
    name = f"billing_icd_{run_meta['model']}"
    path = os.path.join(results_dir, f"{name}.json")
    payload = {
        "run": run_meta,
        "variants": {v: b["overall"] for v, b in result.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="MedGemma ICD-10 code assignment on pediatric encounter notes"
    )
    parser.add_argument("--sample-file", default=SAMPLE_FILE)
    parser.add_argument("--variant", action="append", choices=list(VARIANTS),
                        help="restrict to one variant; repeatable "
                             f"(default: all three, {DEFAULT_VARIANT} is the "
                             "headline)")
    parser.add_argument("--oracle", action="store_true",
                        help="replay gold through the parser and scorer; no GPU")
    parser.add_argument("--score-only", action="store_true",
                        help="rescore what is already cached; no model call")
    parser.add_argument("--backend", default="medgemma",
                        choices=("medgemma", "anthropic"),
                        help="anthropic sends the NOTE TEXT to the API")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help=f"default {REPETITION_PENALTY}; pass 1.0 to turn "
                             "it off, which replays the no-penalty run rather "
                             "than re-running it")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dump-replies", action="store_true",
                        help="keep raw replies (they quote note text)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    result, run_meta, records = run_eval(
        sample_file=args.sample_file,
        variants=args.variant,
        oracle=args.oracle,
        model_id=args.model,
        model_name=args.model_name,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        resume=not args.no_resume,
        dump_replies=args.dump_replies,
        output_dir=args.output_dir,
        score_only=args.score_only,
    )

    _print_summary(result, run_meta, records)

    if args.oracle:
        _check_oracle(result)
    else:
        path = write_results(result, run_meta)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
