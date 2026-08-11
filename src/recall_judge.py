"""L5 — LLM adjudication of the pairs L2, L3 and L4 newly accepted.

A PLANNED STEP, NOT A CONTINGENCY. The measured pair table in `recall_matching`
shows why: no string threshold separates the good matches from the bad ones, so
every level above L1 admits pairs that nobody has checked. L2 in particular
knowingly admits `diabetes` inside `diabetes insipidus` (a different disease) and
`sepsis` inside `no evidence of sepsis` (a negation).

RUN IT ON WHAT EACH LEVEL NEWLY ACCEPTED, NOT ON EVERYTHING. L1 is exact string
equality and needs no judge. Adjudicating only the newly-accepted pairs keeps
the cost bounded and proportional to how much the ladder actually bought.

WHAT IT PRODUCES. A verdict file per level, and an adjudicated recall: the same
ladder with rejected pairs removed. Adjudicated recall is reported ALONGSIDE the
unadjudicated figure, never silently in place of it — the gap between them is
the interesting quantity, because it is how much of the ladder's gain was real.

THE JUDGE IS PLUGGABLE, AND THE CHOICE MATTERS. `--judge medgemma` uses the same
4-bit MedGemma already loaded on the Colab box, which is free but is the model
under test grading its own matches. `--judge none` writes the questions out for a
human or an external model to answer and reads the answers back. Whichever ran is
recorded in the verdict file and printed in the summary, because a benchmark
adjudicated by its own subject has to say so.

PHI: the pairs quote note text, so both the input and the verdict files stay
under the gitignored run dir. Only counts leave it.
"""

import argparse
import glob
import json
import os
import re

from .recall_config import GEN_CONFIG, LOAD_IN_4BIT, MODEL_ID, OUTPUT_DIR

_SYSTEM = (
    "You are a careful clinical terminologist. You judge whether two phrases "
    "refer to the same clinical finding. You answer with one word."
)

_QUESTION = (
    "A model reading a clinical note extracted this phrase:\n"
    "  MODEL: {span}\n"
    "A human medical coder recorded this phrase as the justification for a "
    "billing code:\n"
    "  CODER: {gold}\n"
    "\n"
    "Do these refer to the SAME clinical finding in the same patient?\n"
    "\n"
    "Answer NO if:\n"
    "  - they name different diseases, even if the words overlap "
    "(diabetes vs diabetes insipidus)\n"
    "  - one of them is negated or ruled out "
    "(sepsis vs no evidence of sepsis)\n"
    "  - they differ in acuity, laterality or body site in a way that changes "
    "the diagnosis (acute vs chronic renal failure)\n"
    "\n"
    "Answer YES if:\n"
    "  - one is an abbreviation or synonym of the other (HTN vs hypertension)\n"
    "  - one is the catalogue or SNOMED wording of the other\n"
    "  - they differ only in word order, spelling or an unimportant qualifier\n"
    "\n"
    "Answer with exactly one word: YES or NO."
)

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)


def build_messages(pair):
    """Chat messages for one pair."""
    question = _QUESTION.format(span=pair.get("span") or pair.get("name") or "",
                                gold=pair.get("gold_form", ""))
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]


def parse_verdict(reply):
    """``True``/``False``/``None`` from a judge reply.

    None means unreadable, and an unreadable verdict is KEPT rather than
    rejected: the judge failing to answer is not evidence that the match was
    wrong, and silently dropping those pairs would understate recall for a
    reason that has nothing to do with the model under test.
    """
    if not isinstance(reply, str):
        return None
    head = reply.strip()[:200]
    yes, no = _YES.search(head), _NO.search(head)
    if yes and not no:
        return True
    if no and not yes:
        return False
    if yes and no:
        return yes.start() < no.start()
    return None


def load_pairs(run_dir, levels=("L2", "L3", "L4")):
    """``{level: [pair]}`` from a run directory's new_pairs dumps."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "new_pairs_*.jsonl"))):
        level = os.path.basename(path)[len("new_pairs_"):-len(".jsonl")]
        if level not in levels:
            continue
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if pairs:
            out[level] = pairs
    return out


def judge_pairs(pairs, run_fn, resume=None):
    """Adjudicate `pairs`. Returns the list with a `verdict` field added.

    `run_fn(messages) -> reply`. `resume` is a ``{key: verdict}`` map from a
    previous partial pass; a judge run is as interruptible as the eval run.
    """
    resume = resume or {}
    out = []
    for pair in pairs:
        key = pair_key(pair)
        if key in resume:
            verdict, reply = resume[key], None
        else:
            try:
                reply = run_fn(build_messages(pair))
            except Exception as e:  # noqa: BLE001 - one bad pair must not kill the pass
                print(f"[warn] judge failed on {key[:40]}...: {e}")
                reply = None
            verdict = parse_verdict(reply)
        out.append({**pair, "verdict": verdict,
                    "judge_reply": (reply or "")[:200] if reply else ""})
    return out


def pair_key(pair):
    """Stable identity for one pair, for resuming a partial judge pass."""
    return "|".join(str(pair.get(k, "")) for k in
                    ("note_id", "level", "span", "name", "gold_form"))


def summarize(judged):
    """``{level: counts}`` plus the totals the report quotes."""
    out = {}
    for level, pairs in judged.items():
        kept = sum(1 for p in pairs if p["verdict"] is not False)
        rejected = sum(1 for p in pairs if p["verdict"] is False)
        unreadable = sum(1 for p in pairs if p["verdict"] is None)
        out[level] = {
            "n_pairs": len(pairs),
            "kept": kept,
            "rejected": rejected,
            "unreadable": unreadable,
            "reject_rate": (rejected / len(pairs)) if pairs else 0.0,
        }
    return out


def medgemma_judge(model_id=None, max_new_tokens=16):
    """A `run_fn` backed by the same 4-bit MedGemma the benchmark runs.

    Free on a box that already has the model loaded, and honest only if the
    report says the model graded its own matches — which it does.
    """
    from .model import load_medgemma, run_messages

    pipe = load_medgemma(model_id or MODEL_ID, load_in_4bit=LOAD_IN_4BIT)

    def run(messages):
        return run_messages(pipe, messages,
                            gen_config={"max_new_tokens": max_new_tokens},
                            default_gen=GEN_CONFIG)
    return run


def write_questions(judged, path):
    """Write the unanswered questions for a human or an external model.

    CONTAINS NOTE TEXT. One JSON object per line with a `verdict` field left
    null; fill it in with true/false and feed the file back via `--verdicts`.
    """
    with open(path, "w", encoding="utf-8") as f:
        for level, pairs in judged.items():
            f.writelines(json.dumps({**pair, "level": level,
                                    "verdict": None}) + "\n" for pair in pairs)
    return path


def load_verdicts(path):
    """``{pair_key: verdict}`` from a filled-in questions file."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("verdict") is not None:
                out[pair_key(rec)] = bool(rec["verdict"])
    return out


def run_judge(run_dir=None, judge="medgemma", levels=("L2", "L3", "L4"),
              verdicts_file=None, model_id=None):
    run_dir = run_dir or _latest_run_dir()
    pairs = load_pairs(run_dir, levels)
    if not pairs:
        # Two different situations, and confusing them wastes an afternoon: no
        # run in this directory at all, versus a run where every match was exact
        # and there is genuinely nothing above L1 to adjudicate.
        import glob as _glob

        if _glob.glob(os.path.join(run_dir, "new_pairs_*.jsonl")):
            print(f"Nothing to adjudicate in {run_dir}: no pairs were accepted "
                  f"above L1 ({', '.join(levels)} are all empty). Every match "
                  "was exact.")
        else:
            print(f"No new_pairs_*.jsonl in {run_dir}. Run "
                  "`python -m src.evaluate_recall` first.")
        return {}

    total = sum(len(v) for v in pairs.values())
    print(f"run dir:  {run_dir}")
    print(f"judge:    {judge}")
    print(f"pairs:    {total} across {', '.join(sorted(pairs))}")

    resume = load_verdicts(verdicts_file)
    if resume:
        print(f"verdicts: {len(resume)} supplied from {verdicts_file}")

    if judge == "none":
        run_fn = lambda messages: None
    elif judge == "medgemma":
        print("loading judge model ...")
        run_fn = medgemma_judge(model_id)
    else:
        raise ValueError(f"unknown judge {judge!r}; use medgemma or none")

    judged = {level: judge_pairs(items, run_fn, resume)
              for level, items in pairs.items()}

    for level, items in judged.items():
        path = os.path.join(run_dir, f"verdicts_{level}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps({**item, "judge": judge}) + "\n" for item in items)

    counts = summarize(judged)
    summary_path = os.path.join(run_dir, "l5_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"judge": judge, "run_dir": run_dir, "levels": counts},
                  f, indent=2, sort_keys=True)

    print("\n--- L5 adjudication " + "-" * 47)
    print(f"  {'level':6} {'pairs':>7} {'kept':>7} {'rejected':>9} "
          f"{'unreadable':>11} {'reject rate':>12}")
    for level in sorted(counts):
        c = counts[level]
        print(f"  {level:6} {c['n_pairs']:7} {c['kept']:7} {c['rejected']:9} "
              f"{c['unreadable']:11} {c['reject_rate']:12.4f}")
    if judge == "medgemma":
        print("\n  NOTE: the model under test graded its own matches. Say so "
              "wherever\n  these numbers are quoted.")
    if judge == "none":
        questions = write_questions(pairs, os.path.join(run_dir,
                                                        "l5_questions.jsonl"))
        print(f"\n  No judge ran. Questions written to {questions}\n"
              "  Fill in each `verdict` with true/false and re-run with "
              "--verdicts.")
    print("-" * 67)
    print(f"Wrote {summary_path}")
    return counts


def _latest_run_dir():
    """The most recently modified run directory under OUTPUT_DIR."""
    candidates = [d for d in glob.glob(os.path.join(OUTPUT_DIR, "*"))
                  if os.path.isdir(d)]
    if not candidates:
        raise FileNotFoundError(
            f"no run directories under {OUTPUT_DIR}. Run "
            "`python -m src.evaluate_recall` first."
        )
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="L5: adjudicate the pairs L2-L4 newly accepted"
    )
    parser.add_argument("--run-dir", default=None,
                        help="run directory holding new_pairs_*.jsonl "
                             "(default: the most recent one)")
    parser.add_argument("--judge", default="medgemma",
                        choices=("medgemma", "none"),
                        help="medgemma grades its own matches, which the "
                             "summary says out loud; none writes the questions "
                             "out for a human or an external model")
    parser.add_argument("--levels", default="L2,L3,L4")
    parser.add_argument("--verdicts", default=None,
                        help="a filled-in l5_questions.jsonl to read answers "
                             "back from")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    run_judge(run_dir=args.run_dir, judge=args.judge,
              levels=tuple(x.strip() for x in args.levels.split(",") if x.strip()),
              verdicts_file=args.verdicts, model_id=args.model)


if __name__ == "__main__":
    main()


def load_rejected(run_dir, levels=("L2", "L3", "L4")):
    """``{(note_id, span, name, gold_form)}`` the judge ruled NOT the same finding.

    Only explicit rejections. A verdict of None means the judge could not be
    read, and an unreadable verdict is not evidence that a match was wrong --
    dropping those would understate recall for a reason with nothing to do with
    the model under test.
    """
    rejected = set()
    for path in glob.glob(os.path.join(run_dir, "verdicts_*.jsonl")):
        level = os.path.basename(path)[len("verdicts_"):-len(".jsonl")]
        if level not in levels:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("verdict") is False:
                    rejected.add((rec.get("note_id"), rec.get("span", ""),
                                  rec.get("name", ""), rec.get("gold_form", "")))
    return rejected
