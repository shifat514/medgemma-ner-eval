"""Diagnose raw model replies — repetition loops, truncation, item duplication.

Run this where the replies are (e.g. the Colab VM):

    python -m src.analyze_replies outputs/mimic/<tag>/raw_replies.jsonl

    # compare two runs to see whether the SAME chunks hit the cap
    python -m src.analyze_replies run_1024.jsonl run_1536.jsonl

Why it exists: a smoke run hit ``max_new_tokens`` on the same 5 of 21 chunks at
BOTH 1024 and 1536. A cap that binds identically at two very different limits is
not a length problem — the usual cause is greedy decoding falling into a
repetition loop, emitting the same JSON object until it runs out of budget.

SAFETY: this prints structure and counts only — item counts, duplicate rates,
repeat-run lengths, truncation flags, and note/chunk identifiers. It never prints
span text, note text, or any reply content. The input file DOES quote note text
(the prompt echoes the chunk), so keep it in the gitignored run directory and do
not commit or upload it.
"""

import argparse
import json
import os
import re
from collections import Counter


def _items(reply):
    """Best-effort list of entity-ish objects in a reply, for counting only."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", reply or ""):
        try:
            obj = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _fingerprints(items):
    """Order-preserving hashes of each item, so repeats are detectable.

    Hashed, not stored: fingerprints never reveal the underlying text.
    """
    return [hash(json.dumps(o, sort_keys=True)) for o in items]


def _max_run(seq):
    """Longest run of identical consecutive values."""
    best = run = 0
    prev = object()
    for x in seq:
        run = run + 1 if x == prev else 1
        prev = x
        best = max(best, run)
    return best


def _looks_truncated(reply):
    """True when the reply does not parse as complete JSON.

    Checking the final character is NOT sufficient: a list cut off mid-stream
    ends with ``}`` from its last complete object, which looks like a clean
    close. The only reliable test is whether the whole payload parses.

    Tolerates JSONL (several complete objects in sequence) and a markdown fence.
    """
    s = (reply or "").strip()
    if not s:
        return False

    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    try:
        json.loads(s)
        return False
    except (json.JSONDecodeError, ValueError):
        pass

    # JSONL / concatenated objects: complete iff decoding consumes everything.
    decoder = json.JSONDecoder()
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            return False
        try:
            _, i = decoder.raw_decode(s, i)
        except ValueError:
            return True
    return False


def analyze_file(path, repeat_threshold=3):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    per_chunk, loops = [], []
    for rec in rows:
        reply = rec.get("reply") or ""
        items = _items(reply)
        fps = _fingerprints(items)
        n, uniq = len(fps), len(set(fps))
        run = _max_run(fps)
        info = {
            "note_id": rec.get("note_id"),
            "chunk": tuple(rec.get("chunk") or ()),
            "shape": rec.get("shape"),
            "n_kept": rec.get("n_kept"),
            "chars": len(reply),
            "items": n,
            "unique_items": uniq,
            "dup_items": n - uniq,
            "max_repeat_run": run,
            "truncated": _looks_truncated(reply),
        }
        per_chunk.append(info)
        if run >= repeat_threshold or (n and uniq / n < 0.5):
            loops.append(info)
    return per_chunk, loops


def _report(path, per_chunk, loops):
    n = len(per_chunk)
    print(f"\n=== {os.path.basename(path)} — {n} chunk replies ===")
    if not n:
        print("  (no replies; was the run started with --dump-replies?)")
        return

    tot_items = sum(c["items"] for c in per_chunk)
    tot_dups = sum(c["dup_items"] for c in per_chunk)
    trunc = [c for c in per_chunk if c["truncated"]]
    longest = max(per_chunk, key=lambda c: c["chars"])

    print(f"  JSON-ish items parsed        {tot_items:,}")
    print(f"  duplicate items              {tot_dups:,}"
          f"  ({0.0 if not tot_items else 100.0 * tot_dups / tot_items:.1f}%)")
    print(f"  replies ending mid-structure {len(trunc)} / {n}"
          "   <- consistent with hitting max_new_tokens")
    print(f"  longest reply                {longest['chars']:,} chars, "
          f"{longest['items']} items")
    print(f"  chunks flagged as looping    {len(loops)} / {n}")

    if loops:
        print("\n  REPETITION-LOOP CANDIDATES")
        print(f"  {'note_id':<20} {'chunk':<14} {'items':>6} {'uniq':>6} "
              f"{'maxrun':>7} {'chars':>7}  trunc")
        for c in sorted(loops, key=lambda c: -c["max_repeat_run"])[:20]:
            print(f"  {str(c['note_id']):<20} {str(c['chunk']):<14} "
                  f"{c['items']:>6} {c['unique_items']:>6} "
                  f"{c['max_repeat_run']:>7} {c['chars']:>7}  {c['truncated']}")
        print("\n  A high maxrun with low uniq means greedy decoding emitted the")
        print("  same object repeatedly until the cap. Raising max_new_tokens")
        print("  cannot fix that — it just buys more repetitions. Try")
        print("  repetition_penalty ~1.1, or no_repeat_ngram_size.")
    else:
        print("\n  No repetition loops detected. Chunks hitting the cap are")
        print("  genuinely dense, and the cap is a real (if costly) limit.")

    dist = Counter(c["shape"] for c in per_chunk)
    print(f"\n  reply shapes: " + ", ".join(f"{k}={v}" for k, v in dist.most_common()))


def _compare(a_path, a_chunks, b_path, b_chunks):
    """Do the SAME chunks truncate in both runs?"""
    def key(c):
        return (c["note_id"], c["chunk"])

    a_tr = {key(c) for c in a_chunks if c["truncated"]}
    b_tr = {key(c) for c in b_chunks if c["truncated"]}
    both = a_tr & b_tr

    print("\n=== comparison ===")
    print(f"  truncated in {os.path.basename(a_path)}: {len(a_tr)}")
    print(f"  truncated in {os.path.basename(b_path)}: {len(b_tr)}")
    print(f"  truncated in BOTH:                      {len(both)}")
    if a_tr and len(both) == len(a_tr) == len(b_tr):
        print("  -> identical set. The cap is not a length limit; these chunks")
        print("     would truncate at any budget. Repetition loop is the likely")
        print("     cause — check the maxrun column above.")
    elif both:
        print("  -> overlapping but not identical.")

    a_by = {key(c): c for c in a_chunks}
    grew = [(k, a_by[k]["items"], c["items"]) for c in b_chunks
            if (k := key(c)) in a_by and c["items"] > a_by[k]["items"]]
    if grew:
        extra = sum(b - a for _, a, b in grew)
        print(f"  chunks emitting more items in the second run: {len(grew)} "
              f"(+{extra} items)")


def main():
    ap = argparse.ArgumentParser(
        description="Repetition/truncation analysis of raw_replies.jsonl "
                    "(prints structure only — never note text)")
    ap.add_argument("paths", nargs="+",
                    help="one or two raw_replies.jsonl files")
    ap.add_argument("--repeat-threshold", type=int, default=3,
                    help="flag a reply when N identical items repeat in a row")
    args = ap.parse_args()

    results = []
    for path in args.paths[:2]:
        per_chunk, loops = analyze_file(path, args.repeat_threshold)
        _report(path, per_chunk, loops)
        results.append((path, per_chunk))

    if len(results) == 2:
        _compare(results[0][0], results[0][1], results[1][0], results[1][1])


if __name__ == "__main__":
    main()
