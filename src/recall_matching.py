"""The matching ladder: four levels, each a superset of the one above.

Recall is monotonically non-decreasing up the ladder and the gain at each level
is attributable to that level, which is the whole point — one threshold would
give one number and no way to argue about it.

    L1  exact after normalization
    L2  whole-token containment, either direction
    L3  token-set Dice and/or difflib ratio, thresholded
    L4  biomedical sentence-embedding cosine

WHY A LADDER AND NOT A THRESHOLD. Measured on real pairs:

    model says            gold says                  contains   dice   char
    hyperlipidema         hyperlipidemia                  no    0.00   0.96
    back pain             chronic back pain              yes    0.80   0.69
    acute kidney injury   kidney injury, acute            no    1.00   0.68
    HTN                   hypertension                    no    0.00   0.40
    CHF                   congestive heart failure        no    0.00   0.22
    acute renal failure   chronic renal failure           no    0.67   0.75
    sepsis                no evidence of sepsis          yes    0.40   0.44
    diabetes              diabetes insipidus             yes    0.67   0.62

No string threshold separates the good rows from the bad ones. Anything loose
enough to catch CHF at 0.22 accepts acute-vs-chronic renal failure at 0.75
several times over. That is the argument for L4, and for dumping each level's
newly-matched pairs so L5 adjudicates exactly those.

The defaults follow from that table: Dice 0.80 keeps "kidney injury, acute"
(1.00) and drops acute-vs-chronic (0.67); char ratio 0.90 keeps "hyperlipidema"
(0.96) and drops the same pair (0.75). L2 knowingly admits two bad shapes —
`diabetes` inside `diabetes insipidus` is a false match and `sepsis` inside `no
evidence of sepsis` is a negation — which is why L2's new pairs are dumped
rather than trusted.

ONE PREDICTION TO AT MOST ONE GOLD FORM. Otherwise a single vague prediction
satisfies several gold entries at once and recall measures vagueness. Matching
is therefore bipartite and 1:1.

WHY NOT PLAIN GREEDY. Greedy best-first breaks on ties, and it breaks in exactly
the case the harness check depends on: feed the gold accept-sets back as model
output, several findings compete for the same form, one takes a form another
needed, and recall lands under 1.0000 with no bug present. So each level runs
greedy by rule strictness then score — best-first survives wherever it is
meaningful — and then augments to maximum cardinality over what is left.

Assignments from lower levels are FROZEN: a pair credited to L2 is never
reassigned at L3, so per-level attribution holds and the dumped pairs stay
stable. Unused lower-level edges are re-offered at every higher level, so
freezing costs nothing except in the rare case where re-optimizing an earlier
level would have freed a form — and that direction can only understate recall,
which is the safe direction for a benchmark.
"""

import difflib

from .datasets.mdace_recall import normalize_term
from .recall_config import COSINE_MIN, DICE_MIN, LEVELS, RATIO_MIN

# Rule strictness, used to order the greedy pass. Lower is stricter, and a
# stricter rule is always offered first — that is what keeps an exact match from
# being displaced by a fuzzy one that happens to score higher on its own scale.
RULE_RANK = {"exact": 0, "contains": 1, "dice": 2, "ratio": 2, "cosine": 3}

# Which level first admits each rule.
RULE_LEVEL = {"exact": "L1", "contains": "L2", "dice": "L3", "ratio": "L3",
              "cosine": "L4"}

# Cosine measured on `NeuML/pubmedbert-base-embeddings`, the default L4 backend:
# ``(model phrase, gold phrase, should_match, cosine)``.
#
# THIS IS A RESULT, NOT A FIXTURE. The plan expected L4 to be the level that
# finally separates a real synonym from a near-miss, because it is the only one
# that reaches abbreviations at all. It reaches them — and it does not separate
# them. Sorted by cosine, the pairs we want and the pairs we do not are
# interleaved from top to bottom: `acute renal failure` against `chronic renal
# failure` scores 0.833, above eight of the ten pairs L4 exists to catch.
#
# So the argument the string table makes about L1-L3 is true one level higher up
# as well, and L5 is not a refinement of L4 — it is what makes L4's gain
# interpretable. `no_threshold_separates` asserts exactly that, and the report
# quotes these numbers rather than describing them.
#
# Two other biomedical encoders were measured and are worse:
# `pritamdeka/S-PubMedBert-MS-MARCO` saturates — every pair lands in 0.85-0.99,
# `pneumonia` against `fracture of left wrist` included — and
# `abhinand/MedEmbed-small-v0.1` puts the `no evidence of sepsis` negation
# (0.843) above `htn`/`hypertension` (0.621), `chf`/`congestive heart failure`
# (0.577) and `mi`/`myocardial infarction` (0.557). A general-purpose encoder
# was not measured because it does not know `HTN` at all, which is the premise.
MEASURED_COSINE = (
    ("acute kidney injury", "kidney injury acute",                   True,  0.978),
    ("back pain",           "chronic back pain",                     True,  0.933),
    ("acute renal failure", "chronic renal failure",                 False, 0.833),
    ("anemia",              "sickle cell anemia",                    False, 0.744),
    ("hyperlipidema",       "hyperlipidemia",                        True,  0.723),
    ("chf",                 "congestive heart failure",              True,  0.721),
    ("mi",                  "myocardial infarction",                 True,  0.719),
    ("sepsis",              "no evidence of sepsis",                 False, 0.668),
    ("htn",                 "hypertension",                          True,  0.662),
    ("chest pain",          "abdominal pain",                        False, 0.659),
    ("cabg",                "coronary artery bypass graft",          True,  0.653),
    ("afib",                "atrial fibrillation",                   True,  0.646),
    ("hypertension",        "pulmonary hypertension",                False, 0.642),
    ("esrd",                "end stage renal disease",               True,  0.641),
    ("copd",                "chronic obstructive pulmonary disease", True,  0.610),
    ("diabetes",            "diabetes insipidus",                    False, 0.545),
    ("pneumonia",           "fracture of left wrist",                False, 0.177),
)


def no_threshold_separates(table=MEASURED_COSINE):
    """True when some unwanted pair outscores some wanted one in `table`.

    The single fact the L4 threshold has to be chosen in spite of: there is no
    cutoff that admits every pair we want and no pair we do not.
    """
    want = [score for *_pair, keep, score in table if keep]
    dont = [score for *_pair, keep, score in table if not keep]
    return bool(want) and bool(dont) and min(want) < max(dont)


# ---------------------------------------------------------------------------
# String rules
# ---------------------------------------------------------------------------

def token_contains(a, b):
    """Whole-token containment in either direction, on normalized strings.

    Space-padded so `ca` does not match inside `cabg`. Direction is discarded on
    purpose: the model saying less than gold ("back pain" for "chronic back
    pain") and more than gold are both L2 matches, and both get dumped for
    adjudication.
    """
    if not a or not b:
        return False
    pa, pb = f" {a} ", f" {b} "
    return pa in pb or pb in pa


def dice(a, b):
    """Token-set Dice coefficient. Order-insensitive, so it reaches inversions."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def char_ratio(a, b):
    """difflib similarity on the normalized strings. Reaches typos."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def string_rule(a, b, dice_min=DICE_MIN, ratio_min=RATIO_MIN):
    """The strictest rule admitting ``(a, b)``, or ``None``.

    Returns ``(rule, score)``. Score orders candidates within a rule; it is not
    a probability and is not comparable across rules.
    """
    if not a or not b:
        return None
    if a == b:
        return "exact", 1.0
    if token_contains(a, b):
        # Length ratio: "back pain" inside "chronic back pain" scores higher
        # than "sepsis" inside "no evidence of sepsis", which is the right order
        # to offer them in.
        lo, hi = sorted((len(a), len(b)))
        return "contains", (lo / hi if hi else 0.0)
    d, c = dice(a, b), char_ratio(a, b)
    if d >= dice_min and d >= c:
        return "dice", d
    if c >= ratio_min:
        return "ratio", c
    if d >= dice_min:
        return "dice", d
    return None


# ---------------------------------------------------------------------------
# L4 — biomedical sentence embeddings
# ---------------------------------------------------------------------------

class Embedder:
    """Cosine similarity over a biomedical sentence encoder, with a cache.

    A general-purpose MiniLM does not know that HTN is hypertension, and
    reaching abbreviations is the only reason L4 exists — so the model id is a
    biomedical one and is printed in the report next to the threshold.

    The backend is imported lazily and `load` returns None when it is absent, so
    L1-L3 stay unit-testable on a CPU-only laptop and only L4 needs Colab.
    """

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self._cache = {}

    @classmethod
    def load(cls, model_name=None, quiet=False):
        """Build an Embedder, or return None with a clear message."""
        from .recall_config import EMBED_MODEL

        model_name = model_name or EMBED_MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            if not quiet:
                print("[info] L4 skipped: sentence-transformers is not "
                      "installed. `uv sync --extra embed` enables it; L1-L3 "
                      "are unaffected.")
            return None
        try:
            return cls(SentenceTransformer(model_name), model_name)
        except Exception as e:  # noqa: BLE001 - a missing model must not kill a run
            if not quiet:
                print(f"[info] L4 skipped: could not load {model_name!r}: {e}")
            return None

    def encode(self, strings):
        """L2-normalized vectors for `strings`, cached across notes."""
        import numpy as np

        missing = [s for s in dict.fromkeys(strings) if s not in self._cache]
        if missing:
            vecs = np.asarray(
                self.model.encode(missing, convert_to_numpy=True,
                                  show_progress_bar=False), dtype="float32")
            vecs = vecs / np.clip(
                np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
            for s, v in zip(missing, vecs):
                self._cache[s] = v
        return np.stack([self._cache[s] for s in strings])

    def similarity(self, left, right):
        """``{(l, r): cosine}`` for every pair. Both sides are small per note."""
        if not left or not right:
            return {}
        left = list(dict.fromkeys(left))
        right = list(dict.fromkeys(right))
        matrix = self.encode(left) @ self.encode(right).T
        return {(l, r): float(matrix[i, j])
                for i, l in enumerate(left) for j, r in enumerate(right)}


# ---------------------------------------------------------------------------
# Bipartite matching
# ---------------------------------------------------------------------------

def _augment(node, adjacency, taken, visited):
    """Kuhn's augmenting step. `taken` maps gold form -> finding index."""
    for form in adjacency.get(node, ()):
        if form in visited:
            continue
        visited.add(form)
        holder = taken.get(form)
        if holder is None or _augment(holder, adjacency, taken, visited):
            taken[form] = node
            return True
    return False


def assign(edges, free_findings, free_forms):
    """Assign as many free findings to free forms as `edges` allows.

    `edges` is ``[(rule, score, finding_index, form)]``. Stricter rules and
    higher scores are offered first, so best-first survives where it is
    meaningful; augmenting paths then pick up what a purely greedy pass would
    have stranded, which is what makes the oracle check come out at exactly
    1.0000 instead of "nearly".

    Returns ``{finding_index: (form, rule, score)}``.
    """
    usable = sorted(
        (e for e in edges if e[2] in free_findings and e[3] in free_forms),
        key=lambda e: (RULE_RANK.get(e[0], 99), -e[1], e[2], e[3]),
    )
    if not usable:
        return {}

    detail, adjacency = {}, {}
    for rule, score, finding, form in usable:
        # First write wins, and `usable` is sorted strictest-first, so a pair
        # reachable by two rules is credited to the stricter one.
        if (finding, form) not in detail:
            detail[(finding, form)] = (rule, score)
            adjacency.setdefault(finding, []).append(form)

    taken, claimed = {}, set()
    for rule, score, finding, form in usable:
        if finding not in claimed and form not in taken:
            taken[form] = finding
            claimed.add(finding)
    for finding in sorted(adjacency):
        if finding not in claimed and _augment(finding, adjacency, taken, set()):
            claimed = set(taken.values())

    out = {}
    for form, finding in taken.items():
        rule, score = detail[(finding, form)]
        out[finding] = (form, rule, score)
    return out


def build_edges(findings, gold_forms, embedder=None, dice_min=DICE_MIN,
                ratio_min=RATIO_MIN, cosine_min=COSINE_MIN, blocked=None):
    """Every admissible ``(rule, score, finding_index, form)`` edge.

    `findings` is a list of candidate-string tuples: one finding contributes its
    `span` and its `name` and EITHER may match, which is the point of the
    two-field prompt. A pair reachable by several rules keeps only the strictest.
    """
    edges, placed = [], set()
    blocked = blocked or set()
    for i, cands in enumerate(findings):
        for form in gold_forms:
            if (i, form) in blocked:
                continue
            best = None
            for cand in cands:
                got = string_rule(cand, form, dice_min, ratio_min)
                if got is None:
                    continue
                if best is None or (RULE_RANK[got[0]], -got[1]) < \
                        (RULE_RANK[best[0]], -best[1]):
                    best = got
            if best is not None:
                edges.append((best[0], best[1], i, form))
                placed.add((i, form))

    if embedder is not None:
        flat = [c for cands in findings for c in cands]
        sims = embedder.similarity(flat, sorted(gold_forms))
        for i, cands in enumerate(findings):
            for form in gold_forms:
                if (i, form) in placed or (i, form) in blocked:
                    continue
                score = max((sims.get((c, form), 0.0) for c in cands),
                            default=0.0)
                if score >= cosine_min:
                    edges.append(("cosine", score, i, form))
    return edges


def match(findings, gold_forms, embedder=None, dice_min=DICE_MIN,
          ratio_min=RATIO_MIN, cosine_min=COSINE_MIN, levels=LEVELS,
          blocked=None):
    """Run the ladder for one note against one gold form set.

    Returns ``{level: {"pairs": {finding_index: (form, rule, score)},
                       "matched_forms": set,
                       "new": [(finding_index, form, rule, score)]}}``,
    cumulative at each level, with `new` holding only what that level added.
    """
    findings = [tuple(dict.fromkeys(s for s in cands if s))
                for cands in findings]
    gold_forms = sorted(gold_forms)
    edges = build_edges(findings, gold_forms, embedder, dice_min, ratio_min,
                        cosine_min, blocked=blocked)

    results, pairs, matched_forms = {}, {}, set()
    for level in levels:
        allowed = {r for r, lv in RULE_LEVEL.items()
                   if lv in levels and levels.index(lv) <= levels.index(level)}
        free_findings = {i for i in range(len(findings)) if i not in pairs}
        free_forms = {f for f in gold_forms if f not in matched_forms}
        added = assign([e for e in edges if e[0] in allowed],
                       free_findings, free_forms)
        pairs.update(added)
        matched_forms.update(form for form, _rule, _score in added.values())
        results[level] = {
            "pairs": dict(pairs),
            "matched_forms": set(matched_forms),
            "new": sorted((i, form, rule, score)
                          for i, (form, rule, score) in added.items()),
        }
    return results


def candidate_strings(finding):
    """The normalized strings one finding offers to the matcher.

    Two fields per finding — the phrase as written in the note, and the standard
    clinical name — and either may match. That is the point of the two-field
    prompt: MedGemma already knows HTN expands to hypertension, and using its
    expansion beats making the matcher infer one.
    """
    return tuple(dict.fromkeys(
        s for s in (normalize_term(finding.get("span")),
                    normalize_term(finding.get("name"))) if s))
