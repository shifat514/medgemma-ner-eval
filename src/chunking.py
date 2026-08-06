"""Chunk long notes into overlapping token windows; merge per-chunk predictions.

Why this exists: the NCBI/BC5CDR pipeline handled single sentences. A MIMIC
discharge summary is 181-5,425 whitespace tokens and can carry 381 gold
entities, so one call per note would blow past any sane ``max_new_tokens``.

The two rules that keep scoring honest:

1. **Per-chunk alignment.** ``align.align_entities_to_bio`` tags *every*
   non-overlapping occurrence of a predicted string. Called on a whole 10k-char
   note, one prediction of "aspirin" becomes ~15 predicted entities. Calling it
   on a 400-token window instead confines that expansion to the window the model
   actually read. It narrows the effect; it does not eliminate it, so the
   evaluation logs the expansion factor.

2. **Dedupe after stitching.** Overlapping windows re-read the same tokens, so
   an entity in an overlap region is predicted twice. Merging is done on
   ``(token_start, token_end, type)`` — mapped to char offsets for reporting —
   before the final BIO sequence is painted.

Everything here is pure Python over token lists: no torch, no model, so it is
fully unit-testable with synthetic data.
"""

import bisect
import re

_TOKEN_RE = re.compile(r"\S+")


def tokenize_with_spans(text):
    """Whitespace-tokenize `text`, returning ``(tokens, char_spans)``.

    Same tokenization as the NCBI/BC5CDR adapters (``\\S+``), so gold char
    offsets land on token boundaries and BIO tags line up with what
    ``align_entities_to_bio`` expects.
    """
    tokens, spans = [], []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def chunk_windows(n_tokens, chunk_words, overlap_words):
    """Overlapping ``[start, end)`` token windows covering **all** `n_tokens`.

    Coverage is total by construction — the final window is clamped to
    `n_tokens` — so no note content is ever dropped. `overlap_words` is clamped
    below `chunk_words` so the stride is always positive.
    """
    if n_tokens <= 0:
        return []
    chunk = max(1, int(chunk_words))
    overlap = max(0, min(int(overlap_words), chunk - 1))
    stride = chunk - overlap

    windows = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk, n_tokens)
        windows.append((start, end))
        if end >= n_tokens:
            break
        start += stride
    return windows


def bio_to_spans(bio):
    """Decode a BIO tag list into ``(token_start, token_end, type)`` triples.

    `token_end` is exclusive. A stray ``I-`` with no opening ``B-`` is ignored
    (``align_entities_to_bio`` never emits one, but a hand-built sequence might).
    """
    spans = []
    i, n = 0, len(bio)
    while i < n:
        tag = bio[i]
        if isinstance(tag, str) and tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < n and bio[j] == f"I-{etype}":
                j += 1
            spans.append((i, j, etype))
            i = j
        else:
            i += 1
    return spans


def spans_to_bio(n_tokens, spans, priority=None):
    """Paint ``(token_start, token_end, type)`` spans onto a BIO list.

    All-or-nothing: a span is painted only if every token it covers is still
    ``O``. Partially-overlapping spans are therefore dropped rather than
    producing a corrupt ``I-`` -led sequence, and the count of dropped spans is
    returned so it can be reported.

    Paint order is deterministic: earliest start first, then longest first (so
    an outer span wins over a span nested inside it), then `priority` order as a
    final tiebreak. This mirrors ``align_entities_to_bio``'s first-come-first-
    served token claiming, so gold and predictions are flattened symmetrically.

    Returns ``(bio, dropped_spans)``.
    """
    order = {t: i for i, t in enumerate(priority or [])}
    fallback = len(order)

    def sort_key(s):
        start, end, etype = s
        return (start, -(end - start), order.get(etype, fallback))

    bio = ["O"] * n_tokens
    dropped = []
    for start, end, etype in sorted(spans, key=sort_key):
        start = max(0, start)
        end = min(n_tokens, end)
        if end <= start:
            dropped.append((start, end, etype))
            continue
        if any(bio[i] != "O" for i in range(start, end)):
            dropped.append((start, end, etype))
            continue
        bio[start] = f"B-{etype}"
        for i in range(start + 1, end):
            bio[i] = f"I-{etype}"
    return bio, dropped


def merge_chunk_spans(chunk_results):
    """Merge per-chunk BIO into note-level token spans, deduped.

    `chunk_results` is a list of ``(window_start, chunk_bio)``. Chunk-local token
    indices are shifted by `window_start` into note space, then deduped on
    ``(token_start, token_end, type)``.

    Returns ``(spans, n_duplicates)`` where `spans` is sorted and unique and
    `n_duplicates` counts the redundant re-predictions removed — i.e. how much
    the overlap regions double-counted.
    """
    counts = {}
    for window_start, chunk_bio in chunk_results:
        for start, end, etype in bio_to_spans(chunk_bio):
            key = (window_start + start, window_start + end, etype)
            counts[key] = counts.get(key, 0) + 1

    duplicates = sum(c - 1 for c in counts.values())
    return sorted(counts), duplicates


def token_spans_to_char(spans, char_spans):
    """Map ``(token_start, token_end, type)`` to ``(char_start, char_end, type)``.

    Uses the token->char table from ``tokenize_with_spans``. Spans that fall
    outside the table are skipped.
    """
    out = []
    n = len(char_spans)
    for start, end, etype in spans:
        if start < 0 or end > n or end <= start:
            continue
        out.append((char_spans[start][0], char_spans[end - 1][1], etype))
    return out


def char_spans_to_token_spans(spans, char_spans, with_index=False):
    """Map ``(char_start, char_end, type)`` onto token-index spans.

    A token is covered if its char range overlaps the span's. Gold span *starts*
    land on token boundaries (verified across the corpus), but ~4% of span *ends*
    fall inside a token — usually a trailing ``.`` or ``,``, occasionally a
    run-together string like ``DAILY:PRN``. Such a span widens to the whole
    token. That is the standard consequence of whitespace tokenization and it
    applies identically to gold and predicted spans, so neither side benefits;
    ``build_gold`` counts how often it happens so the report can state it.

    Spans covering no token (e.g. a span landing entirely in whitespace) are
    skipped. With `with_index`, each result is ``(tok_start, tok_end, type, i)``
    where `i` indexes into `spans`, so callers can tell which inputs survived.

    Binary search rather than a linear scan per span: a note can pair 5,400
    tokens with 381 gold spans, and the quadratic version dominated runtime.
    """
    starts = [s for s, _ in char_spans]
    ends = [e for _, e in char_spans]

    out = []
    for i, (char_start, char_end, etype) in enumerate(spans):
        # first token whose end > char_start; last token whose start < char_end
        first = bisect.bisect_right(ends, char_start)
        last = bisect.bisect_left(starts, char_end) - 1
        if first > last or first >= len(char_spans) or last < 0:
            continue
        out.append((first, last + 1, etype, i) if with_index
                   else (first, last + 1, etype))
    return out
