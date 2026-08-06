"""Align predicted entity spans back onto the original token list to build BIO tags.

This is the trickiest part of the generative-NER pipeline: MedGemma returns free
text spans, but seqeval scores BIO tags over the SAME whitespace tokens the gold
labels use. We match spans case- and punctuation-insensitively, as contiguous
token runs, and tag every non-overlapping occurrence without overwriting a token
already claimed by an earlier span.
"""

import re

# Strip leading/trailing non-word characters (punctuation) but keep internals,
# so "cancer." -> "cancer", "(hepatitis)" -> "hepatitis", "5-FU" -> "5-fu".
_STRIP_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


def normalize_token(token):
    """Lowercase and strip surrounding punctuation for robust matching."""
    return _STRIP_RE.sub("", token).lower()


def align_entities_to_bio(tokens, entities, first_only=False):
    """Convert predicted ``(text, type)`` spans into a BIO tag list over `tokens`.

    - Matching is case/punctuation-insensitive (see ``normalize_token``).
    - A multi-word span must match a contiguous run of tokens.
    - Every non-overlapping occurrence of a span is tagged; tokens already tagged
      by an earlier entity are never overwritten (first-come-first-served).

    `first_only=True` stops after the first match for each entity instead of
    tagging every occurrence. Default False preserves the original behavior the
    NCBI/BC5CDR evaluation depends on; the MIMIC evaluation uses it to trade the
    multi-occurrence expansion against recall (see --align-mode).
    """
    norm = [normalize_token(t) for t in tokens]
    n = len(tokens)
    bio = ["O"] * n

    for text, etype in entities:
        span = [w for w in (normalize_token(w) for w in text.split()) if w]
        length = len(span)
        if length == 0:
            continue
        i = 0
        while i + length <= n:
            window = norm[i:i + length]
            if window == span and all(bio[i + k] == "O" for k in range(length)):
                bio[i] = f"B-{etype}"
                for k in range(1, length):
                    bio[i + k] = f"I-{etype}"
                if first_only:
                    break
                i += length  # non-overlapping: skip past the matched run
            else:
                i += 1
    return bio
