"""Claude Haiku backend for zero-shot NER — the API-based sibling of model.py.

Reuses the SAME prompt as the MedGemma run (``prompt._SYSTEM`` + ``build_prompt``)
so the only variable between the two runs is the model. Each sentence is sent to
the Anthropic Messages API via the official ``anthropic`` SDK — no GPU, no
transformers, runs on CPU locally.

The API key is read from a ``.env`` file (python-dotenv) via ``ANTHROPIC_API_KEY``;
it is never hardcoded, printed, or committed. ``anthropic`` and ``dotenv`` are
imported lazily inside the functions so the CPU-only unit tests can mock them.
"""

import time

from .config import (
    ANTHROPIC_API_KEY_ENV,
    HAIKU_BACKOFF_BASE,
    HAIKU_MAX_RETRIES,
    HAIKU_MAX_TOKENS,
    HAIKU_MODEL,
    HAIKU_REQUEST_DELAY,
    HAIKU_TEMPERATURE,
)
from .prompt import _SYSTEM, build_prompt


def load_haiku_client():
    """Load the API key from .env and build an Anthropic client.

    Raises a clear error (with no key material) if the key is absent, so a
    misconfigured .env fails fast rather than deep inside the request loop.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()  # populate os.environ from a .env in the working directory
    if not os.environ.get(ANTHROPIC_API_KEY_ENV):
        raise RuntimeError(
            f"{ANTHROPIC_API_KEY_ENV} is not set. Copy .env.example to .env and add "
            "your key (see the README). The key is never read from anywhere else."
        )

    import anthropic

    # The SDK reads ANTHROPIC_API_KEY from the environment on its own.
    return anthropic.Anthropic()


def _reply_text(response):
    """Concatenate the text blocks of an Anthropic Messages response."""
    return "".join(b.text for b in response.content if b.type == "text")


def run_haiku(
    client,
    sentence,
    usage_acc=None,
    model_id=None,
    delay=None,
    max_retries=None,
    backoff_base=None,
):
    """Prompt Claude Haiku with one sentence and return the raw reply string.

    Uses the identical prompt as MedGemma (system + user turn). ``temperature=0``
    for deterministic output. Rate-limit / transient errors are retried with
    exponential backoff; a client (4xx) error other than rate-limit is raised
    immediately. If ``usage_acc`` (a dict with input_tokens/output_tokens/calls)
    is given, token usage is accumulated into it for the end-of-run cost estimate.
    """
    import anthropic

    model_id = model_id or HAIKU_MODEL
    delay = HAIKU_REQUEST_DELAY if delay is None else delay
    max_retries = HAIKU_MAX_RETRIES if max_retries is None else max_retries
    backoff_base = HAIKU_BACKOFF_BASE if backoff_base is None else backoff_base

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=HAIKU_MAX_TOKENS,
                temperature=HAIKU_TEMPERATURE,
                system=_SYSTEM,
                messages=[{"role": "user", "content": build_prompt(sentence)}],
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError):
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_base * (2 ** attempt))
            continue
        except anthropic.APIStatusError as e:
            # Retry only on server-side errors; surface real client errors.
            if e.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(backoff_base * (2 ** attempt))
                continue
            raise

        if usage_acc is not None:
            usage_acc["input_tokens"] += response.usage.input_tokens
            usage_acc["output_tokens"] += response.usage.output_tokens
            usage_acc["calls"] += 1
        if delay:
            time.sleep(delay)  # be polite between successful calls
        return _reply_text(response)
