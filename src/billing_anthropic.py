"""Anthropic API backend for the billing ICD evaluation — MedGemma's API sibling.

Requested by Ehtesham Bhai on 2026-08-27, after the MedGemma result: "Also try
Claude, gpt. I see better results w claude."

THE ONLY VARIABLE IS THE MODEL. Same prompt (`prompt_billing`), same reply
parser (`parse_codes`), same scorer, same gold, same three input variants. That
is the whole design, and it is the same rule `haiku_model.py` follows on the
`add-haiku-baseline` branch. Anything else that differs between the two runs
makes the comparison worthless.

WHY STRUCTURED OUTPUTS ARE DELIBERATELY NOT USED. The API can guarantee schema-
valid JSON via ``output_config.format``, and it is tempting because MedGemma's
malformed and truncated replies cost real debugging time. It is not used, for
exactly that reason: MedGemma had to produce parseable JSON unaided, so handing
Claude a hard guarantee would compare a model against a model-plus-a-constraint.
If Claude's raw JSON is cleaner, that is a real result and it should show up as
one — in the malformed count, which is already reported.

DETERMINISM IS NOT AVAILABLE HERE, AND THAT IS A REPORTED DIFFERENCE. The
MedGemma runs were byte-identical across repeats, which is how the silently-
dropped `repetition_penalty` was caught at all. Claude Sonnet 5 removed
`temperature` (sending it is a 400) and runs adaptive thinking, so two runs of
this backend may differ. A number from here is one sample, not a fixed point.

REAL PATIENT DATA LEAVES THE MACHINE. The note text goes to Anthropic's API.
That is a third recipient after Google (Colab) and the practice itself, and it
is Shifat's call — recorded here so it is not invisible later.

The key is read from `.env` via python-dotenv as `ANTHROPIC_API_KEY`, never
hardcoded, printed or committed. `anthropic` and `dotenv` are imported lazily so
the CPU-only unit tests never need either installed.
"""

import os

from .prompt_billing import build_messages, parse_codes

# --- Model -----------------------------------------------------------------
#
# Sonnet because Ehtesham Bhai named Sonnet. The repo default everywhere else
# would be Opus; do not silently upgrade it — the question asked was whether
# Sonnet does better than MedGemma-4B, and answering it with a stronger model
# answers a different question.
ANTHROPIC_MODEL = os.environ.get("BILLING_ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_MODEL_NAME = os.environ.get(
    "BILLING_ANTHROPIC_MODEL_NAME", "claude-sonnet-5"
)

# 2048, against MedGemma's 1024. Not an advantage handed to Claude: the cap
# exists to stop a reply being cut off, and truncation is counted and reported
# for both. MedGemma's 1024 was raised from 512 for the same reason and its
# replies still truncated because it looped; a cap only matters when it binds.
ANTHROPIC_MAX_TOKENS = int(os.environ.get("BILLING_ANTHROPIC_MAX_TOKENS", "2048"))

# NO TEMPERATURE. `haiku_model.py` sets temperature=0 for determinism and that is
# correct for Haiku 4.5. Sending it to Sonnet 5 is a 400 — the parameter was
# removed on the 4.6-and-later family. There is no deterministic-decoding knob
# to substitute; see the module docstring.

# The SDK retries 408/409/429/5xx and connection errors with exponential backoff
# on its own. `haiku_model.py` hand-rolled that loop; this does not, because
# reimplementing it only adds a second, worse backoff on top of the SDK's.
ANTHROPIC_MAX_RETRIES = int(os.environ.get("BILLING_ANTHROPIC_MAX_RETRIES", "5"))

# claude-sonnet-5 pricing, USD per 1M tokens. Used only for the cost line.
PRICE_INPUT_PER_MTOK = 2.00
PRICE_OUTPUT_PER_MTOK = 10.00

API_KEY_ENV = "ANTHROPIC_API_KEY"


def load_client(max_retries=ANTHROPIC_MAX_RETRIES):
    """Load the key from `.env` and build a client. Fails fast, without the key.

    The SDK also accepts an `ant auth login` profile, so an unset env var does
    not necessarily mean no credentials — but this repo's convention is the
    `.env` file, and a clear error beats a 401 from inside the request loop.
    """
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get(API_KEY_ENV):
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Copy .env.example to .env and add your "
            "key. It is never read from anywhere else, never printed, and .env "
            "is gitignored."
        )

    import anthropic

    return anthropic.Anthropic(max_retries=max_retries)


def _reply_text(response):
    """Concatenate the text blocks. Thinking blocks are skipped, not parsed."""
    return "".join(b.text for b in response.content if b.type == "text")


def new_usage():
    """A fresh accumulator for the cost line."""
    return {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def estimate_cost(usage):
    """USD for the tokens in `usage`. Twelve calls on four short notes is cents."""
    return (usage["input_tokens"] / 1e6 * PRICE_INPUT_PER_MTOK
            + usage["output_tokens"] / 1e6 * PRICE_OUTPUT_PER_MTOK)


def predict_note(client, note_text, model_id=None, max_tokens=None,
                 usage_acc=None):
    """One API call. Returns ``(codes, reply, n_tokens, truncated)``.

    Signature matches ``evaluate_billing.predict_note`` so the evaluator can swap
    backends at one call site and change nothing else.

    `truncated` is read from `stop_reason == "max_tokens"`, which is exact —
    better than MedGemma's side, where truncation is inferred from a token count
    landing within `CAP_MARGIN` of the cap. Both feed the same reported field,
    so the difference is worth remembering when the two runs are compared.
    """
    import anthropic

    model_id = model_id or ANTHROPIC_MODEL
    max_tokens = max_tokens or ANTHROPIC_MAX_TOKENS

    messages = build_messages(note_text)
    system = messages[0]["content"][0]["text"]
    user = messages[1]["content"][0]["text"]

    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.NotFoundError:
        raise RuntimeError(
            f"model {model_id!r} was not found. Check the id — it takes no date "
            "suffix."
        ) from None
    except anthropic.AuthenticationError:
        raise RuntimeError(
            f"{API_KEY_ENV} was rejected. Check the key in .env."
        ) from None
    except anthropic.BadRequestError as e:
        raise RuntimeError(
            f"the API rejected the request: {e.message}\n"
            "If this mentions `temperature`, something has reintroduced it — "
            "Sonnet 5 removed that parameter. See this module's docstring."
        ) from None

    if usage_acc is not None:
        usage_acc["input_tokens"] += response.usage.input_tokens
        usage_acc["output_tokens"] += response.usage.output_tokens
        usage_acc["calls"] += 1

    # A safety refusal is not an empty answer and must not score as one.
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        raise RuntimeError(
            "the model declined this request"
            + (f" ({detail.category})" if detail else "")
            + ". Clinical note content should not trigger this; if it recurs, "
            "the note text is worth looking at before the harness is."
        )

    reply = _reply_text(response)
    n_tokens = response.usage.output_tokens
    truncated = response.stop_reason == "max_tokens"
    return parse_codes(reply), reply, n_tokens, truncated
