"""The Anthropic backend, with a fake client. No network, no key, no SDK needed.

WHAT THESE ARE FOR. The comparison against MedGemma is only worth anything if
the two backends differ in exactly one thing — the model. Everything else must
be shared, and the way that breaks is silently: a second prompt, a friendlier
parser, a truncation flag that means something different on each side. So the
tests here mostly assert *sameness* rather than behaviour.

The one that would flatter Claude if it broke is
``test_backend_uses_the_same_parser_as_medgemma``. If this backend ever grew its
own parsing — or asked the API for guaranteed-valid JSON via structured outputs
— Claude would be scored under rules MedGemma never got.

`anthropic` is imported lazily inside the functions under test, so a stub in
`sys.modules` is enough and the suite stays CPU-only and dependency-free.
"""

import sys
import types

import pytest

from src import billing_anthropic as api
from src.prompt_billing import build_messages, instruction


class _Usage:
    def __init__(self, i=100, o=50):
        self.input_tokens = i
        self.output_tokens = o


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Response:
    def __init__(self, text, stop_reason="end_turn", usage=None):
        self.content = [_Text(text)]
        self.stop_reason = stop_reason
        self.usage = usage or _Usage()
        self.stop_details = None


class _Messages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _Client:
    def __init__(self, response):
        self.messages = _Messages(response)


@pytest.fixture(autouse=True)
def _stub_anthropic(monkeypatch):
    """A stub `anthropic` module carrying the exception classes used in except."""
    mod = types.ModuleType("anthropic")

    class APIError(Exception):
        pass

    class BadRequestError(APIError):
        def __init__(self, message="bad request"):
            super().__init__(message)
            self.message = message

    mod.APIError = APIError
    mod.BadRequestError = BadRequestError
    mod.NotFoundError = type("NotFoundError", (APIError,), {})
    mod.AuthenticationError = type("AuthenticationError", (APIError,), {})
    mod.RateLimitError = type("RateLimitError", (APIError,), {})
    mod.APIConnectionError = type("APIConnectionError", (APIError,), {})
    mod.APIStatusError = type("APIStatusError", (APIError,), {})
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


# --- the same-as-MedGemma guarantees ----------------------------------------


def test_backend_sends_the_shared_prompt_verbatim():
    """Not "a similar prompt" — the same bytes `prompt_billing` gives MedGemma."""
    client = _Client(_Response('{"codes": []}'))
    api.predict_note(client, "NOTE TEXT HERE")

    sent = client.messages.last_kwargs
    expected = build_messages("NOTE TEXT HERE")
    assert sent["system"] == expected[0]["content"][0]["text"]
    assert sent["messages"][0]["content"] == expected[1]["content"][0]["text"]
    assert instruction() in sent["messages"][0]["content"]
    assert "NOTE TEXT HERE" in sent["messages"][0]["content"]


def test_backend_uses_the_same_parser_as_medgemma():
    """The one that would flatter Claude if it broke. See the module docstring."""
    reply = ('Here you go:\n```json\n{"codes": [{"code": "J11.1", '
             '"description": "Influenza"}]}\n```')
    client = _Client(_Response(reply))
    codes, _, _, _ = api.predict_note(client, "note")
    assert [c["code"] for c in codes] == ["J11.1"]


def test_backend_does_not_request_structured_outputs():
    """Guaranteed-valid JSON is an advantage MedGemma never had."""
    client = _Client(_Response('{"codes": []}'))
    api.predict_note(client, "note")
    assert "output_config" not in client.messages.last_kwargs
    assert "tools" not in client.messages.last_kwargs


def test_malformed_codes_are_still_counted_against_claude():
    client = _Client(_Response('{"codes": [{"code": "99213"}]}'))
    codes, _, _, _ = api.predict_note(client, "note")
    assert codes[0]["well_formed"] is False


# --- Sonnet 5 request shape -------------------------------------------------


def test_temperature_is_never_sent():
    """Sonnet 5 removed it; sending it is a 400. haiku_model.py does send it."""
    client = _Client(_Response('{"codes": []}'))
    api.predict_note(client, "note")
    assert "temperature" not in client.messages.last_kwargs


def test_model_id_has_no_date_suffix():
    assert api.ANTHROPIC_MODEL == "claude-sonnet-5"


def test_a_bad_request_names_temperature_as_the_likely_cause(_stub_anthropic):
    client = _Client(_stub_anthropic.BadRequestError("temperature: unsupported"))
    with pytest.raises(RuntimeError, match="temperature"):
        api.predict_note(client, "note")


# --- truncation and refusal -------------------------------------------------


def test_truncation_comes_from_stop_reason_not_a_token_estimate():
    client = _Client(_Response('{"codes": [', stop_reason="max_tokens"))
    _, _, _, truncated = api.predict_note(client, "note")
    assert truncated is True


def test_a_complete_reply_is_not_flagged_truncated():
    client = _Client(_Response('{"codes": []}', stop_reason="end_turn"))
    _, _, _, truncated = api.predict_note(client, "note")
    assert truncated is False


def test_a_refusal_raises_rather_than_scoring_as_zero_codes():
    """A declined request is not an empty answer, and must not score as one."""
    client = _Client(_Response("", stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="declined"):
        api.predict_note(client, "note")


# --- usage and cost ---------------------------------------------------------


def test_usage_accumulates_across_calls():
    client = _Client(_Response('{"codes": []}', usage=_Usage(1000, 200)))
    usage = api.new_usage()
    api.predict_note(client, "a", usage_acc=usage)
    api.predict_note(client, "b", usage_acc=usage)
    assert usage == {"input_tokens": 2000, "output_tokens": 400, "calls": 2}


def test_cost_estimate_uses_sonnet_pricing():
    cost = api.estimate_cost({"input_tokens": 1_000_000,
                              "output_tokens": 1_000_000, "calls": 1})
    assert cost == pytest.approx(api.PRICE_INPUT_PER_MTOK
                                 + api.PRICE_OUTPUT_PER_MTOK)


# --- key handling -----------------------------------------------------------


def test_missing_key_fails_fast_without_leaking_anything(monkeypatch):
    monkeypatch.delenv(api.API_KEY_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "dotenv",
                        types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
    with pytest.raises(RuntimeError) as exc:
        api.load_client()
    assert api.API_KEY_ENV in str(exc.value)
    assert "sk-ant" not in str(exc.value)
