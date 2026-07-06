"""Haiku backend tests — the Anthropic API is MOCKED; no real network calls.

Confirms that a Claude Haiku reply flows through the SAME parse -> align -> BIO
pipeline as MedGemma, that token usage is accumulated for the cost estimate, and
that the request is shaped correctly (temperature 0, shared system prompt).
"""

from unittest.mock import MagicMock

from src.evaluate import predict_example
from src.haiku_model import run_haiku
from src.prompt import _SYSTEM, parse_entities


def _fake_response(text, in_tok=100, out_tok=20):
    """A stand-in for anthropic's Message: .content[*].text + .usage.*_tokens."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens = in_tok
    response.usage.output_tokens = out_tok
    return response


def _client_returning(text):
    client = MagicMock()
    client.messages.create.return_value = _fake_response(text)
    return client


def test_run_haiku_returns_raw_reply_and_parses():
    reply = '{"entities": [{"text": "lung cancer", "type": "Disease"}]}'
    client = _client_returning(reply)
    out = run_haiku(client, "History of lung cancer.", delay=0)
    assert out == reply
    assert parse_entities(out) == [("lung cancer", "Disease")]


def test_run_haiku_handles_fenced_and_prose_replies():
    reply = 'Sure!\n```json\n{"entities": [{"text": "aspirin", "type": "Chemical"}]}\n```'
    client = _client_returning(reply)
    out = run_haiku(client, "Given aspirin.", delay=0)
    assert parse_entities(out) == [("aspirin", "Chemical")]


def test_run_haiku_accumulates_usage():
    client = _client_returning('{"entities": []}')
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    run_haiku(client, "No entities here.", usage_acc=usage, delay=0)
    run_haiku(client, "Still none.", usage_acc=usage, delay=0)
    assert usage == {"input_tokens": 200, "output_tokens": 40, "calls": 2}


def test_run_haiku_request_shape():
    client = _client_returning('{"entities": []}')
    run_haiku(client, "Patient has flu.", model_id="claude-haiku-4-5", delay=0)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["temperature"] == 0          # deterministic
    assert kwargs["system"] == _SYSTEM         # SAME prompt as MedGemma
    assert kwargs["messages"][0]["role"] == "user"
    assert "Patient has flu." in kwargs["messages"][0]["content"]


def test_run_haiku_retries_on_rate_limit(monkeypatch):
    import anthropic

    import src.haiku_model as hm
    monkeypatch.setattr(hm.time, "sleep", lambda *_: None)  # no real backoff wait

    calls = {"n": 0}

    def flaky_create(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise anthropic.RateLimitError(
                message="slow down",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        return _fake_response('{"entities": [{"text": "gout", "type": "Disease"}]}')

    client = MagicMock()
    client.messages.create.side_effect = flaky_create
    out = run_haiku(client, "Has gout.", delay=0, max_retries=3, backoff_base=0)
    assert calls["n"] == 2  # retried once, then succeeded
    assert parse_entities(out) == [("gout", "Disease")]


def test_predict_example_end_to_end_with_mocked_haiku():
    tokens = ["History", "of", "lung", "cancer", "treated", "with", "aspirin"]
    reply = (
        '{"entities": ['
        '{"text": "lung cancer", "type": "Disease"}, '
        '{"text": "aspirin", "type": "Chemical"}]}'
    )
    client = _client_returning(reply)
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def run_fn(c, sentence):
        return run_haiku(c, sentence, usage_acc=usage, delay=0)

    bio = predict_example(client, tokens, run_fn=run_fn)
    assert bio == ["O", "O", "B-Disease", "I-Disease", "O", "O", "B-Chemical"]
    assert usage["calls"] == 1
