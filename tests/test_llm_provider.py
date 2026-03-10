"""Unit tests for llm_provider.py — factory + call_llm edge cases."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_provider import _is_retryable, call_llm, create_llm_client

# ---------------------------------------------------------------------------
# create_llm_client
# ---------------------------------------------------------------------------

def test_create_llm_client_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_llm_client("nonexistent")


def test_create_llm_client_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        # Ensure GROQ_API_KEY is not set
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(ValueError, match="API key required"):
            create_llm_client("groq")


def test_create_llm_client_explicit_key():
    client = create_llm_client("groq", api_key="test-key-123")
    assert client is not None


def test_create_llm_client_env_var():
    with patch.dict(os.environ, {"GROQ_API_KEY": "env-key-456"}):
        client = create_llm_client("groq")
        assert client is not None


# ---------------------------------------------------------------------------
# call_llm — edge cases
# ---------------------------------------------------------------------------

def _mock_client(content="test answer", choices=None):
    """Create a mock client with configurable response."""
    client = MagicMock()
    if choices is not None:
        client.chat.completions.create.return_value.choices = choices
    else:
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        client.chat.completions.create.return_value.choices = [choice]
    return client


def test_call_llm_success():
    client = _mock_client("  hello world  ")
    result = call_llm(client, "prompt", "model")
    assert result == "hello world"


def test_call_llm_none_content():
    client = _mock_client(content=None)
    with pytest.raises(ValueError, match="empty content"):
        call_llm(client, "prompt", "model")


def test_call_llm_empty_choices():
    client = _mock_client(choices=[])
    with pytest.raises(ValueError, match="no choices"):
        call_llm(client, "prompt", "model")


def test_call_llm_max_attempts_zero():
    client = _mock_client()
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        call_llm(client, "prompt", "model", max_attempts=0)


def test_call_llm_retries_on_429(monkeypatch):
    """Should retry on 429 and succeed on second attempt."""
    monkeypatch.setattr("llm_provider.time.sleep", lambda _: None)
    client = MagicMock()
    msg = MagicMock()
    msg.content = "success"
    ok_choice = MagicMock()
    ok_choice.message = msg
    client.chat.completions.create.side_effect = [
        Exception("Rate limit: 429 Too Many Requests"),
        MagicMock(choices=[ok_choice]),
    ]
    result = call_llm(client, "prompt", "model", max_attempts=3, initial_wait=0.01)
    assert result == "success"
    assert client.chat.completions.create.call_count == 2


def test_call_llm_retries_on_500(monkeypatch):
    """Should retry on 500 server error and succeed on second attempt."""
    monkeypatch.setattr("llm_provider.time.sleep", lambda _: None)
    client = MagicMock()
    msg = MagicMock()
    msg.content = "success"
    ok_choice = MagicMock()
    ok_choice.message = msg
    client.chat.completions.create.side_effect = [
        Exception("Internal Server Error 500"),
        MagicMock(choices=[ok_choice]),
    ]
    result = call_llm(client, "prompt", "model", max_attempts=3, initial_wait=0.01)
    assert result == "success"


def test_call_llm_no_retry_on_400():
    """Should NOT retry on 400 (client error, not transient)."""
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("Bad Request 400")
    with pytest.raises(Exception, match="400"):
        call_llm(client, "prompt", "model", max_attempts=3)
    assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------

def test_is_retryable_429():
    assert _is_retryable(Exception("429 Too Many Requests"))


def test_is_retryable_500():
    assert _is_retryable(Exception("Internal Server Error 500"))


def test_is_retryable_timeout():
    assert _is_retryable(Exception("Connection timeout"))


def test_is_retryable_rate_limit_error():
    import openai
    e = openai.RateLimitError(
        message="rate limited", response=MagicMock(), body=None,
    )
    assert _is_retryable(e)


def test_not_retryable_400():
    assert not _is_retryable(Exception("Bad Request 400"))


def test_not_retryable_auth():
    assert not _is_retryable(Exception("Authentication failed 401"))
