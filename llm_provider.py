"""Unified LLM provider: factory + call_llm with retry.

Supports multiple OpenAI-compatible providers via a single interface.
Provider is selected at runtime via --provider flag; model via --model.

Providers:
  groq   — https://api.groq.com/openai/v1  (env: GROQ_API_KEY)
  github — https://models.inference.ai.azure.com (env: GITHUB_TOKEN)
  ollama — http://localhost:11434/v1         (no key required)
"""

import json
import logging
import os
import time

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "github": {
        "base_url": "https://models.inference.ai.azure.com",
        "env_key": "GITHUB_TOKEN",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
    },
}

_RETRYABLE = ("429", "500", "502", "503", "504", "timeout", "connection")


def _is_retryable(e: Exception) -> bool:
    """Check if an exception is retryable (rate limit or transient server error)."""
    if isinstance(e, openai.RateLimitError):
        return True
    return any(t in str(e).lower() for t in _RETRYABLE)


def create_llm_client(provider: str = "groq", api_key: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client for the given provider.

    Args:
        provider: One of 'groq' or 'github'.
        api_key: Explicit API key. Falls back to provider-specific env var.

    Raises:
        ValueError: Unknown provider or missing API key.
    """
    cfg = PROVIDERS.get(provider)
    if not cfg:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    if cfg["env_key"] is None:
        key = api_key or "ollama"
    else:
        key = api_key or os.environ.get(cfg["env_key"])
        if not key:
            raise ValueError(f"API key required: pass --api-key or set {cfg['env_key']}")
    return OpenAI(base_url=cfg["base_url"], api_key=key, timeout=120)


def call_llm(
    client, prompt: str, model: str,
    max_attempts: int = 5, initial_wait: float = 10.0,
    max_tokens: int = 1000, temperature: float = 0,
) -> str:
    """Make an LLM call with retry on transient errors (429, 500, 503, timeout).

    Works with any OpenAI-compatible client (Groq, GitHub Models, etc.).
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not resp.choices:
                raise ValueError("LLM returned no choices")
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content (None)")
            # Run 2 instrumentation: gated Groq-fingerprint capture for post-hoc drift diagnosis.
            # No-op unless LLM_FP_LOG env var is set (phase-02 enables it per cell).
            _fp_log = os.environ.get("LLM_FP_LOG")
            if _fp_log:
                try:
                    usage = getattr(resp, "usage", None)
                    usage_dict = usage.model_dump() if usage is not None and hasattr(
                        usage, "model_dump"
                    ) else None
                    with open(_fp_log, "a") as fh:
                        fh.write(json.dumps({
                            "t": time.time(),
                            "model": getattr(resp, "model", None),
                            "fingerprint": getattr(resp, "system_fingerprint", None),
                            "usage": usage_dict,
                        }) + "\n")
                except Exception as log_err:
                    logger.debug("LLM_FP_LOG write failed (non-fatal): %s", log_err)
            return content.strip()
        except Exception as e:
            if _is_retryable(e) and attempt < max_attempts - 1:
                wait = initial_wait * (2 ** attempt)
                logger.info(
                    "Retryable error, waiting %.0fs (attempt %d/%d): %s",
                    wait, attempt + 1, max_attempts, e,
                )
                time.sleep(wait)
            else:
                raise
