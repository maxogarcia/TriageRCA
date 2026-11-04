"""Disk cache + throttle/backoff wrapper shared by all LLMClient implementations.

Every real client (gemini.py, openai_compat.py) should route generate() calls through
CachedThrottledClient rather than hitting the SDK directly, so free-tier rate limits and
repeated dev runs don't burn quota.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import diskcache
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.llm.base import LLMResponse, Message, ToolCall, ToolSpec

DEFAULT_CACHE_DIR = Path(".cache/llm")


_RETRYABLE_STATUS_CODES = {429, 500, 503}


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    # A per-day quota (Gemini free tier: "...PerDayPerProjectPerModel...") won't reset within any
    # backoff window, so fail fast instead of burning ~4 minutes of retries per call.
    if "perday" in text:
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    return any(needle in text for needle in ("429", "rate limit", "503", "unavailable", "high demand"))


def _cache_key(model: str, messages: list[Message], tools: list[ToolSpec] | None, params: dict[str, Any]) -> str:
    payload = {
        "model": model,
        "messages": [asdict(m) for m in messages],
        "tools": [asdict(t) for t in tools] if tools else None,
        "params": params,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CachedThrottledClient:
    """Wraps a raw `call_fn(messages, tools, response_schema, **params) -> LLMResponse`
    with a disk cache (keyed on the full request) and exponential backoff on 429s."""

    def __init__(
        self,
        call_fn: Callable[..., LLMResponse],
        model: str,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        min_interval_s: float = 0.0,
        max_attempts: int = 8,
    ) -> None:
        self._call_fn = call_fn
        self._model = model
        self._cache = diskcache.Cache(str(cache_dir))
        self._min_interval_s = min_interval_s
        self._max_attempts = max_attempts
        self._last_call_ts = 0.0

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: Any = None,
        **params: Any,
    ) -> LLMResponse:
        key = _cache_key(self._model, messages, tools, params)
        cached = self._cache.get(key)
        if cached is not None:
            tool_calls = [ToolCall(**tc) for tc in cached["tool_calls"]]
            return LLMResponse(**{**cached, "tool_calls": tool_calls, "cached": True})

        self._throttle()
        response = self._call_with_backoff(messages, tools, response_schema, **params)
        self._last_call_ts = time.monotonic()

        self._cache.set(key, {**asdict(response), "cached": False})
        return response

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

    def _call_with_backoff(self, *args: Any, **kwargs: Any) -> LLMResponse:
        # Waits 4, 8, 16, 32, 60, 60, 60s (~4 min total). Gemini's 429s ask for up to ~60s and its
        # 503 "high demand" spikes last minutes; the old 1-8s schedule gave up after ~15s.
        @retry(
            retry=retry_if_exception(_is_rate_limit_error),
            wait=wait_exponential(multiplier=4, min=4, max=60),
            stop=stop_after_attempt(self._max_attempts),
            reraise=True,
        )
        def _call() -> LLMResponse:
            return self._call_fn(*args, **kwargs)

        return _call()
