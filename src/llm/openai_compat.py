"""LLMClient for any OpenAI-compatible endpoint (vLLM, Ollama) -- the open-weight model."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from src.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from src.llm.cache import CachedThrottledClient


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": str(tc.args)},
                }
                for tc in m.tool_calls
            ]
        out.append(msg)
    return out


def _to_openai_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class OpenAICompatClient:
    """Real client for an OpenAI-compatible endpoint. Wrap in CachedThrottledClient (see `build()`)."""

    def __init__(self, model: str, base_url: str, api_key: str = "not-needed") -> None:
        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def _raw_generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        **params: Any,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(params)
        openai_tools = _to_openai_tools(tools)
        if openai_tools:
            kwargs["tools"] = openai_tools
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        result = self._client.chat.completions.create(
            model=self._model,
            messages=_to_openai_messages(messages),
            **kwargs,
        )
        latency_s = time.monotonic() - start

        choice = result.choices[0]
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, args=_safe_json_args(tc.function.arguments))
            for tc in (choice.message.tool_calls or [])
        ]
        usage = result.usage
        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency_s,
            model=self._model,
            cached=False,
        )


def _safe_json_args(raw: str) -> dict[str, Any]:
    import json

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def build(
    model: str,
    base_url: str,
    api_key: str = "not-needed",
    cache_dir: str = ".cache/llm",
    min_interval_s: float = 0.0,
) -> CachedThrottledClient:
    """Construct the open-weight client teammates should actually use: cached + throttled."""
    raw = OpenAICompatClient(model=model, base_url=base_url, api_key=api_key)
    return CachedThrottledClient(
        call_fn=raw._raw_generate, model=model, cache_dir=cache_dir, min_interval_s=min_interval_s
    )
