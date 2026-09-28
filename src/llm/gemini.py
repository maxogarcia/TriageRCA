"""Gemini LLMClient via the Google Gen AI SDK (google-genai). Free tier, model name from config."""

from __future__ import annotations

import os
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from src.llm.cache import CachedThrottledClient

_ROLE_MAP = {"system": "user", "user": "user", "assistant": "model", "tool": "user"}


def _to_genai_contents(messages: list[Message]) -> list[types.Content]:
    contents = []
    for m in messages:
        contents.append(types.Content(role=_ROLE_MAP[m.role], parts=[types.Part.from_text(text=m.content)]))
    return contents


def _to_genai_tools(tools: list[ToolSpec] | None) -> list[types.Tool] | None:
    if not tools:
        return None
    declarations = [
        types.FunctionDeclaration(name=t.name, description=t.description, parameters=t.parameters) for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


class GeminiClient:
    """Real Gemini client. Wrap in CachedThrottledClient (see `build()`) rather than using directly."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def _raw_generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        **params: Any,
    ) -> LLMResponse:
        config_kwargs: dict[str, Any] = dict(params)
        genai_tools = _to_genai_tools(tools)
        if genai_tools:
            config_kwargs["tools"] = genai_tools
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        start = time.monotonic()
        result = self._client.models.generate_content(
            model=self._model,
            contents=_to_genai_contents(messages),
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )
        latency_s = time.monotonic() - start

        tool_calls = []
        text_parts = []
        candidate = result.candidates[0] if result.candidates else None
        if candidate is not None:
            for part in candidate.content.parts or []:
                if part.function_call is not None:
                    tool_calls.append(
                        ToolCall(
                            id=part.function_call.id or f"call_{len(tool_calls)}",
                            name=part.function_call.name,
                            args=dict(part.function_call.args or {}),
                        )
                    )
                elif part.text:
                    text_parts.append(part.text)

        usage = result.usage_metadata
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_s=latency_s,
            model=self._model,
            cached=False,
        )


def build(
    model: str,
    api_key: str | None = None,
    cache_dir: str = ".cache/llm",
    min_interval_s: float = 4.0,
) -> CachedThrottledClient:
    """Construct the Gemini client teammates should actually use: cached + throttled."""
    raw = GeminiClient(model=model, api_key=api_key)
    return CachedThrottledClient(
        call_fn=raw._raw_generate, model=model, cache_dir=cache_dir, min_interval_s=min_interval_s
    )
