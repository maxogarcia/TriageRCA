"""Scripted LLM client for offline tests (hard rule #6) -- never hits a real API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.llm.base import LLMResponse, Message, ToolSpec


class FakeLLMClient:
    """Returns pre-scripted LLMResponses in order, one per call to generate().

    Usage:
        client = FakeLLMClient([
            LLMResponse(text="", tool_calls=[ToolCall(id="1", name="metric_query", args={...})], ...),
            LLMResponse(text="final report json", tool_calls=[], ...),
        ])
    """

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        self._responses = list(scripted_responses)
        self._index = 0
        self.calls: list[dict[str, Any]] = []  # recorded for assertions in tests

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        **params: Any,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "response_schema": response_schema,
                "params": params,
            }
        )
        if self._index >= len(self._responses):
            raise AssertionError(
                f"FakeLLMClient exhausted: {len(self._responses)} scripted responses, "
                f"but generate() was called {self._index + 1} times."
            )
        response = self._responses[self._index]
        self._index += 1
        return response
