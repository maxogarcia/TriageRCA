"""Shared LLM interface. All model calls in this codebase go through an LLMClient (hard rule #4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None  # set on role="tool" messages, echoes the ToolCall.id being answered
    tool_calls: list[ToolCall] = field(default_factory=list)  # set on role="assistant" messages that call tools


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the tool's arguments


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    cached: bool


class LLMClient(Protocol):
    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        **params: Any,
    ) -> LLMResponse: ...
