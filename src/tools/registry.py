"""Tool registry: maps tool name -> (ToolSpec, callable) so the Executor can look up and invoke
tools by name from the LLM's function-call output, and log every result to the evidence ledger.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.evidence import EvidenceLedger
from src.llm.base import ToolSpec

# A tool function takes its arguments plus the case's data handle and returns a compact string
# summary (hard rule #2: never dump raw telemetry into a prompt).
ToolFn = Callable[..., str]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    fn: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        self._tools[spec.name] = RegisteredTool(spec=spec, fn=fn)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def invoke(self, name: str, args: dict[str, Any], ledger: EvidenceLedger) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Registered tools: {list(self._tools)}")
        result = self._tools[name].fn(**args)
        ledger.add(source=name, args=args, content=result)
        return result
