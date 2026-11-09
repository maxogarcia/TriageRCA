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
# summary (hard rule #2: never dump raw telemetry into a prompt), or a list of strings when the
# result is several separately citable pieces (e.g. one per RAG chunk).
ToolFn = Callable[..., str | list[str]]


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
        """Run a tool, log its result to the ledger, and return it tagged with its evidence ID(s)
        so the model can cite them -- otherwise it has to guess IDs, and one wrong guess fails
        citation validation. A list result gets one ledger entry (and one ID) per part, so each
        RAG chunk is cited and judged on its own (hard rule #3).
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Registered tools: {list(self._tools)}")
        result = self._tools[name].fn(**args)
        if isinstance(result, str):
            result = [result]
        tagged = [f"[{ledger.add(source=name, args=args, content=part).id}] {part}" for part in result]
        return "\n".join(tagged)
