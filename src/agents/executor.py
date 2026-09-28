"""Executor: runs a tool-calling loop (bounded by max_tool_calls) to gather evidence per the
Planner's Plan, then synthesizes a candidate Hypothesis citing evidence IDs.
"""

from __future__ import annotations

from pydantic import ValidationError

from src.evidence import EvidenceLedger
from src.llm.base import LLMClient, Message, ToolCall
from src.schemas import Hypothesis, Plan
from src.tools.registry import ToolRegistry

_SYSTEM_PROMPT = """You are the Executor in a root-cause-analysis (RCA) agent for microservice incidents.

Follow the investigation plan by calling the available tools to gather evidence. You may deviate
from the plan if a tool result suggests a more promising lead. Once you have enough evidence,
stop calling tools and, instead of a tool call, respond with ONLY a JSON object matching this shape
(no prose, no markdown fences): {"ranked_services": [...], "indicator": ..., "fault_category": ...,
"explanation": ..., "citations": [...], "mitigation": ..., "confidence": ...}

Every claim in your final explanation must cite evidence IDs (e.g. "E001", "E002") from the tool
results you were shown -- never cite evidence you were not actually given, and never invent metric
values or log lines not present in a tool result.
"""

# Most backends (e.g. Gemini) can't combine function-calling `tools` with a structured
# `response_schema` in the same call, so the loop asks the model to emit raw JSON text once it's
# done calling tools (see _SYSTEM_PROMPT) and only falls back to a dedicated structured-output
# call -- one extra LLM call -- if that text doesn't parse, or if the tool-call budget ran out
# before the model stopped on its own.


class ToolCallBudgetExceeded(Exception):
    pass


class Executor:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, max_tool_calls: int = 10) -> None:
        self._llm = llm
        self._tools = tools
        self._max_tool_calls = max_tool_calls

    def run(self, incident_query: str, plan: Plan, ledger: EvidenceLedger) -> Hypothesis:
        plan_summary = "\n".join(f"- {s.tool}({s.args}): {s.rationale}" for s in plan.steps)
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Incident: {incident_query}\n\nInvestigation plan:\n{plan_summary}",
            ),
        ]

        tool_calls_made = 0
        while tool_calls_made < self._max_tool_calls:
            response = self._llm.generate(messages, tools=self._tools.specs())

            if not response.tool_calls:
                try:
                    hypothesis = Hypothesis.model_validate_json(response.text)
                except ValidationError:
                    messages.append(Message(role="assistant", content=response.text))
                    hypothesis = self._force_synthesize(messages, forced=False)
                self._validate_citations(hypothesis, ledger)
                return hypothesis

            messages.append(Message(role="assistant", content=response.text, tool_calls=response.tool_calls))
            for call in response.tool_calls:
                if tool_calls_made >= self._max_tool_calls:
                    break
                result = self._invoke_tool(call, ledger)
                messages.append(Message(role="tool", content=result, tool_call_id=call.id))
                tool_calls_made += 1

        hypothesis = self._force_synthesize(messages, forced=True)
        self._validate_citations(hypothesis, ledger)
        return hypothesis

    def _invoke_tool(self, call: ToolCall, ledger: EvidenceLedger) -> str:
        try:
            return self._tools.invoke(call.name, call.args, ledger)
        except KeyError as exc:
            return f"Error: {exc}"

    def _force_synthesize(self, messages: list[Message], forced: bool) -> Hypothesis:
        prompt = "Produce your final Hypothesis now, citing evidence IDs for every claim."
        if forced:
            prompt = (
                f"Tool call budget ({self._max_tool_calls}) reached. "
                + prompt
                + " Base it only on the evidence gathered so far."
            )
        synth_messages = [*messages, Message(role="user", content=prompt)]
        response = self._llm.generate(synth_messages, response_schema=Hypothesis)
        return Hypothesis.model_validate_json(response.text)

    def _validate_citations(self, hypothesis: Hypothesis, ledger: EvidenceLedger) -> None:
        known = ledger.known_ids()
        unknown = [c for c in hypothesis.citations if c not in known]
        if unknown:
            raise ValueError(f"Hypothesis cites evidence IDs not in the ledger: {unknown}")
