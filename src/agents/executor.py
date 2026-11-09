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

`ranked_services` lists up to 5 candidate root-cause services, most likely first. `indicator` is the
single metric that best shows the fault, written as "<service>_<metric>" exactly as metric_query
names it (e.g. "cartservice_cpu"). A service whose metrics shift sharply at the onset is a stronger
root-cause candidate than one that is only slow because it calls that service. `confidence` is a
number between 0 and 1 (e.g. 0.8), not a percentage. Once again, confidence must be a number between 0.0 and 1.0 inclusive.
For example, use 0.85 for 85% confidence. Never output 85.

Every claim in your final explanation must cite evidence IDs (e.g. "E001", "E002") from the tool
results you were shown -- never cite evidence you were not actually given, and never invent metric
values or log lines not present in a tool result.

The citations array must contain bare evidence IDs only:
["E001", "E005"].

Do not include "Evidence", descriptions, punctuation, or explanations
inside citation strings. Put supporting explanations in explanation.
Only cite IDs that actually exist in the supplied tool results.

Use only the supplied canonical service names, not familiar names from another system.
Use Unix seconds for tool windows. For onset comparisons, request a window extending before
AND after onset, preferably onset minus 300 seconds to onset plus 300 seconds. Respect coverage
warnings: min/max and mean/last are NOT before/after changes. A ratio below 1 is a decrease.
No matches, missing sources, and invalid queries do not establish normal behavior.
Anomaly scores identify candidates, not causal proof. Trace duration units are source-native
unless explicitly specified; never assume milliseconds. Attribute each citation to its actual service.
If the evidence cannot support a diagnosis, abstain: ranked_services=[], indicator="unknown",
fault_category="unknown", confidence=0.0, explanation describing missing evidence, and no
speculative mitigation. Do not fill an evidence gap with a plausible story.

fault_category must be exactly one of:
cpu, mem, disk, delay, loss, socket, unknown.

Use the category supported by the observed evidence. If the fault type cannot
be determined, use unknown. Generic descriptions such as "performance issue",
"overload", or "database performance issue" are not valid category values.s
"""

# Most backends (e.g. Gemini) can't combine function-calling `tools` with a structured
# `response_schema` in the same call, so the loop asks the model to emit raw JSON text once it's
# done calling tools (see _SYSTEM_PROMPT) and only falls back to a dedicated structured-output
# call if the report is invalid or the tool budget is exhausted. Structured synthesis
# gets at most two attempts; validation errors are returned once for repair.


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
        if ledger.items:
            # These are already capped tool summaries, not raw telemetry. Replanning must
            # see the same evidence whose IDs remain valid in the shared ledger.
            messages.append(Message(
                role="user",
                content="Previously collected evidence (read before investigating):\n" + "\n\n".join(
                    f"[{item.id}] {item.source}:\n{item.content}" for item in ledger.items
                ),
            ))

        tool_calls_made = 0
        while tool_calls_made < self._max_tool_calls:
            response = self._llm.generate(messages, tools=self._tools.specs())

            if not response.tool_calls:
                try:
                    hypothesis = Hypothesis.model_validate_json(response.text)
                    self._validate_citations(hypothesis, ledger)
                except (ValidationError, ValueError) as exc:
                    messages.append(Message(role="assistant", content=response.text))
                    hypothesis = self._force_synthesize(
                        messages, forced=False, ledger=ledger, validation_error=str(exc)
                    )
                return hypothesis

            messages.append(Message(role="assistant", content=response.text, tool_calls=response.tool_calls))
            for call in response.tool_calls:
                if tool_calls_made >= self._max_tool_calls:
                    messages.append(Message(
                        role="tool", tool_call_id=call.id,
                        content="Not executed: tool-call budget exhausted. This is not telemetry evidence.",
                    ))
                    continue
                result = self._invoke_tool(call, ledger)
                messages.append(Message(role="tool", content=result, tool_call_id=call.id))
                tool_calls_made += 1

        # The last response may contain tool calls and empty text, not a report.
        # The final tool results must reach a separate synthesis request.
        return self._force_synthesize(messages, forced=True, ledger=ledger)

    def _invoke_tool(self, call: ToolCall, ledger: EvidenceLedger) -> str:
        try:
            result = self._tools.invoke(call.name, call.args, ledger)
        except (KeyError, TypeError, ValueError) as exc:
            # Unknown tool or bad/missing args from the model: report back so it can retry,
            # rather than failing the whole case.
            return f"Error: {exc}"
        # Tag the result with its ledger ID so the model can cite it -- otherwise it has to guess
        # IDs from call order, and one wrong guess fails _validate_citations.
        return f"[{ledger.items[-1].id}] {result}"

    def _force_synthesize(
        self,
        messages: list[Message],
        forced: bool,
        ledger: EvidenceLedger,
        validation_error: str | None = None,
    ) -> Hypothesis:
        prompt = (
            "Produce your final Hypothesis now, citing evidence IDs "
            "for every claim. confidence must be between 0.0 and 1.0 "
            "inclusive; use 0.85, not 85."
        )
        known = sorted(ledger.known_ids())
        prompt += (
            f" Allowed citation IDs: {known}. "
            'citations must contain only exact bare IDs, such as ["E001"], '
            "with no descriptions or prefixes. Use only IDs from the allowed list; "
            "if it is empty, citations must be []. Do not invent evidence."
        )
        if validation_error:
            prompt += f" Your previous report failed validation:\n{validation_error}"

        if forced:
            prompt = (
                f"Tool call budget ({self._max_tool_calls}) reached. "
                + prompt
                + " Base it only on the evidence gathered so far."
            )

        synth_messages = [
            *messages,
            Message(role="user", content=prompt),
        ]

        # First synthesis attempt, then at most one repair attempt.
        for attempt in range(2):
            response = self._llm.generate(
                synth_messages,
                response_schema=Hypothesis,
            )

            try:
                if response.tool_calls:
                    raise ValueError("Final synthesis must return JSON, not tool calls")
                hypothesis = Hypothesis.model_validate_json(response.text)
                self._validate_citations(hypothesis, ledger)
                return hypothesis
            except (ValidationError, ValueError) as exc:
                if attempt == 1:
                    raise

                synth_messages.extend([
                    Message(
                        role="assistant",
                        content=response.text,
                    ),
                    Message(
                        role="user",
                        content=(
                            "Your report failed schema or citation validation:\n"
                            f"{exc}\n\n"
                            "Return a corrected complete JSON object. "
                            "confidence must be between 0.0 and 1.0, "
                            "not a percentage. "
                            f"Allowed citation IDs: {known}. "
                            "Use exact bare IDs, not descriptions or invented IDs. "
                            "No further tools are available in this synthesis step. "
                            "Repair the formatting "
                            "without adding evidence or changing your "
                            "diagnosis solely to satisfy validation."
                        ),
                    ),
                ])

        raise RuntimeError("Unreachable synthesis state")

    def _validate_citations(self, hypothesis: Hypothesis, ledger: EvidenceLedger) -> None:
        known = ledger.known_ids()
        unknown = [c for c in hypothesis.citations if c not in known]
        if unknown:
            raise ValueError(f"Hypothesis cites evidence IDs not in the ledger: {unknown}")
        if hypothesis.ranked_services and not hypothesis.citations:
            raise ValueError("A ranked diagnosis requires cited evidence; otherwise use the abstention format")
        if not hypothesis.ranked_services and (
            hypothesis.confidence != 0.0 or hypothesis.indicator != "unknown"
            or hypothesis.fault_category != "unknown"
        ):
            raise ValueError("Abstention requires confidence=0.0, indicator='unknown', fault_category='unknown'")
