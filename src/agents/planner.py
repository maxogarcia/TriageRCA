"""Planner: turns an incident query (+ service list, + critic feedback on re-plan) into an
ordered investigation Plan for the Executor to carry out.
"""

from __future__ import annotations

from src.llm.base import LLMClient, Message
from src.schemas import CriticVerdict, Plan

_SYSTEM_PROMPT = """You are the Planner in a root-cause-analysis (RCA) agent for microservice incidents.

Given an incident description and the list of services involved, produce an ordered investigation
plan: a sequence of tool calls (tool name + args + rationale) that will gather the evidence needed
to identify the root-cause service, the specific indicator (metric/log signal), and the fault
category. Prefer checking likely-culprit services first. Do not guess a root cause yourself --
your job is only to plan what to look at.

Available tools: metric_query(service, start_time, end_time), log_search(service, start_time,
end_time, level=None, keyword=None), trace_query(service, start_time, end_time),
runbook_retrieval(query, k=5).

Respond with a Plan: a list of steps, each with `tool`, `rationale`, and `args` -- args is a
JSON-encoded object as a STRING, e.g. "{\\"service\\": \\"checkoutservice\\", \\"start_time\\": 100,
\\"end_time\\": 200}", not a nested object.
"""


class Planner:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def plan(
        self,
        incident_query: str,
        services: list[str],
        inject_time: float,
        critic_feedback: CriticVerdict | None = None,
    ) -> Plan:
        user_content = (
            f"Incident: {incident_query}\n"
            f"Services involved: {', '.join(services)}\n"
            f"Approximate onset time (unix): {inject_time}\n"
        )
        if critic_feedback is not None:
            user_content += (
                f"\nA previous hypothesis was rejected by the critic.\n"
                f"Reasons: {'; '.join(critic_feedback.reasons)}\n"
                f"Missing evidence: {'; '.join(critic_feedback.missing_evidence)}\n"
                f"Revise the plan to gather the missing evidence.\n"
            )

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]
        response = self._llm.generate(messages, response_schema=Plan)
        return Plan.model_validate_json(response.text)
