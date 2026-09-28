from __future__ import annotations

import json

from src.agents.planner import Planner
from src.llm.base import LLMResponse
from src.llm.fake import FakeLLMClient
from src.schemas import CriticVerdict


def _plan_response(steps: list[dict]) -> LLMResponse:
    return LLMResponse(
        text=json.dumps({"steps": steps, "notes": ""}),
        tool_calls=[],
        input_tokens=10,
        output_tokens=10,
        latency_s=0.01,
        model="fake",
        cached=False,
    )


def test_planner_produces_ordered_steps(sample_case):
    llm = FakeLLMClient(
        [
            _plan_response(
                [
                    {
                        "tool": "metric_query",
                        "args": '{"service": "cartservice"}',
                        "rationale": "check cpu first",
                    }
                ]
            )
        ]
    )
    planner = Planner(llm=llm)

    plan = planner.plan(sample_case.incident_query, sample_case.services, sample_case.inject_time)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "metric_query"


def test_planner_includes_critic_feedback_in_prompt(sample_case):
    llm = FakeLLMClient([_plan_response([])])
    planner = Planner(llm=llm)
    feedback = CriticVerdict(decision="reject", reasons=["not enough evidence"], missing_evidence=["logs for frontend"])

    planner.plan(
        sample_case.incident_query,
        sample_case.services,
        sample_case.inject_time,
        critic_feedback=feedback,
    )

    sent_prompt = llm.calls[0]["messages"][-1].content
    assert "not enough evidence" in sent_prompt
    assert "logs for frontend" in sent_prompt
