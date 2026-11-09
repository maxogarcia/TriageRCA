from __future__ import annotations

import json

import pytest

from src.llm.base import LLMResponse
from src.llm.fake import FakeLLMClient
from src.pipeline import Pricing, run_case
from src.rag.retriever import StubRetriever
from src.schemas import CriticVerdict
from src.tools.metric_query import SPEC as METRIC_QUERY_SPEC
from src.tools.metric_query import make_metric_query
from src.tools.registry import ToolRegistry
from src.tools.runbook_retrieval import SPEC as RUNBOOK_RETRIEVAL_SPEC
from src.tools.runbook_retrieval import make_runbook_retrieval

_HYPOTHESIS = {
    "ranked_services": ["cartservice"],
    "indicator": "cartservice_cpu",
    "fault_category": "cpu",
    "explanation": "Candidate based on the initial metric screen, E001.",
    "citations": ["E001"],
    "mitigation": "n/a",
    "confidence": 0.5,
}


def _hypothesis_response() -> LLMResponse:
    return LLMResponse(
        text=json.dumps(_HYPOTHESIS),
        tool_calls=[],
        input_tokens=1,
        output_tokens=1,
        latency_s=0.01,
        model="fake",
        cached=False,
    )


def _registry(case) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(METRIC_QUERY_SPEC, make_metric_query(case))
    registry.register(RUNBOOK_RETRIEVAL_SPEC, make_runbook_retrieval(StubRetriever()))
    return registry


def test_run_case_react_produces_report(sample_case):
    llm = FakeLLMClient([_hypothesis_response()])
    record = run_case(sample_case, config="react", llm=llm, tools=_registry(sample_case))

    assert record.report is not None
    assert record.report.ranked_services == ["cartservice"]
    assert record.iteration_count == 1
    assert len(record.llm_calls) == 1
    assert not record.errors


def test_run_case_triagerca_no_critic_accepts_first_hypothesis(sample_case):
    plan_response = LLMResponse(
        text=json.dumps({"steps": [], "notes": ""}),
        tool_calls=[],
        input_tokens=1,
        output_tokens=1,
        latency_s=0.01,
        model="fake",
        cached=False,
    )
    llm = FakeLLMClient([plan_response, _hypothesis_response()])
    record = run_case(sample_case, config="triagerca_no_critic", llm=llm, tools=_registry(sample_case))

    assert record.report is not None
    assert record.iteration_count == 1
    assert record.critic_verdicts[0].decision == "accept"


def _response(payload: dict, input_tokens: int = 1, output_tokens: int = 1) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(payload),
        tool_calls=[],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=0.01,
        model="fake",
        cached=False,
    )


class _AlwaysRejectCritic:
    def review(self, hypothesis, ledger):
        return CriticVerdict(decision="reject", reasons=["no evidence"], missing_evidence=["cartservice logs"])


def test_run_case_retrieval_only_logs_retrieval_and_makes_one_call(sample_case):
    hypothesis = {**_HYPOTHESIS, "citations": ["E001"]}
    llm = FakeLLMClient([_response(hypothesis)])
    record = run_case(sample_case, config="retrieval_only", llm=llm, tools=_registry(sample_case))

    assert not record.errors
    assert record.report.citations == ["E001"]
    assert [e["source"] for e in record.evidence_ledger] == ["runbook_retrieval"]
    assert len(record.llm_calls) == 1
    assert llm.calls[0]["tools"] is None  # no telemetry tools for this baseline


def test_run_case_triagerca_replans_on_reject_up_to_max_iterations(sample_case):
    plan = {"steps": [], "notes": ""}
    llm = FakeLLMClient([_response(plan), _response(_HYPOTHESIS)] * 2)
    record = run_case(
        sample_case,
        config="triagerca",
        llm=llm,
        tools=_registry(sample_case),
        critic=_AlwaysRejectCritic(),
        max_iterations=2,
    )

    assert record.iteration_count == 2
    assert [v.decision for v in record.critic_verdicts] == ["reject", "reject"]
    assert record.report is not None  # last hypothesis still reported...
    assert "did not accept within 2 iterations" in record.errors[0]  # ...but flagged
    replan_prompt = llm.calls[2]["messages"][-1].content
    assert "cartservice logs" in replan_prompt  # critic feedback reached the planner


def test_run_case_computes_cost_from_pricing(sample_case):
    llm = FakeLLMClient([_response(_HYPOTHESIS, input_tokens=1_000_000, output_tokens=200_000)])
    record = run_case(
        sample_case,
        config="react",
        llm=llm,
        tools=_registry(sample_case),
        pricing=Pricing(input_per_1m=0.75, output_per_1m=3.75),
    )

    assert record.llm_calls[0].cost_usd == pytest.approx(0.75 + 0.75)
    assert record.cost_usd == pytest.approx(1.5)


@pytest.mark.parametrize("config", ["react", "triagerca", "triagerca_no_critic"])
def test_telemetry_variants_receive_same_label_free_overview(sample_case, config):
    responses = [_hypothesis_response()]
    if config != "react":
        responses.insert(0, _response({"steps": [], "notes": ""}))
    llm = FakeLLMClient(responses)
    record = run_case(sample_case, config=config, llm=llm, tools=_registry(sample_case))
    assert not record.errors
    assert [e["source"] for e in record.evidence_ledger] == ["metric_overview"]
    assert sample_case.case_id not in record.evidence_ledger[0]["content"]
    assert any("[E001] metric_overview" in m.content for m in llm.calls[-1]["messages"])


def test_zero_budget_does_not_collect_overview(sample_case):
    abstention = {**_HYPOTHESIS, "ranked_services": [], "indicator": "unknown",
                  "fault_category": "unknown", "confidence": 0.0, "citations": []}
    llm = FakeLLMClient([_response(abstention)])
    record = run_case(sample_case, config="react", llm=llm, tools=_registry(sample_case), max_tool_calls=0)
    assert not record.errors
    assert record.evidence_ledger == []
    assert record.report.ranked_services == []


def test_overview_consumes_first_iteration_budget(sample_case):
    llm = FakeLLMClient([_hypothesis_response()])
    record = run_case(sample_case, config="react", llm=llm, tools=_registry(sample_case), max_tool_calls=1)
    assert not record.errors
    assert len(record.evidence_ledger) == 1
    assert llm.calls[0]["tools"] is None  # no remaining discretionary tool calls


def test_replanning_preserves_one_overview_and_evidence_context(sample_case):
    llm = FakeLLMClient([_response({"steps": [], "notes": ""}), _hypothesis_response()] * 2)
    record = run_case(sample_case, config="triagerca", llm=llm, tools=_registry(sample_case),
                      critic=_AlwaysRejectCritic(), max_iterations=2)
    assert [e["source"] for e in record.evidence_ledger] == ["metric_overview"]
    assert any("[E001] metric_overview" in m.content for m in llm.calls[-1]["messages"])


class _RecordingCritic:
    """Rejects first, then accepts, and keeps `calls` like LLMCritic does."""

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls: list[LLMResponse] = []

    def review(self, hypothesis, ledger):
        self.calls.append(_response({}, input_tokens=7, output_tokens=3))
        return CriticVerdict(decision=self._decisions.pop(0), reasons=[], missing_evidence=["logs"])


def test_run_case_records_critic_calls_and_every_hypothesis(sample_case):
    plan = {"steps": [], "notes": ""}
    second = {**_HYPOTHESIS, "ranked_services": ["frontend"], "indicator": "frontend_cpu"}
    llm = FakeLLMClient([_response(plan), _response(_HYPOTHESIS), _response(plan), _response(second)])
    critic = _RecordingCritic(["reject", "accept"])
    record = run_case(
        sample_case, config="triagerca", llm=llm, tools=_registry(sample_case), critic=critic, max_iterations=3
    )

    assert not record.errors
    assert [h.ranked_services for h in record.hypotheses] == [["cartservice"], ["frontend"]]
    assert record.report.ranked_services == ["frontend"]  # report is the last (accepted) hypothesis
    roles = [c.role for c in record.llm_calls]
    assert roles.count("pipeline") == 4 and roles.count("critic") == 2
    assert all(c.cost_usd == 0.0 for c in record.llm_calls if c.role == "critic")


def test_run_case_counts_only_this_cases_critic_calls(sample_case):
    # run.py builds one critic for all cases, so its `calls` list keeps growing.
    plan = {"steps": [], "notes": ""}
    critic = _RecordingCritic(["accept", "accept"])
    for _ in range(2):
        llm = FakeLLMClient([_response(plan), _response(_HYPOTHESIS)])
        record = run_case(sample_case, config="triagerca", llm=llm, tools=_registry(sample_case), critic=critic)
        assert [c.role for c in record.llm_calls].count("critic") == 1


def test_run_case_react_records_its_single_hypothesis(sample_case):
    llm = FakeLLMClient([_hypothesis_response()])
    record = run_case(sample_case, config="react", llm=llm, tools=_registry(sample_case))
    assert len(record.hypotheses) == 1
