from __future__ import annotations

import json

from src.llm.base import LLMResponse
from src.llm.fake import FakeLLMClient
from src.pipeline import run_case
from src.rag.retriever import StubRetriever
from src.tools.metric_query import SPEC as METRIC_QUERY_SPEC
from src.tools.metric_query import make_metric_query
from src.tools.registry import ToolRegistry
from src.tools.runbook_retrieval import SPEC as RUNBOOK_RETRIEVAL_SPEC
from src.tools.runbook_retrieval import make_runbook_retrieval

_HYPOTHESIS = {
    "ranked_services": ["cartservice"],
    "indicator": "cartservice_cpu",
    "fault_category": "cpu",
    "explanation": "no evidence cited",
    "citations": [],
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
