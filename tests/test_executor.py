from __future__ import annotations

import json

from src.agents.executor import Executor
from src.evidence import EvidenceLedger
from src.llm.base import LLMResponse, ToolCall
from src.llm.fake import FakeLLMClient
from src.rag.retriever import StubRetriever
from src.schemas import Plan, PlanStep
from src.tools.log_search import SPEC as LOG_SEARCH_SPEC
from src.tools.log_search import make_log_search
from src.tools.metric_query import SPEC as METRIC_QUERY_SPEC
from src.tools.metric_query import make_metric_query
from src.tools.registry import ToolRegistry
from src.tools.runbook_retrieval import SPEC as RUNBOOK_RETRIEVAL_SPEC
from src.tools.runbook_retrieval import make_runbook_retrieval


def _make_registry(case) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(METRIC_QUERY_SPEC, make_metric_query(case))
    registry.register(LOG_SEARCH_SPEC, make_log_search(case))
    registry.register(RUNBOOK_RETRIEVAL_SPEC, make_runbook_retrieval(StubRetriever()))
    return registry


def test_executor_calls_tool_then_synthesizes_hypothesis(sample_case):
    registry = _make_registry(sample_case)

    tool_call_response = LLMResponse(
        text="",
        tool_calls=[
            ToolCall(
                id="1",
                name="metric_query",
                args={"service": "cartservice", "start_time": 100, "end_time": 130},
            )
        ],
        input_tokens=10,
        output_tokens=5,
        latency_s=0.01,
        model="fake",
        cached=False,
    )
    final_hypothesis = {
        "ranked_services": ["cartservice"],
        "indicator": "cartservice_cpu",
        "fault_category": "cpu",
        "explanation": "CPU spiked to 95, see E001.",
        "citations": ["E001"],
        "mitigation": "Scale up cartservice replicas.",
        "confidence": 0.8,
    }
    synth_response = LLMResponse(
        text=json.dumps(final_hypothesis),
        tool_calls=[],
        input_tokens=20,
        output_tokens=15,
        latency_s=0.02,
        model="fake",
        cached=False,
    )

    llm = FakeLLMClient([tool_call_response, synth_response])
    executor = Executor(llm=llm, tools=registry, max_tool_calls=5)
    ledger = EvidenceLedger()
    plan = Plan(steps=[PlanStep(tool="metric_query", args='{"service": "cartservice"}', rationale="check cpu")])

    hypothesis = executor.run(sample_case.incident_query, plan, ledger)

    assert hypothesis.ranked_services == ["cartservice"]
    assert hypothesis.citations == ["E001"]
    assert len(ledger.items) == 1
    assert ledger.items[0].source == "metric_query"


def test_executor_rejects_hypothesis_citing_unknown_evidence(sample_case):
    registry = _make_registry(sample_case)
    bad_hypothesis = {
        "ranked_services": ["cartservice"],
        "indicator": "cartservice_cpu",
        "fault_category": "cpu",
        "explanation": "made up",
        "citations": ["E999"],
        "mitigation": "n/a",
        "confidence": 0.5,
    }
    synth_response = LLMResponse(
        text=json.dumps(bad_hypothesis),
        tool_calls=[],
        input_tokens=5,
        output_tokens=5,
        latency_s=0.01,
        model="fake",
        cached=False,
    )
    llm = FakeLLMClient([synth_response])
    executor = Executor(llm=llm, tools=registry, max_tool_calls=5)
    ledger = EvidenceLedger()
    plan = Plan(steps=[])

    try:
        executor.run(sample_case.incident_query, plan, ledger)
        raise AssertionError("expected ValueError for unknown citation")
    except ValueError as exc:
        assert "E999" in str(exc)
