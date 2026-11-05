"""Executor regression tests: no network or model calls."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agents.executor import Executor
from src.llm.base import ToolCall
from src.schemas import Plan


class StubLedger:
    def __init__(self):
        self.ids = {"E001"}
        self.items = [SimpleNamespace(id="E001", source="metric_query", content="carts CPU: 80")]

    def known_ids(self):
        return self.ids


class StubTools:
    def specs(self):
        return []


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        return next(self.responses)


def report(**changes):
    data = dict(
        ranked_services=["carts"], indicator="carts_cpu",
        fault_category="cpu", explanation="Observed CPU metrics.",
        citations=["E001"], mitigation="Investigate CPU load.", confidence=0.8,
    )
    data.update(changes)
    return SimpleNamespace(text=json.dumps(data), tool_calls=[])


def run_executor(responses, budget=1):
    llm = SequenceLLM(responses)
    ledger = StubLedger()
    executor = Executor(llm=llm, tools=StubTools(), max_tool_calls=budget)
    return executor, llm, ledger


def test_valid_direct_report_uses_one_call():
    executor, llm, ledger = run_executor([report()])
    result = executor.run("Investigate", Plan(steps=[]), ledger)
    assert result.ranked_services == ["carts"]
    assert len(llm.calls) == 1


def test_budget_exhaustion_synthesizes_after_empty_tool_response():
    call = ToolCall(id="t1", name="metric_query", args={"service": "carts"})
    tool_response = SimpleNamespace(text="", tool_calls=[call])
    executor, llm, ledger = run_executor([tool_response, report(citations=["E001"])])

    def invoke(call, ledger):
        ledger.ids.add("E001")
        return "[E001] carts CPU: 80"

    executor._invoke_tool = invoke
    result = executor.run("Investigate", Plan(steps=[]), ledger)
    assert result.citations == ["E001"]
    assert len(llm.calls) == 2
    messages, kwargs = llm.calls[1]
    assert "response_schema" in kwargs
    assert "tools" not in kwargs
    assert any(m.role == "tool" and "CPU: 80" in m.content for m in messages)


def test_zero_tool_budget_still_synthesizes():
    executor, llm, ledger = run_executor([report()], budget=0)
    assert executor.run("Investigate", Plan(steps=[]), ledger).confidence == 0.8
    assert len(llm.calls) == 1


@pytest.mark.parametrize("invalid", [
    report(confidence=85),
    report(citations=["Evidence E001: CPU high"]),
    report(citations=["E999"]),
    SimpleNamespace(text="", tool_calls=[]),
])
def test_invalid_direct_report_enters_repair(invalid):
    executor, llm, ledger = run_executor([invalid, report()])
    executor.run("Investigate", Plan(steps=[]), ledger)
    assert len(llm.calls) == 2
    assert "failed validation" in llm.calls[1][0][-1].content


def test_structured_synthesis_retries_citation_errors():
    executor, llm, ledger = run_executor(
        [report(citations=["Evidence E001: CPU high"]), report(citations=["E001"])],
        budget=0,
    )
    ledger.ids.add("E001")
    result = executor.run("Investigate", Plan(steps=[]), ledger)
    assert result.citations == ["E001"]
    assert len(llm.calls) == 2
    assert "Allowed citation IDs" in llm.calls[1][0][-1].content


def test_persistent_unknown_citations_fail_after_bounded_retry():
    executor, llm, ledger = run_executor([report(citations=["E999"])] * 3)
    with pytest.raises(ValueError, match="E999"):
        executor.run("Investigate", Plan(steps=[]), ledger)
    # One direct report + at most two structured synthesis attempts.
    assert len(llm.calls) == 3

