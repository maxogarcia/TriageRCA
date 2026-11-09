"""Protocol, replanning-context, citation and abstention checks."""
import json
from types import SimpleNamespace

import pytest

from src.agents.executor import Executor
from src.evidence import EvidenceLedger
from src.llm.base import ToolCall
from src.schemas import Plan
from src.tools.registry import ToolRegistry
from src.tools.metric_query import SPEC


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        return next(self.responses)


def report(**changes):
    data = dict(ranked_services=["carts"], indicator="carts_cpu", fault_category="cpu",
                explanation="See E001", citations=["E001"], mitigation="Investigate", confidence=0.8)
    data.update(changes)
    return SimpleNamespace(text=json.dumps(data), tool_calls=[])


def seeded():
    ledger = EvidenceLedger()
    ledger.add("metric_query", {}, "carts CPU increased from 1 to 10")
    return ledger


def test_replanning_includes_actual_existing_evidence():
    llm = SequenceLLM([report()])
    Executor(llm, ToolRegistry()).run("Investigate", Plan(steps=[]), seeded())
    text = "\n".join(m.content for m in llm.calls[0][0])
    assert "[E001] metric_query" in text
    assert "increased from 1 to 10" in text


def test_every_announced_call_has_response_when_budget_exhausted():
    tools = ToolRegistry()
    tools.register(SPEC, lambda **kwargs: "carts CPU increased")
    batch = SimpleNamespace(text="", tool_calls=[
        ToolCall(id="1", name="metric_query", args={}),
        ToolCall(id="2", name="metric_query", args={}),
    ])
    llm = SequenceLLM([batch, report()])
    ledger = EvidenceLedger()
    Executor(llm, tools, max_tool_calls=1).run("Investigate", Plan(steps=[]), ledger)
    responses = [m for m in llm.calls[-1][0] if m.role == "tool"]
    assert [m.tool_call_id for m in responses] == ["1", "2"]
    assert "Not executed" in responses[1].content
    assert len(ledger.items) == 1
    assert "response_schema" in llm.calls[-1][1]


def test_empty_ledger_allows_explicit_abstention():
    llm = SequenceLLM([report(ranked_services=[], indicator="unknown", fault_category="unknown",
                              confidence=0.0, citations=[], explanation="Insufficient evidence")])
    result = Executor(llm, ToolRegistry(), max_tool_calls=0).run("Investigate", Plan(steps=[]), EvidenceLedger())
    assert result.ranked_services == []


def test_ranked_uncited_report_fails_after_bounded_retry():
    bad = report(citations=[])
    llm = SequenceLLM([bad, bad, bad])
    with pytest.raises(ValueError, match="requires cited evidence"):
        Executor(llm, ToolRegistry()).run("Investigate", Plan(steps=[]), EvidenceLedger())
    assert len(llm.calls) == 3


@pytest.mark.parametrize("bad", [report(confidence=85), report(citations=["E999"]),
                                report(citations=["Evidence E001: CPU increased"]),
                                SimpleNamespace(text="", tool_calls=[])])
def test_invalid_direct_report_repairs_with_known_evidence(bad):
    llm = SequenceLLM([bad, report()])
    Executor(llm, ToolRegistry()).run("Investigate", Plan(steps=[]), seeded())
    assert len(llm.calls) == 2


def test_unknown_citation_still_rejected():
    llm = SequenceLLM([report(citations=["E999"])] * 3)
    with pytest.raises(ValueError, match="E999"):
        Executor(llm, ToolRegistry()).run("Investigate", Plan(steps=[]), seeded())
