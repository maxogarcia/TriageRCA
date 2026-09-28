"""ReAct baseline: a single agent using the same tools, no planner/critic loop.

NOTE: ownership of this baseline is unresolved (see CLAUDE.md open decisions -- listed under
Daniel but it runs on Max's tools/executor). Implemented here for now since it's a thin wrapper
around Executor with an empty Plan; move/rename freely once that's settled.
"""

from __future__ import annotations

from src.agents.executor import Executor
from src.evidence import EvidenceLedger
from src.schemas import Hypothesis, Plan


def run_react(executor: Executor, incident_query: str, ledger: EvidenceLedger) -> Hypothesis:
    empty_plan = Plan(steps=[], notes="ReAct baseline: no upfront plan, agent decides tool calls as it goes.")
    return executor.run(incident_query, empty_plan, ledger)
