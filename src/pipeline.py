"""Orchestration loop for each config: retrieval_only, react, triagerca, triagerca_no_critic.

triagerca: Planner -> Executor -> Critic, re-planning on reject/need_more_evidence, capped at
max_iterations. triagerca_no_critic: same but skips the critic (auto-accepts the first
hypothesis). react: single agent, same tools, no planner/critic.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from src.agents.critic import AlwaysAcceptCritic, Critic
from src.agents.executor import Executor
from src.agents.planner import Planner
from src.agents.react import run_react
from src.data.case import Case
from src.evidence import EvidenceLedger
from src.llm.base import LLMResponse
from src.schemas import CriticVerdict, LLMCallRecord, RCAReport, RunRecord

MAX_ITERATIONS = 3


class RecordingLLM:
    """Wraps an LLMClient to record every call's usage for the RunRecord, without changing
    the LLMClient interface Planner/Executor depend on."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.records: list[LLMCallRecord] = []

    def generate(self, *args, **kwargs) -> LLMResponse:
        response = self._inner.generate(*args, **kwargs)
        self.records.append(
            LLMCallRecord(
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_s=response.latency_s,
                cached=response.cached,
            )
        )
        return response


def run_case(case: Case, config: str, llm, tools, critic: Critic | None = None, max_tool_calls: int = 10) -> RunRecord:
    start = time.monotonic()
    recording_llm = RecordingLLM(llm)
    ledger = EvidenceLedger()
    critic_verdicts: list[CriticVerdict] = []
    errors: list[str] = []
    report: RCAReport | None = None
    iteration_count = 0

    executor = Executor(llm=recording_llm, tools=tools, max_tool_calls=max_tool_calls)

    try:
        if config == "react":
            hypothesis = run_react(executor, case.incident_query, ledger)
            report = RCAReport(**hypothesis.model_dump())
            iteration_count = 1

        elif config in ("triagerca", "triagerca_no_critic"):
            planner = Planner(llm=recording_llm)
            active_critic = (
                AlwaysAcceptCritic() if config == "triagerca_no_critic" else (critic or AlwaysAcceptCritic())
            )
            feedback: CriticVerdict | None = None

            for iteration_count in range(1, MAX_ITERATIONS + 1):
                plan = planner.plan(case.incident_query, case.services, case.inject_time, critic_feedback=feedback)
                hypothesis = executor.run(case.incident_query, plan, ledger)
                verdict = active_critic.review(hypothesis, ledger)
                critic_verdicts.append(verdict)

                if verdict.decision == "accept":
                    report = RCAReport(**hypothesis.model_dump())
                    break
                feedback = verdict
            else:
                # Loop exhausted without acceptance -- report the last hypothesis anyway,
                # flagged via errors so eval can tell it never passed the critic.
                report = RCAReport(**hypothesis.model_dump())
                errors.append(f"Critic did not accept within {MAX_ITERATIONS} iterations.")

        else:
            raise ValueError(f"Unknown config: {config!r}")

    except Exception as exc:  # noqa: BLE001 -- run harness must not crash on one bad case
        errors.append(f"{type(exc).__name__}: {exc}")

    return RunRecord(
        case_id=case.case_id,
        config=config,
        report=report,
        evidence_ledger=[asdict(i) for i in ledger.items],
        llm_calls=recording_llm.records,
        critic_verdicts=critic_verdicts,
        iteration_count=iteration_count,
        wall_clock_s=time.monotonic() - start,
        errors=errors,
    )
