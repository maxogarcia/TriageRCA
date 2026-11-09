"""Orchestration loop for each config: retrieval_only, react, triagerca, triagerca_no_critic.

retrieval_only: one runbook retrieval + one LLM call, no telemetry tools.
triagerca: Planner -> Executor -> Critic, re-planning on reject/need_more_evidence, capped at
max_iterations. triagerca_no_critic: same but skips the critic (auto-accepts the first
hypothesis). react: single agent, same tools, no planner/critic.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from src.agents.critic import AlwaysAcceptCritic, Critic
from src.agents.executor import Executor
from src.agents.planner import Planner
from src.agents.react import run_react
from src.agents.retrieval_only import run_retrieval_only
from src.data.case import Case
from src.evidence import EvidenceLedger
from src.llm.base import LLMResponse
from src.schemas import CriticVerdict, Hypothesis, LLMCallRecord, RCAReport, RunRecord
from src.tools.metric_query import metric_overview

MAX_ITERATIONS = 3


@dataclass(frozen=True)
class Pricing:
    """List price per 1M tokens (configs/models.yaml), for cost-per-incident reporting."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_1m + output_tokens * self.output_per_1m) / 1_000_000


class RecordingLLM:
    """Wraps an LLMClient to record every call's usage for the RunRecord, without changing
    the LLMClient interface Planner/Executor depend on."""

    def __init__(self, inner, pricing: Pricing | None = None) -> None:
        self._inner = inner
        self._pricing = pricing or Pricing()
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
                cost_usd=self._pricing.cost(response.input_tokens, response.output_tokens),
            )
        )
        return response


def _critic_call_records(critic, since: int) -> list[LLMCallRecord]:
    """LLM calls the critic made after index `since`. The critic object is shared across cases,
    so only the new calls belong to this case. Stub critics have no `calls` and make no calls.
    Cost is 0: the critic runs on a local open-weight model."""
    return [
        LLMCallRecord(
            model=r.model,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            latency_s=r.latency_s,
            cached=r.cached,
            cost_usd=0.0,
            role="critic",
        )
        for r in getattr(critic, "calls", [])[since:]
    ]


def run_case(
    case: Case,
    config: str,
    llm,
    tools,
    critic: Critic | None = None,
    max_tool_calls: int = 10,
    max_iterations: int = MAX_ITERATIONS,
    pricing: Pricing | None = None,
) -> RunRecord:
    start = time.monotonic()
    recording_llm = RecordingLLM(llm, pricing)
    ledger = EvidenceLedger()
    critic_verdicts: list[CriticVerdict] = []
    critic_calls: list[LLMCallRecord] = []
    hypotheses: list[Hypothesis] = []
    errors: list[str] = []
    report: RCAReport | None = None
    iteration_count = 0

    try:
        if config not in ("retrieval_only", "react", "triagerca", "triagerca_no_critic"):
            raise ValueError(f"Unknown config: {config!r}")
        if max_tool_calls < 0 or max_iterations < 1:
            raise ValueError("max_tool_calls must be nonnegative and max_iterations must be positive")
        first_budget = max_tool_calls
        if config != "retrieval_only" and max_tool_calls > 0:
            # One fixed preprocessing query, identical across telemetry-enabled variants.
            # No CaseLabel, path, or case ID is passed into the evidence summary.
            ledger.add(
                source="metric_overview",
                args={"start_time": case.inject_time - 300, "end_time": case.inject_time + 300},
                content=metric_overview(case),
            )
            first_budget -= 1
        executor = Executor(llm=recording_llm, tools=tools, max_tool_calls=first_budget)
        if config == "retrieval_only":
            hypothesis = run_retrieval_only(recording_llm, tools, case.incident_query, case.services, ledger)
            hypotheses.append(hypothesis)
            report = RCAReport(**hypothesis.model_dump())
            iteration_count = 1

        elif config == "react":
            hypothesis = run_react(executor, case.incident_query, ledger)
            hypotheses.append(hypothesis)
            report = RCAReport(**hypothesis.model_dump())
            iteration_count = 1

        elif config in ("triagerca", "triagerca_no_critic"):
            planner = Planner(llm=recording_llm)
            active_critic = (
                AlwaysAcceptCritic() if config == "triagerca_no_critic" else (critic or AlwaysAcceptCritic())
            )
            feedback: CriticVerdict | None = None

            for iteration_count in range(1, max_iterations + 1):
                if iteration_count > 1:
                    # Preserve the original per-iteration cap; overview was collected only once.
                    executor = Executor(llm=recording_llm, tools=tools, max_tool_calls=max_tool_calls)
                plan = planner.plan(case.incident_query, case.services, case.inject_time, critic_feedback=feedback)
                hypothesis = executor.run(case.incident_query, plan, ledger)
                hypotheses.append(hypothesis)
                calls_before = len(getattr(active_critic, "calls", []))
                try:
                    verdict = active_critic.review(hypothesis, ledger)
                finally:
                    # Record critic calls even if review() raised, so failed calls still count.
                    critic_calls.extend(_critic_call_records(active_critic, calls_before))
                critic_verdicts.append(verdict)

                if verdict.decision == "accept":
                    report = RCAReport(**hypothesis.model_dump())
                    break
                feedback = verdict
            else:
                # Loop exhausted without acceptance -- report the last hypothesis anyway,
                # flagged via errors so eval can tell it never passed the critic.
                report = RCAReport(**hypothesis.model_dump())
                errors.append(f"Critic did not accept within {max_iterations} iterations.")

        else:
            raise ValueError(f"Unknown config: {config!r}")

    except Exception as exc:  # noqa: BLE001 -- run harness must not crash on one bad case
        errors.append(f"{type(exc).__name__}: {exc}")

    return RunRecord(
        case_id=case.case_id,
        config=config,
        report=report,
        evidence_ledger=[asdict(i) for i in ledger.items],
        llm_calls=[*recording_llm.records, *critic_calls],
        critic_verdicts=critic_verdicts,
        hypotheses=hypotheses,
        iteration_count=iteration_count,
        wall_clock_s=time.monotonic() - start,
        cost_usd=sum(r.cost_usd for r in recording_llm.records),
        errors=errors,
    )
