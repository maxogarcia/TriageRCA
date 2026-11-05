"""Shared Pydantic schemas for plans, hypotheses, critic verdicts, and final reports/run records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    tool: str
    # JSON-encoded object, e.g. '{"service": "checkoutservice", "start_time": 100, "end_time": 200}".
    # Not a dict: Gemini's structured-output mode (response_schema) rejects open-ended object
    # fields (it can't emit additionalProperties), and this is descriptive guidance for the
    # Executor, not something parsed and executed directly -- see Executor.run()'s plan_summary.
    args: str
    rationale: str


class Plan(BaseModel):
    steps: list[PlanStep]
    notes: str = ""


class Hypothesis(BaseModel):
    ranked_services: list[str] = Field(max_length=5)
    indicator: str
    fault_category: str
    explanation: str
    citations: list[str]  # evidence IDs
    mitigation: str
    confidence: float = Field(ge=0.0, le=1.0)


class CriticVerdict(BaseModel):
    decision: Literal["accept", "reject", "need_more_evidence"]
    reasons: list[str]
    missing_evidence: list[str] = Field(default_factory=list)


# RCAReport is the accepted Hypothesis -- same shape, kept as a separate name for the eval harness boundary.
class RCAReport(Hypothesis):
    pass


class LLMCallRecord(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool
    cost_usd: float = 0.0  # at list price, even when cached or on the free tier


class RunRecord(BaseModel):
    case_id: str
    config: str
    report: RCAReport | None
    evidence_ledger: list[dict]
    llm_calls: list[LLMCallRecord]
    critic_verdicts: list[CriticVerdict]
    iteration_count: int
    wall_clock_s: float
    cost_usd: float = 0.0  # sum of llm_calls[].cost_usd
    errors: list[str] = Field(default_factory=list)


ClaimLabel = Literal["supported", "partial", "unsupported", "contradicted"]


class FieldCheck(BaseModel):
    """The critic's check of one hypothesis field (top-ranked service or indicator).

    These are schema fields rather than list items so the model cannot skip them.
    """

    evidence_ids: list[str]  # ledger IDs the label is based on, cited or not
    comparison: str  # the value(s) the field's value is compared against, with IDs; "" if none
    reason: str  # one sentence quoting the values that decide the label
    label: ClaimLabel


class ClaimCheck(BaseModel):
    """One atomic claim from a hypothesis explanation, checked against the evidence ledger.

    Field order matters: models fill JSON fields in order, so the comparison and reason (with
    the actual numbers or log lines) are written before the label is chosen.
    """

    quote: str  # exact words from the explanation this claim comes from (claims not found are dropped)
    claim: str  # standalone sentence, spec v1.1 sections 2.1-2.2 (no hedges, no recommendations)
    is_root_cause: bool  # true if the claim names the root-cause service, indicator, or fault
    evidence_ids: list[str]  # ledger IDs the label is based on, cited or not
    comparison: str  # baseline or other services' values the claim is judged against; "" if none
    reason: str  # one sentence quoting the values or lines that decide the label
    label: ClaimLabel


class ClaimCheckResult(BaseModel):
    """Structured output of the critic's single LLM call."""

    root_cause_check: FieldCheck  # "The root cause is <ranked_services[0]>."
    indicator_check: FieldCheck  # "<indicator> is the metric that shows the fault."
    claims: list[ClaimCheck]
    # Concrete tool checks that would settle weak claims, e.g. "metric_query checkoutservice
    # over a window spanning the inject time". Passed to the Planner as missing_evidence.
    suggested_checks: list[str] = Field(default_factory=list, max_length=3)
