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


class RunRecord(BaseModel):
    case_id: str
    config: str
    report: RCAReport | None
    evidence_ledger: list[dict]
    llm_calls: list[LLMCallRecord]
    critic_verdicts: list[CriticVerdict]
    iteration_count: int
    wall_clock_s: float
    errors: list[str] = Field(default_factory=list)
