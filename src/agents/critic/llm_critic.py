"""LLMCritic: an LLM falsifier that checks a hypothesis against the evidence ledger.

Flow (see the critic plan, Nov 4):
  1. Code check, no LLM call: every cited evidence ID must exist in the ledger.
  2. One LLM call splits the explanation into atomic claims (grounding spec v1.1 sections
     2.1-2.2) and labels each supported | partial | unsupported | contradicted.
     The top-ranked service and the indicator are checked as schema fields so they can't be
     skipped. Optionally (comparisons=True) the evidence includes a cross-service metric table
     computed from the ledger.
  3. Code clean-up (assemble_claims()): drop claims whose quote isn't in the explanation and
     evidence IDs not in the ledger; cap comparative claims with no stated comparison at partial.
  4. The verdict comes from fixed code rules (decide()), not from the model, so the
     accept/reject policy is the same for every model and easy to report.

The critic model is fixed (qwen2.5:7b-instruct via Ollama by default) and runs at temperature 0,
independent of which model drives the Planner/Executor.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from src.evidence import EvidenceLedger, format_evidence
from src.llm.base import LLMClient, LLMResponse, Message
from src.schemas import ClaimCheck, ClaimCheckResult, CriticVerdict, FieldCheck, Hypothesis

# Words that make a claim comparative: it can only be "supported" if a comparison is stated.
_COMPARATIVE = re.compile(
    r"\b(increas\w*|decreas\w*|spik\w*|surg\w*|drop\w*|significant\w*|high(er|est)?|low(er|est)?|"
    r"elevat\w*|abnormal\w*|anomal\w*|unusual\w*|degrad\w*|caus\w*|due to|because)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d")


def _has_value(comparison: str) -> bool:
    """A comparison only counts if it states a value to compare against, not just a metric name."""
    return bool(_NUMBER.search(comparison))


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.\s]", " ", text.lower()).split())


def _quote_in(quote: str, text: str) -> bool:
    """True if the quote really comes from the explanation (case, punctuation, spacing ignored)."""
    q = _normalize(quote)
    return len(q) >= 8 and q in _normalize(text)


def _cap_uncompared(check: ClaimCheck) -> ClaimCheck:
    """A comparative claim with no stated comparison cannot be fully supported."""
    if check.label == "supported" and not _has_value(check.comparison) and _COMPARATIVE.search(check.claim):
        check.label = "partial"
        check.reason += " [capped at partial: comparative claim with no comparison stated]"
    return check


def _field_claim(text: str, field: FieldCheck) -> ClaimCheck:
    return ClaimCheck(
        quote="(hypothesis field)",
        claim=text,
        is_root_cause=True,
        evidence_ids=field.evidence_ids,
        comparison=field.comparison,
        reason=field.reason,
        label=field.label,
    )


def assemble_claims(result: ClaimCheckResult, hypothesis: Hypothesis, known_ids: set[str]) -> list[ClaimCheck]:
    """Field checks first, then explanation claims whose quote is really in the explanation.

    Also drops evidence IDs the model invented and applies the comparative-claim cap.
    Field claims are always comparative ("the root cause", "best shows the fault").
    """
    top = hypothesis.ranked_services[0] if hypothesis.ranked_services else "(none)"
    fields = [
        _field_claim(f"The root cause is {top}.", result.root_cause_check),
        _field_claim(f"{hypothesis.indicator} is the metric that best shows the fault.", result.indicator_check),
    ]
    for f in fields:
        if f.label == "supported" and not _has_value(f.comparison):
            f.label = "partial"
            f.reason += " [capped at partial: no comparison stated]"
    from_text = [c for c in result.claims if _quote_in(c.quote, hypothesis.explanation)]
    claims = fields + [_cap_uncompared(c) for c in from_text]
    for c in claims:
        c.evidence_ids = [e for e in c.evidence_ids if e in known_ids]
    return claims


SYSTEM_PROMPT = """\
You are the critic in an incident root-cause analysis (RCA) system. Your job is to falsify: \
find every claim in a hypothesis that the evidence does not back up. You do not propose your \
own diagnosis.

Labels, judged ONLY against the evidence below, including evidence the hypothesis did not cite:
- supported: the evidence directly shows it, with matching values, AND any comparison it makes \
("increase", "spike", "significant", "high", "abnormal", "caused") is backed by a baseline or \
by other values in the evidence.
- partial: the values are right but the comparison or conclusion is not shown, e.g. "significant \
increase" with no earlier window and no other service to compare against.
- unsupported: nothing in the evidence shows it, or it is speculation.
- contradicted: some evidence shows the opposite, e.g. other services have the same supposedly \
abnormal value, or a different metric of the same service is far more anomalous.

Comparing values: a before/after comparison of the same service is the strongest evidence. \
Differences between services can be normal, because services do different work (a frontend or \
checkout service has higher latency than a leaf service, a catalog gets more requests), so a \
large gap across services is a lead, not proof. Write what you compared against in \
"comparison", with the values and IDs (e.g. "<service> <metric> mean <value> [<ID>]"); leave it empty \
if there is nothing to compare against.

Step 1. root_cause_check: is the top-ranked service the root cause, according to the evidence?
Step 2. indicator_check: is the named indicator the metric that best shows the fault? Label it \
contradicted if another metric of that service is clearly more anomalous.
Step 3. claims: split the explanation into atomic claims. An atomic claim attributes a behavior \
or value to a service, metric, or resource; asserts a causal or temporal relationship; states a \
root cause; or references a threshold, baseline, or norm. Skip hedges with no factual content, \
restatements of the task, general knowledge, and recommendations. Only take claims from the \
explanation text: put the exact words you took it from in "quote". Mark claims that name the \
root cause with is_root_cause = true.
Step 4. suggested_checks: up to 3 concrete tool checks that would settle the weakest claims, \
naming the tool, service, and metric or keyword, e.g. "metric_query <service> over a window that \
includes the inject time". Leave empty if everything is supported.

In evidence_ids, list the IDs your label is based on. In reason, quote the specific values or \
lines that decide the label, in one sentence. Return JSON only, matching the schema.\
"""


def build_user_prompt(hypothesis: Hypothesis, evidence_text: str) -> str:
    return (
        "# Hypothesis\n"
        f"ranked_services (first = root cause): {', '.join(hypothesis.ranked_services)}\n"
        f"indicator: {hypothesis.indicator}\n"
        f"fault_category: {hypothesis.fault_category}\n"
        f"citations: {', '.join(hypothesis.citations) or '(none)'}\n"
        f"explanation: {hypothesis.explanation}\n\n"
        "# Evidence\n"
        f"{evidence_text}\n"
    )


def decide(
    claims: list[ClaimCheck], suggested_checks: list[str], citations: list[str], max_weak: int = 1
) -> CriticVerdict:
    """Turn labeled claims into a verdict with fixed rules.

    reject:             any claim contradicted, or a root-cause claim unsupported.
    accept:             every root-cause claim supported, and at most `max_weak`
                        partial/unsupported claims overall.
    need_more_evidence: everything else (including: no root-cause claim at all).
    """
    root = [c for c in claims if c.is_root_cause]
    weak = [c for c in claims if c.label in ("partial", "unsupported")]

    reasons = [
        f"[{c.label}{', root cause' if c.is_root_cause else ''}] {c.claim} ({c.reason})"
        for c in claims
        if c.label != "supported"
    ]
    missing = list(suggested_checks)

    # Supported claims that lean on evidence the hypothesis forgot to cite.
    cited = set(citations)
    for c in claims:
        uncited = [e for e in c.evidence_ids if e not in cited]
        if c.label == "supported" and uncited:
            missing.append(f"cite {', '.join(uncited)} for: {c.claim}")

    if any(c.label == "contradicted" for c in claims) or any(c.label == "unsupported" for c in root):
        decision = "reject"
    elif not root:
        decision = "need_more_evidence"
        reasons.append("the explanation never states a root cause")
    elif all(c.label == "supported" for c in root) and len(weak) <= max_weak:
        decision = "accept"
    else:
        decision = "need_more_evidence"

    if not reasons:
        reasons = ["all claims supported by the evidence"]
    return CriticVerdict(decision=decision, reasons=reasons, missing_evidence=missing)


class LLMCritic:
    def __init__(
        self,
        llm: LLMClient,
        evidence_mode: str = "full",
        comparisons: bool = True,
        max_weak: int = 1,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm
        self._evidence_mode = evidence_mode
        self._comparisons = comparisons
        self._max_weak = max_weak
        self._temperature = temperature
        # Every critic LLM call, so the pipeline can record them in RunRecord.llm_calls.
        self.calls: list[LLMResponse] = []
        # Claim checks from the most recent review(), for debugging and the grounding analysis.
        self.last_claims: list[ClaimCheck] = []

    def review(self, hypothesis: Hypothesis, ledger: EvidenceLedger) -> CriticVerdict:
        self.last_claims = []

        # 1. Free code check: cited IDs must exist.
        if not hypothesis.citations:
            return CriticVerdict(
                decision="reject",
                reasons=["the hypothesis cites no evidence"],
                missing_evidence=["cite the evidence IDs that show the root cause"],
            )
        unknown = [e for e in hypothesis.citations if e not in ledger.known_ids()]
        if unknown:
            return CriticVerdict(
                decision="reject",
                reasons=[f"cited evidence IDs not in the ledger: {', '.join(unknown)}"],
                missing_evidence=["cite only evidence IDs that exist in the ledger"],
            )

        # 2. One LLM call: claims + labels.
        evidence_text = format_evidence(
            ledger, hypothesis.citations, mode=self._evidence_mode, comparisons=self._comparisons
        )
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=build_user_prompt(hypothesis, evidence_text)),
        ]
        response = self._llm.generate(messages, response_schema=ClaimCheckResult, temperature=self._temperature)
        self.calls.append(response)
        try:
            result = ClaimCheckResult.model_validate_json(response.text)
        except ValidationError as e:
            # Fail open: a broken critic reply must not count against the hypothesis. The run
            # then behaves like the no-critic config, and the reason makes it visible in eval.
            return CriticVerdict(
                decision="accept",
                reasons=[f"critic output could not be parsed, accepted unchecked: {e.error_count()} errors"],
            )

        # 3. Code clean-up: field checks first, invented claims and IDs dropped, comparative cap.
        claims = assemble_claims(result, hypothesis, ledger.known_ids())
        self.last_claims = claims

        # 4. Verdict from code rules.
        return decide(claims, result.suggested_checks, hypothesis.citations, max_weak=self._max_weak)
