from __future__ import annotations

import json

import pytest

from src.agents.critic.llm_critic import LLMCritic
from src.evidence import EvidenceLedger, compare_metrics, format_evidence
from src.llm.base import LLMResponse
from src.llm.fake import FakeLLMClient
from src.schemas import ClaimCheckResult, Hypothesis


def _ledger() -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.add(
        "metric_query",
        {"service": "cartservice"},
        "Metrics for 'cartservice':\n  - cartservice_cpu: min=10.00 max=95.00 mean=90.00 last=95.00",
    )
    ledger.add(
        "metric_query",
        {"service": "frontend"},
        "Metrics for 'frontend':\n  - frontend_cpu: min=4.00 max=6.00 mean=5.00 last=5.00",
    )
    ledger.add("log_search", {"service": "cartservice"}, "OOM killed process")
    return ledger


def _hypothesis(citations: list[str] | None = None) -> Hypothesis:
    return Hypothesis(
        ranked_services=["cartservice", "frontend"],
        indicator="cartservice_cpu",
        fault_category="cpu",
        explanation="cartservice CPU was at 90 while frontend stayed at 5. The cart logs show an OOM kill.",
        citations=["E001"] if citations is None else citations,
        mitigation="scale cartservice",
        confidence=0.9,
    )


def _claim(
    label: str,
    root: bool = False,
    ids: list[str] | None = None,
    quote: str = "cartservice CPU was at 90",
    text: str | None = None,
    comparison: str = "frontend 5 [E002]",
) -> dict:
    return {
        "quote": quote,
        "claim": text or f"a {label} claim",
        "is_root_cause": root,
        "evidence_ids": ["E001"] if ids is None else ids,
        "comparison": comparison,
        "reason": "because",
        "label": label,
    }


def _field(label: str = "supported", comparison: str = "frontend 5 [E002]") -> dict:
    return {"evidence_ids": ["E001"], "comparison": comparison, "reason": "because", "label": label}


def _reply(
    claims: list[dict],
    checks: list[str] | None = None,
    root: dict | None = None,
    indicator: dict | None = None,
) -> LLMResponse:
    body = {
        "root_cause_check": root or _field(),
        "indicator_check": indicator or _field(),
        "claims": claims,
        "suggested_checks": checks or [],
    }
    return LLMResponse(
        text=json.dumps(body),
        tool_calls=[],
        input_tokens=10,
        output_tokens=10,
        latency_s=0.01,
        model="fake",
        cached=False,
    )


def _review(claims: list[dict], checks: list[str] | None = None, root=None, indicator=None, **kwargs):
    llm = FakeLLMClient([_reply(claims, checks, root, indicator)])
    critic = LLMCritic(llm=llm, **kwargs)
    return critic.review(_hypothesis(), _ledger()), critic, llm


# --- 1. code check, no LLM call -------------------------------------------------------------


def test_unknown_citation_rejects_without_llm_call():
    llm = FakeLLMClient([])  # any generate() call would raise
    verdict = LLMCritic(llm=llm).review(_hypothesis(["E001", "E009"]), _ledger())
    assert verdict.decision == "reject"
    assert "E009" in verdict.reasons[0]
    assert llm.calls == []


def test_no_citations_rejects_without_llm_call():
    llm = FakeLLMClient([])
    verdict = LLMCritic(llm=llm).review(_hypothesis([]), _ledger())
    assert verdict.decision == "reject"
    assert llm.calls == []


# --- 2. the LLM call ------------------------------------------------------------------------


def test_llm_call_uses_schema_temperature_zero_and_full_evidence():
    _, critic, llm = _review([_claim("supported", root=True)])
    call = llm.calls[0]
    assert call["response_schema"] is ClaimCheckResult
    assert call["params"]["temperature"] == 0.0
    user_prompt = call["messages"][1].content
    assert "[E002]" in user_prompt  # uncited item is visible in full mode
    assert len(critic.calls) == 1


def test_cited_mode_hides_uncited_evidence():
    _, _, llm = _review([_claim("supported", root=True)], evidence_mode="cited")
    assert "[E002]" not in llm.calls[0]["messages"][1].content


def test_unparseable_reply_fails_open():
    bad = LLMResponse(
        text="not json", tool_calls=[], input_tokens=1, output_tokens=1, latency_s=0.0, model="fake", cached=False
    )
    verdict = LLMCritic(llm=FakeLLMClient([bad])).review(_hypothesis(), _ledger())
    assert verdict.decision == "accept"
    assert "could not be parsed" in verdict.reasons[0]


def test_invented_evidence_ids_are_dropped():
    _, critic, _ = _review([_claim("supported", root=True, ids=["E001", "E042"])])
    assert critic.last_claims[2].evidence_ids == ["E001"]


# --- 3. verdict rules -----------------------------------------------------------------------


def test_all_supported_accepts():
    verdict, _, _ = _review([_claim("supported", root=True), _claim("supported")])
    assert verdict.decision == "accept"


def test_one_weak_side_claim_still_accepts():
    verdict, _, _ = _review([_claim("supported", root=True), _claim("partial")])
    assert verdict.decision == "accept"


def test_weak_side_claims_do_not_block_accept():
    # Fields decide the verdict; weak explanation claims only add reasons.
    verdict, _, _ = _review([_claim("supported", root=True), _claim("partial"), _claim("unsupported")])
    assert verdict.decision == "accept"
    assert any("a partial claim" in r for r in verdict.reasons)
    assert any("an unsupported claim" in r or "a unsupported claim" in r for r in verdict.reasons)


def test_contradicted_field_rejects():
    verdict, _, _ = _review([_claim("supported", root=True)], root=_field("contradicted"))
    assert verdict.decision == "reject"


def test_unsupported_root_cause_rejects():
    verdict, _, _ = _review([_claim("supported", root=True)], root=_field("unsupported"))
    assert verdict.decision == "reject"


def test_partial_root_cause_needs_more_evidence_and_passes_checks():
    verdict, _, _ = _review([_claim("supported", root=True)], root=_field("partial"),
                            checks=["metric_query cartservice cpu"])
    assert verdict.decision == "need_more_evidence"
    assert "metric_query cartservice cpu" in verdict.missing_evidence


def test_field_checks_come_first_and_are_root_claims():
    _, critic, _ = _review([_claim("supported")])
    first, second = critic.last_claims[:2]
    assert first.claim == "The root cause is cartservice." and first.is_root_cause
    assert second.claim == "cartservice_cpu is the metric that best shows the fault." and second.is_root_cause


def test_contradicted_indicator_rejects():
    verdict, _, _ = _review([_claim("supported", root=True)], indicator=_field("contradicted"))
    assert verdict.decision == "reject"
    assert any("best shows the fault" in r for r in verdict.reasons)


def test_field_check_without_comparison_is_capped():
    verdict, critic, _ = _review([_claim("supported", root=True)], root=_field("supported", comparison=""))
    assert critic.last_claims[0].label == "partial"
    assert verdict.decision == "need_more_evidence"


# --- 4. code clean-up of the model's claims -------------------------------------------------


def test_claim_not_quoted_from_explanation_is_dropped():
    invented = _claim("supported", quote="shippingservice handled many GetQuote requests")
    _, critic, _ = _review([_claim("supported", root=True), invented])
    assert len(critic.last_claims) == 3  # 2 field claims + the real one


def test_quote_match_ignores_case_and_punctuation():
    _, critic, _ = _review([_claim("supported", root=True, quote="The cart logs show an OOM kill")])
    assert len(critic.last_claims) == 3


def test_comparative_claim_without_comparison_is_capped():
    claim = _claim("supported", text="cartservice CPU showed a significant increase", comparison="")
    _, critic, _ = _review([_claim("supported", root=True), claim])
    assert critic.last_claims[3].label == "partial"
    assert "capped" in critic.last_claims[3].reason


def test_comparison_without_a_value_is_capped():
    claim = _claim("supported", text="cartservice CPU showed a significant increase", comparison="frontend_cpu")
    _, critic, _ = _review(
        [_claim("supported", root=True), claim], root=_field("supported", comparison="cartservice_cpu")
    )
    assert critic.last_claims[0].label == "partial"
    assert critic.last_claims[3].label == "partial"


def test_plain_value_claim_without_comparison_stays_supported():
    claim = _claim("supported", text="cartservice CPU mean was 90", comparison="")
    _, critic, _ = _review([_claim("supported", root=True), claim])
    assert critic.last_claims[3].label == "supported"


def test_supported_claim_on_uncited_evidence_asks_for_citation():
    verdict, _, _ = _review([_claim("supported", root=True), _claim("supported", ids=["E003"])])
    assert verdict.decision == "accept"
    assert any(m.startswith("cite E003") for m in verdict.missing_evidence)


# --- format_evidence / compare_metrics -------------------------------------------------------


def test_comparison_table_is_built_from_metric_items():
    table = compare_metrics(_ledger())
    assert "- cpu (mean / max): cartservice 90 / 95 [E001], frontend 5 / 6 [E002]" in table
    assert "median" not in table  # values only, no interpretation


def test_comparison_table_skips_metrics_seen_for_one_service():
    ledger = EvidenceLedger()
    ledger.add("metric_query", {}, "  - cartservice_cpu: min=1 max=2 mean=1.5 last=2")
    assert compare_metrics(ledger) == ""


def test_comparisons_switch_controls_the_table_in_the_prompt():
    _, _, llm_on = _review([_claim("supported", root=True)], comparisons=True)
    _, _, llm_off = _review([_claim("supported", root=True)], comparisons=False)
    assert "## Same metric across services" in llm_on.calls[0]["messages"][1].content
    assert "## Same metric across services" not in llm_off.calls[0]["messages"][1].content


def test_format_evidence_modes():
    ledger = _ledger()
    cited = format_evidence(ledger, ["E002"], mode="cited")
    full = format_evidence(ledger, ["E002"], mode="full")
    assert "[E002]" in cited and "[E001]" not in cited
    assert full.index("[E002]") < full.index("[E001]")  # cited first
    with pytest.raises(ValueError):
        format_evidence(ledger, [], mode="all")


def test_format_evidence_truncates_long_items():
    ledger = EvidenceLedger()
    ledger.add("log_search", {}, "x" * 5000)
    text = format_evidence(ledger, ["E001"], max_chars_per_item=100)
    assert "(truncated)" in text and len(text) < 300


# --- failed_checks (field-aware feedback) ---

def _failed_claim(label, quote="(hypothesis field)", is_root=True):
    from src.schemas import ClaimCheck
    return ClaimCheck(quote=quote, claim="c", is_root_cause=is_root, evidence_ids=["E001"],
                      comparison="E001 5 vs E002 1", reason="r", label=label)


@pytest.mark.parametrize("root,ind,expected", [
    ("supported", "supported", []),
    ("supported", "contradicted", ["indicator"]),
    ("unsupported", "contradicted", ["root_cause", "indicator"]),
    ("partial", "supported", ["root_cause"]),
])
def test_decide_failed_checks(root, ind, expected):
    from src.agents.critic.llm_critic import decide
    verdict = decide([_failed_claim(root), _failed_claim(ind)], [], ["E001"])
    assert verdict.failed_checks == expected


def test_decide_explanation_claims_do_not_set_verdict():
    from src.agents.critic.llm_critic import decide
    claims = [_failed_claim("supported"), _failed_claim("supported"),
              _failed_claim("contradicted", quote="some explanation text", is_root=False)]
    verdict = decide(claims, [], ["E001"])
    assert verdict.decision == "accept"  # fields decide; side claims only add reasons
    assert verdict.failed_checks == []


# Regression: metric_overview lines prefix the service ("svc: svc_metric, before ..."). The doubled
# name the Executor sometimes writes in its indicator (svc_svc_metric) does not come from parsing;
# the service and metric must still split correctly here.
def _overview_ledger():
    ledger = EvidenceLedger()
    item = ledger.add(
        "metric_overview",
        {"start_time": 0.0, "end_time": 600.0},
        "Onset overview:\n"
        "  - productcatalogservice: productcatalogservice_latency-90, before mean=0.004369 -> after mean=0.2033; "
        "direction=increase, change=+4551.8%, x46.52, shift=910.4, n=300/301\n"
        "  - frontend: frontend_latency-50, before mean=0.0100 -> after mean=0.0520; "
        "direction=increase, change=+420.0%, x5.20, shift=516.8, n=300/301\n"
        "  - frontend-external: frontend-external_cpu, before mean=1.00 -> after mean=1.10; "
        "direction=increase, change=+10.0%, x1.10, shift=3.5, n=300/301",
    )
    return ledger, item.id


def test_onset_ranking_parses_service_prefixed_overview_lines():
    from src.evidence import onset_shift_ranking

    ledger, eid = _overview_ledger()
    assert onset_shift_ranking(ledger) == [
        ("productcatalogservice", "latency-90", 910.4, eid),
        ("frontend", "latency-50", 516.8, eid),
        ("frontend-external", "cpu", 3.5, eid),
    ]


def test_compare_metrics_largest_shift_line_puts_top_service_first():
    ledger, eid = _overview_ledger()
    first = compare_metrics(ledger).splitlines()[1]
    assert first.startswith(f"Largest shift per service: productcatalogservice latency-90 910.4 [{eid}], frontend")


def test_fmt_shift_marks_capped_values():
    from src.evidence import SHIFT_CAP, _fmt_shift

    assert SHIFT_CAP == 1e6
    assert _fmt_shift(1e6) == ">=1e6 (capped)"
    assert _fmt_shift(2e6) == ">=1e6 (capped)"
    assert _fmt_shift(910.4) == "910.4"


def test_decide_field_partial_needs_more_evidence():
    from src.agents.critic.llm_critic import decide
    verdict = decide([_failed_claim("partial"), _failed_claim("supported")], [], ["E001"])
    assert verdict.decision == "need_more_evidence"
    assert verdict.failed_checks == ["root_cause"]


def test_decide_field_indicator_unsupported_rejects():
    from src.agents.critic.llm_critic import decide
    verdict = decide([_failed_claim("supported"), _failed_claim("unsupported")], [], ["E001"])
    assert verdict.decision == "reject"
    assert verdict.failed_checks == ["indicator"]


def test_decide_without_field_claims_uses_all_claims_rule():
    from src.agents.critic.llm_critic import decide
    side = lambda lab, root: _failed_claim(lab, quote="explanation text", is_root=root)
    assert decide([side("supported", True), side("contradicted", False)], [], ["E001"]).decision == "reject"
    assert decide([side("supported", True), side("partial", False)], [], ["E001"]).decision == "accept"
    assert decide([side("supported", True), side("partial", False), side("partial", False)],
                  [], ["E001"]).decision == "need_more_evidence"


# OnsetRuleCritic (ablation row "code rule only") shares citation_check and onset_conflict
# with the hybrid critic.
def _h(root, citations):
    return _hypothesis().model_copy(update={"ranked_services": [root], "citations": citations})


def test_onset_rule_rejects_when_other_service_shifted_more():
    from src.agents.critic.onset_critic import OnsetRuleCritic

    ledger, eid = _overview_ledger()
    v = OnsetRuleCritic().review(_h("frontend", [eid]), ledger)
    assert v.decision == "reject"
    assert v.failed_checks == ["root_cause", "indicator"]
    assert v.reasons[0].startswith("code onset check: productcatalogservice")
    assert v.missing_evidence[0].startswith("investigate productcatalogservice")


def test_onset_rule_accepts_when_root_is_top():
    from src.agents.critic.onset_critic import OnsetRuleCritic

    ledger, eid = _overview_ledger()
    v = OnsetRuleCritic().review(_h("productcatalogservice", [eid]), ledger)
    assert v.decision == "accept" and v.failed_checks == []
    assert v.reasons[0].startswith("code onset rule passed")


def test_onset_rule_skips_when_root_has_no_onset_data():
    from src.agents.critic.onset_critic import OnsetRuleCritic

    ledger, eid = _overview_ledger()
    v = OnsetRuleCritic().review(_h("cartservice", [eid]), ledger)
    assert v.decision == "accept"
    assert v.reasons[0].startswith("code onset rule skipped")


def test_onset_rule_cited_mode_only_uses_cited_items():
    from src.agents.critic.onset_critic import OnsetRuleCritic

    ledger, _ = _overview_ledger()
    other = ledger.add("log_search", {"service": "frontend"}, "no matching log lines")
    h = _h("frontend", [other.id])
    assert OnsetRuleCritic(evidence_mode="cited").review(h, ledger).decision == "accept"
    assert OnsetRuleCritic(evidence_mode="full").review(h, ledger).decision == "reject"


def test_onset_rule_matches_hybrid_onset_check():
    from src.agents.critic.llm_critic import _apply_onset_check
    from src.agents.critic.onset_critic import OnsetRuleCritic

    ledger, eid = _overview_ledger()
    for root in ["frontend", "productcatalogservice", "frontend-external", "cartservice"]:
        h = _h(root, [eid])
        _, hint = _apply_onset_check([], h, ledger, None)
        v = OnsetRuleCritic().review(h, ledger)
        assert (v.decision == "reject") == (hint is not None), root
        if hint is not None:
            assert v.missing_evidence == [hint], root


def test_onset_rule_shares_citation_checks_with_hybrid():
    from src.agents.critic.llm_critic import citation_check
    from src.agents.critic.onset_critic import OnsetRuleCritic

    ledger, eid = _overview_ledger()
    for citations in [[], ["E999"], [eid, "E999"]]:
        h = _h("productcatalogservice", citations)
        expected = citation_check(h, ledger)
        assert expected is not None and expected.decision == "reject", citations
        assert OnsetRuleCritic().review(h, ledger) == expected, citations


def test_parse_failure_still_applies_onset_check():
    from types import SimpleNamespace

    from src.agents.critic.llm_critic import LLMCritic

    class BrokenLLM:
        def generate(self, messages, **kwargs):
            return SimpleNamespace(text="not json")

    ledger, eid = _overview_ledger()
    hybrid = LLMCritic(llm=BrokenLLM(), evidence_mode="full", comparisons=True, onset_check=True)
    v = hybrid.review(_h("frontend", [eid]), ledger)
    assert v.decision == "reject" and v.failed_checks == ["root_cause", "indicator"]
    assert v.reasons[0].startswith("critic output could not be parsed")
    assert hybrid.review(_h("productcatalogservice", [eid]), ledger).decision == "accept"
    llm_only = LLMCritic(llm=BrokenLLM(), evidence_mode="full", comparisons=True, onset_check=False)
    assert llm_only.review(_h("frontend", [eid]), ledger).decision == "accept"
