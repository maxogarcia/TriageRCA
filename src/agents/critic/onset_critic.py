"""Code-rule-only critic: ablation row "code rule only".

The hybrid critic with the LLM removed: the same free citation check, then the same onset rule
(onset_conflict), with the same reject verdict as the hybrid's onset check (onset_reject_verdict).
Accepts otherwise, including when the onset rule is skipped (fewer than two services with onset
data, or the claimed root has none). The indicator is not checked on its own, and this critic
never returns need_more_evidence.
"""

from __future__ import annotations

from src.agents.critic.llm_critic import citation_check, onset_conflict, onset_reject_verdict
from src.evidence import EvidenceLedger, onset_shift_ranking
from src.schemas import CriticVerdict, Hypothesis


class OnsetRuleCritic:
    def __init__(self, evidence_mode: str = "full"):
        if evidence_mode not in ("full", "cited"):
            raise ValueError(f"evidence_mode must be 'full' or 'cited', got {evidence_mode!r}")
        self._evidence_mode = evidence_mode

    def review(self, hypothesis: Hypothesis, ledger: EvidenceLedger) -> CriticVerdict:
        cite_verdict = citation_check(hypothesis, ledger)
        if cite_verdict is not None:
            return cite_verdict
        only_ids = None if self._evidence_mode == "full" else set(hypothesis.citations)
        conflict = onset_conflict(hypothesis, ledger, only_ids)
        if conflict is not None:
            return onset_reject_verdict(conflict)
        root = hypothesis.ranked_services[0].strip().lower() if hypothesis.ranked_services else None
        ranked = {svc.lower() for svc, _, _, _ in onset_shift_ranking(ledger, only_ids)}
        if root is None or len(ranked) < 2 or root not in ranked:
            reason = "code onset rule skipped: fewer than two services with onset data, or the claimed root has none"
        else:
            reason = f"code onset rule passed: no other examined service shifted more at onset than {root}"
        return CriticVerdict(decision="accept", reasons=[reason], missing_evidence=[])
