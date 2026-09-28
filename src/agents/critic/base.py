"""Critic interface + a stub implementation.

Aeon owns the real Critic (checks hypotheses against retrieved evidence, spec grounding rules).
AlwaysAcceptCritic exists only so the Planner/Executor loop and the triagerca_no_critic-shaped
control flow are testable before that lands.
"""

from __future__ import annotations

from typing import Protocol

from src.evidence import EvidenceLedger
from src.schemas import CriticVerdict, Hypothesis


class Critic(Protocol):
    def review(self, hypothesis: Hypothesis, ledger: EvidenceLedger) -> CriticVerdict: ...


class AlwaysAcceptCritic:
    def review(self, hypothesis: Hypothesis, ledger: EvidenceLedger) -> CriticVerdict:
        return CriticVerdict(decision="accept", reasons=["stub critic: always accepts"], missing_evidence=[])
