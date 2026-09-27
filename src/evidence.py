"""Evidence ledger (hard rule #3): every tool result and RAG chunk gets a stable ID (E001, ...).

The report cites these IDs and the grounding judge scores claims against exactly this ledger,
so completeness matters more than convenience here -- always go through EvidenceLedger.add(),
never build EvidenceItem lists by hand elsewhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    id: str  # "E001"
    source: str  # "metric_query" | "log_search" | "trace_query" | "rag" | "input"
    args: dict
    content: str  # exactly what the agent saw
    created_at: float


@dataclass
class EvidenceLedger:
    items: list[EvidenceItem] = field(default_factory=list)

    def add(self, source: str, args: dict, content: str) -> EvidenceItem:
        item = EvidenceItem(
            id=f"E{len(self.items) + 1:03d}",
            source=source,
            args=args,
            content=content,
            created_at=time.time(),
        )
        self.items.append(item)
        return item

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.id == evidence_id), None)

    def known_ids(self) -> set[str]:
        return {i.id for i in self.items}
