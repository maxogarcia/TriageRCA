"""Evidence ledger (hard rule #3): every tool result and RAG chunk gets a stable ID (E001, ...).

The report cites these IDs and the grounding judge scores claims against exactly this ledger,
so completeness matters more than convenience here -- always go through EvidenceLedger.add(),
never build EvidenceItem lists by hand elsewhere.
"""

from __future__ import annotations

import re
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


# One summary line of a metric_query result, e.g.
#   "  - checkoutservice_cpu: min=0.39 max=20.03 mean=18.09 last=19.99"
_METRIC_LINE = re.compile(
    r"^\s*-\s*(?P<service>[A-Za-z0-9]+)_(?P<metric>[A-Za-z0-9.\-]+):"
    r"\s*min=(?P<min>[-\d.eE+]+)\s+max=(?P<max>[-\d.eE+]+)\s+mean=(?P<mean>[-\d.eE+]+)"
)


def compare_metrics(ledger: EvidenceLedger, only_ids: set[str] | None = None) -> str:
    """Cross-service comparison table built only from metric_query items already in the ledger.

    For each metric seen for at least two services, list mean / max per service, highest mean
    first. Values only, no ratios: a large cross-service gap can be normal (services do different
    work), so the table states facts and leaves the interpretation to the critic. It adds no
    information the system did not retrieve. only_ids limits it to items the critic is shown
    (cited mode). Returns "" when there is nothing to compare.
    """
    rows: dict[str, list[tuple[str, float, float, str]]] = {}
    for item in ledger.items:
        if item.source != "metric_query" or (only_ids is not None and item.id not in only_ids):
            continue
        for line in item.content.splitlines():
            m = _METRIC_LINE.match(line)
            if m:
                rows.setdefault(m["metric"], []).append((m["service"], float(m["mean"]), float(m["max"]), item.id))

    lines = []
    for metric in sorted(rows):
        entries = sorted(rows[metric], key=lambda r: r[1], reverse=True)
        if len({svc for svc, *_ in entries}) < 2:
            continue
        values = ", ".join(f"{svc} {mean:g} / {mx:g} [{eid}]" for svc, mean, mx, eid in entries)
        lines.append(f"- {metric} (mean / max): {values}")
    if not lines:
        return ""
    return (
        "## Same metric across services (computed from the metric_query items above; "
        "query windows may differ, see each item's args)\n" + "\n".join(lines)
    )


def format_evidence(
    ledger: EvidenceLedger,
    cited_ids: list[str],
    mode: str = "full",
    max_chars_per_item: int = 2000,
    comparisons: bool = False,
) -> str:
    """Render the ledger as text for the critic prompt.

    mode="cited": only items the hypothesis cites.
    mode="full":  cited items first, then every other item (spec v1.1 section 3.1: the evidence
                  set is everything the system saw, not only what it chose to cite).
    Cited IDs missing from the ledger are skipped; the critic's ID check reports them.
    comparisons=True appends compare_metrics() (critic_comparisons ablation switch).
    """
    if mode not in ("cited", "full"):
        raise ValueError(f"Unknown evidence mode: {mode!r} (expected 'cited' or 'full')")

    def render(item: EvidenceItem) -> str:
        args = ", ".join(f"{k}={v}" for k, v in sorted(item.args.items()))
        content = item.content
        if len(content) > max_chars_per_item:
            content = content[:max_chars_per_item] + "\n  ... (truncated)"
        return f"[{item.id}] {item.source} ({args})\n{content}"

    cited_set = set(cited_ids)
    cited = [i for i in ledger.items if i.id in cited_set]
    sections = ["## Cited evidence", "\n\n".join(render(i) for i in cited) or "(none)"]
    if mode == "full":
        others = [i for i in ledger.items if i.id not in cited_set]
        sections += ["## Other evidence (not cited)", "\n\n".join(render(i) for i in others) or "(none)"]
    if comparisons:
        table = compare_metrics(ledger, only_ids=None if mode == "full" else cited_set)
        if table:
            sections.append(table)
    return "\n\n".join(sections)
