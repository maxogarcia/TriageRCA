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


# Metric lines the tools print. Service names may contain "-" (frontend-external); metric names
# contain no "_", so the first "_" splits service from metric.
_SVC_METRIC = r"(?P<service>[A-Za-z0-9\-]+)_(?P<metric>[A-Za-z0-9.\-]+)"
_NUM = r"[-+\d.eE]+"

# Whole-window line (metric_query without onset coverage), e.g.
#   "  - checkoutservice_cpu: min=0.39 max=20.03 mean=18.09 last=19.99"
_METRIC_LINE = re.compile(
    rf"^\s*-\s*{_SVC_METRIC}:\s*min=(?P<min>{_NUM})\s+max=(?P<max>{_NUM})\s+mean=(?P<mean>{_NUM})"
)

# Before/after line from metric_query or metric_overview, e.g.
#   "  - checkoutservice_cpu: before mean=0.46 -> after mean=18.29 (x39.43, ..., change=+3843.0%, ...) shift=453.6"
#   "  - checkoutservice: checkoutservice_cpu, before mean=0.4527 -> after mean=17.12; ..., change=+3682.9%, x37.83, shift=207.9, n=300/301"
_SHIFT_LINE = re.compile(
    rf"^\s*-\s*(?:[A-Za-z0-9\-]+:\s*)?{_SVC_METRIC}[:,]\s*before mean=(?P<before>{_NUM})\s*->\s*after mean=(?P<after>{_NUM})"
)
_CHANGE = re.compile(r"change=(?P<change>[-+\d.]+%)")
_SHIFT = re.compile(rf"shift=(?P<shift>{_NUM})")

_TABLE_SOURCES = ("metric_query", "metric_overview")


def _parse_metric_items(
    ledger: EvidenceLedger, only_ids: set[str] | None = None
) -> tuple[dict[tuple[str, str], tuple[float, float, str, float, str]], dict[str, list[tuple[str, float, float, str]]]]:
    """Parse before/after (shifts) and whole-window (window) metric lines from table-source items."""
    shifts: dict[tuple[str, str], tuple[float, float, str, float, str]] = {}
    window: dict[str, list[tuple[str, float, float, str]]] = {}
    for item in ledger.items:
        if item.source not in _TABLE_SOURCES or (only_ids is not None and item.id not in only_ids):
            continue
        for line in item.content.splitlines():
            s = _SHIFT_LINE.match(line)
            if s:
                ch = _CHANGE.search(line)
                sh = _SHIFT.search(line)
                shift = float(sh["shift"]) if sh else 0.0
                key = (s["service"], s["metric"])
                if key not in shifts or shift > shifts[key][3]:
                    shifts[key] = (float(s["before"]), float(s["after"]), ch["change"] if ch else "?", shift, item.id)
                continue
            m = _METRIC_LINE.match(line)
            if m:
                window.setdefault(m["metric"], []).append((m["service"], float(m["mean"]), float(m["max"]), item.id))
    return shifts, window


def _best_shift_per_service(
    shifts: dict[tuple[str, str], tuple[float, float, str, float, str]]
) -> dict[str, tuple[str, float, str]]:
    """Largest onset shift per service: service -> (metric, shift, evidence id)."""
    best: dict[str, tuple[str, float, str]] = {}
    for (svc, metric), (_, _, _, shift, eid) in shifts.items():
        if svc not in best or shift > best[svc][1]:
            best[svc] = (metric, shift, eid)
    return best


SHIFT_CAP = 1e6  # must match the cap in src/tools/metric_query.py


def _fmt_shift(shift: float) -> str:
    """Shift as shown to the critic; capped values (pre-onset std ~0) are marked so they read as huge."""
    return ">=1e6 (capped)" if shift >= SHIFT_CAP else f"{shift:g}"


def onset_shift_ranking(
    ledger: EvidenceLedger, only_ids: set[str] | None = None
) -> list[tuple[str, str, float, str]]:
    """Services ranked by largest onset shift, largest first: [(service, metric, shift, evidence id)].

    Same numbers as the "Largest shift per service" line in compare_metrics. Callers should
    require at least two entries before comparing, as compare_metrics does.
    """
    best = _best_shift_per_service(_parse_metric_items(ledger, only_ids)[0])
    ranking = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)
    return [(svc, metric, shift, eid) for svc, (metric, shift, eid) in ranking]


def compare_metrics(ledger: EvidenceLedger, only_ids: set[str] | None = None) -> str:
    """Cross-service comparison table built only from metric items already in the ledger.

    Section 1 (onset comparison, from before/after lines in metric_query and metric_overview):
    the largest shift per service, then, for each metric seen for at least two services, the
    before -> after mean, change and shift per service. Section 2 (whole window, from metric_query
    items without pre-onset samples): mean / max per service, which cannot show an onset change.
    Every number is copied from a tool result, so the table adds no information the system did
    not retrieve; a large cross-service gap can be normal, so interpretation is left to the critic.
    only_ids limits it to items the critic is shown (cited mode). Returns "" when there is nothing
    to compare.
    """
    shifts, window = _parse_metric_items(ledger, only_ids)

    sections = []

    best = _best_shift_per_service(shifts)
    if len(best) >= 2:
        ranking = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)
        lines = ["Largest shift per service: " + ", ".join(
            f"{svc} {metric} {_fmt_shift(shift)} [{eid}]" for svc, (metric, shift, eid) in ranking
        )]
        by_metric: dict[str, list] = {}
        for (svc, metric), vals in shifts.items():
            by_metric.setdefault(metric, []).append((svc, *vals))
        for metric in sorted(by_metric):
            entries = sorted(by_metric[metric], key=lambda r: r[4], reverse=True)
            if len(entries) < 2:
                continue
            values = ", ".join(
                f"{svc} {b:g} -> {a:g} ({ch}, shift {_fmt_shift(sh)}) [{eid}]" for svc, b, a, ch, sh, eid in entries
            )
            lines.append(f"- {metric} (before -> after mean): {values}")
        sections.append(
            "## Onset comparison across services (copied from before/after lines in the metric_query "
            "and metric_overview items above; shift = mean change in pre-onset std units, larger = stronger change; a shift shown as >=1e6 (capped) means the metric was flat before onset "
            "(std ~0), so it is larger than every uncapped shift)\n"
            + "\n".join(lines)
        )

    lines = []
    for metric in sorted(window):
        entries = sorted(window[metric], key=lambda r: r[1], reverse=True)
        if len({svc for svc, *_ in entries}) < 2:
            continue
        values = ", ".join(f"{svc} {mean:g} / {mx:g} [{eid}]" for svc, mean, mx, eid in entries)
        lines.append(f"- {metric} (mean / max): {values}")
    if lines:
        sections.append(
            "## Same metric across services, whole window only (from metric_query items with too few "
            "pre-onset samples, so these values cannot show an onset change; query windows may differ, "
            "see each item's args)\n" + "\n".join(lines)
        )

    return "\n\n".join(sections)


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
