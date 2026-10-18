"""metric_query tool: compact stats for one service's metrics over a time window.

When the window spans the incident onset (case.inject_time, which the agent is already given as the
approximate onset), each metric is split into before/after the onset and ranked by how far it shifted
-- whole-window min/max/mean hides a 100x CPU jump behind a service that is merely busy all the time.

Hard rule #2: never return the raw series, only a capped summary.
"""

from __future__ import annotations

import math

import pandas as pd

from src.data.case import Case
from src.llm.base import ToolSpec

SPEC = ToolSpec(
    name="metric_query",
    description=(
        "Get summary statistics for a service's metrics in a time window. If the window spans the "
        "incident onset, each metric is compared before vs after onset and ranked by how much it "
        "shifted. Use this to check whether a service's CPU/memory/latency/etc looks abnormal."
    ),
    parameters={
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "Service name, e.g. 'cartservice'"},
            "start_time": {"type": "number", "description": "Unix timestamp, window start"},
            "end_time": {"type": "number", "description": "Unix timestamp, window end"},
        },
        "required": ["service", "start_time", "end_time"],
    },
)

_MAX_METRICS_REPORTED = 8


def make_metric_query(case: Case):
    def metric_query(service: str, start_time: float, end_time: float) -> str:
        df = case.metrics
        window = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
        service_cols = [c for c in df.columns if c.startswith(f"{service}_")]

        if not service_cols:
            return f"No metrics found for service '{service}'. Known services: {', '.join(case.services)}"
        if window.empty:
            return f"No metric samples for '{service}' in window [{start_time}, {end_time}]."

        before = window[window["time"] < case.inject_time]
        after = window[window["time"] >= case.inject_time]
        if before.empty or after.empty:
            return _whole_window_summary(service, service_cols, window, start_time, end_time)

        rows = []
        for col in service_cols:
            b, a = before[col].dropna(), after[col].dropna()
            if b.empty or a.empty:
                continue
            rows.append((_shift_score(b, a), col[len(service) + 1 :], b.mean(), a.mean(), a.max()))
        rows.sort(key=lambda r: r[0], reverse=True)

        lines = [
            f"Metrics for '{service}' in [{start_time:.0f}, {end_time:.0f}], split at onset "
            f"{case.inject_time:.0f} ({len(before)} samples before, {len(after)} after), "
            f"most-shifted first. shift = |after mean - before mean| / before std:"
        ]
        for score, name, b_mean, a_mean, a_max in rows[:_MAX_METRICS_REPORTED]:
            lines.append(
                f"  - {service}_{name}: before mean={b_mean:.2f} -> after mean={a_mean:.2f} "
                f"(x{_ratio(b_mean, a_mean)}, after max={a_max:.2f}) shift={score:.1f}"
            )
        if len(rows) > _MAX_METRICS_REPORTED:
            lines.append(f"  ... ({len(rows) - _MAX_METRICS_REPORTED} more metrics omitted)")
        return "\n".join(lines)

    return metric_query


def _shift_score(before: pd.Series, after: pd.Series) -> float:
    # Floor the std at 5% of the baseline mean: integer-valued metrics (e.g. socket counts) are
    # often constant before onset, and an unfloored std of ~0 turns a 4 -> 4.03 wobble into a huge
    # score that outranks a real 40x CPU jump.
    std = max(before.std(ddof=0), 0.05 * abs(before.mean()), 1e-9)
    score = abs(after.mean() - before.mean()) / std
    return min(score, 1e6) if math.isfinite(score) else 0.0


def _ratio(before_mean: float, after_mean: float) -> str:
    if abs(before_mean) < 1e-9:
        return "n/a"
    return f"{after_mean / before_mean:.2f}"


def _whole_window_summary(service: str, service_cols: list[str], window: pd.DataFrame, start: float, end: float) -> str:
    lines = [
        f"Metrics for '{service}' in [{start:.0f}, {end:.0f}] ({len(window)} samples, window does not span onset):"
    ]
    for col in service_cols[:_MAX_METRICS_REPORTED]:
        series = window[col].dropna()
        if series.empty:
            continue
        lines.append(
            f"  - {col}: min={series.min():.2f} max={series.max():.2f} "
            f"mean={series.mean():.2f} last={series.iloc[-1]:.2f}"
        )
    if len(service_cols) > _MAX_METRICS_REPORTED:
        lines.append(f"  ... ({len(service_cols) - _MAX_METRICS_REPORTED} more metrics omitted)")
    return "\n".join(lines)
