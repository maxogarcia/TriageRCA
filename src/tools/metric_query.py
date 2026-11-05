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
from src.tools.validation import validate_query

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
_MIN_SAMPLES = 5


def make_metric_query(case: Case):
    def metric_query(service: str, start_time: float, end_time: float) -> str:
        validate_query(case, service, start_time, end_time)
        df = case.metrics
        window = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
        service_cols = [c for c in df.columns if c != "time" and next(
            (s for s in sorted(case.services, key=len, reverse=True) if c.startswith(s + "_")), None
        ) == service]

        if not service_cols:
            return f"No metrics found for service '{service}'. Known services: {', '.join(case.services)}"
        if window.empty:
            return f"No metric samples for '{service}' in window [{start_time}, {end_time}]."

        before = window[window["time"] < case.inject_time]
        after = window[window["time"] >= case.inject_time]
        if len(before) < _MIN_SAMPLES or len(after) < _MIN_SAMPLES:
            return (
                f"INSUFFICIENT COMPARISON COVERAGE: {len(before)} samples before, {len(after)} after; "
                f"need at least {_MIN_SAMPLES} per side. Whole-window statistics do not establish "
                "an onset change or normal behavior.\n"
                + _whole_window_summary(service, service_cols, window, start_time, end_time)
            )

        rows = []
        for col in service_cols:
            b, a = before[col].dropna(), after[col].dropna()
            if len(b) < _MIN_SAMPLES or len(a) < _MIN_SAMPLES:
                continue
            rows.append((_shift_score(b, a), col[len(service) + 1 :], b.mean(), a.mean(), a.max()))
        rows.sort(key=lambda r: r[0], reverse=True)

        lines = [
            f"Metrics for '{service}' in [{start_time:.0f}, {end_time:.0f}], split at onset "
            f"{case.inject_time:.0f} ({len(before)} samples before, {len(after)} after), "
            f"most-shifted first. shift uses before std floored at 5% of |before mean| "
            f"and 1e-9; only metrics with >= {_MIN_SAMPLES} non-null samples per side:"
        ]
        for score, name, b_mean, a_mean, a_max in rows[:_MAX_METRICS_REPORTED]:
            lines.append(
                f"  - {service}_{name}: before mean={b_mean:.2f} -> after mean={a_mean:.2f} "
                f"(x{_ratio(b_mean, a_mean)}, {_direction(b_mean, a_mean)}, "
                f"after max={a_max:.2f}) shift={score:.1f}"
            )
        if len(rows) > _MAX_METRICS_REPORTED:
            lines.append(f"  ... ({len(rows) - _MAX_METRICS_REPORTED} more metrics omitted)")
        if not rows:
            lines.append("INSUFFICIENT COMPARISON COVERAGE: no metric has enough non-null values.")
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


def _direction(before_mean: float, after_mean: float) -> str:
    delta = after_mean - before_mean
    direction = "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
    pct = f"{100 * delta / abs(before_mean):+.1f}%" if abs(before_mean) >= 1e-9 else "n/a"
    return f"direction={direction}, change={pct}"


def metric_overview(case, window_seconds: float = 300.0) -> str:
    """Fixed, capped all-service screen; no labels, case IDs, or raw series."""
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite")
    df = case.metrics
    start, end = case.inject_time - window_seconds, case.inject_time + window_seconds
    before = df[(df.time >= start) & (df.time < case.inject_time)]
    after = df[(df.time >= case.inject_time) & (df.time <= end)]
    rows, insufficient = [], []
    # Longest canonical prefix avoids attributing foo_bar_cpu to service foo.
    services = sorted(case.services, key=len, reverse=True)
    grouped = {s: [] for s in services}
    for col in df.columns:
        if col == "time":
            continue
        owner = next((s for s in services if col.startswith(s + "_")), None)
        if owner:
            b, a = before[col].dropna(), after[col].dropna()
            if len(b) >= _MIN_SAMPLES and len(a) >= _MIN_SAMPLES:
                grouped[owner].append((_shift_score(b, a), col, b.mean(), a.mean(), len(b), len(a)))
    for service, metrics in grouped.items():
        if not metrics:
            insufficient.append(service)
        else:
            rows.append((max(metrics, key=lambda r: (r[0], r[1])), service))
    rows.sort(key=lambda r: (-r[0][0], r[1]))
    lines = [
        f"All-service metric screen: requested Unix-second window [{start:.0f}, {end:.0f}], "
        f"onset={case.inject_time:.0f}; {len(before)} rows before, {len(after)} after.",
        "Valid services: " + ", ".join(sorted(case.services)),
        "Top 12 services by maximum standardized mean shift (not causal proof). "
        "Scale is max(before std, 5% of |before mean|, 1e-9). "
        "Each eligible service was screened; at least 5 non-null samples per side required.",
    ]
    for (score, col, b, a, nb, na), service in rows[:12]:
        lines.append(f"  - {service}: {col}, before mean={b:.4g} -> after mean={a:.4g}; "
                     f"{_direction(b, a)}, x{_ratio(b, a)}, shift={score:.1f}, n={nb}/{na}")
    lines.append(f"Eligible services={len(rows)}; omitted from top list={max(0, len(rows)-12)}.")
    if insufficient:
        lines.append("Insufficient coverage (not evidence of health): " + ", ".join(sorted(insufficient)))
    return "\n".join(lines)


def _whole_window_summary(service: str, service_cols: list[str], window: pd.DataFrame, start: float, end: float) -> str:
    lines = [
        f"Metrics for '{service}' in [{start:.0f}, {end:.0f}] ({len(window)} samples; "
        "whole-window only, no reliable onset comparison):"
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
