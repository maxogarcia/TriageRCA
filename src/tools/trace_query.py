"""trace_query tool: latency/error-rate summary per downstream operation for a service+window."""

from __future__ import annotations

import pandas as pd

from src.data.case import Case
from src.data.service_names import trace_service_names
from src.llm.base import ToolSpec

SPEC = ToolSpec(
    name="trace_query",
    description=(
        "Summarize traces touching a service in a time window: per-operation call counts and "
        "latency, compared before vs after the incident onset when the window spans it, to spot "
        "downstream calls that got slower."
    ),
    parameters={
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "serviceName to filter on"},
            "start_time": {"type": "number", "description": "Unix ms or seconds, window start"},
            "end_time": {"type": "number"},
        },
        "required": ["service", "start_time", "end_time"],
    },
)

_MAX_OPERATIONS = 10


def _fmt(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def make_trace_query(case: Case):
    def trace_query(service: str, start_time: float, end_time: float) -> str:
        df = case.traces
        # startTimeMillis may be seconds or ms depending on source export; normalize by magnitude.
        start_ms = start_time * 1000 if start_time < 1e12 else start_time
        end_ms = end_time * 1000 if end_time < 1e12 else end_time

        mask = (
            (df["startTimeMillis"] >= start_ms)
            & (df["startTimeMillis"] <= end_ms)
            & (df["serviceName"].isin(trace_service_names(service)))
        )
        matched = df[mask]
        if matched.empty:
            return f"No spans found for service='{service}' in window. Known services vary by source naming."

        if "duration" not in matched.columns:
            counts = matched["operationName"].value_counts().head(_MAX_OPERATIONS)
            lines = [f"{len(matched)} spans for '{service}' across {matched['operationName'].nunique()} operations:"]
            lines += [f"  - {op}: count={count}" for op, count in counts.items()]
            return "\n".join(lines)

        # Split at onset (as in metric_query): an operation that is always slow is background,
        # not a symptom -- what matters is which operations got slower after the onset.
        onset_ms = case.inject_time * 1000
        is_after = matched["startTimeMillis"] >= onset_ms
        if is_after.all() or not is_after.any():
            stats = matched.groupby("operationName")["duration"].agg(["count", "mean", "max"])
            lines = [
                f"{len(matched)} spans for '{service}' across {len(stats)} operations "
                f"(window does not span onset):"
            ]
            for op, row in stats.sort_values("count", ascending=False).head(_MAX_OPERATIONS).iterrows():
                lines.append(f"  - {op}: count={int(row['count'])} mean_dur={row['mean']:.1f} max_dur={row['max']:.1f}")
            return "\n".join(lines)

        before = matched[~is_after].groupby("operationName")["duration"].agg(["count", "mean"])
        after = matched[is_after].groupby("operationName")["duration"].agg(["count", "mean", "max"])
        stats = after.join(before, how="outer", lsuffix="_after", rsuffix="_before")
        stats["slowdown"] = stats["mean_after"] / stats["mean_before"]
        stats = stats.sort_values("slowdown", ascending=False, na_position="last")

        lines = [
            f"{len(matched)} spans for '{service}' across {len(stats)} operations, split at onset "
            f"{case.inject_time:.0f}, biggest slowdown first (durations as recorded in the traces):"
        ]
        for op, row in stats.head(_MAX_OPERATIONS).iterrows():
            lines.append(
                f"  - {op}: mean_dur before={_fmt(row['mean_before'])} -> after={_fmt(row['mean_after'])} "
                f"(x{_fmt(row['slowdown'], 2)}), count before={_fmt(row['count_before'], 0)} "
                f"after={_fmt(row['count_after'], 0)}"
            )
        if len(stats) > _MAX_OPERATIONS:
            lines.append(f"  ... ({len(stats) - _MAX_OPERATIONS} more operations omitted)")
        return "\n".join(lines)

    return trace_query
