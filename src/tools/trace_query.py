"""trace_query tool: latency/error-rate summary per downstream operation for a service+window."""

from __future__ import annotations

import pandas as pd

from src.data.case import Case
from src.data.service_names import trace_service_names
from src.llm.base import ToolSpec
from src.tools.validation import validate_query

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
            "start_time": {"type": "number", "description": "Unix seconds, window start"},
            "end_time": {"type": "number"},
        },
        "required": ["service", "start_time", "end_time"],
    },
)

_MAX_OPERATIONS = 10


def _fmt(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def _timestamp_seconds(values: pd.Series, unit: str | None = None) -> pd.Series:
    """Prefer an explicit source unit. Auto mode supports modern Unix exports;
    ambiguous small timestamps honor the startTimeMillis column's millisecond contract.
    Do not infer units from the requested window: a bad window must not change the data.
    """
    numeric = pd.to_numeric(values, errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    divisors = {"s": 1, "ms": 1000, "us": 1_000_000, "ns": 1_000_000_000}
    if unit is None:
        magnitude = numeric.abs().max()
        if pd.isna(magnitude):
            return numeric
        # Compatibility with modern Unix seconds and ms/us/ns source exports.
        # Relative/small timestamps are inherently ambiguous: default to the column
        # name's declared unit (ms); callers with other units must specify them.
        unit = ("ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14
                else "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e8 else "ms")
    if unit not in divisors:
        raise ValueError("Trace timestamp_unit must be one of: s, ms, us, ns")
    return numeric / divisors[unit]


def make_trace_query(case: Case, *, timestamp_unit: str | None = None):
    """Query bounds are seconds. An explicit timestamp_unit overrides DataFrame
    attrs['startTimeMillis_unit']; absent both, use the documented auto fallback.
    """
    def trace_query(service: str, start_time: float, end_time: float) -> str:
        validate_query(case, service, start_time, end_time)
        df = case.traces
        if df is None or df.empty:
            return "Trace source unavailable or empty; this is not evidence of normal behavior."
        unit = timestamp_unit if timestamp_unit is not None else df.attrs.get("startTimeMillis_unit")
        timestamps = _timestamp_seconds(df["startTimeMillis"], unit)

        mask = (
            (timestamps >= start_time)
            & (timestamps <= end_time)
            & (df["serviceName"].isin(trace_service_names(service)))
        )
        matched = df[mask]
        if matched.empty:
            return f"No matching spans for service='{service}'; source naming/coverage may differ. Not proof of health."

        if "duration" not in matched.columns:
            counts = matched["operationName"].value_counts().head(_MAX_OPERATIONS)
            lines = [f"{len(matched)} spans for '{service}' across {matched['operationName'].nunique()} operations:"]
            lines += [f"  - {op}: count={count}" for op, count in counts.items()]
            return "\n".join(lines)

        # Split at onset (as in metric_query): an operation that is always slow is background,
        # not a symptom -- what matters is which operations got slower after the onset.
        is_after = timestamps.loc[matched.index] >= case.inject_time
        if int(is_after.sum()) < 5 or int((~is_after).sum()) < 5:
            stats = matched.groupby("operationName")["duration"].agg(["count", "mean", "max"])
            lines = [
                f"{len(matched)} spans for '{service}' across {len(stats)} operations "
                f"(insufficient onset comparison: {int((~is_after).sum())} before, "
                f"{int(is_after.sum())} after; need 5 per side). "
                "Durations are source-native units, NOT assumed milliseconds; no onset-change conclusion:"
            ]
            for op, row in stats.sort_values("count", ascending=False).head(_MAX_OPERATIONS).iterrows():
                lines.append(f"  - {op}: count={int(row['count'])} mean_dur={row['mean']:.1f} max_dur={row['max']:.1f}")
            return "\n".join(lines)

        before = matched[~is_after].groupby("operationName")["duration"].agg(["count", "mean"])
        after = matched[is_after].groupby("operationName")["duration"].agg(["count", "mean", "max"])
        stats = after.join(before, how="outer", lsuffix="_after", rsuffix="_before")
        stats["slowdown"] = stats["mean_after"] / stats["mean_before"]
        eligible = (stats["count_before"] >= 5) & (stats["count_after"] >= 5) & (stats["mean_before"] > 0)
        stats.loc[~eligible, "slowdown"] = float("nan")
        stats = stats.sort_values("slowdown", ascending=False, na_position="last")

        lines = [
            f"{len(matched)} spans for '{service}' across {len(stats)} operations, split at onset "
            f"{case.inject_time:.0f}, biggest slowdown first (source-native duration units, "
            "NOT assumed milliseconds; ratios require 5 observations per side and positive baseline):"
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
