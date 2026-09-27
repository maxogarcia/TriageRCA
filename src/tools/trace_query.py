"""trace_query tool: latency/error-rate summary per downstream operation for a service+window."""

from __future__ import annotations

from src.data.case import Case
from src.data.service_names import trace_service_names
from src.llm.base import ToolSpec

SPEC = ToolSpec(
    name="trace_query",
    description=(
        "Summarize traces touching a service in a time window: per-operation call counts and "
        "latency stats, to spot slow or newly-erroring downstream calls."
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

        grouped = matched.groupby("operationName")["duration"] if "duration" in matched.columns else None
        lines = [f"{len(matched)} spans for '{service}' across {matched['operationName'].nunique()} operations:"]
        if grouped is not None:
            stats = grouped.agg(["count", "mean", "max"]).sort_values("count", ascending=False)
            for op, row in stats.head(_MAX_OPERATIONS).iterrows():
                lines.append(f"  - {op}: count={int(row['count'])} mean_dur={row['mean']:.1f} max_dur={row['max']:.1f}")
        else:
            counts = matched["operationName"].value_counts().head(_MAX_OPERATIONS)
            for op, count in counts.items():
                lines.append(f"  - {op}: count={count}")
        return "\n".join(lines)

    return trace_query
