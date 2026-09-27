"""metric_query tool: compact stats for one service's metrics over a time window.

Hard rule #2: never return the raw series, only a capped summary.
"""

from __future__ import annotations

from src.data.case import Case
from src.llm.base import ToolSpec

SPEC = ToolSpec(
    name="metric_query",
    description=(
        "Get summary statistics (min/max/mean/last and a coarse trend) for a service's metrics "
        "in a time window. Use this to check whether a service's CPU/memory/latency/etc looks abnormal."
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

        lines = [f"Metrics for '{service}' in [{start_time:.0f}, {end_time:.0f}] ({len(window)} samples):"]
        for col in service_cols[:_MAX_METRICS_REPORTED]:
            series = window[col].dropna()
            if series.empty:
                continue
            metric_name = col[len(service) + 1 :]
            lines.append(
                f"  - {metric_name}: min={series.min():.2f} max={series.max():.2f} "
                f"mean={series.mean():.2f} last={series.iloc[-1]:.2f}"
            )
        if len(service_cols) > _MAX_METRICS_REPORTED:
            lines.append(f"  ... ({len(service_cols) - _MAX_METRICS_REPORTED} more metrics omitted)")
        return "\n".join(lines)

    return metric_query
