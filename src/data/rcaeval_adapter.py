"""Thin RCAEval adapter -- temporary, for developing/testing tools and agents against real-shaped
data before Daniel's real loader (case sampler, EDA-driven normalization, etc.) replaces it.

TODO(Daniel): replace with the real loader. Keep load_case()'s return type (Case, CaseLabel)
stable so callers don't need to change.
"""

from __future__ import annotations

from pathlib import Path

from RCAEval.utility import read_logs, read_metrics, read_traces

from src.data.case import Case, CaseLabel

# TODO: fill in once cases.parquet ground truth is loaded (Daniel). Placeholder keeps this
# module importable and testable without the real dataset present.
_PLACEHOLDER_LABELS: dict[str, CaseLabel] = {}


def load_case(case_dir: str | Path) -> tuple[Case, CaseLabel]:
    case_dir = Path(case_dir)
    case_id = case_dir.name

    metrics = read_metrics(str(case_dir))
    logs = read_logs(str(case_dir))
    traces = read_traces(str(case_dir))
    inject_time = float((case_dir / "inject_time.txt").read_text().strip())

    services = sorted({c.rsplit("_", 1)[0] for c in metrics.columns if c != "time"})

    label = _PLACEHOLDER_LABELS.get(
        case_id,
        CaseLabel(case_id=case_id, root_cause_service="unknown", fault_category="unknown", indicator="unknown"),
    )

    case = Case(
        case_id=case_id,
        system="online_boutique",
        services=services,
        inject_time=inject_time,
        incident_query=_template_incident_query(services, inject_time),
        metrics=metrics,
        logs=logs,
        traces=traces,
    )
    return case, label


def _template_incident_query(services: list[str], inject_time: float) -> str:
    # Symptom-level only -- must never name the root cause (see CLAUDE.md Data facts).
    return (
        f"Users are reporting degraded behavior in the online boutique system starting around "
        f"unix time {inject_time:.0f}. Services involved: {', '.join(services)}. "
        f"Please investigate and identify the likely root cause."
    )
