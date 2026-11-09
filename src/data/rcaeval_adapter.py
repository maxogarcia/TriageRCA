"""RCAEval telemetry adapter with manifest-backed evaluation labels.

The return interface remains (Case, CaseLabel). Never pass CaseLabel to agents.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from RCAEval.utility import read_logs, read_metrics, read_traces

from src.data.case import Case, CaseLabel
from src.data.labels import DEFAULT_MANIFEST, ManifestLabels



def load_case(
    case_dir: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    label_provider: ManifestLabels | None = None,
) -> tuple[Case, CaseLabel]:
    case_dir = Path(case_dir)
    case_id = case_dir.name
    if (case_dir / "metrics.csv").exists() and not (case_dir / "metrics.parquet").exists():
        raise ValueError(
            f"{case_dir} is in the Zenodo CSV layout, which the tools don't read. "
            "Download the Hugging Face Parquet layout instead: python scripts/download_data.py"
        )

    provider = label_provider if label_provider is not None else ManifestLabels(manifest_path)
    label = provider.label(case_id)
    system = provider.system(case_id)
    metrics = read_metrics(str(case_dir))
    logs = _normalize_logs(read_logs(str(case_dir)))
    traces = _normalize_traces(read_traces(str(case_dir)))
    inject_time = float((case_dir / "inject_time.txt").read_text().strip())

    services = sorted({c.rsplit("_", 1)[0] for c in metrics.columns if c != "time"})

    case = Case(
        case_id=case_id,
        system=system,
        services=services,
        inject_time=inject_time,
        incident_query=_template_incident_query(services, inject_time, system),
        metrics=metrics,
        logs=logs,
        traces=traces,
    )
    return case, label


def _normalize_logs(logs: pd.DataFrame | None) -> pd.DataFrame | None:
    """Tools compare `timestamp` against unix seconds. Parquet already stores seconds; CSV readers
    return strings, sometimes in nanoseconds, so coerce here rather than in every tool."""
    if logs is None:
        return None
    ts = pd.to_numeric(logs["timestamp"], errors="coerce")
    while ts.max() > 1e11:  # ns/us/ms -> s
        ts = ts / 1000
    return logs.assign(timestamp=ts)


def _normalize_traces(traces: pd.DataFrame | None) -> pd.DataFrame | None:
    if traces is None:
        return None
    numeric = [c for c in ("startTimeMillis", "startTime", "duration", "statusCode") if c in traces.columns]
    return traces.assign(**{c: pd.to_numeric(traces[c], errors="coerce") for c in numeric})


def _template_incident_query(services: list[str], inject_time: float, system: str = "online_boutique") -> str:
    # Symptom-level only: no label-derived service selection or fault description.
    return (
        f"Users are reporting degraded behavior in the {system.replace('_', ' ')} system starting around "
        f"unix time {inject_time:.0f}. Services involved: {', '.join(services)}. "
        f"Please investigate and identify the likely root cause."
    )
