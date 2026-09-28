"""Case / CaseLabel data holders.

Temporary home for these types -- Daniel owns src/data/ long-term. Kept here (and in
rcaeval_adapter.py) only so Max's tools/agents have something concrete to run against
before the real loader lands. Interfaces here should not need to change when it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Case:
    """Everything an agent is allowed to see. No ground truth here (hard rule #1)."""

    case_id: str
    system: str  # e.g. "online_boutique"
    services: list[str]
    inject_time: float  # unix timestamp
    incident_query: str  # generated NL description, symptom-level only
    metrics: pd.DataFrame  # wide: "time" + one column per "<service>_<metric>"
    logs: pd.DataFrame  # columns: timestamp, container_name, message, level, error
    traces: pd.DataFrame  # columns: traceID, spanID, serviceName, methodName, operationName, startTimeMillis


@dataclass
class CaseLabel:
    """Ground truth. Only the eval harness may read this -- never pass it to an agent,
    tool, or prompt (hard rule #1)."""

    case_id: str
    root_cause_service: str
    fault_category: str
    indicator: str
