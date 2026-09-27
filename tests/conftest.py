from __future__ import annotations

import pandas as pd
import pytest

from src.data.case import Case


@pytest.fixture
def sample_case() -> Case:
    metrics = pd.DataFrame(
        {
            "time": [100, 110, 120, 130],
            "cartservice_cpu": [10.0, 15.0, 90.0, 95.0],
            "cartservice_mem": [200.0, 210.0, 205.0, 208.0],
            "frontend_latency-90": [50.0, 52.0, 51.0, 53.0],
        }
    )
    logs = pd.DataFrame(
        {
            "timestamp": [105, 115, 125],
            "container_name": ["cartservice", "cartservice", "frontend"],
            "message": ["ok", "OOM killed process", "ok"],
            "level": ["info", "error", "info"],
            "error": [False, True, False],
        }
    )
    traces = pd.DataFrame(
        {
            "traceID": ["t1", "t2"],
            "spanID": ["s1", "s2"],
            "serviceName": ["cartservice", "frontend"],
            "methodName": ["GetCart", "Home"],
            "operationName": ["GetCart", "Home"],
            "startTimeMillis": [120000, 121000],
            "duration": [300, 40],
        }
    )
    return Case(
        case_id="re2ob_cartservice_cpu_1",
        system="online_boutique",
        services=["cartservice", "frontend"],
        inject_time=115,
        incident_query="Users report checkout is slow starting around unix time 115.",
        metrics=metrics,
        logs=logs,
        traces=traces,
    )
