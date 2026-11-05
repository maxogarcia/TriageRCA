"""Offline synthetic telemetry tests; no downloaded RCAEval data required."""
from dataclasses import replace

import pandas as pd
import pytest

from src.data.case import Case
from src.tools.metric_query import make_metric_query, metric_overview
from src.tools.log_search import make_log_search
from src.tools.trace_query import make_trace_query


@pytest.fixture
def telemetry_case():
    times = list(range(100, 120))
    return Case(
        case_id="do-not-leak-this-id", system="synthetic", services=["carts", "user"],
        inject_time=110, incident_query="Investigate degradation near 110",
        metrics=pd.DataFrame({"time": times, "carts_cpu": [1.0] * 10 + [10.0] * 10,
                              "user_mem": [100.0] * 10 + [80.0] * 10}),
        logs=pd.DataFrame({"timestamp": [110, 111], "container_name": ["carts", "carts"],
                           "message": ["literal [error]", "normal"]}),
        traces=pd.DataFrame({"startTimeMillis": [t * 1000 for t in times],
                             "serviceName": ["carts"] * 20, "operationName": ["read"] * 20,
                             "duration": [10.0] * 10 + [20.0] * 10}),
    )


@pytest.mark.parametrize("factory", [make_metric_query, make_log_search, make_trace_query])
@pytest.mark.parametrize("bounds", [(120, 100), (110, 110), (float("nan"), 120),
                                   (100, float("inf")), (1000000000000, 1000000001000)])
def test_reject_invalid_windows(telemetry_case, factory, bounds):
    with pytest.raises(ValueError):
        factory(telemetry_case)("carts", *bounds)


@pytest.mark.parametrize("factory", [make_metric_query, make_log_search, make_trace_query])
def test_reject_wrong_system_names(telemetry_case, factory):
    with pytest.raises(ValueError, match="Valid services: carts, user"):
        factory(telemetry_case)("cartservice", 100, 120)


def test_one_post_sample_is_not_comparison(telemetry_case):
    result = make_metric_query(telemetry_case)("carts", 100, 110)
    assert "INSUFFICIENT COMPARISON COVERAGE" in result
    assert "shift=" not in result


def test_nonnull_coverage_is_per_metric(telemetry_case):
    telemetry_case.metrics.loc[11:, "carts_cpu"] = float("nan")
    result = make_metric_query(telemetry_case)("carts", 100, 120)
    assert "no metric has enough non-null values" in result


def test_direction_is_explicit(telemetry_case):
    result = make_metric_query(telemetry_case)("user", 100, 120)
    assert "direction=decrease, change=-20.0%" in result


def test_overview_screens_services_without_id_or_labels(telemetry_case):
    result = metric_overview(telemetry_case)
    assert "carts_cpu" in result and "user_mem" in result
    assert "Eligible services=2" in result
    assert telemetry_case.case_id not in result
    assert result.index("  - carts:") < result.index("  - user:")


def test_overview_reports_missing_coverage(telemetry_case):
    result = metric_overview(replace(telemetry_case, inject_time=119))
    assert "Insufficient coverage" in result
    assert "Eligible services=0" in result


@pytest.mark.parametrize("factory,field", [(make_log_search, "logs"), (make_trace_query, "traces")])
def test_absent_source_is_explicit(telemetry_case, factory, field):
    result = factory(replace(telemetry_case, **{field: None}))("carts", 100, 120)
    assert "unavailable" in result and "not evidence" in result


def test_keyword_is_literal_not_regex(telemetry_case):
    result = make_log_search(telemetry_case)("carts", 100, 120, keyword="[error]")
    assert "1 matching log lines" in result


@pytest.mark.parametrize("scale", [1, 1000, 1000000, 1000000000])
def test_trace_source_seconds_ms_us_ns(telemetry_case, scale):
    telemetry_case.traces["startTimeMillis"] = pd.Series(range(100, 120)) + 1700000000
    telemetry_case.traces["startTimeMillis"] *= scale
    telemetry_case.inject_time = 1700000110
    result = make_trace_query(telemetry_case)("carts", 1700000100, 1700000120)
    assert "20 spans" in result
    assert "x2.00" in result
    assert "NOT assumed milliseconds" in result


def test_trace_per_operation_sparse_ratio_is_na(telemetry_case):
    telemetry_case.traces.loc[10:, "operationName"] = "other"
    result = make_trace_query(telemetry_case)("carts", 100, 120)
    assert "xn/a" in result
