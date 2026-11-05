from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.tools.log_search import make_log_search
from src.tools.metric_query import make_metric_query
from src.tools.trace_query import make_trace_query


def test_metric_query_reports_stats(sample_case):
    tool = make_metric_query(sample_case)
    result = tool(service="cartservice", start_time=100, end_time=130)
    assert "cpu" in result
    assert "max=95.00" in result


def test_metric_query_unknown_service(sample_case):
    tool = make_metric_query(sample_case)
    with pytest.raises(ValueError, match="Unknown service"):
        tool(service="nope", start_time=100, end_time=130)


def test_log_search_filters_by_service_and_level(sample_case):
    tool = make_log_search(sample_case)
    result = tool(service="cartservice", start_time=100, end_time=130, level="error")
    assert "OOM killed" in result
    assert "1 matching" in result


def test_trace_query_summarizes_by_operation(sample_case):
    tool = make_trace_query(sample_case)
    result = tool(service="cartservice", start_time=100, end_time=130)
    assert "GetCart" in result


def test_metric_query_splits_at_onset_and_ranks_by_shift(sample_case):
    # Keep the shared sparse fixture unchanged; this test needs >=5 samples per side.
    metrics = pd.DataFrame({
        "time": [100, 103, 106, 109, 112, 115, 118, 121, 124, 127],
        "cartservice_cpu": [12.5] * 5 + [92.5] * 5,
        "cartservice_mem": [205.0] * 5 + [206.5] * 5,
        "frontend_latency-90": [50.0] * 10,
    })
    tool = make_metric_query(replace(sample_case, metrics=metrics))
    result = tool(service="cartservice", start_time=100, end_time=130)  # inject_time=115
    lines = result.splitlines()
    assert "split at onset 115" in lines[0]
    assert lines[1].lstrip().startswith("- cartservice_cpu: before mean=12.50 -> after mean=92.50")
    assert "cartservice_mem" in lines[2]


def test_metric_query_window_without_onset_falls_back_to_whole_window(sample_case):
    tool = make_metric_query(sample_case)
    result = tool(service="cartservice", start_time=120, end_time=130)
    assert "INSUFFICIENT COMPARISON COVERAGE: 0 samples before, 2 after" in result
    assert "shift=" not in result
    assert "cartservice_cpu: min=90.00 max=95.00" in result


def test_trace_query_post_only_reports_insufficient_coverage(sample_case):
    tool = make_trace_query(sample_case)
    result = tool(service="cartservice", start_time=100, end_time=130)  # spans start at 120s
    assert "insufficient onset comparison: 0 before, 1 after" in result
    assert "GetCart" in result


def test_metric_query_sparse_fixture_does_not_claim_reliable_shift(sample_case):
    result = make_metric_query(sample_case)("cartservice", 100, 130)
    assert "INSUFFICIENT COMPARISON COVERAGE: 2 samples before, 2 after" in result
    assert "shift=" not in result


def test_trace_query_compares_before_and_after_onset(sample_case):
    traces = pd.DataFrame({
        "startTimeMillis": [100000, 103000, 106000, 109000, 112000,
                            115000, 118000, 121000, 124000, 127000],
        "serviceName": ["cartservice"] * 10,
        "operationName": ["GetCart"] * 10,
        "duration": [10.0] * 5 + [30.0] * 5,
    })
    result = make_trace_query(replace(sample_case, traces=traces))("cartservice", 100, 130)
    assert "split at onset 115" in result
    assert "x3.00" in result
    assert "count before=5 after=5" in result


def test_trace_small_milliseconds_preserve_column_contract(sample_case):
    result = make_trace_query(sample_case)("cartservice", 119, 120)
    assert "GetCart" in result
    # Timestamp interpretation must not adapt to an incorrect query window.
    assert "No matching spans" in make_trace_query(sample_case)("cartservice", 119999, 120001)


def test_trace_explicit_small_seconds(sample_case):
    traces = sample_case.traces.copy()
    traces["startTimeMillis"] /= 1000
    result = make_trace_query(replace(sample_case, traces=traces), timestamp_unit="s")("cartservice", 100, 130)
    assert "GetCart" in result


def test_trace_source_metadata_declares_small_seconds(sample_case):
    traces = sample_case.traces.copy()
    traces["startTimeMillis"] /= 1000
    traces.attrs["startTimeMillis_unit"] = "s"
    result = make_trace_query(replace(sample_case, traces=traces))("cartservice", 100, 130)
    assert "GetCart" in result


def test_trace_invalid_explicit_unit_rejected(sample_case):
    with pytest.raises(ValueError, match="timestamp_unit"):
        make_trace_query(sample_case, timestamp_unit="minutes")("cartservice", 100, 130)


def test_trace_nonfinite_timestamp_does_not_hang(sample_case):
    traces = sample_case.traces.copy()
    traces["startTimeMillis"] = traces["startTimeMillis"].astype(float)
    traces.loc[1, "startTimeMillis"] = float("inf")
    result = make_trace_query(replace(sample_case, traces=traces))("cartservice", 100, 130)
    assert "GetCart" in result
