from __future__ import annotations

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
    result = tool(service="nope", start_time=100, end_time=130)
    assert "No metrics found" in result


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
    tool = make_metric_query(sample_case)
    result = tool(service="cartservice", start_time=100, end_time=130)  # inject_time=115
    lines = result.splitlines()
    assert "split at onset 115" in lines[0]
    assert lines[1].lstrip().startswith("- cartservice_cpu: before mean=12.50 -> after mean=92.50")
    assert "cartservice_mem" in lines[2]


def test_metric_query_window_without_onset_falls_back_to_whole_window(sample_case):
    tool = make_metric_query(sample_case)
    result = tool(service="cartservice", start_time=120, end_time=130)
    assert "does not span onset" in result
    assert "cartservice_cpu: min=90.00 max=95.00" in result


def test_trace_query_compares_before_and_after_onset(sample_case):
    tool = make_trace_query(sample_case)
    result = tool(service="cartservice", start_time=100, end_time=130)  # spans start at 120s
    assert "does not span onset" in result  # every cartservice span is after onset 115
    assert "GetCart" in result
