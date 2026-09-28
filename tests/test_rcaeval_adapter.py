from __future__ import annotations

import pandas as pd
import pytest

from src.data.rcaeval_adapter import _normalize_logs, _normalize_traces, load_case


def test_normalize_logs_coerces_string_nanoseconds_to_seconds():
    # CSV-layout logs come back as strings in ns; tools compare against unix seconds.
    logs = pd.DataFrame({"timestamp": ["1705353846067349596"], "container_name": ["x"], "message": ["m"]})
    out = _normalize_logs(logs)
    assert out["timestamp"].iloc[0] == pytest.approx(1705353846.067, abs=1e-3)
    assert (out["timestamp"] >= 1705353800).all()  # the comparison that used to raise TypeError


def test_normalize_logs_leaves_seconds_alone():
    logs = pd.DataFrame({"timestamp": [1705353846], "container_name": ["x"], "message": ["m"]})
    assert _normalize_logs(logs)["timestamp"].iloc[0] == 1705353846


def test_normalize_traces_coerces_numeric_columns():
    traces = pd.DataFrame({"startTimeMillis": ["1705353846065"], "duration": ["186"], "statusCode": ["0.0"]})
    out = _normalize_traces(traces)
    assert out["startTimeMillis"].iloc[0] == 1705353846065
    assert out["duration"].iloc[0] == 186


def test_load_case_rejects_zenodo_csv_layout(tmp_path):
    (tmp_path / "metrics.csv").write_text("time\n1\n")
    with pytest.raises(ValueError, match="download_data.py"):
        load_case(tmp_path)
