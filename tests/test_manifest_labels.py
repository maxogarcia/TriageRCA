from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.labels import ManifestLabels
from src.data import rcaeval_adapter as adapter
from scripts.evaluate_agent_runs import evaluate


def make_manifest(tmp_path, **changes):
    row = dict(case="re2ob_checkoutservice_cpu_1", root_cause_service="checkoutservice",
               fault="cpu", system="ob")
    row.update(changes)
    path = tmp_path / "manifest.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_real_label_and_missing_indicator(tmp_path):
    labels = ManifestLabels(make_manifest(tmp_path))
    label = labels.label("re2ob_checkoutservice_cpu_1")
    assert label.root_cause_service == "checkoutservice"
    assert label.fault_category == "cpu"
    assert label.indicator == "unknown"


@pytest.mark.parametrize("column,value", [
    ("root_cause_service", ""), ("fault", "unknown"), ("case", "x "),
    ("system", "unsupported"),
])
def test_invalid_manifest_rejected(tmp_path, column, value):
    with pytest.raises(ValueError):
        ManifestLabels(make_manifest(tmp_path, **{column: value}))


def test_duplicate_ids_rejected(tmp_path):
    path = make_manifest(tmp_path)
    df = pd.read_csv(path)
    pd.concat([df, df]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        ManifestLabels(path)


def test_missing_case_rejected(tmp_path):
    with pytest.raises(KeyError, match="No ground-truth"):
        ManifestLabels(make_manifest(tmp_path)).label("missing")


@pytest.mark.parametrize("system,expected", [
    ("ob", "online_boutique"), ("ss", "sock_shop"), ("tt", "train_ticket"),
])
def test_adapter_preserves_interface_and_uses_system(tmp_path, monkeypatch, system, expected):
    manifest = make_manifest(tmp_path, system=system)
    case_dir = tmp_path / "re2ob_checkoutservice_cpu_1"
    case_dir.mkdir()
    (case_dir / "inject_time.txt").write_text("100")
    monkeypatch.setattr(adapter, "read_metrics", lambda _: pd.DataFrame(
        {"time": [99, 100], "checkoutservice_cpu": [1, 2], "cartservice_cpu": [1, 1]}))
    monkeypatch.setattr(adapter, "read_logs", lambda _: None)
    monkeypatch.setattr(adapter, "read_traces", lambda _: None)
    case, label = adapter.load_case(case_dir, manifest_path=manifest)
    assert case.system == expected
    assert expected.replace("_", " ") in case.incident_query
    assert label.root_cause_service == "checkoutservice"
    assert not hasattr(case, "root_cause_service")
    assert "CPU stress" not in case.incident_query


def test_scoring_correct_wrong_and_failed_reports(tmp_path):
    labels = ManifestLabels(make_manifest(tmp_path))
    runs = tmp_path / "runs"
    runs.mkdir()
    path = runs / "case.json"
    for report, score in [
        ({"ranked_services": ["checkoutservice"], "fault_category": "cpu"}, 1),
        ({"ranked_services": ["cartservice"], "fault_category": "mem"}, 0),
        (None, 0),
    ]:
        path.write_text(json.dumps({"case_id": "re2ob_checkoutservice_cpu_1", "report": report}))
        rows, summary = evaluate(runs, labels)
        assert summary["AC@1"] == score
        assert summary["indicator_accuracy"] is None
        assert summary["indicator_label_coverage"] == 0


def test_missing_runs_stay_in_denominator(tmp_path):
    path = make_manifest(tmp_path)
    df = pd.read_csv(path)
    extra = df.copy()
    extra["case"] = "another"
    pd.concat([df, extra]).to_csv(path, index=False)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "one.json").write_text(json.dumps({
        "case_id": "re2ob_checkoutservice_cpu_1",
        "report": {"ranked_services": ["checkoutservice"]},
    }))
    _, summary = evaluate(runs, ManifestLabels(path))
    assert summary["num_expected_cases"] == 2
    assert summary["AC@1"] == 0.5

