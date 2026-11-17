"""Offline tests for scripts/results_table.py and the --model axis of scripts/run_ablation.py."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import results_table, run_ablation

CASES = ["re2ss_carts_cpu_1", "re2ss_orders_mem_1", "re2tt_ts-order-service_delay_1"]
TRUTH = {"re2ss_carts_cpu_1": ("carts", "cpu", "ss"), "re2ss_orders_mem_1": ("orders", "mem", "ss"),
         "re2tt_ts-order-service_delay_1": ("ts-order-service", "delay", "tt")}


def write_manifest(path: Path) -> str:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "root_cause_service", "fault", "system"])
        for c, (svc, fault, system) in TRUTH.items():
            w.writerow([c, svc, fault, system])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_experiment(root: Path, manifest: Path, sha: str, model: str, rows: dict[str, dict], version="abc123") -> None:
    root.mkdir(parents=True)
    exp_rows = {r: {"model": model, "config_name": cfg["config_name"]} for r, cfg in rows.items()}
    (root / "experiment.json").write_text(json.dumps(
        {"code_version": version, "manifest": str(manifest), "manifest_sha256": sha, "cases": CASES, "rows": exp_rows}))
    for row, cfg in rows.items():
        d = root / row / cfg["config_name"]
        d.mkdir(parents=True)
        for case_id, rec in cfg["records"].items():
            (d / f"{case_id}.json").write_text(json.dumps({"case_id": case_id, **rec}))


def rec(ranked, fault="cpu", cost=0.01, lat=2.0):
    return {"report": {"ranked_services": ranked, "fault_category": fault}, "cost_usd": cost,
            "llm_calls": [{"latency_s": lat}], "wall_clock_s": lat + 1}


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["results_table.py", *argv])
    results_table.main()


def test_table_scores_ac_at_k_and_cost(tmp_path, monkeypatch):
    manifest = tmp_path / "m.csv"
    sha = write_manifest(manifest)
    write_experiment(tmp_path / "qwen", manifest, sha, "open_weight_model", {
        "hybrid": {"config_name": "triagerca", "records": {
            CASES[0]: rec(["carts", "orders"]),                       # rank 1
            CASES[1]: rec(["carts", "user", "orders"], fault="mem"),  # rank 3
            # CASES[2] missing -> wrong, stays in the denominator
        }},
        "never_run": {"config_name": "triagerca", "records": {}},
    })
    (tmp_path / "qwen" / "never_run").rename(tmp_path / "qwen" / "_gone")  # listed in experiment.json, no folder
    write_experiment(tmp_path / "gem", manifest, sha, "pipeline_model", {
        "hybrid": {"config_name": "triagerca",
                   "records": {c: rec([TRUTH[c][0]], fault=TRUTH[c][1], cost=0.02) for c in CASES}},
    })
    out = tmp_path / "out" / "table"
    run_main(monkeypatch, [str(tmp_path / "qwen"), str(tmp_path / "gem"), "--out", str(out)])

    rows = {(r["model"], r["row"], r["group"]): r for r in csv.DictReader(open(out.with_suffix(".csv")))}
    names = results_table.model_names()
    q = rows[(names["open_weight_model"], "hybrid", "ss+tt")]
    assert float(q["AC@1"]) == pytest.approx(1 / 3, abs=1e-3)
    assert float(q["AC@3"]) == pytest.approx(2 / 3, abs=1e-3)
    assert int(q["missing"]) == 1
    assert float(q["cost_per_case_usd"]) == pytest.approx(0.01)  # over the 2 cases that ran
    g = rows[(names["pipeline_model"], "hybrid", "ss+tt")]
    assert float(g["AC@1"]) == 1.0 and float(g["fault_acc"]) == 1.0
    assert not any(k[1] == "never_run" for k in rows)
    md = out.with_suffix(".md").read_text()
    assert md.index("### ss+tt") < md.index("### ss\n")  # headline group first


def test_refuses_mixed_code_and_different_cases(tmp_path, monkeypatch):
    both = [str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "t")]
    manifest = tmp_path / "m.csv"
    sha = write_manifest(manifest)
    one = {"hybrid": {"config_name": "triagerca", "records": {}}}
    write_experiment(tmp_path / "a", manifest, sha, "open_weight_model", one, version="aaa")
    write_experiment(tmp_path / "b", manifest, sha, "pipeline_model", one, version="bbb")
    with pytest.raises(SystemExit, match="different code versions"):
        run_main(monkeypatch, both)
    run_main(monkeypatch, [*both, "--allow-mixed-code"])

    exp = json.loads((tmp_path / "b" / "experiment.json").read_text())
    exp["cases"] = CASES[:2]
    (tmp_path / "b" / "experiment.json").write_text(json.dumps(exp))
    with pytest.raises(SystemExit, match="different cases"):
        run_main(monkeypatch, [*both, "--allow-mixed-code"])


def test_ablation_model_axis_keeps_default_lock():
    base = {"data_dir": "data/raw/rcaeval"}
    out, cache = Path("runs/heldout"), Path("runs/heldout/llm_cache")
    # Default model: identical config to before the --model flag, so existing experiment.json locks still match.
    assert run_ablation.row_config(base, "hybrid", out, cache)["model"] == "open_weight_model"
    gem = run_ablation.row_config(base, "hybrid", out, cache, "pipeline_model")
    assert gem["model"] == "pipeline_model" and gem["critic_model"] == "critic_14b"  # critic stays pinned
    assert run_ablation.default_out("open_weight_model") == Path("runs/heldout")
    assert run_ablation.default_out("pipeline_model") == Path("runs/heldout_pipeline_model")
    assert {"pipeline_model", "open_weight_model", "critic_14b"} <= set(run_ablation.model_keys())


def test_grounding_matched_by_run_path_not_row_name(tmp_path, monkeypatch):
    manifest = tmp_path / "m.csv"
    sha = write_manifest(manifest)
    # Same row name in two experiments (two models): the judge's row column alone would collide.
    for name, model in (("qwen", "open_weight_model"), ("gem", "pipeline_model")):
        write_experiment(tmp_path / name, manifest, sha, model, {
            "hybrid": {"config_name": "triagerca", "records": {c: rec([TRUTH[c][0]]) for c in CASES}},
        })
    judged = tmp_path / "grounding_cases.csv"
    with open(judged, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "case_id", "run_path", "grounding_expl"])
        for c in CASES[:2]:  # qwen: 2 of 3 judged, one without a score
            score = "0.5" if c == CASES[0] else ""
            w.writerow(["hybrid", c, tmp_path / "qwen" / "hybrid" / "triagerca" / f"{c}.json", score])
        for c in CASES:
            w.writerow(["hybrid", c, tmp_path / "gem" / "hybrid" / "triagerca" / f"{c}.json", "0.9"])
    out = tmp_path / "t"
    exps = [str(tmp_path / "qwen"), str(tmp_path / "gem")]
    run_main(monkeypatch, [*exps, "--grounding", str(judged), "--out", str(out)])

    names = results_table.model_names()
    rows = {(r["model"], r["group"]): r for r in csv.DictReader(open(out.with_suffix(".csv")))}
    q, g = rows[(names["open_weight_model"], "ss+tt")], rows[(names["pipeline_model"], "ss+tt")]
    assert float(q["grounding"]) == 0.5 and q["grounding_n"] == "1"
    assert float(g["grounding"]) == pytest.approx(0.9) and g["grounding_n"] == "3"
    assert "0.50 (n=1)" in out.with_suffix(".md").read_text()
