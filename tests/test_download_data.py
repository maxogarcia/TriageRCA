"""Downloader tests use temporary manifests and a fake transfer; no network/data download."""
import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[1] / "scripts/download_data.py"
    spec = importlib.util.spec_from_file_location("manifest_downloader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(path, ids, column="case"):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([column])
        writer.writerows([value] for value in ids)
    return path


def local_files(root, ids):
    for case_id in ids:
        directory = root / case_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.parquet").touch()
        (directory / "inject_time.txt").touch()


def test_reads_fixed_case_ids(downloader, tmp_path):
    ids = ["re2ob_checkoutservice_cpu_1", "re2ss_user_mem_3", "re2tt_ts-route-service_cpu_3"]
    assert downloader.read_case_ids(manifest(tmp_path / "m.csv", ids), 3) == ids


@pytest.mark.parametrize("ids", [[], ["re2ss_user_mem_3"] * 2, [""], ["../x"],
                                 ["re2ob_*"], [" re2ss_user_mem_3"], ["re2ss_x/../y"]])
def test_rejects_empty_duplicate_or_unsafe_ids(downloader, tmp_path, ids):
    with pytest.raises(ValueError):
        downloader.read_case_ids(manifest(tmp_path / "m.csv", ids))


def test_requires_case_column(downloader, tmp_path):
    with pytest.raises(ValueError, match="no 'case' column"):
        downloader.read_case_ids(manifest(tmp_path / "m.csv", ["x"], "wrong"))


def test_rejects_wrong_expected_count(downloader, tmp_path):
    with pytest.raises(ValueError, match="Expected 90 cases, found 1"):
        downloader.read_case_ids(manifest(tmp_path / "m.csv", ["re2ss_user_mem_3"]), 90)


def test_snapshot_requests_only_manifest_directories(downloader, tmp_path, monkeypatch):
    ids = ["re2ss_user_mem_3", "re2tt_ts-route-service_cpu_3"]
    m = manifest(tmp_path / "m.csv", ids)
    received = []

    def fake_snapshot(repo, **kwargs):
        received.append((repo, kwargs))
        local_files(Path(kwargs["local_dir"]), ids)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot))
    out = tmp_path / "raw"
    downloader.main(["--manifest", str(m), "--out", str(out), "--expected-cases", "2", "--revision", "abc123"])
    assert received == [(downloader.REPO_ID, dict(repo_type="dataset", local_dir=str(out),
                         allow_patterns=[f"{i}/*" for i in ids], revision="abc123"))]


def test_download_missing_required_files_fails(downloader, tmp_path, monkeypatch):
    m = manifest(tmp_path / "m.csv", ["re2ss_user_mem_3"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: None))
    with pytest.raises(SystemExit, match="Missing required files"):
        downloader.main(["--manifest", str(m), "--out", str(tmp_path / "raw")])


def test_check_only_uses_project_root_not_cwd(downloader, tmp_path, monkeypatch):
    project = tmp_path / "project"
    manifests = project / "data/manifests"
    manifests.mkdir(parents=True)
    ids = ["re2ss_user_mem_3"]
    manifest(manifests / "sample_manifest.csv", ids)
    local_files(project / "data/raw/rcaeval", ids)
    local_files(project / "data/raw/rcaeval", ["re2ob_extra_cpu_1"])
    monkeypatch.setattr(downloader, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    # Any accidental network/transfer attempt fails the test.
    def forbidden(*args, **kwargs):
        raise AssertionError("check-only must not transfer")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=forbidden))
    downloader.main(["--check-only", "--expected-cases", "1"])


def test_exploratory_pattern_is_explicit(downloader, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda *a, **k: calls.append((a, k))))
    downloader.main(["--pattern", "re2ob_*", "--out", str(tmp_path / "raw")])
    assert calls[0][1]["allow_patterns"] == ["re2ob_*"]


@pytest.mark.parametrize("args", [["--manifest", "m.csv", "--pattern", "re2*"],
                                  ["--pattern", "re2*", "--check-only"],
                                  ["--pattern", "re2*", "--expected-cases", "90"],
                                  ["--expected-cases", "0"]])
def test_rejects_conflicting_options(downloader, args):
    with pytest.raises(SystemExit) as exc:
        downloader.main(args)
    assert exc.value.code == 2
