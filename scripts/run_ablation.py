"""Run the critic ablation: one `python -m src.run` per row, with resume.

Output layout under --out (default runs/heldout):
    experiment.json                            lock file: commit, manifest, case list, every row's config
    llm_cache/                                 LLM cache for this experiment only (shared by its rows)
    <row>/config.yaml                          full config for the row (critic settings included)
    <row>/git_commit.txt                       commit the experiment runs at
    <row>/<config_name>/<case_id>.json         records written by src.run (per-case time: wall_clock_s)
    <row>/timing.jsonl                         one line per src.run invocation (start, end, seconds, cases)
    <row>.log                                  src.run output, timestamped, appended every invocation

Resume: a case whose JSON already exists and parses is skipped; only missing cases are passed to
src.run. After a crash, run the same command again. Every invocation is checked against
experiment.json first (completed rows included); any difference stops the run before anything
starts. A dirty working tree is refused unless --allow-dirty (smoke tests only).

Cache: rows share one fresh cache inside --out. Identical calls (e.g. the Planner and the first
Executor call, which do not depend on the critic) are computed once, so all rows start from the
same first hypothesis. Later rows therefore have cache hits; count them with llm_calls[*].cached.

Usage:
    python scripts/run_ablation.py --dry-run                 # show the plan, run nothing
    python scripts/run_ablation.py                           # all rows, held-out cases
    python scripts/run_ablation.py --rows hybrid llm_only    # some rows only
    python scripts/run_ablation.py --cases "re2ob_checkoutservice_*_1" --cases-limit 1 --out runs/ablation_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.labels import DEFAULT_MANIFEST, ManifestLabels  # noqa: E402

PIPELINE_MODEL = "open_weight_model"
HELDOUT_CASES = ["re2ss_*", "re2tt_*"]
# Explicit case IDs are passed to src.run, so its cases_limit only needs to be large enough.
SRC_RUN_CASES_LIMIT = 1000

# Run order: fast rows first. Keys override the base config (configs/dev.yaml).
ROWS: dict[str, dict] = {
    "no_critic": {"config_name": "triagerca_no_critic", "critic": "stub"},
    "code_rule": {"config_name": "triagerca", "critic": "onset_rule"},
    "hybrid": {"config_name": "triagerca", "critic": "llm", "critic_model": "critic_14b", "critic_onset_check": True},
    "llm_only": {"config_name": "triagerca", "critic": "llm", "critic_model": "critic_14b", "critic_onset_check": False},
    "hybrid_7b": {"config_name": "triagerca", "critic": "llm", "critic_model": "open_weight_model", "critic_onset_check": True},
}


def select_cases(base_cfg: dict, manifest: Path, patterns: list[str], limit: int | None) -> list[str]:
    """Same selection as src.run: case dirs with inject_time.txt, in the manifest, matching a pattern."""
    labels = ManifestLabels(manifest)
    data_dir = Path(base_cfg["data_dir"])
    names = sorted(p.name for p in data_dir.iterdir() if (p / "inject_time.txt").is_file())
    names = [n for n in names if n in labels.rows.index]
    names = [n for n in names if any(fnmatch(n, pat) for pat in patterns)]
    return names[:limit] if limit is not None else names


def is_done(path: Path) -> bool:
    """A case counts as done only if its JSON exists and parses (a half-written file is rerun)."""
    try:
        json.loads(path.read_text())
        return True
    except (OSError, ValueError):
        return False


def missing_cases(row_dir: Path, config_name: str, cases: list[str]) -> list[str]:
    out_dir = row_dir / config_name
    return [c for c in cases if not is_done(out_dir / f"{c}.json")]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def code_version(allow_dirty: bool) -> str:
    """HEAD, or HEAD plus a hash of the uncommitted diff, so different edits never share an ID."""
    head = _git("rev-parse", "HEAD").strip()
    if not _git("status", "--porcelain", "--untracked-files=no").strip():
        return head
    if not allow_dirty:
        raise SystemExit(
            "Tracked files have uncommitted changes. Commit them (or stash) before a formal run.\n"
            "For a smoke test only, pass --allow-dirty.\n" + _git("status", "--short", "--untracked-files=no")
        )
    diff_hash = hashlib.sha256(_git("diff", "HEAD").encode("utf-8")).hexdigest()[:12]
    return f"{head}-dirty-{diff_hash}"


def row_config(base_cfg: dict, row: str, out: Path, cache_dir: Path) -> dict:
    cfg = dict(base_cfg)
    cfg.update(ROWS[row])
    cfg["model"] = PIPELINE_MODEL
    cfg["cases_limit"] = SRC_RUN_CASES_LIMIT
    cfg["runs_dir"] = (out / row).as_posix()
    cfg["cache_dir"] = cache_dir.as_posix()
    return cfg


def check_experiment(path: Path, experiment: dict) -> list[str]:
    """Return the top-level fields that differ from an existing experiment.json ([] if new or equal)."""
    if not path.exists():
        return []
    old = json.loads(path.read_text())
    return [k for k in sorted(set(old) | set(experiment)) if old.get(k) != experiment.get(k)]


def run_row(row: str, cmd: list[str], row_dir: Path, log_path: Path, n_cases: int) -> int:
    env = dict(os.environ, PYTHONUNBUFFERED="1")  # stream src.run output line by line
    start = time.time()
    with log_path.open("a") as log:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [run_ablation] {row}: {n_cases} cases: {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{row}] {line}", end="")
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}")
            log.flush()
        rc = proc.wait()
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [run_ablation] {row}: exit code {rc}\n")
    end = time.time()
    with (row_dir / "timing.jsonl").open("a") as f:
        f.write(json.dumps({"start": start, "end": end, "seconds": round(end - start, 1), "cases": n_cases, "exit_code": rc}) + "\n")
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="+", choices=list(ROWS), default=list(ROWS), help="rows to run (always in the fixed order)")
    parser.add_argument("--cases", nargs="+", default=HELDOUT_CASES, help="case IDs or globs (default: held-out re2ss_* re2tt_*)")
    parser.add_argument("--cases-limit", type=int, help="keep only the first N selected cases (for smoke tests)")
    parser.add_argument("--out", type=Path, default=Path("runs/heldout"), help="output root (default: runs/heldout)")
    parser.add_argument("--base-config", type=Path, default=Path("configs/dev.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    parser.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes (smoke tests only)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and commands, run nothing")
    args = parser.parse_args()

    os.chdir(REPO)  # src.run resolves configs/models.yaml, data_dir, cache_dir relative to the repo root
    version = code_version(args.allow_dirty)
    base_cfg = yaml.safe_load(args.base_config.read_text())
    cases = select_cases(base_cfg, args.manifest, args.cases, args.cases_limit)
    if not cases:
        raise SystemExit(f"No cases matched {args.cases} in the manifest under {base_cfg['data_dir']}.")
    cache_dir = args.out / "llm_cache"

    # Every row's config is part of the lock, not only the rows selected now, so a later
    # --rows invocation cannot run under a different setup than the rows already finished.
    configs = {row: row_config(base_cfg, row, args.out, cache_dir) for row in ROWS}
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    experiment = {
        "code_version": version,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,  # labels can change without the path changing
        "cases": cases,
        "rows": configs,
    }
    exp_path = args.out / "experiment.json"
    diffs = check_experiment(exp_path, experiment)

    print(f"code version: {version}")
    print(f"cases: {len(cases)} selected by {args.cases}" + (f" (limit {args.cases_limit})" if args.cases_limit else ""))
    print(f"out: {args.out}  (cache: {cache_dir})")
    print(f"experiment.json: {'new' if not exp_path.exists() else ('MISMATCH in ' + ', '.join(diffs) if diffs else 'matches')}")
    rows = [r for r in ROWS if r in args.rows]
    plan = []
    for row in rows:
        cfg = configs[row]
        todo = missing_cases(args.out / row, cfg["config_name"], cases)
        plan.append((row, cfg, todo))
        print(f"  {row:10s} critic={cfg['critic']:10s} done {len(cases) - len(todo):3d}/{len(cases)}, to run {len(todo)}")

    if diffs:
        raise SystemExit(
            f"\n{exp_path} differs from this run in: {', '.join(diffs)}.\n"
            "Resuming would mix results. Use a new --out, or restore the code/config/case selection it was started with."
        )

    if args.dry_run:
        for row, cfg, todo in plan:
            print(f"\n[{row}] config.yaml would be:\n{yaml.safe_dump(cfg, sort_keys=False)}")
            if todo:
                print(f"[{row}] would run: {sys.executable} -m src.run --config {args.out / row / 'config.yaml'} "
                      f"--manifest {args.manifest} --cases {' '.join(todo)}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    if not exp_path.exists():
        exp_path.write_text(json.dumps(experiment, indent=2) + "\n")

    for row, cfg, todo in plan:
        if not todo:
            print(f"\n[{row}] all {len(cases)} cases done, skipping")
            continue
        row_dir = args.out / row
        row_dir.mkdir(parents=True, exist_ok=True)
        config_path = row_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        (row_dir / "git_commit.txt").write_text(version + "\n")

        cmd = [sys.executable, "-m", "src.run", "--config", str(config_path), "--manifest", str(args.manifest), "--cases", *todo]
        print(f"\n[{row}] running {len(todo)} cases (log: {args.out / (row + '.log')})")
        rc = run_row(row, cmd, row_dir, args.out / f"{row}.log", len(todo))

        still = missing_cases(row_dir, cfg["config_name"], cases)
        if rc != 0:
            raise SystemExit(f"\n[{row}] src.run exited with code {rc}; {len(still)} cases still missing. "
                             "Fix the problem, then run the same command again to resume.")
        if still:
            # src.run exited cleanly but some JSONs are missing: case_id may differ from the folder name.
            raise SystemExit(f"\n[{row}] src.run finished but {len(still)} cases have no JSON, e.g. {still[:3]}. "
                             f"Check the file names in {row_dir / cfg['config_name']}.")
        print(f"[{row}] done: {len(cases)}/{len(cases)}")

    print("\nAll rows done.")


if __name__ == "__main__":
    main()
