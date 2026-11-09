"""Download only cases selected by a committed manifest (default: team sample).

Relative paths resolve against the project root, independent of working directory.
Use --pattern only for explicit exploratory downloads, not shared experiments.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

REPO_ID = "phamquiluan/RCAEval"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("data/manifests/sample_manifest.csv")
DEFAULT_OUT = Path("data/raw/rcaeval")
REQUIRED_FILES = ("metrics.parquet", "inject_time.txt")


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_case_ids(path: Path, expected_cases: int | None = None) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "case" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest has no 'case' column: {path}")
        ids = [row.get("case") for row in reader]
    if not ids:
        raise ValueError("Manifest contains no cases")
    for case_id in ids:
        if not isinstance(case_id, str) or not re.fullmatch(r"re2(?:ob|ss|tt)_[A-Za-z0-9_-]+", case_id):
            raise ValueError(f"Invalid RE2 case ID: {case_id!r}")
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate case IDs")
    if expected_cases is not None and len(ids) != expected_cases:
        raise ValueError(f"Expected {expected_cases} cases, found {len(ids)}")
    return ids


def missing_required_files(out: Path, case_ids: list[str]) -> list[str]:
    return [f"{case_id}/{filename}" for case_id in case_ids for filename in REQUIRED_FILES
            if not (out / case_id / filename).is_file()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--manifest", type=Path, help=f"Default: {DEFAULT_MANIFEST}")
    selection.add_argument("--pattern", help="Explicit exploratory HF glob; bypasses manifest selection")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-cases", type=int, help="Optional exact manifest row count, e.g. 90")
    parser.add_argument("--revision", help="Optional HF dataset commit SHA/tag; default is upstream main")
    parser.add_argument("--check-only", action="store_true", help="Check required local files without network access")
    args = parser.parse_args(argv)
    if args.expected_cases is not None and args.expected_cases < 1:
        parser.error("--expected-cases must be positive")
    if args.pattern is not None and (args.check_only or args.expected_cases is not None):
        parser.error("--check-only and --expected-cases require manifest mode")
    out = project_path(args.out)
    ids = None
    if args.pattern is None:
        manifest = project_path(args.manifest if args.manifest is not None else DEFAULT_MANIFEST)
        try:
            ids = read_case_ids(manifest, args.expected_cases)
        except (OSError, ValueError, csv.Error) as exc:
            parser.error(str(exc))
        patterns = [f"{case_id}/*" for case_id in ids]
        print(f"Selected {len(ids)} cases from {manifest}")
    else:
        if not args.pattern.strip():
            parser.error("--pattern must not be empty")
        patterns = [args.pattern]
        print("Exploratory pattern mode: not restricted to the shared manifest")
    if not args.check_only:
        # Lazy import: --check-only and the test suite need no network or HF installation.
        from huggingface_hub import snapshot_download
        out.mkdir(parents=True, exist_ok=True)
        kwargs = dict(repo_type="dataset", local_dir=str(out), allow_patterns=patterns)
        if args.revision:
            kwargs["revision"] = args.revision
        snapshot_download(REPO_ID, **kwargs)
    if ids is not None:
        missing = missing_required_files(out, ids)
        if missing:
            raise SystemExit("Missing required files (first 20):\n" + "\n".join(missing[:20])
                             + f"\nTotal missing files: {len(missing)}")
        print(f"Verified metrics.parquet and inject_time.txt for all {len(ids)} selected cases in {out}")
        print("Logs/traces are downloaded when present; their availability is not validated here.")
    else:
        print(f"Downloaded matching files to {out}; no manifest completeness check was performed")


if __name__ == "__main__":
    main()
