"""Download RCAEval cases from Hugging Face in the flat Parquet layout the adapter expects:

    data/re2ob/re2ob_checkoutservice_cpu_1/{metrics,logs,traces}.parquet, inject_time.txt

Use this rather than RCAEval.utility.download_re2ob_dataset(): that pulls the Zenodo zip, whose
nested CSV layout (checkoutservice_cpu/1/metrics.csv, raw metric names, string-typed logs/traces)
the tools don't read.

Usage:
    python scripts/download_data.py                       # all 90 RE2-OB cases
    python scripts/download_data.py --pattern "re2ob_checkoutservice_cpu_1*"   # one case
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

REPO_ID = "phamquiluan/RCAEval"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="re2ob_*", help="glob over case dir names, e.g. 're2ob_cartservice*'")
    parser.add_argument("--out", default="data/re2ob")
    args = parser.parse_args()

    snapshot_download(REPO_ID, repo_type="dataset", local_dir=args.out, allow_patterns=[args.pattern])
    print(f"Downloaded cases matching {args.pattern!r} to {args.out}")


if __name__ == "__main__":
    main()
