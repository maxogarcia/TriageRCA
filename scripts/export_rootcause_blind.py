#!/usr/bin/env python3
"""Export a blind labeling CSV of the fixed root-cause claims, straight from run JSONs (no judge call).

Used to validate a judge prompt change on fresh dev cases: the humans label these claims before the
judge is run on the folder, so the labels stay blind. The claim text and claim_id come from
judge_grounding.py's own structured_claims (claim 1 = ranked_services[0] is the root-cause service,
2 = fault category, 3 = indicator), so they line up with the judge's grounding_claims.csv later.

Read-only on the runs; writes one CSV with the same columns as calibration_blind.csv, which
render_calibration.py turns into the labeling page.

Example:
  python scripts/export_rootcause_blind.py --out runs --rows quickdev_onset \\
      --blind results/grounding_onset/calibration_blind_rootcause.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_grounding import discover, structured_claims  # noqa: E402

KINDS = {"service": 1, "category": 2, "indicator": 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", nargs="+", required=True, help="run output dir(s), same as the judge's --out")
    ap.add_argument("--rows", nargs="*", help="only these rows (first dir under --out)")
    ap.add_argument("--claims", nargs="+", default=["service"], choices=list(KINDS),
                    help="which fixed claims to export (default: service only)")
    ap.add_argument("--blind", required=True, help="output CSV path")
    args = ap.parse_args()

    out = Path(args.blind)
    if out.exists():
        sys.exit(f"{out} already exists; it may hold a labeling batch. Pick a new name or move it first.")

    rows = []
    for row, p in discover(args.out, set(args.rows) if args.rows else None):
        try:
            run = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"skip unreadable {p}")
            continue
        if not isinstance(run, dict) or "evidence_ledger" not in run:
            continue
        report = run.get("report") or {}
        fixed = structured_claims(report)
        # structured_claims skips empty fields, so map each sentence back to its kind by prefix/position
        by_kind = {}
        for i, text in enumerate(fixed, 1):
            kind = ("service" if text.endswith("is the root-cause service of this incident.")
                    else "category" if text.startswith("The fault category is ")
                    else "indicator")
            by_kind[kind] = (i, text)
        for kind in args.claims:
            if kind not in by_kind:
                print(f"note: {row}/{run.get('case_id', p.stem)} has no {kind} claim")
                continue
            cid, text = by_kind[kind]
            rows.append({"row": row, "case_id": run.get("case_id", p.stem), "claim_id": cid,
                         "claim_type": "root-cause", "claim": text, "run_path": str(p),
                         "human_label": "", "human_evidence": ""})
    if not rows:
        sys.exit("no claims found; check --out/--rows (older dev runs: --out runs --rows <folder>)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} claims, {len({(r['row'], r['case_id']) for r in rows})} cases -> {out}")


if __name__ == "__main__":
    main()
