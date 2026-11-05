"""Score saved RunRecord JSONs without sending labels to any model.

Run from the repository root: python -m scripts.evaluate_agent_runs --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.labels import DEFAULT_MANIFEST, ManifestLabels


def evaluate(runs_dir: Path, labels: ManifestLabels):
    # Each directory is one config/model. Every manifest case is in the denominator.
    predictions = {}
    for path in sorted(runs_dir.glob("*.json")):
        record = json.loads(path.read_text())
        case_id = record["case_id"]
        labels.row(case_id)  # Reject unrecognised case IDs rather than quietly mixing samples.
        if case_id in predictions:
            raise ValueError(f"Duplicate run for {case_id}")
        predictions[case_id] = record
    if not predictions:
        raise ValueError(f"No run JSON files found directly in {runs_dir}")

    rows = []
    for case_id in labels.rows.index:
        truth = labels.label(case_id)
        record = predictions.get(case_id)
        report = (record or {}).get("report")
        ranked = (report or {}).get("ranked_services", [])
        if not isinstance(ranked, list) or not all(isinstance(x, str) for x in ranked):
            raise ValueError(f"Invalid ranked_services for {case_id}")
        if len(ranked) != len(set(ranked)):
            raise ValueError(f"Duplicate ranked services for {case_id}")
        rank = ranked.index(truth.root_cause_service) + 1 if truth.root_cause_service in ranked else None
        indicator_known = truth.indicator != "unknown"
        rows.append({
            "case_id": case_id,
            "true_service": truth.root_cause_service,
            "record_present": record is not None,
            "report_present": report is not None,
            "has_errors": bool((record or {}).get("errors")),
            **{f"AC@{k}": int(rank is not None and rank <= k) for k in (1, 3, 5)},
            "MRR": 1 / rank if rank else 0.0,
            "fault_accuracy": int((report or {}).get("fault_category") == truth.fault_category),
            "indicator_accuracy": int((report or {}).get("indicator") == truth.indicator) if indicator_known else None,
        })
    n = len(rows)
    indicator_rows = [r for r in rows if r["indicator_accuracy"] is not None]
    summary = {
        "num_expected_cases": n,
        "num_records": len(predictions),
        "num_reports": sum(r["report_present"] for r in rows),
        "num_records_with_errors": sum(r["has_errors"] for r in rows),
        **{key: sum(r[key] for r in rows) / n for key in ("AC@1", "AC@3", "AC@5", "MRR", "fault_accuracy")},
        "indicator_label_coverage": len(indicator_rows) / n,
        "indicator_accuracy": sum(r["indicator_accuracy"] for r in indicator_rows) / len(indicator_rows) if indicator_rows else None,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-dir", required=True, type=Path, help="one model/config directory, containing case JSON files")
    parser.add_argument("--output", required=True, type=Path, help="evaluation JSON, outside the run-record directory")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.runs_dir.resolve()):
        parser.error("Save evaluation outside --runs-dir to avoid mixing scores with run records")
    rows, summary = evaluate(args.runs_dir, ManifestLabels(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_case": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
