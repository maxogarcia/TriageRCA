import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def reciprocal_rank(ranked_items, target):
    for index, item in enumerate(ranked_items, start=1):
        if item == target:
            return 1 / index
    return 0.0


def item_rank(ranked_items, target):
    for index, item in enumerate(ranked_items, start=1):
        if item == target:
            return index
    return np.nan
def evaluate_predictions(prediction_path, manifest):
    truth = manifest.set_index("case")
    rows = []

    with open(prediction_path) as f:
        predictions = [json.loads(line) for line in f]

    for prediction in predictions:
        case_id = prediction["case_id"]

        if case_id not in truth.index:
            print(f"Warning: {case_id} not found in manifest")
            continue

        true_service = str(truth.loc[case_id, "root_cause_service"])
        ranked_services = prediction.get("ranked_services", [])

        rank = item_rank(ranked_services, true_service)

        rows.append({
            "case_id": case_id,
            "true_service": true_service,
            "predicted_top1": ranked_services[0]
            if ranked_services else None,
            "rank": rank,
            "AC@1": int(true_service in ranked_services[:1]),
            "AC@3": int(true_service in ranked_services[:3]),
            "AC@5": int(true_service in ranked_services[:5]),
            "MRR": reciprocal_rank(ranked_services, true_service),
            "tool_calls": prediction.get("tool_calls", 0),
            "replans": prediction.get("replans", 0),
            "latency_seconds": prediction.get("latency_seconds"),
            "estimated_cost_usd": prediction.get("estimated_cost_usd"),
        })

    per_case = pd.DataFrame(rows)

    summary = {
        "num_cases": len(per_case),
        "AC@1": per_case["AC@1"].mean(),
        "AC@3": per_case["AC@3"].mean(),
        "AC@5": per_case["AC@5"].mean(),
        "MRR": per_case["MRR"].mean(),
        "mean_rank": per_case["rank"].mean(),
        "mean_tool_calls": per_case["tool_calls"].mean(),
        "mean_replans": per_case["replans"].mean(),
    }

    return per_case, summary


def main():
    manifest = pd.read_csv(
        PROJECT_ROOT / "data/manifests/sample_manifest.csv"
    )

    results_dir = PROJECT_ROOT / "results/baselines"
    prediction_path = results_dir / "deterministic_metric.jsonl"

    # Cell 14:
    per_case, summary = evaluate_predictions(
        prediction_path,
        manifest,
    )

    print(pd.Series(summary))
    print(per_case.head().to_string(index=False))

    # Cell 15:
    per_case.to_csv(
        results_dir / "deterministic_metric_per_case.csv",
        index=False,
    )

    with (
        results_dir / "deterministic_metric_summary.json"
    ).open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()