import json
from pathlib import Path

import pandas as pd
from RCAEval.utility import read_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def metric_service(metric_name: str) -> str:
    """
    Initial assumption: metric names start with the service name.
    Verify this against your actual metric columns.
    """
    return metric_name.split("_", 1)[0]
	
def rank_metrics(metrics, inject_time, window_seconds=600):
    metrics = metrics.copy()
    metrics["time"] = pd.to_numeric(metrics["time"], errors="coerce")

    pre = metrics[
        (metrics["time"] >= inject_time - window_seconds)
        & (metrics["time"] < inject_time)
    ]

    post = metrics[
        (metrics["time"] >= inject_time)
        & (metrics["time"] < inject_time + window_seconds)
    ]

    metric_rows = []

    for column in metrics.columns:
        if column == "time":
            continue

        pre_values = pd.to_numeric(pre[column], errors="coerce").dropna()
        post_values = pd.to_numeric(post[column], errors="coerce").dropna()

        if len(pre_values) < 2 or len(post_values) == 0:
            continue

        pre_mean = pre_values.mean()
        post_mean = post_values.mean()
        delta = post_mean - pre_mean
        pre_std = pre_values.std(ddof=0)

        # Prevent division by zero for nearly constant metrics.
        scale = max(pre_std, abs(pre_mean) * 0.01, 1e-8)
        anomaly_score = abs(delta) / scale

        if delta > 0:
            direction = "increase"
        elif delta < 0:
            direction = "decrease"
        else:
            direction = "no_change"

        metric_rows.append({
            "metric": column,
            "service": metric_service(column),
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "delta": delta,
            "direction": direction,
            "score": anomaly_score,
        })

    metric_scores = pd.DataFrame(metric_rows)

    if metric_scores.empty:
        return metric_scores, pd.DataFrame()

    metric_scores = metric_scores.sort_values(
        "score", ascending=False
    ).reset_index(drop=True)

    service_scores = (
        metric_scores
        .groupby("service", as_index=False)["score"]
        .max()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    return metric_scores, service_scores
	
def make_prediction(case_row):
    case_dir = PROJECT_ROOT / Path(case_row["case_dir"])
    inject_time = int(case_row["inject_time"])

    metrics = read_metrics(case_dir)

    metric_scores, service_scores = rank_metrics(
        metrics,
        inject_time=inject_time
    )

    return {
        "case_id": str(case_row["case"]),
        "ranked_services": service_scores["service"].tolist(),
        "ranked_indicators": metric_scores["metric"].tolist(),
        "indicator_scores": metric_scores.head(20).to_dict("records"),

        # Reserved for later systems:
        "evidence_ids": [],
        "claims": [],
        "tool_calls": 0,
        "replans": 0,
        "latency_seconds": None,
        "estimated_cost_usd": None,
    }


def main():
    manifest = pd.read_csv(
        PROJECT_ROOT / "data/manifests/sample_manifest.csv"
    )

    # Cell 10:
    results = []

    for _, case_row in manifest.iterrows():
        try:
            prediction = make_prediction(case_row)
            results.append(prediction)
        except Exception as exc:
            print(f"Failed on {case_row['case']}: {exc}")

    # Cell 11:
    results_dir = PROJECT_ROOT / "results/baselines"
    results_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = results_dir / "deterministic_metric.jsonl"

    with prediction_path.open("w") as f:
        for prediction in results:
            f.write(json.dumps(prediction) + "\n")

    print(f"Saved {len(results)}/{len(manifest)} predictions")
    print(prediction_path)


if __name__ == "__main__":
    main()