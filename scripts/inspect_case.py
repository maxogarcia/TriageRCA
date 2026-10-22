from pathlib import Path

import pandas as pd
from RCAEval.utility import read_logs, read_metrics, read_traces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "rcaeval"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "sample_manifest.csv"
)

manifest = pd.read_csv(MANIFEST_PATH)
row = manifest.iloc[0]

case_dir = Path(row["case_dir"])

print("Case:", row["case"])
print("Dataset:", row["dataset"])
print("Root-cause service:", row["root_cause_service"])
print("Fault:", row["fault"])
print("Directory:", case_dir)

inject_time = int(
    (case_dir / "inject_time.txt").read_text().strip()
)

metrics = read_metrics(case_dir)
logs = read_logs(case_dir)
traces = read_traces(case_dir)

print("\nInjection time:", inject_time)
print("Metrics shape:", metrics.shape)
print("Metric columns:", list(metrics.columns[:10]))

if logs is not None:
    print("Logs shape:", logs.shape)
    print("Log columns:", list(logs.columns))
else:
    print("Logs: unavailable")

if traces is not None:
    print("Traces shape:", traces.shape)
    print("Trace columns:", list(traces.columns))
else:
    print("Traces: unavailable")