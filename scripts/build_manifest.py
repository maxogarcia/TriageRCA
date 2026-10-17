from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "rcaeval_re2_cases.csv"
)

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "rcaeval"
OUTPUT_PATH = PROJECT_ROOT / "data" / "manifests" / "sample_manifest.csv"

index = pd.read_csv(INDEX_PATH)

# Select one case for every system × root-cause service × fault combination.
# This produces a balanced development sample of approximately 90 cases.
sample = (
    index
    .groupby(
        ["dataset", "root_cause_service", "fault"],
        group_keys=False
    )
    .sample(n=1, random_state=42)
    .sort_values(["dataset", "fault", "root_cause_service"])
)

sample["case_dir"] = sample["case"].apply(
    lambda case: str(DATA_ROOT / case)
)

sample["exists"] = sample["case_dir"].apply(
    lambda path: Path(path).exists()
)

print(f"Selected cases: {len(sample)}")
print(f"Missing case directories: {(~sample['exists']).sum()}")
print(sample.groupby(["dataset", "fault"]).size())

sample.to_csv(OUTPUT_PATH, index=False)

print(f"Saved manifest to: {OUTPUT_PATH}")