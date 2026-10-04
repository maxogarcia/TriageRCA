from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

index_path = hf_hub_download(
    repo_id="phamquiluan/RCAEval",
    repo_type="dataset",
    filename="cases.parquet",
    local_dir=str(OUTPUT_DIR),
)

index = pd.read_parquet(index_path)

index.to_parquet(
    OUTPUT_DIR / "rcaeval_cases.parquet",
    index=False,
)

re2 = index[
    index["dataset"]
    .astype(str)
    .str.upper()
    .str.startswith("RE2")
]

re2.to_csv(
    OUTPUT_DIR / "rcaeval_re2_cases.csv",
    index=False,
)

print(f"All cases: {len(index)}")
print(f"RE2 cases: {len(re2)}")
print(re2.groupby(["dataset", "fault"]).size())