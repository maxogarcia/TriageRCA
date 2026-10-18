from pathlib import Path
from huggingface_hub import snapshot_download

data_dir = Path("data/raw/rcaeval")
data_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="phamquiluan/RCAEval",
    repo_type="dataset",
    allow_patterns="re2*",
    local_dir=str(data_dir),
)

print(f"RE2 data downloaded to: {data_dir}")