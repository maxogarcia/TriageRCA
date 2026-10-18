# TriageRCA RCAEval Setup

This guide describes how to set up the TriageRCA repository, install RCAEval as a Python dependency, download the RE2 index and telemetry, build the manifest, run the deterministic baseline, evaluate its predictions, and execute the EDA notebook.

The upstream RCAEval source tree is not copied into TriageRCA. Only the project scripts and notebook are maintained here.

## Repository layout

```text
TriageRCA/
├── scripts/
│   ├── download_index.py
│   ├── download_re2.py
│   ├── inspect_case.py
│   ├── build_manifest.py
│   ├── run_deterministic_baseline.py
│   ├── evaluate_results.py
│   └── __init__.py
├── notebooks/
│   └── 01_rcaeval_eda.ipynb
├── data/
├── results/
├── docs/
├── requirements.txt
└── README.md
```

## 1. Clone TriageRCA

On Ubuntu or WSL:

```bash
mkdir -p ~/projects
cd ~/projects
git clone <TriageRCA-repository-url> TriageRCA
cd TriageRCA
```

Replace `<TriageRCA-repository-url>` with the repository URL used by the team.

## 2. Create and activate the environment

Python 3.12 is the recommended team version:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file should pin the RCAEval package without its conflicting default dependency bundle:
```text
RCAEval==1.8.0
```




Verify the package installation:

```bash
python -c "import importlib.metadata as m; print('RCAEval', m.version('RCAEval'))"
```

The project scripts use the installed RCAEval package for telemetry loading. They do not require an upstream RCAEval checkout.

## 3. Download the index and RE2 data

Run these commands from the TriageRCA root:

```bash
python scripts/download_index.py
python scripts/download_re2.py
```

The downloaded files are stored under `data/`.


## 4. Build the manifest

```bash
python scripts/build_manifest.py
```

The expected output is:

```text
data/manifests/sample_manifest.csv
```

The manifest includes fields such as `dataset`, `case`, `case_dir`, `root_cause_service`, `fault`, and `inject_time`. Each teammate should build the manifest locally because it uses paths relative to their own TriageRCA checkout.

Verify it with:

```bash
python - <<'PY'
import pandas as pd

manifest = pd.read_csv("data/manifests/sample_manifest.csv")
print("Shape:", manifest.shape)
print("Columns:", manifest.columns.tolist())
print(manifest.head())
print(manifest["dataset"].value_counts())
PY
```

## 5. Inspect a case

After downloading RE2, run:

```bash
python scripts/inspect_case.py
```

This verifies that a case directory exists and that available telemetry and the injection timestamp can be loaded. Some datasets have no traces; missing traces alone do not indicate a failed download.

## 6. Run and evaluate the deterministic baseline

```bash
python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py
```

The baseline ranks services and indicators using metric changes around injection. The evaluator compares the predicted service rankings with the manifest's root-cause service labels.

Expected output files:

```text
results/baselines/deterministic_metric.jsonl
results/baselines/deterministic_metric_per_case.csv
results/baselines/deterministic_metric_summary.json
```

For the current 90-case sample, the reference run saved 90/90 predictions and produced:

| Metric | Reference result |
| --- | ---: |
| AC@1 | 0.755556 |
| AC@3 | 0.944444 |
| AC@5 | 0.977778 |
| MRR | 0.849815 |
| Mean rank among retrieved cases | 1.566667 |

These values depend on using the same sample manifest, data, and code. Tool calls and replans are zero for this baseline; latency and estimated cost are currently placeholders.

## 7. Register and run the EDA notebook

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install \
  --user \
  --name triagerca312 \
  --display-name "Python 3.12 (TriageRCA)"
```

Run the baseline and evaluator first if the notebook displays saved evaluation results. Then launch JupyterLab from `notebooks/` because the notebook uses `PROJECT_ROOT = Path.cwd().parent`:

```bash
cd ~/projects/TriageRCA/notebooks
python -m jupyterlab
```

In JupyterLab:

1. Open `01_rcaeval_eda.ipynb`.
2. Select the `Python 3.12 (TriageRCA)` kernel.
3. Choose **Kernel → Restart Kernel and Run All Cells**.
4. Confirm that the manifest, metrics, logs, traces, and baseline outputs load.

The notebook imports `rank_metrics` , `evaluate_predictions`and `make_prediction` from `scripts/`. The scripts use `Path(__file__).resolve().parents[1]` to find the TriageRCA root, while the notebook uses its working directory's parent. If imports fail, confirm that JupyterLab was launched from `TriageRCA/notebooks`.

## 8. Portable case-directory paths

The manifest should store relative paths such as:

```text
data/raw/rcaeval/RE2-OB/...
```

It should not store a user-specific path such as `/home/<username>/projects/TriageRCA/...`. The scripts resolve relative paths against the TriageRCA root and also support absolute paths when needed.

## 9. Clean rebuild from scratch

To remove generated artifacts and reproduce the data setup, first confirm that the terminal is at the TriageRCA root:

```bash
pwd
```

Then remove only generated directories:

```bash
rm -rf data results output
```

Cache cleanup is optional:

```bash
rm -rf __pycache__ .pytest_cache notebooks/.ipynb_checkpoints
```

Do not delete `.venv/`, `scripts/`, `notebooks/`, `requirements.txt`, or `docs/` unless you intentionally want to recreate those files or reinstall the environment. A teammate starting from a fresh clone does not need to delete anything.

After cleaning, repeat:

```bash
python scripts/download_index.py
python scripts/download_re2.py
python scripts/build_manifest.py
python scripts/inspect_case.py
python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py
```

## 10. Everyday command sequence

```bash
cd ~/projects/TriageRCA
source .venv/bin/activate

python scripts/download_index.py
python scripts/download_re2.py
python scripts/build_manifest.py
python scripts/inspect_case.py
python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py

cd notebooks
python -m jupyterlab
```

Downloads and manifest generation only need to be repeated when rebuilding or changing the data selection. Rerun the baseline and evaluator after changing their code or inputs.

