# TriageRCA/RCAEval Team Setup

This guide describes how to set up RCAEval, download the index and RE2 data, inspect a case, build the manifest, run and evaluate the project's deterministic baseline, and execute the EDA notebook. RCAEval's built-in baselines are a separate optional step.

The repository currently contains these helper scripts in the RCAEval scripts directory:

```text
download_index.py
download_re2.py
inspect_case.py
build_manifest.py
__init__.py
run_deterministic_baseline.py
evaluate_results.py
```

The notebook can be found in the notebooks directory:

```text
01_rcaeval_eda.ipynb
```


## 1. Clone RCAEval

On Ubuntu or WSL:

```bash
mkdir -p ~/projects
cd ~/projects

git clone https://github.com/phamquiluan/RCAEval.git
cd RCAEval
```

## 2. Create and activate the environment

Python 3.12 is recommended for team consistency:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[default]"
python -m pip install jupyterlab ipykernel
```

If Python 3.14 is used instead, replace `python3.12` with `python3.14` when creating the environment. The later commands remain the same.

Verify the installation:

```bash
python -m pytest tests/test.py::test_basic
```

## 3. Download the index and RE2

Run the project scripts from the RCAEval root directory:

```bash
python scripts/download_index.py
python scripts/download_re2.py
```

The exact downloaded files should be placed under the repository's `data/` directory.

## 4. Inspect a case

After downloading RE2, run:

```bash
python scripts/inspect_case.py
```

This should verify that a case directory exists and that its available telemetry and injection timestamp can be loaded. Some datasets have no traces; unavailable traces alone do not indicate a failed download.

## 5. Build the manifest

Run:

```bash
python scripts/build_manifest.py
```

The expected output is:

```text
data/manifests/sample_manifest.csv
```

The manifest should contain fields such as:

```text
dataset
case
case_dir
root_cause_service
fault
inject_time
```

The current `build_manifest.py` does not use a hardcoded home-directory path, so each teammate can generate the manifest locally.

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

## 6. Run and evaluate the project's deterministic baseline

From the RCAEval root, run:

```bash
python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py
```

The first script uses metric changes before and after injection to rank services and indicators. The second compares the saved rankings against the manifest's true root-cause service labels.

Expected output files:

```text
results/baselines/deterministic_metric.jsonl
results/baselines/deterministic_metric_per_case.csv
results/baselines/deterministic_metric_summary.json
```

For the current 90-case sample, the baseline should print:

```text
Saved 90/90 predictions
```

Confirm the saved count matches the manifest count. The current implementation skips failed cases, so a smaller count means evaluation could exclude failures.

With the same sample manifest, data, and code, the reference results are:

| Metric | Reference result |
| --- | ---: |
| AC@1 | 0.755556 |
| AC@3 | 0.944444 |
| AC@5 | 0.977778 |
| MRR | 0.849815 |
| Mean rank among retrieved cases | 1.566667 |

These values apply to the current sample, not every RE2 subset. AC@k measures whether the true service occurs within the top k predictions. MRR averages the reciprocal of its rank, with zero for missing services. The current mean rank excludes cases where the true service is absent.

Tool calls and replans are zero for this baseline. Latency and estimated cost are currently `None` placeholders.

## 7. Optional: run RCAEval's built-in baselines

These commands run methods provided by RCAEval. They are separate from our deterministic baseline and are not required to execute the EDA notebook.

From the RCAEval root, first check the available options:

```bash
python main.py --help
```

For example, run multi-source BARO on RE2 Online Boutique:

```bash
python main.py \
  --method mmbaro \
  --dataset re2-ob \
  --length 20 \
  --report-chance
```

Run it on the other RE2 datasets with:

```bash
python main.py --method mmbaro --dataset re2-ss --length 20 --report-chance
python main.py --method mmbaro --dataset re2-tt --length 20 --report-chance
```

Multi-source CIRCA can be run with:

```bash
python main.py \
  --method mmcirca \
  --dataset re2-ob \
  --length 20 \
  --report-chance
```

The generated baseline files are normally written under `output/` or another output directory reported by the script.

## 8. Register the Jupyter kernel

Install the kernel package inside the activated environment:

```bash
python -m pip install ipykernel

python -m ipykernel install \
  --user \
  --name rcaeval312 \
  --display-name "Python 3.12 (RCAEval)"
```

## 9. Run the EDA notebook

Run the two project baseline scripts first if the notebook displays saved evaluation results. The notebook now imports the baseline functions instead of defining and running the entire batch pipeline itself.

The current notebook uses:

```python
PROJECT_ROOT = Path.cwd().parent
```

Therefore, launch JupyterLab from the `notebooks` directory:

```bash
cd ~/projects/RCAEval/notebooks
python -m jupyterlab
```

In JupyterLab:

1. Open `01_rcaeval_eda.ipynb`.
2. Select the `Python 3.12 (RCAEval)` kernel.
3. Choose **Kernel → Restart Kernel and Run All Cells**.
4. Confirm that the manifest loads and that metrics, logs, and traces are displayed for the sample case.

Do not run only the final cells. The notebook must execute from top to bottom because later cells depend on variables and functions defined earlier.

The notebook import cell should make the project root importable:

```python
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_deterministic_baseline import rank_metrics
from scripts.evaluate_results import evaluate_predictions
```

An empty `scripts/__init__.py` is included with the project files. Both scripts use `if __name__ == "__main__":` guards so importing their functions does not execute a batch run.

The scripts resolve their root with `Path(__file__).resolve().parents[1]`; the notebook retains `Path.cwd().parent`. If imports fail, check that the `scripts/` files exist and that the notebook's `PROJECT_ROOT` points to RCAEval.

## 10. Portable case-directory paths

The manifest should preferably store relative paths, for example:

```text
data/RE2/RE2-OB/...
```

It should not store a user-specific path such as:

```text
/home/(username)/projects/RCAEval/data/RE2/RE2-OB/...
```

The notebook can support either form with:

```python
case_dir_value = Path(str(case["case_dir"]))

if case_dir_value.is_absolute():
    case_dir = case_dir_value
else:
    case_dir = PROJECT_ROOT / case_dir_value

assert case_dir.exists(), f"Case directory not found: {case_dir}"
```

## 11. Clean rebuild from scratch

If the goal is to reproduce the data setup from scratch, it is safe to remove generated data and result artifacts.

From the RCAEval root, remove or empty:

```text
data/
results/
output/
```

You may also remove generated Python and notebook caches:

```text
__pycache__/
.pytest_cache/
notebooks/.ipynb_checkpoints/
```

On Linux or WSL, examples are:

```bash
rm -rf data results output
```

Only remove generated contents from `data/`; retain any hand-written configuration or source files if present. Cache cleanup is optional and does not affect reproducing the downloads. Leave installed package metadata such as `*.egg-info` intact.

Only run those commands from the RCAEval repository root after confirming the path with:

```bash
pwd
```

Do **not** delete the following unless you intentionally want to reinstall the environment or reclone the repository:

```text
.venv/
download_index.py
download_re2.py
inspect_case.py
build_manifest.py
scripts/
notebooks/
RCAEval/
main.py
```

After cleaning, repeat the setup sequence:

```bash
python download_index.py
python download_re2.py
python inspect_case.py
python build_manifest.py
python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py
```

For a teammate starting from a fresh clone, nothing needs to be deleted. Do not distribute your `.venv`; each teammate creates their own environment. Keep a copy of the reference results before cleaning if you want to compare the rebuild.

## 12. Everyday command sequence

After installation and copying the team files:

```bash
cd ~/projects/RCAEval
source .venv/bin/activate

python scripts/download_index.py
python scripts/download_re2.py
python scripts/inspect_case.py
python scripts/build_manifest.py

python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py

cd notebooks
python -m jupyterlab
```

Downloads and manifest generation need only be repeated when rebuilding or changing the data selection. Rerun the baseline and evaluator after changes to their code or inputs.

Then launch JupyterLab from `RCAEval/notebooks` and use **Restart Kernel and Run All Cells**.
