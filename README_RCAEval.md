# TriageRCA — RCAEval Data and Baselines

This workspace contains the first TriageRCA implementation stage: loading RCAEval RE2 telemetry, building a case manifest, exploring the data, and establishing a deterministic metric baseline with a shared evaluator.

RCAEval is installed as a Python dependency; its upstream source tree is not copied into this repository. Retain any required upstream license files included with the project. See the [team setup guide](TriageRCA_RCAEval_Setup.md) for detailed instructions.

## Setup

Use Ubuntu or WSL with the team's Python 3.12 environment. From the TriageRCA repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import importlib.metadata as m; print('RCAEval', m.version('RCAEval'))"
```



## Download and prepare data

Run all commands from the TriageRCA root:

```bash
python scripts/download_index.py
python scripts/download_re2.py
python scripts/inspect_case.py
python scripts/build_manifest.py
```

These steps download the case index and RE2 telemetry, inspect a case, and generate `data/manifests/sample_manifest.csv`. Each teammate should build the manifest locally so its paths remain portable.

## Run and evaluate the baseline

```bash
python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py
```

The deterministic baseline ranks metrics by normalized mean change around injection, then ranks services by their highest metric score. The evaluator reports service AC@1/3/5 and MRR.

Results are saved under `results/baselines/`:

- `deterministic_metric.jsonl`: case predictions and metric scores.
- `deterministic_metric_per_case.csv`: per-case evaluation.
- `deterministic_metric_summary.json`: aggregate scores.

Confirm that all manifest cases produce predictions. The current 90-case reference run achieved AC@1 = 0.755556, AC@3 = 0.944444, AC@5 = 0.977778, and MRR = 0.849815. These values depend on using the same manifest, data, and code.

## Explore the notebook

Register the environment as a kernel and launch JupyterLab from `notebooks/`:

```bash
python -m ipykernel install --user --name triagerca312 \
  --display-name "Python 3.12 (TriageRCA)"
cd notebooks
python -m jupyterlab
```

Open `01_rcaeval_eda.ipynb`, select the TriageRCA kernel, and choose **Kernel → Restart Kernel and Run All Cells**. Launch from `notebooks/` because the notebook uses `PROJECT_ROOT = Path.cwd().parent`. The notebook imports functions from `scripts/`; run the baseline and evaluator first if it displays saved results.

## Project files

| Location | Purpose |
| --- | --- |
| `scripts/` | Download, inspection, manifest, baseline, and evaluation scripts; includes `__init__.py` |
| `notebooks/01_rcaeval_eda.ipynb` | Data exploration, plots, and single-case ranking previews |
| `data/` | Downloaded telemetry and generated manifests |
| `results/baselines/` | Project baseline predictions and evaluation |
| `requirements.txt` | Pinned Python dependencies, including RCAEval |

Agent baselines, indicator accuracy, and independent evidence-grounding evaluation are later project stages. Tool calls and replans are zero for the deterministic baseline; latency and cost fields are placeholders.

