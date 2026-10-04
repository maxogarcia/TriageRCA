# TriageRCA — RCAEval Data and Baselines

This workspace supports the first stage of **TriageRCA**: loading RCAEval RE2 telemetry, building a case manifest, exploring the data, and establishing a deterministic metric baseline with a shared evaluator.

It builds on [RCAEval](https://github.com/phamquiluan/RCAEval). Retain the upstream `LICENSE` and `LICENSES/` files. See [the team setup guide](TriageRCA_RCAEval_Setup.md) for detailed instructions.

## Setup

Use Ubuntu or WSL and the team's Python 3.12 environment. From the RCAEval repository root, with the team's scripts and notebook present:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[default]"
python -m pip install jupyterlab ipykernel
python -m pytest tests/test.py::test_basic
```

## Download and prepare data

Run all terminal commands from the repository root:

```bash
python scripts/download_index.py
python scripts/download_re2.py
python scripts/inspect_case.py
python scripts/build_manifest.py
```

These steps download the case index and RE2 telemetry, inspect a case, and generate `data/manifests/sample_manifest.csv`. Each teammate should build the manifest locally.

## Run and evaluate the baseline

```bash
python scripts/run_deterministic_baseline.py
python scripts/evaluate_results.py
```

The deterministic baseline ranks metrics by their normalized mean change across the injection time, then ranks services by their highest metric score. The evaluator reports service AC@1/3/5 and MRR.

Results are saved under `results/baselines/`:

- `deterministic_metric.jsonl`: case predictions and metric scores.
- `deterministic_metric_per_case.csv`: per-case evaluation.
- `deterministic_metric_summary.json`: aggregate scores.

Confirm that all manifest cases produce predictions. The current 90-case reference run achieved AC@1 = 0.755556, AC@3 = 0.944444, AC@5 = 0.977778, and MRR = 0.849815. These values depend on using the same manifest, data, and code.

## Explore the notebook

```bash
python -m ipykernel install --user --name rcaeval312 \
  --display-name "Python 3.12 (RCAEval)"
cd notebooks
python -m jupyterlab
```

Open `01_rcaeval_eda.ipynb`, select the RCAEval kernel, and **restart the kernel and run all cells**. Launch from `notebooks/` because the notebook uses `PROJECT_ROOT = Path.cwd().parent`. The notebook imports functions from `scripts/`; run the baseline and evaluator first to generate any saved results it displays.

## Project files

| Location | Purpose |
| --- | --- |
| `scripts/` | Download, inspection, manifest, baseline, and evaluation scripts; includes `__init__.py` |
| `notebooks/01_rcaeval_eda.ipynb` | Data exploration, plots, and single-case ranking previews |
| `data/` | Downloaded telemetry and generated manifests |
| `results/baselines/` | Project baseline predictions and evaluation |
| `main.py` | Upstream RCAEval benchmark runner |

The project baseline is separate from RCAEval's built-in methods. Running `main.py` is optional for the notebook workflow; use `python main.py --help` for its available options.

Agent baselines, indicator accuracy, and independent evidence-grounding evaluation are later project stages. Tool calls and replans are currently zero for the deterministic baseline; latency and cost fields are placeholders.
