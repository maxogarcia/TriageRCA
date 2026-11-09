# TriageRCA: A Planner-Executor-Critic Agent for Root Cause Analysis

**Course:** CMPE 258 — Deep Learning (Fall 2026)
**Track:** LLMs + AI Agent System (Evaluation-First)

## Team Members

- Maximilian (Max) Garcia
- Yiang (Aeon) Shen
- Daniel Cai

## Abstract

When a cloud microservice degrades, on-call engineers must sift through logs, metrics, and traces across dozens of interdependent services to find what actually broke. This is slow, stressful, and error-prone, and it is exactly the kind of grounded reasoning task where a single-shot large language model tends to hallucinate a plausible but wrong answer.

TriageRCA is a multi-agent system that performs automated root cause analysis (RCA) for microservice incidents. It takes an incident window of telemetry plus a natural-language incident description and produces a structured, evidence-grounded diagnosis: the root-cause service, the specific indicator (a metric or log line), the likely fault category, and a proposed mitigation, with a human making the final decision. The core of the system is a **Planner-Executor-Critic** loop: a Planner decomposes the investigation, an Executor calls telemetry and runbook-retrieval tools, and a Critic verifies every hypothesis against retrieved evidence before it is reported. This verification step directly targets the dominant failure mode recent work has documented in LLM RCA agents, namely confidently misinterpreting data that does not support the conclusion.

We evaluate on **RCAEval**, a public benchmark of 735 labeled failure cases across three microservice systems, and we report not only localization accuracy (AC@k) but also an evidence-grounding (faithfulness) score and a controlled cost/latency comparison across three models (a GPT-class model, a Claude-class model, and one open-weight model). The project's contribution is a reliability-focused, reproducible RCA agent rather than a raw accuracy record.

## Problem, in one paragraph

Given telemetry (metrics, logs, traces) for a failing microservice system and a short incident query, output the root-cause service and indicator, a grounded explanation, and a suggested fix. Success is measured by top-k localization accuracy, indicator correctness, evidence-grounding rate, and per-incident cost and latency, benchmarked against a ReAct single-agent baseline and a non-agent retrieval baseline.

## Approach at a glance

- **Planner** — turns the incident into an investigation plan, guided by the service-dependency graph so it inspects likely-culprit services first.
- **Executor** — a tool-augmented agent that queries logs, metrics, and traces and retrieves relevant runbooks / past postmortems (RAG).
- **Critic** — checks each candidate root cause against the retrieved evidence, rejects unsupported claims, and can send the Planner back for more investigation. This is the reliability core of the system.
- **Human-in-the-loop** — the agent produces a report; a human makes the final call, as recommended in our proposal review.

## Data

- **Primary:** [RCAEval](https://github.com/phamquiluan/RCAEval) — 735 labeled failure cases across Online Boutique, Sock Shop, and Train Ticket; metrics, logs, and traces; 11 injected fault types; ground-truth root-cause service and indicator. CC BY 4.0, installable from PyPI.
- **RAG corpus:** 30–100 runbooks / SOPs / postmortems (curated + synthesized) for grounded retrieval.
- **Extension set:** a small authored set of authorization / access-violation incidents layered on real telemetry, to cover incident types beyond trivial crashes.

### Setup

Python 3.12 (team standard). Run these commands from the project root in Linux/WSL.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py --expected-cases 90
python scripts/download_data.py --expected-cases 90 --check-only
python -m src.rag.embed  # build the runbook index -> data/chroma_db/
```

The downloader defaults to the committed `data/manifests/sample_manifest.csv` and requests
only those **90 selected RE2 case directories across the three systems**, rather than all
Online Boutique cases or all RE2 data. Telemetry is stored in the flat Parquet layout:

```text
data/raw/rcaeval/<case_id>/metrics.parquet
data/raw/rcaeval/<case_id>/inject_time.txt
data/raw/rcaeval/<case_id>/logs.parquet    # when available
data/raw/rcaeval/<case_id>/traces.parquet  # when available
```

Both `data/manifests/sample_manifest.csv` and `data/manifests/rcaeval_re2_cases.csv` are shared
Git-tracked metadata. Raw telemetry remains ignored. Do not regenerate the shared sample
during normal setup: `build_manifest.py` is for intentional sample rebuilding, not a
prerequisite to downloading the committed cases. Local availability columns in a committed
CSV may be stale; use `--check-only` to check this machine. This checks required file existence,
not telemetry contents, label validity, or optional log/trace completeness.

`scripts/download_re2.py` delegates to the same downloader; it no longer downloads all RE2
cases by default. Existing extra data is not deleted. `--pattern` remains available for
explicit exploratory downloads, but does not define the shared experiment sample.
Use `--manifest path/to/another_manifest.csv` for another fixed sample and `--out` to change
the destination. Relative downloader paths resolve against the project root.

The manifest fixes case selection, not upstream file contents. For byte-level reproducibility,
agree on a Hugging Face dataset commit and pass the same `--revision <commit_sha>`; without
that option, downloads use upstream main.

Use the Hugging Face Parquet layout, not `RCAEval.utility.download_re2ob_dataset()` (Zenodo zip:
nested `service_fault/N/` directories with `metrics.csv`, which the adapter rejects).

Set `data_dir: data/raw/rcaeval` in `configs/dev.yaml`. The runner must filter case directories
against `--manifest` before applying `--cases` and `cases_limit`, so previously downloaded
non-manifest cases cannot enter the experiment. Labels are used offline, not passed to agents.
The current sample has root-cause service and fault-category labels; verified indicator labels
are not populated, so indicator accuracy is unavailable until annotation is completed.

Configure the selected hosted model's credentials in `.env` (for the Gemini configuration,
`GEMINI_API_KEY`), then run tests and a one-case smoke test:

```bash
python -m pytest -q
python -m src.run --config configs/dev.yaml \
  --manifest data/manifests/sample_manifest.csv --cases-limit 1
```

### Open-weight model

The open-weight slot (`open_weight_model` in `configs/models.yaml`) is Qwen2.5-7B-Instruct served by
[Ollama](https://ollama.com) through its OpenAI-compatible endpoint. GPU memory requirements
depend on quantization and context length; verify the actual allocation on your hardware.

```bash
# In a dedicated terminal, only if an Ollama server is not already running:
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

Leave that terminal running. In another terminal with the project environment activated:

```bash
ollama pull qwen2.5:7b-instruct
python -m src.run --config configs/dev.yaml --model open_weight_model \
  --manifest data/manifests/sample_manifest.csv --cases-limit 1 --runs-dir runs/qwen_smoke
```

While the model is loaded, verify the actual server allocation:

```bash
ollama ps
```

Check `CONTEXT` is `32768` and inspect `PROCESSOR` for GPU/CPU placement. Setting a variable in
a separate shell does not reconfigure an already-running server. If Ollama is managed by an
app, service, or container, configure that launch mechanism instead of starting a second
server on the same port. See [Ollama context configuration](https://docs.ollama.com/context-length).
Record the active context setting with results and check that replanning prompts, tool
definitions and generated output fit; a 32K allocation alone is not a guarantee.

Use a separate `--runs-dir` per model: records are written to `<runs_dir>/<config_name>/<case_id>.json`, so
two models sharing one runs dir overwrite each other. Also use a fresh directory per experiment
version. A fresh runs directory does not disable the LLM cache; configure a separate `cache_dir`
for fresh-inference measurements.

### Investigation tool budget

All telemetry-enabled variants (`react`, `triagerca`, and `triagerca_no_critic`) receive one
deterministic, label-free all-service metric overview before investigation. This overview
consumes one call from the first iteration's tool budget.

With `max_tool_calls: 6`, five discretionary tool calls remain in the first iteration.
Subsequent investigation iterations retain six calls; the overview is not collected again.
This is a per-iteration limit, not a six-call limit for the entire incident. A zero budget
collects no overview. The `retrieval_only` baseline receives no telemetry overview.
Reported LLM-call counts are separate from this tool-call budget.

### Offline scoring

Score each configuration/model directory against the manifest used for that run:

```bash
python -m scripts.evaluate_agent_runs \
  --manifest data/manifests/sample_manifest.csv \
  --runs-dir runs/qwen_smoke/triagerca \
  --output results/qwen_smoke.json
```

This path assumes `config_name: triagerca`; substitute the actual configuration directory.
A one-case smoke run scored against the 90-case manifest counts the other cases as missing,
so this is not a full-sample accuracy result. For subset experiments, use a subset manifest
consistently for running and scoring. Valid citation IDs establish reference validity, not
semantic grounding; grounding requires independent evaluation.

## Repository Structure

```
triage-rca/
├── README.md                      # this file (Deliverable A)
├── docs/
│   ├── LITERATURE_SURVEY.md       # Deliverable B: 10-paper SOTA survey
│   ├── NOVELTY_AUDIT.md           # Deliverable D: AI novelty & feasibility audit
│   └── TriageRCA_Proposal.pdf     # Deliverable C: proposal document (also submitted to Canvas)
├── src/
│   ├── agents/                    # planner, executor, critic
│   ├── tools/                     # log_search, metric_query, trace_query, runbook_retrieval
│   ├── rag/                       # embedding + vector store over runbooks/postmortems
│   ├── data/                      # RCAEval loaders, case sampler, extension incidents
│   └── eval/                      # AC@k, indicator accuracy, grounding score, cost/latency
├── configs/                       # model + experiment configs (3-model comparison)
├── scripts/                       # manifest-based download, manifest building, offline evaluation
├── data/
│   ├── manifests/                 # committed shared index and fixed sample CSVs
│   ├── raw/rcaeval/               # downloaded telemetry (not committed)
│   └── runbooks/                  # curated/synthesized retrieval corpus
├── notebooks/                     # EDA + result analysis
└── requirements.txt
```

