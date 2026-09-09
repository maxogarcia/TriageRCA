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
├── notebooks/                     # EDA + result analysis
└── requirements.txt
```

