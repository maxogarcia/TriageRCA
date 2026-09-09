# Literature & SOTA Survey — TriageRCA

**Project:** TriageRCA: A Planner-Executor-Critic Agent for Root Cause Analysis
**Course:** CMPE 258 — Deep Learning (Fall 2026)
**Scope:** Ten recent (2024–2026) papers on LLM-based and agentic root cause analysis (RCA), incident diagnosis, and RCA benchmarking, each with its relevance to our design.

The papers cluster into four groups that map onto our project: **benchmarks** we evaluate against, **agent / multi-agent RCA methods** that inform our architecture, **failure-mode and evaluation** work that motivates our Critic, and **surveys** that place the work in context.

---

## Benchmarks

### 1. RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data
*Luan Pham et al. — The Web Conference (WWW) 2025 (arXiv 2412.17015, Dec 2024).*

RCAEval provides 735 labeled failure cases across three microservice systems (Online Boutique, Sock Shop, Train Ticket) with full metrics, logs, and traces, eleven injected fault types, and ground-truth root-cause service and indicator for each case. It ships with reproducible baselines and standard metrics (AC@k, Avg@k) and is installable from PyPI.
**Relevance:** This is our primary dataset and evaluation harness. Its labels and metrics define the exact task and scoring TriageRCA is measured on.
Link: https://arxiv.org/abs/2412.17015 · https://github.com/phamquiluan/RCAEval

### 2. OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures?
*Junjie Xu et al. — ICLR 2025.*

OpenRCA frames RCA as an LLM task: given a natural-language query over large volumes of telemetry (KPIs, traces, logs) from three production-scale systems, the model must locate the root-cause component, reason, and time. It contributes 335 incident tasks and an RCA-agent baseline.
**Relevance:** OpenRCA's query-driven formulation is almost exactly our input/output contract, and it is our planned second benchmark for showing the agent generalizes beyond RCAEval.
Link: https://github.com/microsoft/OpenRCA

---

## Agent and Multi-Agent RCA Methods

### 3. Flow-of-Action: SOP-Enhanced LLM-Based Multi-Agent System for Root Cause Analysis
*Changhua Pei et al. — WWW 2025 (Industry) (arXiv 2502.08224, Feb 2025).*

Flow-of-Action constrains a multi-agent LLM system with Standard Operating Procedures (SOPs) rather than letting it reason freely, and adds auxiliary agents to filter noise. It reports 64.0% RCA accuracy versus 35.5% for a ReAct baseline, showing that procedural grounding sharply reduces hallucination.
**Relevance:** This is the closest method to ours. Our runbook/SOP RAG layer and our ReAct baseline comparison are directly informed by it; we extend it with an explicit evidence-verifying Critic.
Link: https://arxiv.org/abs/2502.08224

### 4. RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models
*Zefan Wang et al. — CIKM 2024 (arXiv 2310.16340).*

RCAgent is an autonomous, tool-augmented agent for industrial cloud RCA, emphasizing practical deployment concerns such as using smaller/local models, tool orchestration, and self-consistency aggregation of evidence.
**Relevance:** Our Executor is a tool-augmented agent in this mold (log/metric/trace/runbook tools). RCAgent's evidence-aggregation ideas inform how our Critic consolidates findings.
Link: https://arxiv.org/pdf/2310.16340

### 5. mABC: Multi-Agent Blockchain-Inspired Collaboration for Root Cause Analysis in Micro-Services
*Wei Zhang et al. — EMNLP 2024 (Findings).*

mABC coordinates multiple specialized agents that reach a decision through a structured voting/collaboration protocol for microservice fault localization, demonstrating that agent collaboration outperforms a single monolithic agent.
**Relevance:** Provides evidence for splitting responsibilities across Planner, Executor, and Critic rather than using one agent, and offers a collaboration/aggregation pattern to compare against.

### 6. GALA: Graph-Augmented LLM Agents for Root Cause Analysis and Incident Response in Microservices
*Yifang Tian et al. — arXiv 2608.08968 (2026).*

GALA guides LLM investigation with a service-dependency graph and refines diagnoses with localized multi-modal evidence, reporting a 25+ percentage-point improvement in AC@1 over the strongest LLM baseline plus favorable SRE-expert ratings.
**Relevance:** Our Planner uses a dependency graph to prioritize which services to inspect; GALA is our reference for that mechanism and a natural stretch/future-work direction.
Link: https://arxiv.org/html/2608.08968

### 7. Exploring LLM-Based Agents for Root Cause Analysis
*Devjeet Roy et al. — FSE 2024 (Industry) (arXiv 2403.04123).*

An early, influential study of whether ReAct-style LLM agents with retrieval and tool access can perform RCA on real incident data, characterizing where agents help and where they still fall short versus retrieval-only approaches.
**Relevance:** Establishes the ReAct-agent and retrieval-only baselines that TriageRCA must beat, and frames the open problems our Critic aims to address.
Link: https://arxiv.org/abs/2403.04123

---

## Failure Modes and Evaluation

### 8. Why Do AI Agents Systematically Fail at Cloud Root Cause Analysis?
*Taeyoon Kim et al. — arXiv 2602.09937 (2026).*

Running five models on OpenRCA's 335 incidents, the authors find that the dominant failure is "hallucinated data interpretation" (71.2% of runs) and incomplete exploration, that these breakdowns are architectural rather than model-specific (prompting alone cannot fix them), and that enriching inter-agent communication reduces such failures by up to 15 points.
**Relevance:** This is the central motivation for our Critic and evidence-grounding metric. It turns "add a verification agent" from a nice-to-have into a research-justified design choice.
Link: https://arxiv.org/abs/2602.09937

### 9. AIOpsLab: A Holistic Framework for Evaluating AI Agents for Enabling Autonomous Cloud
*Microsoft Research et al. — MLSys 2025.*

AIOpsLab provides an environment and evaluation framework spanning the incident lifecycle (detection, localization, root cause, mitigation) for benchmarking autonomous cloud agents in a reproducible way.
**Relevance:** Informs our evaluation methodology beyond single-number accuracy, especially how to score localization and proposed mitigation separately.

---

## Survey

### 10. A Survey of AIOps in the Era of Large Language Models
*(2025).*

A current survey organizing LLM-for-AIOps work across data, failure management, and evaluation, with a taxonomy of methods and open challenges.
**Relevance:** Our related-work anchor; used to position TriageRCA and to source additional citations. (Companion reference: *A Survey of AIOps for Failure Management in the Era of LLMs*, Chen et al., arXiv 2406.11213, 2024.)

---

## Further reading (used but not counted in the ten)

- **ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks** (IBM, ICML 2025) — source of our authorization/access-incident extension idea; ITBench-Lite offers static k8s snapshots usable without a live cluster.
- **A Survey of AIOps for Failure Management in the Era of LLMs** (Chen et al., arXiv 2406.11213, 2024) — the standard failure-management taxonomy.

## How this survey shapes TriageRCA

The takeaway across these papers is a clear throughline: multi-agent and SOP-grounded methods (Flow-of-Action, mABC, RCAgent, GALA) show that agentic RCA works and that grounding beats free reasoning; the failure-mode analysis (Kim et al.) shows that the remaining bottleneck is unverified, hallucinated interpretation; and RCAEval/OpenRCA give us the labeled benchmarks to prove a fix. TriageRCA's Planner-Executor-**Critic** design, with an explicit evidence-grounding metric, targets exactly that bottleneck.
