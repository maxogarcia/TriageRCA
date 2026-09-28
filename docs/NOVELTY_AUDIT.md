# AI Novelty & Feasibility Audit — TriageRCA

**Project:** TriageRCA: A Planner-Executor-Critic Agent for Root Cause Analysis
**Course:** CMPE 258 — Deep Learning (Fall 2026)
**Nature of this document:** This is a candid AI-generated critique of the project's novelty, market/research saturation ("red ocean") risk, and feasibility. It is intentionally critical so that the team goes in with clear eyes. Nothing here is fatal; each risk has a concrete mitigation the project already adopts.

---

## 1. Novelty Assessment

**Honest verdict: the general idea is not novel; the specific contribution can be.**

"An LLM agent that does root cause analysis on microservices" is, as of 2026, a well-trodden idea. There are strong entries from Microsoft (OpenRCA), Alibaba (RCAgent), and multiple 2025–2026 multi-agent systems (Flow-of-Action, mABC, GALA). A proposal that simply reads "we build a ReAct agent that finds the root cause" would be judged derivative.

Where TriageRCA can claim legitimate, defensible novelty for a course project:

1. **Verification-first architecture targeting a documented failure mode.** Kim et al. (2026) show that the dominant reason RCA agents fail is *hallucinated data interpretation* (71.2% of runs), and that this is architectural, not fixable by prompting. TriageRCA's Critic is an explicit, evidence-checking agent whose only job is to reject unsupported conclusions. Most existing systems add agents for *planning* or *tool use*, not for *falsification*. Framing the Critic as a falsifier is a fresh angle.
2. **An evaluation dimension the field under-reports: grounding/faithfulness.** Nearly all RCA papers report only localization accuracy (AC@k). TriageRCA adds an evidence-grounding score (what fraction of the diagnosis's claims are supported by retrieved telemetry). This measures *reliability*, not just correctness, and is directly aligned with the failure-mode literature.
3. **A controlled, reproducible three-model cost/latency/accuracy comparison** on a common public benchmark. This is engineering-survey value: practitioners genuinely want to know the accuracy-per-dollar tradeoff between a frontier API model and an open-weight model on RCA.
4. **A small benchmark extension** with authorization/access-violation incidents layered on real telemetry, addressing the "incidents beyond trivial crashes" gap noted in our proposal review.

**Recommendation:** In the writeup, do *not* claim to beat SOTA accuracy. Claim a *reliability and reproducibility* contribution. That claim is both true and achievable.

---

## 2. Red-Ocean / Oversaturation Risk

**Risk level: HIGH at the research frontier, MODERATE for a course project.**

- **Crowded space.** Microservice RCA on RCAEval/OpenRCA is actively contested by well-resourced industry labs with large compute budgets and production telemetry access. Competing on raw leaderboard accuracy is a losing game for a student team.
- **Commoditized components.** ReAct agents, RAG over documents, and tool-calling are now standard. None of these is a contribution on its own; reviewers will expect them as table stakes.
- **Benchmark-building is saturated too.** Do *not* pivot to "we built a new RCA benchmark." RCAEval, OpenRCA, ITBench, and AIOpsLab already cover that ground; a new one would be both redundant and infeasible in a semester.

**Areas to explicitly avoid:**
- Yet-another-ReAct-agent with no verification or new metric.
- Generic log anomaly detection (a different, older problem than root-cause localization).
- From-scratch model training or fine-tuning a base LLM for RCA (out of scope and unnecessary; save fine-tuning for a dedicated project).

**How TriageRCA stays out of the red ocean:** compete on *method reliability and honest evaluation*, not on SOTA numbers. The grounding metric, the falsifying Critic, and the cost/accuracy study are contributions that do not require beating Microsoft on accuracy to be interesting.

---

## 3. Feasibility Assessment

**Verdict: HIGH feasibility for a one-semester, 3-person team.**

**What makes it feasible:**
- **Data is solved.** RCAEval is public, labeled, CC BY 4.0, and pip-installable. No data collection, no labeling. This removes the single biggest risk in an RCA project (getting realistic incidents with ground truth), which was the main concern raised in our proposal review.
- **No training required for the core.** The system is built on hosted APIs plus one open-weight model for inference. There is no GPU training loop on the critical path, so there is no long, fragile training phase to derail the timeline.
- **Incremental scope.** Start with one system (Online Boutique) and a subset of fault types, hosted APIs only; expand to more systems, the open model, and the graph-guided Planner as stretch goals. Every milestone produces a working, gradeable artifact.

**Top risks and mitigations:**

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Telemetry volume exceeds LLM context window | High | Retrieval + summarization of telemetry; feed the agent slices on demand via tools, never the full dump. |
| API cost of a 3-model × N-case sweep | Medium | Cap the eval set at ~100–150 stratified cases; cache model outputs; run the full sweep once, late. |
| "Are the incidents realistic?" (review concern) | Medium | Use RCAEval's injected faults with ground truth as the realism backbone; add a small authored auth/access set for coverage, not as the main corpus. |
| Open-weight model underperforms and skews comparison | Low | Report it honestly as a finding (accuracy-per-dollar), which is the point of the comparison, not a failure. |
| Scope creep (graph module, second benchmark) | Medium | Keep GALA-style graph guidance and OpenRCA as explicitly labeled stretch goals with a hard cutoff date. |

**Compute needs:** modest. Core path needs only API access; the open-weight model can run on a single mid-range GPU or a hosted inference endpoint. No cluster required.

---

## 4. Bottom Line

TriageRCA is **feasible and worth doing**, provided the team frames the contribution correctly. The winning framing is: *"Agentic RCA works but hallucinates; we add a falsifying Critic and a grounding metric, and we measure reliability and cost honestly on a public benchmark."* That is a defensible, achievable, and technically meaningful semester project. The losing framing is *"we will build the best RCA agent,"* which invites direct comparison to industry systems the team cannot out-resource. Choose the first framing and the red-ocean risk drops from a threat to a well-understood context.
