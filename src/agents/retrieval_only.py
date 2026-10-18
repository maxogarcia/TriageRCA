"""retrieval_only baseline: RAG over runbooks/postmortems + one LLM call, no telemetry tools,
no planner/critic. Measures how far retrieval alone gets without investigating the incident.
"""

from __future__ import annotations

from src.evidence import EvidenceLedger
from src.llm.base import LLMClient, Message
from src.schemas import Hypothesis
from src.tools.registry import ToolRegistry

_SYSTEM_PROMPT = """You are a root-cause-analysis (RCA) assistant for microservice incidents.

You are given an incident description, the list of services, and runbook/postmortem excerpts
retrieved for the incident. You cannot query metrics, logs, or traces. Using only the retrieved
excerpts, rank up to 5 likely root-cause services (most likely first), name the most likely
indicator (a "<service>_<metric>" signal), the fault category, an explanation, and a mitigation.

Cite evidence IDs (e.g. "E001") for every claim, and only IDs you were shown. If the excerpts do
not support a conclusion, say so and give a low confidence rather than inventing evidence.
"""


def run_retrieval_only(
    llm: LLMClient,
    tools: ToolRegistry,
    incident_query: str,
    services: list[str],
    ledger: EvidenceLedger,
    k: int = 5,
) -> Hypothesis:
    # Goes through the registry (not the Retriever directly) so the retrieval lands in the
    # evidence ledger exactly as it would when the Executor calls the tool (hard rule #3).
    args = {"query": incident_query, "k": k}
    result = tools.invoke("runbook_retrieval", args, ledger)
    evidence_id = ledger.items[-1].id

    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                f"Incident: {incident_query}\n"
                f"Services: {', '.join(services)}\n\n"
                f"[{evidence_id}] runbook_retrieval({args}):\n{result}"
            ),
        ),
    ]
    response = llm.generate(messages, response_schema=Hypothesis)
    hypothesis = Hypothesis.model_validate_json(response.text)

    unknown = [c for c in hypothesis.citations if c not in ledger.known_ids()]
    if unknown:
        raise ValueError(f"Hypothesis cites evidence IDs not in the ledger: {unknown}")
    return hypothesis
