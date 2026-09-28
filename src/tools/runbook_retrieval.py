"""runbook_retrieval tool: wraps a Retriever (Aeon's RAG system, or the stub) as a tool the
Executor can call, so runbook/postmortem hits land in the evidence ledger like any other tool.
"""

from __future__ import annotations

from src.llm.base import ToolSpec
from src.rag.retriever import Retriever

SPEC = ToolSpec(
    name="runbook_retrieval",
    description="Retrieve relevant runbooks, SOPs, or past postmortems for a free-text query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, e.g. 'cartservice high cpu'",
            },
            "k": {"type": "integer", "description": "Number of results to return", "default": 5},
        },
        "required": ["query"],
    },
)

_MAX_CHUNK_CHARS = 500


def make_runbook_retrieval(retriever: Retriever):
    def runbook_retrieval(query: str, k: int = 5) -> str:
        chunks = retriever.retrieve(query, k=k)
        if not chunks:
            return f"No runbooks/postmortems found for query: {query!r}"

        lines = [f"{len(chunks)} result(s) for {query!r}:"]
        for c in chunks:
            snippet = c.text if len(c.text) <= _MAX_CHUNK_CHARS else c.text[:_MAX_CHUNK_CHARS] + "..."
            lines.append(f"  - [{c.source}] (score={c.score:.2f}) {snippet}")
        return "\n".join(lines)

    return runbook_retrieval
