"""ChromaRetriever — bridges Aeon's ChromaDB-backed retrieve() to Max's Retriever Protocol.

This adapter lets the Planner-Executor pipeline call the real RAG system
through the same Retriever interface that the StubRetriever satisfies.

Usage in src/run.py:
    from src.rag.chroma_retriever import ChromaRetriever
    registry.register(RUNBOOK_RETRIEVAL_SPEC, make_runbook_retrieval(ChromaRetriever()))
"""

from __future__ import annotations

from src.rag.retrieve import retrieve as rag_retrieve
from src.rag.retriever import Chunk


class ChromaRetriever:
    """Wraps Aeon's ChromaDB-backed retrieve() as a Retriever for the pipeline."""

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        results = rag_retrieve(query, top_k=k)
        return [
            Chunk(
                text=r["text"],
                source=r["metadata"].get("source_file", "unknown"),
                score=r["score"],
            )
            for r in results
        ]
