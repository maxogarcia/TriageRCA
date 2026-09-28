"""Retriever interface + a stub implementation.

Aeon owns the real RAG system (embeddings + vector store over runbooks/postmortems). This stub
exists only so Max's runbook_retrieval tool and executor loop are testable before that lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Chunk:
    text: str
    source: str
    score: float = 0.0


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]: ...


class StubRetriever:
    """Returns nothing found, always. Placeholder until Aeon's Retriever is wired in."""

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return []
