from src.evidence import EvidenceLedger
from src.rag.retriever import Chunk, StubRetriever
from src.tools.registry import ToolRegistry
from src.tools.runbook_retrieval import _MAX_CHUNK_CHARS, SPEC, make_runbook_retrieval


class _FixedRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return self._chunks[:k]


def _registry(retriever) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SPEC, make_runbook_retrieval(retriever))
    return registry


def test_each_chunk_gets_its_own_evidence_id():
    chunks = [Chunk(text=f"runbook {i}", source=f"r{i}.md", score=0.5) for i in range(3)]
    ledger = EvidenceLedger()

    result = _registry(_FixedRetriever(chunks)).invoke("runbook_retrieval", {"query": "cpu"}, ledger)

    assert [i.id for i in ledger.items] == ["E001", "E002", "E003"]
    assert ledger.items[1].content == "[r1.md] (score=0.50) runbook 1"
    for item in ledger.items:
        assert f"[{item.id}] {item.content}" in result


def test_long_chunks_are_capped():
    ledger = EvidenceLedger()
    retriever = _FixedRetriever([Chunk(text="x" * (_MAX_CHUNK_CHARS + 100), source="r.md")])

    _registry(retriever).invoke("runbook_retrieval", {"query": "cpu"}, ledger)

    assert ledger.items[0].content.endswith("x...")
    assert ledger.items[0].content.count("x") == _MAX_CHUNK_CHARS


def test_no_results_is_still_one_citable_item():
    ledger = EvidenceLedger()

    result = _registry(StubRetriever()).invoke("runbook_retrieval", {"query": "cpu"}, ledger)

    assert len(ledger.items) == 1
    assert result.startswith("[E001] No runbooks")
