"""
Retrieval interface for TriageRCA.
Takes an incident query, returns top-k runbook chunks with similarity scores.
This is what the Executor agent will call as a tool.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# Must match embed.py settings
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "runbooks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Module-level singletons (loaded once)
_model = None
_collection = None


def _load():
    """Lazy-load the model and ChromaDB collection."""
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def retrieve(
    query: str,
    top_k: int = 5,
    filter_fault_type: str | None = None,
    filter_origin: str | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant runbook chunks for a given query.

    Args:
        query: Natural language incident description,
               e.g. "checkoutservice latency spike, upstream timeout errors"
        top_k: Number of results to return (default 5)
        filter_fault_type: Optional exact Golden Template value, such as
                           "Resource", "Network", or "Code-level"
        filter_origin: Optional — restrict to "real" or "synthesized"

    Returns:
        List of dicts, each with:
          - text: the chunk content
          - score: cosine similarity (higher = more relevant)
          - metadata: source_file, document_title, fault_id, scenario,
                      fault_type, services, origin
    """
    model, collection = _load()

    # Build optional metadata filter
    where_filter = None
    conditions = []
    if filter_fault_type:
        conditions.append({"fault_type": {"$eq": filter_fault_type}})
    if filter_origin:
        conditions.append({"origin": {"$eq": filter_origin}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # Embed the query
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "distances", "metadatas"],
    )

    # The collection uses cosine space, so cosine similarity = 1 - distance.
    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance

        output.append({
            "text": results["documents"][0][i],
            "score": round(similarity, 4),
            "metadata": results["metadatas"][0][i],
        })

    return output


def retrieve_formatted(query: str, top_k: int = 5, **kwargs) -> str:
    """
    Same as retrieve() but returns a formatted string for the Executor agent.
    This is the version the agent will actually call.
    """
    results = retrieve(query, top_k, **kwargs)

    if not results:
        return "No relevant runbook entries found for this query."

    parts = [f"Found {len(results)} relevant runbook chunks:\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"--- Result {i} (score: {r['score']}) ---")
        parts.append(f"Source: {r['metadata'].get('source_file', 'unknown')}")
        parts.append(f"Runbook: {r['metadata'].get('document_title', 'unknown')}")
        parts.append(
            "Scenario: "
            f"{r['metadata'].get('fault_id', 'unknown')} — "
            f"{r['metadata'].get('scenario', 'unknown')}"
        )
        parts.append(f"Fault Type: {r['metadata'].get('fault_type', 'unknown')}")
        parts.append(f"Services: {r['metadata'].get('services', 'unknown')}")
        parts.append(f"\n{r['text']}\n")

    return "\n".join(parts)


# --- Quick test ---
if __name__ == "__main__":
    test_queries = [
        "checkoutservice high latency, payment timeout",
        "memory usage spike in orders service, OOMKilled pod",
        "network packet loss between frontend and cart service",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print("=" * 60)
        print(retrieve_formatted(q, top_k=3))
