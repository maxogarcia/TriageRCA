"""
Embedding pipeline for the TriageRCA runbook corpus.
Validates the Golden Template, creates one chunk per H2 scenario,
embeds with all-MiniLM-L6-v2, and stores the chunks in ChromaDB.
"""

import hashlib
import re
from pathlib import Path

import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer

# --- Config ---
CORPUS_DIR = "data/runbooks"          # contains real/ and synthesized/
CHROMA_DIR = "data/chroma_db"         # persistent ChromaDB storage
COLLECTION_NAME = "runbooks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, runs locally

REQUIRED_FIELDS = [
    "Affected Services",
    "Fault Type",
    "Symptom",
    "Root Cause",
    "Diagnostic Steps",
    "Remediation & Mitigation",
]

ALLOWED_FAULT_TYPES = {
    "Resource",
    "Network",
    "Code-level",
    "Concurrency",
    "Config",
    "Data",
    "Procedure",
}

FIELD_RE = re.compile(r"^- \*\*(.+?)\*\*: ?(.*)$", re.MULTILINE)

HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "scenario")],
    strip_headers=False,
)


def _validate_document(content: str, filepath: Path) -> str:
    """Validate document-level rules and return the single H1 title."""
    if content.startswith("---"):
        raise ValueError(f"{filepath}: YAML frontmatter is not allowed")

    h1_titles = re.findall(r"^# (?!#)(.+)$", content, flags=re.MULTILINE)
    if len(h1_titles) != 1:
        raise ValueError(f"{filepath}: expected exactly one H1, found {len(h1_titles)}")

    if not re.search(r"^## .+", content, flags=re.MULTILINE):
        raise ValueError(f"{filepath}: expected at least one H2 scenario")

    return h1_titles[0].strip()


def _validate_scenario(text: str, scenario: str, filepath: Path) -> dict[str, str]:
    """Validate one H2 chunk and return its six parsed fields."""
    matches = FIELD_RE.findall(text)
    field_order = [name for name, _ in matches]

    if field_order != REQUIRED_FIELDS:
        raise ValueError(
            f"{filepath} [{scenario}]: expected fields {REQUIRED_FIELDS}, "
            f"found {field_order}"
        )

    fields = {name: value.strip() for name, value in matches}
    fault_type = fields["Fault Type"]
    if fault_type not in ALLOWED_FAULT_TYPES:
        raise ValueError(
            f"{filepath} [{scenario}]: invalid Fault Type {fault_type!r}"
        )

    diagnostic_block = re.search(
        r"^- \*\*Diagnostic Steps\*\*:\s*\n"
        r"(?P<steps>(?:[ \t]*\d+\..*(?:\n|$))+)",
        text,
        flags=re.MULTILINE,
    )
    if not diagnostic_block:
        raise ValueError(f"{filepath} [{scenario}]: numbered diagnostics required")

    mitigation = fields["Remediation & Mitigation"]
    if "Immediate" not in mitigation or "Long-term" not in mitigation:
        raise ValueError(
            f"{filepath} [{scenario}]: mitigation needs Immediate and Long-term actions"
        )

    return fields


def chunk_runbook(filepath: Path) -> list[dict]:
    """
    Validate a runbook and return one chunk per H2 fault scenario.
    """
    content = filepath.read_text(encoding="utf-8")
    document_title = _validate_document(content, filepath)
    split_documents = HEADER_SPLITTER.split_text(content)

    chunks = []
    for document in split_documents:
        scenario = document.metadata.get("scenario")
        if not scenario:
            # Ignore the H1 preamble; only H2 scenarios are indexed.
            continue

        text = document.page_content.strip()
        fields = _validate_scenario(text, scenario, filepath)
        fault_id, separator, short_title = scenario.partition(":")
        if not separator:
            raise ValueError(
                f"{filepath} [{scenario}]: H2 must be 'Fault-ID: Short Title'"
            )

        relative_path = filepath.relative_to(Path(CORPUS_DIR)).as_posix()
        stable_key = f"{relative_path}:{fault_id.strip()}"
        chunk_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:24]

        chunks.append({
            "id": chunk_id,
            "text": f"# {document_title}\n\n{text}",
            "metadata": {
                "source_file": relative_path,
                "document_title": document_title,
                "fault_id": fault_id.strip(),
                "scenario": short_title.strip(),
                "fault_type": fields["Fault Type"],
                "services": fields["Affected Services"],
                "origin": filepath.parent.name,  # real or synthesized
            },
        })

    return chunks


def build_index():
    """
    Main pipeline: read all runbooks, chunk, embed, store in ChromaDB.
    """
    # 1. Collect all runbooks in a stable order.
    corpus_root = Path(CORPUS_DIR)
    files = sorted([
        *corpus_root.joinpath("real").glob("*.md"),
        *corpus_root.joinpath("synthesized").glob("*.md"),
    ])

    if not files:
        print(f"No .md files found in {CORPUS_DIR}/real or {CORPUS_DIR}/synthesized")
        return

    print(f"Found {len(files)} runbook files")

    # 2. Chunk all documents
    all_chunks = []
    for filepath in files:
        chunks = chunk_runbook(filepath)
        all_chunks.extend(chunks)
        print(f"  {filepath.name}: {len(chunks)} scenario chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # 3. Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Embed all chunk texts
    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )

    # 5. Store in ChromaDB
    print(f"\nStoring in ChromaDB at {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete the prior collection when re-indexing.
    existing_names = {item.name for item in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "TriageRCA Golden Template runbook corpus",
            "hnsw:space": "cosine",
        },
    )

    # ChromaDB expects lists of ids, documents, embeddings, metadatas
    ids = [c["id"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}'")
    print("Done!")


if __name__ == "__main__":
    build_index()
