"""Build and load the persistent indexes for both systems."""
from __future__ import annotations

import json

from .chunking import fixed_size_chunks, hierarchical_chunks
from .config import CONFIG
from .ingest import Document, load_documents
from .types import Chunk
from .vectorstore import VectorStore

_PARENTS_PATH = CONFIG.storage_dir / "parents_advanced.json"


# --------------------------------------------------------------------------- #
def build_basic_index(docs: list[Document]) -> int:
    """Basic: documents -> fixed-size chunks -> embeddings -> vector DB."""
    store = VectorStore(CONFIG.basic_collection, reset=True)
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(fixed_size_chunks(doc))
    store.add(all_chunks)
    return len(all_chunks)


# --------------------------------------------------------------------------- #
def build_advanced_index(docs: list[Document]) -> dict:
    """Advanced: context-aware section splitting + hierarchical parent/child chunks.

    Children (small, context-headed windows) are embedded and retrieved; parents
    (whole sections) are stored for parent-expansion at answer time.
    """
    store = VectorStore(CONFIG.advanced_collection, reset=True)
    parents: dict[str, Chunk] = {}
    all_children: list[Chunk] = []
    for doc in docs:
        children, doc_parents = hierarchical_chunks(doc)
        parents.update(doc_parents)
        all_children.extend(children)
    store.add(all_children)

    _save_parents(parents)
    return {"children": len(all_children), "parents": len(parents)}


# --------------------------------------------------------------------------- #
def _save_parents(parents: dict[str, Chunk]) -> None:
    data = {pid: {"text": p.text, "doc_id": p.doc_id, "metadata": p.metadata}
            for pid, p in parents.items()}
    _PARENTS_PATH.write_text(json.dumps(data, indent=2))


def load_parents() -> dict[str, Chunk]:
    if not _PARENTS_PATH.exists():
        return {}
    data = json.loads(_PARENTS_PATH.read_text())
    return {pid: Chunk(id=pid, text=v["text"], doc_id=v["doc_id"],
                       metadata=v["metadata"]) for pid, v in data.items()}


def build_all() -> dict:
    docs = load_documents()
    basic = build_basic_index(docs)
    adv = build_advanced_index(docs)
    return {"documents": len(docs), "basic_chunks": basic, "advanced": adv}
