"""Self-contained, NumPy-backed vector store, persisted to disk.

Embeddings are L2-normalised and stored in one matrix, so cosine similarity is a
single matrix-vector product. Saved as .npz + .json. Both systems use this same
storage layer, so the comparison isolates the techniques, not the database. The
interface (add, search) is what a Chroma/FAISS swap-in would expose.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .config import CONFIG
from .embeddings import Embedder
from .types import Chunk, Scored


class VectorStore:
    def __init__(self, collection: str, reset: bool = False):
        self.name = collection
        self.dir = CONFIG.storage_dir / "vectors"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.dir / f"{collection}.npz"
        self.meta_path = self.dir / f"{collection}.json"
        self.embedder = Embedder.get()

        self.embeddings: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.records: list[dict[str, Any]] = []  # {id, text, doc_id, metadata}

        if reset:
            self.vec_path.unlink(missing_ok=True)
            self.meta_path.unlink(missing_ok=True)
        elif self.vec_path.exists() and self.meta_path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    def add(self, chunks: list[Chunk], batch: int = 64) -> None:
        vecs = []
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            vecs.append(self.embedder.embed_documents([c.text for c in part]))
        if vecs:
            new = np.vstack(vecs).astype(np.float32)
            self._append(chunks, new)

    def add_with_embeddings(self, chunks: list[Chunk], embeddings) -> None:
        """Add chunks whose embeddings were already computed (e.g. late chunking)."""
        if len(chunks) == 0:
            return
        self._append(chunks, np.asarray(embeddings, dtype=np.float32))

    def _append(self, chunks: list[Chunk], new_vecs: np.ndarray) -> None:
        self.embeddings = np.vstack([self.embeddings, new_vecs]) if len(self.embeddings) \
            else new_vecs
        for c in chunks:
            self.records.append({"id": c.id, "text": c.text, "doc_id": c.doc_id,
                                 "metadata": c.metadata})
        self._save()

    def count(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int | None = None) -> list[Scored]:
        k = k or CONFIG.top_k
        emb = self.embedder.embed_query(query)
        return self.search_vector(emb, k)

    def search_vector(self, emb: np.ndarray, k: int) -> list[Scored]:
        if len(self.records) == 0:
            return []
        sims = self.embeddings @ emb.astype(np.float32)  # cosine (vectors normalised)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        out = []
        for idx in top:
            r = self.records[idx]
            chunk = Chunk(id=r["id"], text=r["text"], doc_id=r["doc_id"],
                          metadata=dict(r["metadata"]))
            out.append(Scored(chunk=chunk, score=float(sims[idx]), source="vector"))
        return out

    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        np.savez_compressed(self.vec_path, embeddings=self.embeddings)
        self.meta_path.write_text(json.dumps(self.records))

    def _load(self) -> None:
        self.embeddings = np.load(self.vec_path)["embeddings"].astype(np.float32)
        self.records = json.loads(self.meta_path.read_text())
