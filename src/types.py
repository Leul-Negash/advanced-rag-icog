"""Shared data structures used across both pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A unit of retrievable text.

    metadata may carry: doc_id, doc_title, section, parent_id, context_header
    (the natural-language situating sentence used for contextual retrieval), and
    chunk_type ("fixed", "child", "parent").
    """
    id: str
    text: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.metadata.get("doc_title", self.doc_id)


@dataclass
class Scored:
    """A chunk paired with a retrieval score and the retriever that produced it."""
    chunk: Chunk
    score: float
    source: str = "vector"  # vector | graph | rerank | rrf

    @property
    def id(self) -> str:
        return self.chunk.id


@dataclass
class RAGResult:
    """The full output of a RAG query, kept rich so evaluation can inspect it."""
    question: str
    answer: str
    contexts: list[Scored] = field(default_factory=list)
    refused: bool = False
    # trace holds technique-specific diagnostics (expanded queries, grades,
    # corrective rounds, graph hits, etc.) for the comparison report.
    trace: dict[str, Any] = field(default_factory=dict)

    def context_ids(self) -> list[str]:
        return [c.chunk.id for c in self.contexts]

    def context_docs(self) -> list[str]:
        # de-duplicated source doc ids, in order
        seen: list[str] = []
        for c in self.contexts:
            if c.chunk.doc_id not in seen:
                seen.append(c.chunk.doc_id)
        return seen
