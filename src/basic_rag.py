"""Basic RAG baseline.

Documents -> fixed-size chunking -> embeddings -> vector DB ->
retrieve top-k -> LLM generates the answer. No query rewriting, re-ranking, or
relevance checking - this is the baseline for comparison.
"""
from __future__ import annotations

from .config import CONFIG
from .generation import generate_answer, looks_like_refusal
from .types import RAGResult
from .vectorstore import VectorStore


class BasicRAG:
    def __init__(self):
        self.store = VectorStore(CONFIG.basic_collection)

    def query(self, question: str, k: int | None = None) -> RAGResult:
        k = k or CONFIG.basic_top_k
        contexts = self.store.search(question, k=k)
        answer = generate_answer(question, contexts)
        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            refused=looks_like_refusal(answer),
            trace={"retrieved_ids": [c.chunk.id for c in contexts],
                   "n_retrieved": len(contexts)},
        )
