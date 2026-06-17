"""Re-ranking (Bonus): re-order candidate chunks by true relevance before the LLM.

Two methods are provided:
  - CrossEncoderReranker: a local cross-encoder (ms-marco-MiniLM) that scores
    (query, chunk) pairs jointly - the standard high-precision approach.
  - LLMReranker: asks the LLM to score each chunk 0-10 for relevance.

The cross-encoder is the default (fast, local, no API cost). Both improve
*precision* - pushing the strongest evidence to the top and dropping noise.
"""
from __future__ import annotations

import threading

from .config import CONFIG
from .llm import get_llm
from .types import Scored


def _chunk_text(s: Scored) -> str:
    # prefer the raw (un-contextualized) text for scoring readability
    return s.chunk.metadata.get("raw_text", s.chunk.text)


class CrossEncoderReranker:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name or CONFIG.rerank_model)

    @classmethod
    def get(cls) -> "CrossEncoderReranker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        if not candidates:
            return []
        pairs = [(query, _chunk_text(c)) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        out = []
        for cand, sc in ranked[:top_k]:
            out.append(Scored(chunk=cand.chunk, score=float(sc), source="rerank"))
        return out


class LLMReranker:
    """Alternative reranker: an LLM scores each candidate's relevance 0-10."""

    def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        if not candidates:
            return []
        listing = "\n\n".join(
            f"[{i}] {_chunk_text(c)[:500]}" for i, c in enumerate(candidates)
        )
        prompt = (
            f"Question: {query}\n\n"
            f"Score how well each passage helps answer the question, 0 (useless) to "
            f"10 (directly answers it).\n\nPassages:\n{listing}\n\n"
            'Return a JSON object mapping passage index (as string) to integer score, '
            'e.g. {"0": 8, "1": 2}.'
        )
        try:
            grades = get_llm().generate_json(prompt, temperature=0.0)
            scored = []
            for i, c in enumerate(candidates):
                g = float(grades.get(str(i), 0))
                scored.append(Scored(chunk=c.chunk, score=g, source="rerank"))
            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:top_k]
        except Exception:
            return candidates[:top_k]


def get_reranker(method: str = "cross-encoder"):
    return LLMReranker() if method == "llm" else CrossEncoderReranker.get()
