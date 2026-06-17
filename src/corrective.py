"""Corrective RAG (Optional): judge evidence before trusting it.

Flow:
  retrieve -> grade relevance of each chunk
            -> if enough strong evidence: answer
            -> if weak: rewrite the query and retrieve again (up to N rounds)
            -> if still weak: REFUSE rather than hallucinate.

This is what lets the system correctly decline "answer-unavailable" questions
instead of producing a confident but unsupported answer.
"""
from __future__ import annotations

from .config import CONFIG
from .llm import get_llm
from .types import Scored

_GRADE_SYS = (
    "You are a retrieval-quality grader. You judge whether a passage provides "
    "information useful for answering the question OR ANY PART of it. For "
    "multi-step / relationship questions, a passage that supplies just one hop "
    "(one entity, link, or fact the question asks about) still counts as RELEVANT. "
    "Judge only from the passage text; never use outside knowledge."
)


def grade_relevance(question: str, candidates: list[Scored]) -> list[float]:
    """Return a 0-1 relevance grade per candidate (LLM-judged, batched)."""
    if not candidates:
        return []
    listing = "\n\n".join(
        f"[{i}] {c.chunk.metadata.get('raw_text', c.chunk.text)[:500]}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Question: {question}\n\nPassages:\n{listing}\n\n"
        "For each passage, decide if it is RELEVANT. A passage is RELEVANT if it "
        "helps answer the question OR any sub-part of it (for multi-step questions, "
        "covering a single hop is enough). Mark it NOT relevant only if it is "
        "off-topic. Return a JSON object mapping passage index (string) to 1 "
        '(relevant) or 0 (not), e.g. {"0":1,"1":0}.'
    )
    try:
        out = get_llm().generate_json(prompt, system=_GRADE_SYS, temperature=0.0)
        return [float(out.get(str(i), 0)) for i in range(len(candidates))]
    except Exception:
        # if grading fails, assume relevant so we don't wrongly refuse
        return [1.0] * len(candidates)


def rewrite_query(question: str, weak_context: list[Scored]) -> str:
    """Rewrite the query after a weak retrieval round (corrective step)."""
    prompt = (
        f"The search query '{question}' returned weak or off-topic results. "
        "Rewrite it to retrieve better evidence: make implicit terms explicit, "
        "expand acronyms, and focus on the key entities. Return only the new query."
    )
    try:
        return get_llm().generate(prompt, temperature=0.3).strip() or question
    except Exception:
        return question


def is_sufficient(grades: list[float], threshold: float | None = None) -> bool:
    """Evidence is sufficient if at least one chunk is clearly relevant."""
    threshold = CONFIG.relevance_threshold if threshold is None else threshold
    if not grades:
        return False
    return max(grades) >= threshold and sum(grades) >= 1.0


def filter_relevant(candidates: list[Scored], grades: list[float],
                    threshold: float = 0.5) -> list[Scored]:
    return [c for c, g in zip(candidates, grades) if g >= threshold]
