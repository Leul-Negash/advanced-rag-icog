"""Better query handling for the Advanced system.

Implements BOTH techniques the assignment requires:
  - Query expansion: enrich the original query with related terms/synonyms.
  - Multi-query RAG: generate several paraphrases, retrieve for each, then merge
    and de-duplicate the results.

Merging uses Reciprocal Rank Fusion (RRF), which combines rankings from many
queries robustly without needing comparable raw scores.
"""
from __future__ import annotations

from .config import CONFIG
from .llm import QuotaExhausted, get_llm
from .types import Scored

_EXPAND_SYS = (
    "You rewrite search queries for a retrieval system over a corpus about "
    "OpenCog Hyperon, the MeTTa language, the AtomSpace, iCog Labs, and related "
    "AGI projects. Be precise and use domain vocabulary."
)


def expand_query(question: str) -> str:
    """Query expansion: add useful related terms / acronyms / context."""
    prompt = (
        f"Original search query: {question}\n\n"
        "Rewrite it into ONE richer search query that adds closely related terms, "
        "synonyms, and expanded acronyms that are likely to appear in the relevant "
        "documents. Keep it on-topic. Return only the rewritten query text."
    )
    try:
        return get_llm().generate(prompt, system=_EXPAND_SYS, temperature=0.3).strip()
    except Exception:
        return question  # degrade gracefully


def multi_queries(question: str, n: int = 3) -> list[str]:
    """Multi-query RAG: produce n diverse paraphrases of the question."""
    prompt = (
        f"User question: {question}\n\n"
        f"Generate {n} alternative versions of this question that a search engine "
        "could use to find relevant documents. Vary the wording and phrasing; some "
        "should be keyword-style, some natural-language. Return a JSON array of "
        "strings only."
    )
    try:
        out = get_llm().generate_json(prompt, system=_EXPAND_SYS, temperature=0.5)
        qs = [str(q).strip() for q in out if str(q).strip()]
        return qs[:n] if qs else []
    except Exception:
        return []


def expand_and_multiquery(question: str, n_variants: int = 2) -> dict:
    """Do BOTH query techniques in ONE LLM call (quota-efficient).

    Returns {"expanded": <enriched query>, "variants": [<paraphrases>]}.
    Functionally identical to calling expand_query + multi_queries separately,
    but uses a single request instead of two.
    """
    prompt = (
        f"User question: {question}\n\n"
        "Produce search queries to retrieve relevant documents. Return JSON:\n"
        '{"expanded": "one richer query that adds related terms, synonyms and '
        'expanded acronyms", '
        f'"variants": [{n_variants} alternative phrasings of the question]}}\n'
        "Stay on-topic and use domain vocabulary."
    )
    try:
        out = get_llm().generate_json(prompt, system=_EXPAND_SYS, temperature=0.4)
        expanded = str(out.get("expanded", "")).strip()
        variants = [str(v).strip() for v in out.get("variants", []) if str(v).strip()]
        return {"expanded": expanded or question, "variants": variants[:n_variants]}
    except QuotaExhausted:
        raise
    except Exception:
        return {"expanded": question, "variants": []}


def reciprocal_rank_fusion(ranked_lists: list[list[Scored]],
                           k: int = None) -> list[Scored]:
    """Merge multiple ranked lists and de-duplicate by chunk id using RRF.

    RRF score = sum over lists of 1 / (rrf_k + rank). Identical chunks retrieved
    by several query variants rise to the top; duplicates are collapsed.
    """
    k = k or CONFIG.rrf_k
    scores: dict[str, float] = {}
    best: dict[str, Scored] = {}
    for lst in ranked_lists:
        for rank, scored in enumerate(lst):
            cid = scored.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # keep the representative with the highest individual vector score
            if cid not in best or scored.score > best[cid].score:
                best[cid] = scored
    fused = []
    for cid, fscore in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        s = best[cid]
        fused.append(Scored(chunk=s.chunk, score=fscore, source="rrf"))
    return fused
