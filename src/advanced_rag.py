"""Advanced RAG.

Pipeline (each stage is toggleable so the evaluation can run ablations):

  question
    -> QUERY HANDLING      : query expansion + multi-query paraphrases
    -> RETRIEVE            : vector search per query
    -> FUSE + DEDUP        : Reciprocal Rank Fusion across query variants
    -> RE-RANK             : cross-encoder picks the strongest evidence (bonus)
    -> CORRECTIVE          : grade relevance; if weak, rewrite & retry; else REFUSE
    -> PARENT EXPANSION    : swap child chunks for their parent sections
    -> GENERATE            : grounded answer with citations
"""
from __future__ import annotations

from .config import CONFIG
from .corrective import grade_relevance, is_sufficient, rewrite_query
from .generation import REFUSAL, generate_answer, looks_like_refusal
from .index import load_parents
from .query_handling import expand_and_multiquery, reciprocal_rank_fusion
from .rerank import get_reranker
from .types import RAGResult, Scored
from .vectorstore import VectorStore


class AdvancedRAG:
    def __init__(self, *, use_query_expansion=True, use_multi_query=True,
                 use_rerank=True, use_corrective=True,
                 use_parent_expansion=True, rerank_method="cross-encoder"):
        self.store = VectorStore(CONFIG.advanced_collection)
        self.parents = load_parents()
        self.use_query_expansion = use_query_expansion
        self.use_multi_query = use_multi_query
        self.use_rerank = use_rerank
        self.use_corrective = use_corrective
        self.use_parent_expansion = use_parent_expansion
        self.rerank_method = rerank_method
        self.reranker = get_reranker(rerank_method) if use_rerank else None

    # ------------------------------------------------------------------ #
    def _build_queries(self, question: str) -> list[str]:
        queries = [question]
        # query expansion + multi-query are produced together in one LLM call
        if self.use_query_expansion or self.use_multi_query:
            qv = expand_and_multiquery(question, n_variants=2)
            if self.use_query_expansion and qv["expanded"]:
                queries.append(qv["expanded"])
            if self.use_multi_query:
                queries.extend(qv["variants"])
        # de-duplicate, preserve order
        seen, out = set(), []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(q.strip())
        return out

    def _vector_retrieve(self, queries: list[str]) -> list[Scored]:
        ranked_lists = [self.store.search(q, k=CONFIG.candidate_k) for q in queries]
        fused = reciprocal_rank_fusion(ranked_lists)
        return fused[:CONFIG.candidate_k]

    def _rerank(self, question: str, candidates: list[Scored]) -> list[Scored]:
        if self.use_rerank and candidates:
            return self.reranker.rerank(question, candidates, CONFIG.top_k)
        return candidates[:CONFIG.top_k]

    def _expand_parents(self, contexts: list[Scored]) -> list[Scored]:
        """Swap each child chunk for its parent section (de-duplicated)."""
        if not self.use_parent_expansion:
            return contexts
        out, seen = [], set()
        for s in contexts:
            pid = s.chunk.metadata.get("parent_id")
            if pid and pid in self.parents:
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(Scored(chunk=self.parents[pid], score=s.score,
                                  source=s.source))
            else:
                out.append(s)
        return out

    # ------------------------------------------------------------------ #
    def query(self, question: str) -> RAGResult:
        trace: dict = {"techniques": self._active_techniques()}
        cur_q = question

        rounds = 0
        all_queries: list[str] = []
        while True:
            queries = self._build_queries(cur_q)
            all_queries.extend(queries)
            candidates = self._vector_retrieve(queries)
            reranked = self._rerank(question, candidates)

            if not self.use_corrective:
                trace["queries"] = all_queries
                return self._finalize(question, reranked, trace)

            grades = grade_relevance(question, reranked)
            trace.setdefault("corrective_rounds", []).append(
                {"round": rounds, "query": cur_q,
                 "max_grade": max(grades) if grades else 0.0,
                 "n_relevant": int(sum(g >= 0.5 for g in grades))})

            if is_sufficient(grades):
                # drop clearly-irrelevant chunks before answering
                kept = [c for c, g in zip(reranked, grades) if g >= 0.5] or reranked
                trace["queries"] = all_queries
                return self._finalize(question, kept, trace)

            rounds += 1
            if rounds >= CONFIG.max_corrective_rounds:
                # evidence stayed weak -> refuse instead of hallucinating
                trace["queries"] = all_queries
                trace["refused_reason"] = "insufficient evidence after corrective retries"
                return RAGResult(question=question, answer=REFUSAL, contexts=reranked,
                                 refused=True, trace=trace)
            cur_q = rewrite_query(question, reranked)

    def _finalize(self, question: str, contexts: list[Scored],
                  trace: dict) -> RAGResult:
        final = self._expand_parents(contexts)
        answer = generate_answer(question, final)
        return RAGResult(question=question, answer=answer, contexts=final,
                         refused=looks_like_refusal(answer), trace=trace)

    def _active_techniques(self) -> list[str]:
        t = []
        if self.use_query_expansion: t.append("query_expansion")
        if self.use_multi_query: t.append("multi_query")
        if self.use_rerank: t.append(f"rerank:{self.rerank_method}")
        if self.use_corrective: t.append("corrective")
        if self.use_parent_expansion: t.append("parent_expansion")
        return t
