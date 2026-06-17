"""Evaluation: scores a RAGResult against the gold test set.

Metrics captured per question:
  - retrieval_hit : did at least one relevant document appear in the contexts?
  - refused       : did the system decline to answer?
  - correct       : final judgement (LLM-as-judge for answerable; correct refusal
                    for unanswerable).
  - answer_score  : LLM-judged 0/1 correctness vs the gold answer (answerable only).
  - grounded      : is the answer supported by the retrieved contexts?
"""
from __future__ import annotations

import re

from .generation import format_context
from .llm import get_llm
from .types import RAGResult

_STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "was",
         "were", "by", "for", "with", "that", "which", "what", "who", "how", "does",
         "stand", "it", "its", "this", "as", "at", "be", "from", "into", "version"}


def heuristic_judge(question: str, gold: str, answer: str) -> dict:
    """Zero-LLM correctness check: do the key gold terms appear in the answer?

    Used to score answers without spending LLM quota. Robust enough for factual
    gold answers with distinctive keywords (e.g. 'Meta Type Talk', 'Addis Ababa',
    'Warren Abstract Machine').
    """
    def toks(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9.]+", s.lower())
                if w not in _STOP and len(w) > 2}

    gold_t = toks(gold)
    ans_t = toks(answer)
    if not gold_t:
        return {"correct": 0, "reason": "no gold terms"}
    covered = gold_t & ans_t
    ratio = len(covered) / len(gold_t)
    ok = ratio >= 0.6
    return {"correct": int(ok), "reason": f"key-term coverage {ratio:.0%}",
            "method": "heuristic"}


def retrieval_hit(result: RAGResult, relevant_docs: list[str]) -> bool | None:
    if not relevant_docs:
        return None  # not applicable for unanswerable questions
    got = set(result.context_docs())
    return any(d in got for d in relevant_docs)


def judge_answer(question: str, gold: str, answer: str) -> dict:
    """LLM-as-judge: is the answer correct vs the gold answer?"""
    prompt = (
        f"Question: {question}\n"
        f"Reference (gold) answer: {gold}\n"
        f"Candidate answer: {answer}\n\n"
        "Is the candidate answer factually correct and does it convey the key "
        "information of the reference answer? Ignore wording/style differences. "
        'Return JSON: {"correct": 0 or 1, "reason": "short reason"}.'
    )
    try:
        out = get_llm().generate_json(prompt, temperature=0.0)
        return {"correct": int(out.get("correct", 0)), "reason": out.get("reason", "")}
    except Exception as e:
        return {"correct": 0, "reason": f"judge error: {e}"}


def judge_grounded(answer: str, result: RAGResult) -> int:
    """Is the answer supported by the retrieved context (no hallucination)?"""
    if result.refused or not result.contexts:
        return 1 if result.refused else 0
    ctx = format_context(result.contexts)
    prompt = (
        f"Context:\n{ctx}\n\nAnswer: {answer}\n\n"
        "Is every factual claim in the answer supported by the context above? "
        'Return JSON: {"grounded": 0 or 1}.'
    )
    try:
        out = get_llm().generate_json(prompt, temperature=0.0)
        return int(out.get("grounded", 0))
    except Exception:
        return 0


def evaluate_one(result: RAGResult, q: dict, judge: str = "heuristic") -> dict:
    """judge: 'heuristic' (no LLM), 'llm' (LLM-as-judge), or 'none'."""
    answerable = q.get("answerable", True)
    hit = retrieval_hit(result, q.get("relevant_docs", []))
    row = {
        "id": q["id"],
        "category": q["category"],
        "answerable": answerable,
        "retrieval_hit": hit,
        "refused": result.refused,
        "answer": result.answer,
        "context_docs": result.context_docs(),
    }
    if not answerable:
        # correct iff the system refused
        row["correct"] = bool(result.refused)
        row["answer_score"] = None
        row["grounded"] = 1 if result.refused else 0
        return row

    if result.refused:
        row.update(answer_score=0, grounded=0, correct=False,
                   judge_reason="refused an answerable question")
        return row

    if judge == "llm":
        j = judge_answer(q["question"], q.get("gold", ""), result.answer)
        row["grounded"] = judge_grounded(result.answer, result)
    elif judge == "heuristic":
        j = heuristic_judge(q["question"], q.get("gold", ""), result.answer)
        row["grounded"] = 1 if hit else 0  # proxy: grounded if a relevant doc was used
    else:
        j = {"correct": None, "reason": "not judged"}
        row["grounded"] = None
    row["answer_score"] = j["correct"]
    row["judge_reason"] = j["reason"]
    row["correct"] = bool(j["correct"]) if j["correct"] is not None else None
    return row


def aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def frac(items, key):
        vals = [bool(i[key]) for i in items if i.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n_total": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval_recall": frac([r for r in answerable if r["retrieval_hit"] is not None],
                                 "retrieval_hit"),
        "answer_accuracy": frac(answerable, "correct"),
        "grounded_rate": frac(answerable, "grounded"),
        "refusal_accuracy_unanswerable": frac(unanswerable, "correct"),
        "hallucination_on_unanswerable": (
            round(sum(not r["refused"] for r in unanswerable) / len(unanswerable), 3)
            if unanswerable else None),
        "overall_correct": frac(rows, "correct"),
    }
