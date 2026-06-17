"""Grounded answer generation, shared by both systems.

The prompt restricts the model to the retrieved context and tells it to say it
cannot answer when the context is insufficient, so unanswerable questions get a
refusal instead of a hallucinated answer.
"""
from __future__ import annotations

from .llm import get_llm
from .types import Scored

REFUSAL = ("I cannot answer this question based on the provided documents. "
           "The available evidence does not contain the necessary information.")

_SYS = (
    "You are a precise question-answering assistant for a documentation corpus "
    "about OpenCog Hyperon, MeTTa, the AtomSpace, and iCog Labs. You answer ONLY "
    "using the provided context passages. If the context does not contain the "
    "answer, you MUST say you cannot answer from the documents - never guess or "
    "use outside knowledge. Cite the sources you used by their [n] markers."
)


def format_context(contexts: list[Scored]) -> str:
    blocks = []
    for i, s in enumerate(contexts, 1):
        text = s.chunk.metadata.get("raw_text", s.chunk.text)
        title = s.chunk.title
        blocks.append(f"[{i}] (source: {title})\n{text}")
    return "\n\n".join(blocks)


def generate_answer(question: str, contexts: list[Scored]) -> str:
    if not contexts:
        return REFUSAL
    ctx = format_context(contexts)
    prompt = (
        f"Context passages:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above. If the context does not "
        f"contain the answer, reply exactly with: \"{REFUSAL}\" "
        "Otherwise give a concise, accurate answer and cite sources with [n]."
    )
    try:
        return get_llm().generate(prompt, system=_SYS, temperature=0.1).strip()
    except Exception as e:
        return f"[generation error: {e}]"


def looks_like_refusal(answer: str) -> bool:
    a = answer.lower()
    markers = ["cannot answer", "could not find", "does not contain",
               "no information", "not contain the necessary", "unable to answer",
               "isn't in the provided", "not in the provided", "i don't have"]
    return any(m in a for m in markers)
