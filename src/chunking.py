"""Chunking strategies.

Basic system:    naive fixed-size character windows (the baseline).
Advanced system: two techniques combined --
  1. context-aware section splitting (boundaries fall on real markdown structure)
  2. hierarchical parent/child chunks (small children are retrieved; their parent
     section is fetched at answer time for fuller context).
A short situating header is prepended to each child so an isolated window still
carries which document/section it came from.
"""
from __future__ import annotations

import re

from .config import CONFIG
from .ingest import Document
from .types import Chunk

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


# --------------------------------------------------------------------------- #
# BASIC: fixed-size chunking                                                   #
# --------------------------------------------------------------------------- #
def fixed_size_chunks(doc: Document,
                      size: int | None = None,
                      overlap: int | None = None) -> list[Chunk]:
    """Fixed-size character chunking with overlap (the baseline).

    Slices the raw text every `size` characters regardless of headings or
    sentences, which is the weakness the advanced chunker addresses.
    """
    size = size or CONFIG.basic_chunk_size
    overlap = overlap or CONFIG.basic_chunk_overlap
    text = doc.text
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(1, size - overlap)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(Chunk(
                id=f"{doc.doc_id}::f{idx}",
                text=piece,
                doc_id=doc.doc_id,
                metadata={"doc_title": doc.title, "chunk_type": "fixed"},
            ))
            idx += 1
        start += step
    return chunks


# --------------------------------------------------------------------------- #
# ADVANCED: context-aware section parsing                                      #
# --------------------------------------------------------------------------- #
def _split_sections(doc: Document) -> list[tuple[str, str]]:
    """Split a markdown doc into (section_path, body) at heading boundaries.

    This is *context-aware*: boundaries fall on real structure (## headings), so a
    chunk never starts in the middle of a heading, definition, or list.
    """
    lines = doc.text.split("\n")
    sections: list[tuple[str, str]] = []
    cur_head = doc.title
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((cur_head, body))

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            heading = m.group(2).strip()
            cur_head = heading if level <= 1 else f"{doc.title} > {heading}"
        else:
            buf.append(line)
    flush()
    return sections


def _pack_paragraphs(body: str, max_chars: int) -> list[str]:
    """Greedily pack paragraphs into <= max_chars blocks (no mid-paragraph cuts)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    blocks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            blocks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        blocks.append(cur)
    return blocks


def _window_sentences(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over sentence boundaries (context-aware child splitting).

    Each sentence is consumed exactly once; on overflow we flush the current
    window and seed the next one with the trailing sentences that fit in `overlap`
    characters. This always advances, even for sentences longer than `size`.
    """
    sents = [s for s in _SENT.split(text) if s.strip()]
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > size:
            windows.append(" ".join(cur).strip())
            # seed overlap from the tail of the window just flushed
            keep: list[str] = []
            klen = 0
            for ps in reversed(cur):
                if klen + len(ps) > overlap:
                    break
                keep.insert(0, ps)
                klen += len(ps) + 1
            cur, cur_len = keep, klen
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        windows.append(" ".join(cur).strip())
    return [w for w in windows if w]


def context_header(doc_title: str, section: str) -> str:
    """Deterministic situating header prepended to a child chunk before embedding."""
    return f"Document: {doc_title}. Section: {section}."


def hierarchical_chunks(doc: Document) -> tuple[list[Chunk], dict[str, Chunk]]:
    """Build parent (section) and child (window) chunks.

    Returns (children, parents_by_id). Children are what gets embedded/retrieved;
    parents are fetched at answer time to give the LLM fuller surrounding context.

    Each child's text is prefixed with a situating header so an isolated window
    still carries which document/section it's from.
    """
    children: list[Chunk] = []
    parents: dict[str, Chunk] = {}
    p_idx = 0
    for section, body in _split_sections(doc):
        for block in _pack_paragraphs(body, CONFIG.parent_max_chars):
            parent_id = f"{doc.doc_id}::p{p_idx}"
            parents[parent_id] = Chunk(
                id=parent_id, text=block, doc_id=doc.doc_id,
                metadata={"doc_title": doc.title, "section": section,
                          "chunk_type": "parent"},
            )
            windows = _window_sentences(block, CONFIG.child_chunk_size,
                                        CONFIG.child_chunk_overlap)
            for c_idx, win in enumerate(windows):
                header = context_header(doc.title, section)
                contextualized = f"[{header}]\n{win}"
                children.append(Chunk(
                    id=f"{parent_id}c{c_idx}",
                    text=contextualized,
                    doc_id=doc.doc_id,
                    metadata={
                        "doc_title": doc.title,
                        "section": section,
                        "parent_id": parent_id,
                        "context_header": header,
                        "raw_text": win,
                        "chunk_type": "child",
                    },
                ))
            p_idx += 1
    return children, parents
