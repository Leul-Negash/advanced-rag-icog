"""Document loading and light cleaning."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    path: str


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)        # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse blank runs
    return text.strip()


def _title_from(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def load_documents(raw_dir: Path | None = None) -> list[Document]:
    raw_dir = raw_dir or CONFIG.raw_dir
    docs: list[Document] = []
    for path in sorted(raw_dir.glob("*.md")):
        text = _clean(path.read_text(encoding="utf-8"))
        doc_id = path.stem
        docs.append(Document(
            doc_id=doc_id,
            title=_title_from(text, doc_id),
            text=text,
            path=str(path),
        ))
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {raw_dir}")
    return docs
