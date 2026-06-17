"""Offline smoke test for the pure-logic components (no API key, no model load).

Verifies chunking, section parsing, RRF fusion, and JSON parsing work before the
heavier (model/LLM) paths are exercised.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import fixed_size_chunks, hierarchical_chunks, _split_sections
from src.ingest import load_documents
from src.llm import _extract_json
from src.query_handling import reciprocal_rank_fusion
from src.types import Chunk, Scored


def main():
    docs = load_documents()
    assert len(docs) >= 10, f"expected >=10 docs, got {len(docs)}"
    print(f"[ok] loaded {len(docs)} documents")

    d = docs[0]
    fx = fixed_size_chunks(d)
    assert fx and all(c.text for c in fx)
    print(f"[ok] fixed-size chunking: {len(fx)} chunks from '{d.title}'")

    secs = _split_sections(d)
    assert len(secs) >= 1
    print(f"[ok] context-aware section split: {len(secs)} sections")

    children, parents = hierarchical_chunks(d)
    assert children and parents
    assert all("parent_id" in c.metadata for c in children)
    assert all(c.metadata["context_header"] for c in children)
    print(f"[ok] hierarchical: {len(parents)} parents, {len(children)} children "
          f"(contextual headers present)")

    # RRF fusion + dedup
    a = Chunk("x1", "a", "d1"); b = Chunk("x2", "b", "d1")
    l1 = [Scored(a, 0.9), Scored(b, 0.5)]
    l2 = [Scored(b, 0.8), Scored(a, 0.4)]
    fused = reciprocal_rank_fusion([l1, l2])
    assert len(fused) == 2, "RRF should de-duplicate to 2 unique chunks"
    print(f"[ok] RRF fusion + dedup: {len(fused)} unique chunks")

    # JSON extraction tolerance
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('noise {"b":2} tail') == {"b": 2}
    print("[ok] tolerant JSON parsing")

    print("\nALL OFFLINE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
