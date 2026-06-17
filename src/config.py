"""Central configuration for both RAG systems.

Everything tunable lives here so the Basic and Advanced systems can be compared
fairly: they share the same embedding model, the same vector store backend, and
the same generation model. Only the *techniques* differ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv optional; env vars still work without it
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- paths ---
    root: Path = ROOT
    raw_dir: Path = ROOT / "data" / "raw"
    storage_dir: Path = ROOT / "storage"

    # --- embeddings (local sentence-transformers) ---
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    # bge models want this instruction prefixed to *queries* (not documents).
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- re-ranking (local cross-encoder) ---
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- LLM (Gemini by default; pluggable) ---
    # flash-lite has the most generous free-tier daily quota, so it is the default.
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    llm_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    # minimum seconds between LLM calls (free tier RPM is low) - set 0 to disable
    llm_min_interval: float = float(os.getenv("LLM_MIN_INTERVAL", "6.0"))

    # --- BASIC RAG chunking (naive fixed-size) ---
    basic_chunk_size: int = 600       # characters
    basic_chunk_overlap: int = 80     # characters

    # --- ADVANCED RAG chunking (hierarchical parent/child + context-aware) ---
    parent_max_chars: int = 1600      # parent = a whole section (context-aware boundary)
    child_chunk_size: int = 450
    child_chunk_overlap: int = 60

    # --- retrieval ---
    # Basic is the naive baseline: it retrieves a small fixed number of top chunks
    # by raw similarity, with no re-ranking. Advanced fetches a wider candidate pool
    # (candidate_k), re-ranks it, and passes the best top_k to the LLM.
    basic_top_k: int = 3              # naive baseline context size
    top_k: int = 6                    # advanced context after re-ranking
    candidate_k: int = 12             # candidates fetched before re-ranking
    rrf_k: int = 60                   # Reciprocal Rank Fusion constant

    # --- corrective RAG ---
    relevance_threshold: float = 0.5  # mean grade below this triggers a corrective retry
    max_corrective_rounds: int = 2

    # --- collection names ---
    basic_collection: str = "basic_rag"
    advanced_collection: str = "advanced_rag"

    def __post_init__(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
