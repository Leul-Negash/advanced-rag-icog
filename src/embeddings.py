"""Local embedding model (sentence-transformers).

Shared by both the Basic and Advanced systems so the comparison is apples-to-
apples. No API key required - the model runs on the CPU.
"""
from __future__ import annotations

import threading

import numpy as np

from .config import CONFIG


class Embedder:
    """Thin wrapper around a sentence-transformers bi-encoder.

    Handles the asymmetry that retrieval models like BGE expect: a special
    instruction prefix on the *query* but not on the documents.
    """

    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or CONFIG.embed_model
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        # Only BGE-style models use the query instruction prefix.
        self._use_instruction = "bge" in self.model_name.lower()

    @classmethod
    def get(cls) -> "Embedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            batch_size=32, convert_to_numpy=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        if self._use_instruction:
            text = CONFIG.query_instruction + text
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
