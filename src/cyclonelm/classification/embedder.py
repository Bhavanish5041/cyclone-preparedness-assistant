"""Sentence embedding wrapper using all-MiniLM-L6-v2 on CPU."""

import logging
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.cyclonelm.classification.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class MiniLMEmbedder:
    """Manages sentence embedding generation using all-MiniLM-L6-v2."""

    _instance: "MiniLMEmbedder" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MiniLMEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = "cpu") -> None:
        if getattr(self, "_initialized", False):
            return
        self.model_name = model_name
        self.device = device
        logger.info("Initializing MiniLMEmbedder with model: %s on %s", model_name, device)
        # Configure torch thread count for CPU inference
        if device == "cpu" and torch.get_num_threads() < 4:
            torch.set_num_threads(8)
        self.model = SentenceTransformer(model_name, device=device)
        self._initialized = True

    def embed_texts(
        self, texts: List[str], batch_size: int = 32, normalize: bool = True
    ) -> np.ndarray:
        """Compute sentence embeddings for a list of strings."""
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Compute 1D embedding vector for a single string."""
        return self.embed_texts([text], batch_size=1, normalize=normalize)[0]
