"""Embedding similarity classifier using all-MiniLM-L6-v2."""

from typing import Dict, List, Optional
import numpy as np

from src.cyclonelm.classification.classifiers.base import BaseIntentClassifier, ClassificationPrediction
from src.cyclonelm.classification.config import (
    ALL_CLASSIFIER_CLASSES,
    SUPPORTED_INTENTS,
)
from src.cyclonelm.classification.embedder import MiniLMEmbedder


class EmbeddingSimilarityClassifier(BaseIntentClassifier):
    """Classifies user queries using cosine similarity against class centroids.

    Note: Raw embedding similarities must not be treated as calibrated probabilities.
    Softmax with a calibrated temperature is applied over cosine similarities.
    """

    def __init__(
        self,
        embedder: Optional[MiniLMEmbedder] = None,
        temperature: float = 0.1,
    ) -> None:
        self.embedder = embedder or MiniLMEmbedder()
        self.temperature = temperature
        self.centroids: Dict[str, np.ndarray] = {}
        self.classes: List[str] = []

    def fit(self, texts: List[str], labels: List[str]) -> "EmbeddingSimilarityClassifier":
        embeddings = self.embedder.embed_texts(texts, normalize=True)
        self.classes = sorted(list(set(labels)))

        self.centroids = {}
        for c in self.classes:
            indices = [i for i, lbl in enumerate(labels) if lbl == c]
            if indices:
                centroid = np.mean(embeddings[indices], axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                self.centroids[c] = centroid

        return self

    def predict(self, text: str) -> ClassificationPrediction:
        if not self.centroids:
            raise ValueError("Classifier must be fitted before predict() is called.")

        q_emb = self.embedder.embed_single(text, normalize=True)

        raw_sims: Dict[str, float] = {}
        for c in self.classes:
            sim = float(np.dot(q_emb, self.centroids[c]))
            raw_sims[c] = sim

        # Temperature-scaled softmax over similarities to produce relative scores
        sim_arr = np.array([raw_sims[c] for c in self.classes])
        exp_sims = np.exp(sim_arr / self.temperature)
        probs = exp_sims / np.sum(exp_sims)

        prob_dict = {c: round(float(probs[i]), 4) for i, c in enumerate(self.classes)}
        best_idx = int(np.argmax(probs))
        best_intent = self.classes[best_idx]
        best_conf = round(float(probs[best_idx]), 4)

        sorted_scores = sorted(
            [(c, round(raw_sims[c], 4)) for c in self.classes],
            key=lambda x: x[1],
            reverse=True,
        )

        return ClassificationPrediction(
            intent=best_intent,
            confidence=best_conf,
            is_supported_tool=(best_intent in SUPPORTED_INTENTS),
            probabilities=prob_dict,
            all_scores=sorted_scores,
        )
