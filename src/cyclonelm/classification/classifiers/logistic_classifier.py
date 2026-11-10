"""Lightweight traditional ML classifier (Logistic Regression) on MiniLM embeddings."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.cyclonelm.classification.classifiers.base import BaseIntentClassifier, ClassificationPrediction
from src.cyclonelm.classification.config import (
    ALL_CLASSIFIER_CLASSES,
    SUPPORTED_INTENTS,
)
from src.cyclonelm.classification.embedder import MiniLMEmbedder

DEFAULT_MODEL_PATH = Path("models/cyclonelm/v0.8/intent/logistic_intent_model.joblib")


class LogisticIntentClassifier(BaseIntentClassifier):
    """Calibrated Multinomial Logistic Regression trained on MiniLM embeddings.

    Provides well-calibrated probabilities, robust class weighting, and
    fast CPU training (< 1 second) and inference (< 2 ms).
    """

    def __init__(
        self,
        embedder: Optional[MiniLMEmbedder] = None,
        c_param: float = 1.0,
        random_state: int = 42,
        model_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ) -> None:
        self.embedder = embedder or MiniLMEmbedder()
        self.clf = LogisticRegression(
            C=c_param,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )
        self.classes: List[str] = []
        self.is_fitted = False

        if auto_load:
            target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            if target_path.exists():
                self.load(target_path)

    def save(self, path: Union[str, Path]) -> None:
        """Serialize fitted model and class list."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "classes": self.classes}, p)

    def load(self, path: Union[str, Path]) -> "LogisticIntentClassifier":
        """Load fitted model and class list from joblib file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        data = joblib.load(p)
        self.clf = data["clf"]
        self.classes = data["classes"]
        self.is_fitted = True
        return self

    def fit(self, texts: List[str], labels: List[str]) -> "LogisticIntentClassifier":
        embeddings = self.embedder.embed_texts(texts, normalize=True)
        self.clf.fit(embeddings, labels)
        self.classes = list(self.clf.classes_)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> ClassificationPrediction:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict() is called.")

        q_emb = self.embedder.embed_single(text, normalize=True).reshape(1, -1)
        probs = self.clf.predict_proba(q_emb)[0]

        prob_dict = {c: round(float(probs[i]), 4) for i, c in enumerate(self.classes)}
        best_idx = int(np.argmax(probs))
        best_intent = self.classes[best_idx]
        best_conf = round(float(probs[best_idx]), 4)

        sorted_scores = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        return ClassificationPrediction(
            intent=best_intent,
            confidence=best_conf,
            is_supported_tool=(best_intent in SUPPORTED_INTENTS),
            probabilities=prob_dict,
            all_scores=sorted_scores,
        )
