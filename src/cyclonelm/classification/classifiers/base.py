"""Base abstraction and result dataclasses for CycloneLM intent classifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ClassificationPrediction:
    """Standardized output from an intent classifier."""

    intent: str
    confidence: float
    is_supported_tool: bool
    probabilities: Dict[str, float] = field(default_factory=dict)
    all_scores: List[Tuple[str, float]] = field(default_factory=list)


class BaseIntentClassifier(ABC):
    """Abstract interface for all CycloneLM intent classifiers."""

    @abstractmethod
    def fit(self, texts: List[str], labels: List[str]) -> "BaseIntentClassifier":
        """Train or fit the classifier on labelled examples."""
        pass

    @abstractmethod
    def predict(self, text: str) -> ClassificationPrediction:
        """Predict the primary intent and confidence for a single input query."""
        pass

    def batch_predict(self, texts: List[str]) -> List[ClassificationPrediction]:
        """Predict intents for a batch of queries."""
        return [self.predict(t) for t in texts]
