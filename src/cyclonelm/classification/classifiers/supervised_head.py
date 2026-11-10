"""Supervised PyTorch classification head on frozen MiniLM embeddings."""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.cyclonelm.classification.classifiers.base import BaseIntentClassifier, ClassificationPrediction
from src.cyclonelm.classification.config import EMBEDDING_DIM, SUPPORTED_INTENTS
from src.cyclonelm.classification.embedder import MiniLMEmbedder


class IntentHeadModule(nn.Module):
    """PyTorch MLP classification head."""

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupervisedHeadClassifier(BaseIntentClassifier):
    """PyTorch MLP classifier operating on frozen sentence embeddings."""

    def __init__(
        self,
        embedder: Optional[MiniLMEmbedder] = None,
        epochs: int = 40,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        random_seed: int = 42,
    ) -> None:
        self.embedder = embedder or MiniLMEmbedder()
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.classes: List[str] = []
        self.model: Optional[IntentHeadModule] = None
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "SupervisedHeadClassifier":
        torch.manual_seed(self.random_seed)
        self.classes = sorted(list(set(labels)))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_train = torch.tensor([class_to_idx[lbl] for lbl in labels], dtype=torch.long)

        embeddings = self.embedder.embed_texts(texts, normalize=True)
        x_train = torch.tensor(embeddings, dtype=torch.float32)

        self.model = IntentHeadModule(
            in_features=EMBEDDING_DIM,
            num_classes=len(self.classes),
            hidden_dim=128,
            dropout=0.15,
        )
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batch_size = 32
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.is_fitted = True
        return self

    def predict(self, text: str) -> ClassificationPrediction:
        if not self.is_fitted or self.model is None:
            raise ValueError("Classifier must be fitted before predict() is called.")

        q_emb = self.embedder.embed_single(text, normalize=True)
        x = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

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
