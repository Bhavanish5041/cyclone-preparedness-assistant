"""CycloneLM configuration module.

Defines strongly typed, serializable configuration schemas for CycloneLM
model architecture, training dynamics, inference generation, and tokenization.
Supports YAML and JSON loading and dumping.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class CycloneLMConfig:
    """Model architecture configuration for CycloneLM decoder-only Transformer."""

    vocab_size: int = 1024
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.1
    bias: bool = True
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def estimate_parameter_count(self) -> int:
        """Estimate the parameter count for this configuration."""
        token_emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        # Per layer:
        # ln1: 2 * d_model
        # qkv: 3 * (d_model * d_model + (d_model if bias else 0))
        # out_proj: d_model * d_model + (d_model if bias else 0)
        # ln2: 2 * d_model
        # mlp1: d_model * d_ff + (d_ff if bias else 0)
        # mlp2: d_ff * d_model + (d_model if bias else 0)
        per_layer = (
            2 * self.d_model  # ln1
            + 3 * (self.d_model * self.d_model + (self.d_model if self.bias else 0))
            + (self.d_model * self.d_model + (self.d_model if self.bias else 0))
            + 2 * self.d_model  # ln2
            + (self.d_model * self.d_ff + (self.d_ff if self.bias else 0))
            + (self.d_ff * self.d_model + (self.d_model if self.bias else 0))
        )
        final_ln = 2 * self.d_model
        lm_head = 0 if self.tie_word_embeddings else (self.vocab_size * self.d_model)
        return token_emb + pos_emb + (self.n_layers * per_layer) + final_ln + lm_head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CycloneLMConfig:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        content = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    @classmethod
    def from_yaml(cls, path_or_content: str | Path) -> CycloneLMConfig:
        p = Path(path_or_content)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(str(path_or_content))
        return cls.from_dict(data or {})


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for model pretraining and fine-tuning."""

    batch_size: int = 8
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 500
    epochs: int = 15
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    early_stopping_patience: Optional[int] = None
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 25
    save_interval: int = 50
    checkpoint_dir: str = "models/cyclonelm/checkpoints"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        content = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    @classmethod
    def from_yaml(cls, path_or_content: str | Path) -> TrainingConfig:
        p = Path(path_or_content)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(str(path_or_content))
        return cls.from_dict(data or {})


@dataclass
class CycloneLMFullConfig:
    """Consolidated system configuration combining model and training configurations."""

    model: CycloneLMConfig = field(default_factory=CycloneLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CycloneLMFullConfig:
        model_data = data.get("model", {})
        training_data = data.get("training", {})
        return cls(
            model=CycloneLMConfig.from_dict(model_data),
            training=TrainingConfig.from_dict(training_data),
        )

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        content = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    @classmethod
    def from_yaml(cls, path_or_content: str | Path) -> CycloneLMFullConfig:
        p = Path(path_or_content)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(str(path_or_content))
        return cls.from_dict(data or {})
