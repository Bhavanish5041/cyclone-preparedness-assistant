"""Decoder-only Transformer implementation for CycloneLM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cyclonelm.config.model_config import CycloneLMConfig
from src.cyclonelm.model.attention import CausalSelfAttention


@dataclass
class CausalLMOutput:
    """Output container for CycloneLM forward passes."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class FeedForward(nn.Module):
    """Feedforward network with GELU activation and dropout."""

    def __init__(self, config: CycloneLMConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block with Pre-LayerNorm architecture."""

    def __init__(self, config: CycloneLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LayerNorm with residual connections
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CycloneLM(nn.Module):
    """CycloneLM: Domain-specialised decoder-only Transformer language model."""

    def __init__(self, config: CycloneLMConfig) -> None:
        super().__init__()
        self.config = config

        # Token and learned positional embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Language Model output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Weight initialization
        self.apply(self._init_weights)

        # Scale weights of residual projections according to GPT-2 scheme
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def count_parameters(self, non_embedding_only: bool = False) -> int:
        """Calculate the total number of trainable parameters."""
        if non_embedding_only:
            return sum(
                p.numel()
                for n, p in self.named_parameters()
                if "embeddings" not in n and p.requires_grad
            )
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> CausalLMOutput:
        """Forward pass for CycloneLM.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            targets: Optional LongTensor of shape (batch_size, seq_len)
            attention_mask: Optional LongTensor of shape (batch_size, seq_len)
            ignore_index: Loss ignore index for padding tokens

        Returns:
            CausalLMOutput containing logits and optional scalar cross-entropy loss.
        """
        device = input_ids.device
        b, t = input_ids.size()

        if t > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {t} exceeds maximum sequence length {self.config.max_seq_len}"
            )

        positions = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # Token + Positional embeddings
        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.pos_embeddings(positions)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Final LayerNorm
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss = None
        if targets is not None:
            # Flatten across sequence dimension for cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=ignore_index,
            )

        return CausalLMOutput(logits=logits, loss=loss)
