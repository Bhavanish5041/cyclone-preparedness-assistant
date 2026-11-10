"""Causal Multi-Head Self-Attention for CycloneLM."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cyclonelm.config.model_config import CycloneLMConfig


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with pre-computed lower-triangular mask."""

    def __init__(self, config: CycloneLMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len

        # Key, Query, Value projections in a combined or separate linear
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: upper triangular values set to -inf
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, self.max_seq_len, self.max_seq_len))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional padding mask of shape (batch_size, seq_len) with 1 for valid and 0 for pad.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.size()

        # Project and reshape into (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        # scores shape: (B, n_heads, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask: mask out future tokens (where causal_mask == 0)
        causal = self.causal_mask[:, :, :T, :T]  # type: ignore[index]
        scores = scores.masked_fill(causal == 0, float("-inf"))

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            pad_mask = attention_mask.view(B, 1, 1, T)
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        # Handle all -inf edge case (e.g. if entirely padded)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Context output: (B, n_heads, T, head_dim) -> (B, T, C)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and residual dropout
        out = self.resid_dropout(self.out_proj(out))
        return out
