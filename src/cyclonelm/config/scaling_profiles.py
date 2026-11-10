"""Architecture scaling profiles and validation routines for CycloneLM experiments.

Defines the progressive model scaling configurations:
- 239k: Baseline architecture (239,360 parameters)
- 1m: 1 Million parameter target (1,071,872 parameters)
- 5m: 5 Million parameter target (5,296,128 parameters)
- 10m: 10 Million parameter target (10,560,640 parameters)
- 20m: 20 Million parameter target (20,029,440 parameters)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch

from src.cyclonelm.config.model_config import CycloneLMConfig
from src.cyclonelm.model.transformer import CycloneLM

SCALING_PROFILES: Dict[str, Dict[str, Any]] = {
    "239k": {
        "name": "239k",
        "description": "Baseline domain Transformer architecture",
        "config": CycloneLMConfig(
            vocab_size=1024,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_seq_len=128,
            dropout=0.1,
            bias=True,
        ),
        "target_params": "~239K",
    },
    "1m": {
        "name": "1m",
        "description": "1M parameter compact domain architecture",
        "config": CycloneLMConfig(
            vocab_size=1024,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            max_seq_len=128,
            dropout=0.1,
            bias=True,
        ),
        "target_params": "~1M",
    },
    "5m": {
        "name": "5m",
        "description": "5M parameter medium domain architecture",
        "config": CycloneLMConfig(
            vocab_size=1024,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            max_seq_len=128,
            dropout=0.1,
            bias=True,
        ),
        "target_params": "~5M",
    },
    "10m": {
        "name": "10m",
        "description": "10M parameter large domain architecture",
        "config": CycloneLMConfig(
            vocab_size=1024,
            d_model=320,
            n_layers=8,
            n_heads=8,
            d_ff=1280,
            max_seq_len=128,
            dropout=0.1,
            bias=True,
        ),
        "target_params": "~10M",
    },
    "20m": {
        "name": "20m",
        "description": "20M parameter apex domain architecture",
        "config": CycloneLMConfig(
            vocab_size=1024,
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            max_seq_len=128,
            dropout=0.1,
            bias=True,
        ),
        "target_params": "~20M",
    },
}


def get_profile_config(name: str) -> CycloneLMConfig:
    """Retrieve the model configuration for a named scaling profile."""
    if name not in SCALING_PROFILES:
        valid = list(SCALING_PROFILES.keys())
        raise KeyError(f"Unknown profile: {name!r}. Available: {valid}")
    return SCALING_PROFILES[name]["config"]


def audit_profile_architecture(name: str) -> Dict[str, Any]:
    """Perform structural and causal audit on an instantiated scaling profile model."""
    cfg = get_profile_config(name)
    model = CycloneLM(cfg)

    # 1. Exact parameter counts
    total_params = model.count_parameters()
    non_emb_params = model.count_parameters(non_embedding_only=True)
    estimated_params = cfg.estimate_parameter_count()

    # 2. Divisibility check
    is_divisible = (cfg.d_model % cfg.n_heads == 0)
    head_dim = cfg.head_dim

    # 3. Causal masking verification (forward pass with perturbed future token)
    seq1 = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    seq2 = torch.tensor([[10, 20, 99, 88]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out1 = model(seq1).logits
        out2 = model(seq2).logits
    pos0_diff = float(torch.max(torch.abs(out1[0, 0] - out2[0, 0])).item())
    pos1_diff = float(torch.max(torch.abs(out1[0, 1] - out2[0, 1])).item())
    causal_mask_verified = (pos0_diff == 0.0 and pos1_diff == 0.0)

    # 4. Context window check
    context_window = cfg.max_seq_len

    # 5. Theoretical memory footprint (model weights in FP32 + Adam states: 4 + 8 = 12 bytes/param)
    weights_mb = (total_params * 4) / (1024 * 1024)
    optimizer_mb = (total_params * 8) / (1024 * 1024)
    estimated_train_ram_mb = (total_params * 16) / (1024 * 1024) + 100.0  # weights + grads + Adam + activations + overhead

    return {
        "profile": name,
        "total_parameters": total_params,
        "non_embedding_parameters": non_emb_params,
        "formula_matches_actual": total_params == estimated_params,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "head_dim": head_dim,
        "d_ff": cfg.d_ff,
        "max_seq_len": context_window,
        "vocab_size": cfg.vocab_size,
        "is_divisible": is_divisible,
        "causal_mask_verified": causal_mask_verified,
        "weights_mb": round(weights_mb, 2),
        "optimizer_mb": round(optimizer_mb, 2),
        "estimated_train_ram_mb": round(estimated_train_ram_mb, 2),
    }
