"""Unit tests for CycloneLM progressive model scaling profiles."""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch

from src.cyclonelm.config.model_config import CycloneLMConfig
from src.cyclonelm.config.scaling_profiles import (
    SCALING_PROFILES,
    audit_profile_architecture,
    get_profile_config,
)
from src.cyclonelm.model.transformer import CycloneLM


@pytest.mark.parametrize("profile_name", ["239k", "1m", "5m", "10m", "20m"])
def test_profile_configuration_exists_and_valid(profile_name: str):
    """Verify profile definition and configuration parameters."""
    cfg = get_profile_config(profile_name)
    assert isinstance(cfg, CycloneLMConfig)
    assert cfg.vocab_size == 1024
    assert cfg.max_seq_len >= 128
    assert cfg.d_model > 0
    assert cfg.n_layers > 0
    assert cfg.n_heads > 0
    assert cfg.d_model % cfg.n_heads == 0, f"d_model {cfg.d_model} not divisible by n_heads {cfg.n_heads}"


@pytest.mark.parametrize("profile_name, expected_range", [
    ("239k", (200_000, 300_000)),
    ("1m", (900_000, 1_200_000)),
    ("5m", (4_500_000, 6_000_000)),
    ("10m", (9_000_000, 12_000_000)),
    ("20m", (18_000_000, 22_000_000)),
])
def test_instantiated_model_parameter_count(profile_name: str, expected_range: tuple):
    """Verify exact parameter count from instantiated model matches target regime."""
    cfg = get_profile_config(profile_name)
    model = CycloneLM(cfg)
    actual_params = model.count_parameters()
    min_p, max_p = expected_range
    assert min_p <= actual_params <= max_p, (
        f"Profile {profile_name} has {actual_params:,} parameters, expected range [{min_p:,}, {max_p:,}]"
    )

    # Check non-embedding parameters are non-zero and less than total
    non_emb = model.count_parameters(non_embedding_only=True)
    assert 0 < non_emb < actual_params


@pytest.mark.parametrize("profile_name", ["239k", "1m", "5m", "10m", "20m"])
def test_causal_masking_invariant(profile_name: str):
    """Verify that earlier token logits are invariant to changes in subsequent tokens."""
    cfg = get_profile_config(profile_name)
    model = CycloneLM(cfg)
    model.eval()

    seq1 = torch.tensor([[12, 34, 56, 78]], dtype=torch.long)
    seq2 = torch.tensor([[12, 34, 999, 500]], dtype=torch.long)

    with torch.no_grad():
        out1 = model(seq1).logits
        out2 = model(seq2).logits

    # Positions 0 and 1 must be identical
    diff0 = torch.max(torch.abs(out1[0, 0] - out2[0, 0])).item()
    diff1 = torch.max(torch.abs(out1[0, 1] - out2[0, 1])).item()

    assert diff0 == 0.0, f"Causal violation at pos 0: diff={diff0}"
    assert diff1 == 0.0, f"Causal violation at pos 1: diff={diff1}"


@pytest.mark.parametrize("profile_name", ["239k", "1m", "5m", "10m", "20m"])
def test_forward_pass_execution(profile_name: str):
    """Verify forward pass with batch dimension and target loss computation."""
    cfg = get_profile_config(profile_name)
    model = CycloneLM(cfg)
    model.eval()

    batch_size = 2
    seq_len = 16
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        out = model(x, targets=y)

    assert out.logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert out.loss is not None
    assert not torch.isnan(out.loss)
    assert out.loss.item() > 0.0


def test_audit_profile_architecture_function():
    """Verify that audit_profile_architecture returns full structural metadata."""
    audit = audit_profile_architecture("239k")
    assert audit["profile"] == "239k"
    assert audit["total_parameters"] == 239360
    assert audit["is_divisible"] is True
    assert audit["causal_mask_verified"] is True
    assert audit["weights_mb"] > 0.0
    assert audit["optimizer_mb"] > 0.0
    assert audit["estimated_train_ram_mb"] > 0.0


def test_invalid_profile_name_raises():
    """Verify error on unknown scaling profile."""
    with pytest.raises(KeyError):
        get_profile_config("unknown_100m")
