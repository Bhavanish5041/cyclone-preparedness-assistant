"""Unit tests for CycloneLM Transformer architecture."""

import pytest
import torch

from src.cyclonelm.config.model_config import CycloneLMConfig
from src.cyclonelm.model.attention import CausalSelfAttention
from src.cyclonelm.model.generate import generate
from src.cyclonelm.model.transformer import CycloneLM


@pytest.fixture
def small_config():
    return CycloneLMConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
    )


def test_parameter_count_limit(small_config):
    model = CycloneLM(small_config)
    params = model.count_parameters()
    assert params < 1_000_000
    assert params > 1_000


def test_forward_pass_output_shape(small_config):
    model = CycloneLM(small_config)
    model.eval()

    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    assert out.logits.shape == (batch_size, seq_len, small_config.vocab_size)
    assert out.loss is None


def test_loss_computation(small_config):
    model = CycloneLM(small_config)
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    out = model(input_ids, targets=targets)
    assert out.loss is not None
    assert out.loss.item() > 0.0
    assert not torch.isnan(out.loss)


def test_causal_masking():
    """Verify that future tokens do not affect past token logits (causality)."""
    cfg = CycloneLMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )
    model = CycloneLM(cfg)
    model.eval()

    # Sequence 1: [10, 20, 30]
    seq1 = torch.tensor([[10, 20, 30]])
    # Sequence 2: [10, 20, 99] (identical first 2 tokens, different 3rd token)
    seq2 = torch.tensor([[10, 20, 45]])

    with torch.no_grad():
        out1 = model(seq1)
        out2 = model(seq2)

    # Logits at position 0 and 1 must be identical despite difference at position 2
    torch.testing.assert_close(out1.logits[:, 0, :], out2.logits[:, 0, :], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out1.logits[:, 1, :], out2.logits[:, 1, :], atol=1e-5, rtol=1e-5)


def test_autoregressive_generation(small_config):
    model = CycloneLM(small_config)
    model.eval()

    prompt = torch.tensor([[5, 10, 15]])
    gen_len = 10
    out = generate(
        model,
        prompt,
        max_new_tokens=gen_len,
        temperature=0.0,  # greedy
    )
    assert out.shape == (1, 3 + gen_len)
    # Check that initial prompt is preserved
    assert torch.equal(out[:, :3], prompt)
