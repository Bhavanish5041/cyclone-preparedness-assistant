"""Unit tests for CycloneLM training pipeline."""

from pathlib import Path

import pytest
import torch

from src.cyclonelm.config.model_config import CycloneLMConfig, TrainingConfig
from src.cyclonelm.model.transformer import CycloneLM
from src.cyclonelm.training.dataset import CycloneDataset, create_dataloaders
from src.cyclonelm.training.trainer import CycloneTrainer, set_seed


def test_dataset_chunking():
    tokens = list(range(100))
    ds = CycloneDataset(tokens, max_seq_len=16, stride=16)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["input_ids"].shape == (16,)
    assert sample["targets"].shape == (16,)
    # targets should be shifted by 1 relative to input_ids
    assert torch.equal(sample["targets"][:-1], sample["input_ids"][1:])


def test_training_loss_decrease(tmp_path):
    set_seed(42)
    cfg = CycloneLMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=16,
    )
    train_cfg = TrainingConfig(
        batch_size=4,
        learning_rate=0.01,
        max_steps=20,
        epochs=5,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        eval_interval=10,
        save_interval=10,
    )

    model = CycloneLM(cfg)
    tokens = [i % 64 for i in range(200)]
    train_loader, val_loader = create_dataloaders(tokens, tokens[:50], max_seq_len=16, batch_size=4)

    trainer = CycloneTrainer(model, train_cfg, train_loader, val_loader)
    initial_loss, _ = trainer.evaluate()
    summary = trainer.train()

    assert summary["global_steps"] >= 10
    # Final val loss should be lower than or equal to initial unoptimized loss
    assert summary["final_val_loss"] < initial_loss


def test_checkpoint_roundtrip(tmp_path):
    set_seed(42)
    cfg = CycloneLMConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
    train_cfg = TrainingConfig(checkpoint_dir=str(tmp_path / "checkpoints"))

    model1 = CycloneLM(cfg)
    train_loader, val_loader = create_dataloaders(list(range(50)), list(range(20)), max_seq_len=16, batch_size=2)
    trainer1 = CycloneTrainer(model1, train_cfg, train_loader, val_loader)

    ckpt_file = tmp_path / "checkpoints" / "test_ckpt.pt"
    trainer1.save_checkpoint(ckpt_file, val_loss=4.5)
    assert ckpt_file.exists()

    model2 = CycloneLM(cfg)
    trainer2 = CycloneTrainer(model2, train_cfg, train_loader, val_loader)
    trainer2.load_checkpoint(ckpt_file)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_instruction_sft_dataset_loss_masking():
    from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer
    from src.cyclonelm.training.dataset import InstructionSFTDataset

    tok = BPETokenizer()
    examples = [
        {
            "instruction": "What is AHP?",
            "response": "Analytic Hierarchy Process.",
        }
    ]
    ds = InstructionSFTDataset(examples, tokenizer=tok, max_seq_len=64, ignore_index=-100)
    assert len(ds) == 1
    sample = ds[0]

    assert sample["input_ids"].shape == (64,)
    assert sample["targets"].shape == (64,)
    assert sample["attention_mask"].shape == (64,)

    # Targets must begin with -100 for masked prompt tokens
    assert sample["targets"][0].item() == -100
    # There must be unmasked target tokens corresponding to the response
    unmasked = [t.item() for t in sample["targets"] if t.item() != -100]
    assert len(unmasked) > 0
    # Trailing padding targets must be -100
    assert sample["targets"][-1].item() == -100
    assert sample["attention_mask"][-1].item() == 0

