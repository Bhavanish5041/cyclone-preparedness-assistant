"""Training command script for CycloneLM v0.1.

Trains the BPE tokenizer on domain data, initializes CycloneLM from random weights,
and runs the training loop on CPU with checkpointing and loss tracking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.cyclonelm.config.model_config import (
    CycloneLMConfig,
    CycloneLMFullConfig,
    TrainingConfig,
)
from src.cyclonelm.model.transformer import CycloneLM
from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer
from src.cyclonelm.training.dataset import create_dataloaders, load_corpus_tokens
from src.cyclonelm.training.trainer import CycloneTrainer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CycloneLM from scratch on domain data.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/cyclonelm/config/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/cyclonelm/processed/cyclonelm_combined_corpus.txt",
        help="Path to pretraining corpus file",
    )
    parser.add_argument(
        "--train_corpus",
        type=str,
        default="data/cyclonelm/processed/train_combined.txt",
        help="Path to pretraining train corpus",
    )
    parser.add_argument(
        "--val_corpus",
        type=str,
        default="data/cyclonelm/processed/val_combined.txt",
        help="Path to pretraining val corpus",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/cyclonelm",
        help="Directory to save checkpoints and tokenizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load configuration
    cfg_path = Path(args.config)
    if cfg_path.exists():
        full_cfg = CycloneLMFullConfig.from_yaml(cfg_path)
    else:
        full_cfg = CycloneLMFullConfig()

    model_cfg = full_cfg.model
    train_cfg = full_cfg.training

    if args.epochs:
        train_cfg.epochs = args.epochs
    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size

    output_dir = Path(args.output_dir)
    tok_dir = output_dir / "tokenizer"
    ckpt_dir = output_dir / "checkpoints"
    tok_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_cfg.checkpoint_dir = str(ckpt_dir)

    set_seed(train_cfg.seed)

    print("=" * 60)
    print("CYCLONELM v0.1 TRAINING PIPELINE")
    print(f"Device: {train_cfg.device} | Seed: {train_cfg.seed}")
    print("=" * 60)

    # 2. Train or Load BPE Tokenizer
    tok_path = tok_dir / "vocab.json"
    tokenizer = BPETokenizer()
    combined_path = Path(args.corpus)

    if combined_path.exists():
        print(f"Training BPE tokenizer on domain corpus: {combined_path}")
        with open(combined_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        tokenizer.train(lines, vocab_size=model_cfg.vocab_size)
        tokenizer.save(tok_path)
        print(f"Tokenizer saved to {tok_path} (vocab size: {tokenizer.vocab_size})")
    elif tok_path.exists():
        print(f"Loading existing tokenizer from {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        raise FileNotFoundError(f"Corpus file not found: {combined_path}")

    # Synchronize vocab_size with actual tokenizer vocab size
    model_cfg.vocab_size = tokenizer.vocab_size

    # 3. Prepare tokenized datasets
    print("Tokenizing corpus...")
    train_tokens = load_corpus_tokens(args.train_corpus, tokenizer)
    val_tokens = load_corpus_tokens(args.val_corpus, tokenizer)

    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

    train_loader, val_loader = create_dataloaders(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        max_seq_len=model_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
    )
    print(f"Train batches per epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 4. Instantiate Model
    model = CycloneLM(model_cfg)
    param_count = model.count_parameters()
    print(f"Initialized CycloneLM Transformer from random weights:")
    print(f"  - Layers: {model_cfg.n_layers}")
    print(f"  - Hidden Dim (d_model): {model_cfg.d_model}")
    print(f"  - Attention Heads: {model_cfg.n_heads} (head_dim: {model_cfg.head_dim})")
    print(f"  - Feed-forward Dim (d_ff): {model_cfg.d_ff}")
    print(f"  - Context Window (max_seq_len): {model_cfg.max_seq_len}")
    print(f"  - Total Parameters: {param_count:,}")

    # 5. Train Model
    trainer = CycloneTrainer(
        model=model,
        train_config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    summary = trainer.train()

    # 6. Save Training Report
    report_path = output_dir / "training_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("TRAINING RUN COMPLETE")
    print(f"  - Global Steps: {summary['global_steps']}")
    print(f"  - Final Train Loss: {summary['final_train_loss']:.4f}")
    print(f"  - Final Val Loss: {summary['final_val_loss']:.4f}")
    print(f"  - Final Val Perplexity: {summary['final_val_perplexity']:.2f}")
    print(f"  - Duration: {summary['elapsed_seconds']:.2f}s")
    print(f"  - Summary saved: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
