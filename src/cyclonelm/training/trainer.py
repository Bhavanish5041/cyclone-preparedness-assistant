"""CycloneLM Trainer for pretraining and fine-tuning."""

from __future__ import annotations

import math
import os
import random
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.cyclonelm.config.model_config import CycloneLMConfig, TrainingConfig
from src.cyclonelm.model.transformer import CycloneLM


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CycloneTrainer:
    """Trainer orchestrating optimization, evaluation, checkpointing, and metrics."""

    def __init__(
        self,
        model: CycloneLM,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device assignment
        self.device = torch.device(
            train_config.device
            if train_config.device == "cpu" or not torch.cuda.is_available()
            else "cuda"
        )
        self.model.to(self.device)

        # Reproducibility
        set_seed(train_config.seed)

        # Optimizer: apply weight decay only to 2D parameters (weights), not biases/norms
        decay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        nodecay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]
        optim_groups = [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history: List[Dict[str, Any]] = []

        # Checkpoints directory
        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_lr(self, step: int) -> float:
        """Compute learning rate with linear warmup and cosine decay."""
        cfg = self.train_config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
        if step > cfg.max_steps:
            return cfg.min_learning_rate
        # Cosine decay
        decay_ratio = (step - cfg.warmup_steps) / max(1, (cfg.max_steps - cfg.warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)

    def evaluate(self) -> Tuple[float, float]:
        """Compute average validation loss and perplexity."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["input_ids"].to(self.device)
                y = batch["targets"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)

                output = self.model(x, targets=y, attention_mask=attn_mask)
                if output.loss is not None:
                    total_loss += output.loss.item()
                    total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20.0))  # guard against overflow
        self.model.train()
        return avg_loss, perplexity

    def save_checkpoint(
        self,
        filepath: str | Path,
        val_loss: Optional[float] = None,
        train_loss: Optional[float] = None,
    ) -> None:
        """Save full model and training state checkpoint."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "history": self.history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Load model state and training state from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("val_loss", float("inf")) or float("inf")
        self.history = checkpoint.get("history", [])

    def train(self, resume_checkpoint: Optional[str | Path] = None) -> Dict[str, Any]:
        """Execute full training loop across epochs and steps with gradient accumulation, early stopping, and telemetry."""
        if resume_checkpoint is not None:
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            self.load_checkpoint(resume_checkpoint)

        self.model.train()
        start_time = time.time()
        start_process_time = time.process_time()
        print(
            f"Starting CycloneLM training: {self.model.count_parameters():,} params | "
            f"device: {self.device} | start_step: {self.global_step} | max_steps: {self.train_config.max_steps} | "
            f"epochs: {self.train_config.epochs} | grad_accum: {self.train_config.grad_accum_steps}"
        )

        accum_steps = max(1, self.train_config.grad_accum_steps)
        patience = self.train_config.early_stopping_patience
        patience_counter = 0
        early_stopped = False
        self.optimizer.zero_grad(set_to_none=True)

        start_epoch = self.current_epoch + 1
        for epoch in range(start_epoch, self.train_config.epochs + 1):
            if early_stopped:
                break
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                self.global_step += 1

                # Update learning rate schedule
                lr = self._get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                x = batch["input_ids"].to(self.device)
                y = batch["targets"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                output = self.model(x, targets=y, attention_mask=attn_mask)
                loss = output.loss

                if loss is None:
                    continue

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_batches += 1

                # Backward pass with gradient accumulation
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.train_config.grad_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Step-level logging
                if self.global_step % self.train_config.log_interval == 0:
                    print(
                        f"Epoch {epoch:02d} | Step {self.global_step:04d} | "
                        f"Train Loss: {loss_val:.4f} | LR: {lr:.6f}"
                    )

                # Periodic evaluation
                if self.global_step % self.train_config.eval_interval == 0:
                    val_loss, val_ppl = self.evaluate()
                    print(
                        f"  -> EVAL @ Step {self.global_step:04d}: Val Loss: {val_loss:.4f} | "
                        f"Val PPL: {val_ppl:.2f}"
                    )
                    log_entry = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "train_loss": loss_val,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                        "learning_rate": lr,
                    }
                    self.history.append(log_entry)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        patience_counter = 0
                        best_path = self.checkpoint_dir / "best_model.pt"
                        self.save_checkpoint(best_path, val_loss=val_loss, train_loss=loss_val)
                    else:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            print(f"Early stopping triggered after {patience} evals without improvement.")
                            early_stopped = True
                            break

                # Periodic checkpointing
                if self.global_step % self.train_config.save_interval == 0:
                    step_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step:04d}.pt"
                    self.save_checkpoint(step_path, train_loss=loss_val)

                if self.global_step >= self.train_config.max_steps:
                    break

            if self.global_step >= self.train_config.max_steps:
                print(f"Reached max_steps ({self.train_config.max_steps}). Finishing training.")
                break

        # Final evaluation
        final_val_loss, final_val_ppl = self.evaluate()
        final_train_loss = epoch_loss / max(1, epoch_batches)
        elapsed_sec = time.time() - start_time
        process_cpu_time = time.process_time() - start_process_time
        cpu_utilization_pct = round((process_cpu_time / max(1e-6, elapsed_sec)) * 100.0, 1)

        # Save latest model checkpoint
        latest_path = self.checkpoint_dir / "latest_model.pt"
        self.save_checkpoint(latest_path, val_loss=final_val_loss, train_loss=final_train_loss)

        total_tokens = self.global_step * self.train_config.batch_size * self.model.config.max_seq_len
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        checkpoint_size_kb = latest_path.stat().st_size / 1024.0 if latest_path.exists() else 0.0

        summary = {
            "total_parameters": self.model.count_parameters(),
            "non_embedding_parameters": self.model.count_parameters(non_embedding_only=True),
            "global_steps": self.global_step,
            "total_epochs": self.current_epoch,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_val_perplexity": final_val_ppl,
            "best_val_loss": self.best_val_loss,
            "elapsed_seconds": elapsed_sec,
            "cpu_utilization_pct": cpu_utilization_pct,
            "peak_rss_mb": peak_rss_mb,
            "checkpoint_size_kb": checkpoint_size_kb,
            "training_tokens_per_second": total_tokens / max(0.001, elapsed_sec),
            "device": str(self.device),
            "early_stopped": early_stopped,
        }
        return summary
