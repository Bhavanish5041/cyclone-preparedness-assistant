"""CycloneLM Inference Engine.

Provides an isolated, clean abstraction for loading model checkpoints,
executing autoregressive token generation, checking model health, and preparing
for future FastAPI and orchestrator integration.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.cyclonelm.config.model_config import CycloneLMConfig
from src.cyclonelm.model.generate import generate as model_generate
from src.cyclonelm.model.transformer import CycloneLM
from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer


class CycloneLMEngine:
    """Isolated inference service engine for CycloneLM."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        tokenizer_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None

        self.model: Optional[CycloneLM] = None
        self.config: Optional[CycloneLMConfig] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.checkpoint_metadata: Dict[str, Any] = {}

        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load_model(self.checkpoint_path, tokenizer_path=self.tokenizer_path)

    def load_model(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: Optional[str | Path] = None,
        config: Optional[CycloneLMConfig] = None,
    ) -> None:
        """Load checkpoint weights, configuration, and tokenizer."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        # Load checkpoint dict
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Restore or create config
        if config is not None:
            self.config = config
        elif "config" in checkpoint:
            self.config = CycloneLMConfig.from_dict(checkpoint["config"])
        else:
            self.config = CycloneLMConfig()

        # Instantiate model
        self.model = CycloneLM(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_metadata = {
            "checkpoint_path": str(ckpt_path),
            "global_step": checkpoint.get("global_step"),
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "train_loss": checkpoint.get("train_loss"),
        }

        # Restore tokenizer
        self.tokenizer = None
        tok_path = Path(tokenizer_path) if tokenizer_path else None
        if tok_path and tok_path.exists():
            self.tokenizer = BPETokenizer.load(tok_path)
        else:
            # Auto-detect versioned tokenizer matching checkpoint path
            if "v0.2" in str(ckpt_path) or "experiments" in str(ckpt_path):
                v02_tok = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
                if v02_tok.exists():
                    self.tokenizer = BPETokenizer.load(v02_tok)
            elif "v0.1" in str(ckpt_path):
                v01_tok = Path("models/cyclonelm/v0.1/tokenizer/vocab.json")
                if v01_tok.exists():
                    self.tokenizer = BPETokenizer.load(v01_tok)

            if self.tokenizer is None:
                default_tok_path = Path("models/cyclonelm/tokenizer/vocab.json")
                if default_tok_path.exists():
                    self.tokenizer = BPETokenizer.load(default_tok_path)
                else:
                    self.tokenizer = BPETokenizer()

        # Check vocabulary and special-token compatibility
        if self.tokenizer is not None and self.config is not None:
            if self.tokenizer.vocab_size != self.config.vocab_size:
                raise ValueError(
                    f"Vocabulary size mismatch: tokenizer has {self.tokenizer.vocab_size} tokens, "
                    f"but model configuration expects {self.config.vocab_size}."
                )

            is_v01_checkpoint = (
                "v0.1" in str(ckpt_path)
                or checkpoint.get("version") == "0.1"
                or (
                    "train_config" in checkpoint
                    and "models/cyclonelm/checkpoints" in str(checkpoint["train_config"].get("checkpoint_dir", ""))
                    and not checkpoint.get("has_closing_special_tokens", False)
                )
            )
            tokenizer_has_closing_tags = "</tool_call>" in self.tokenizer.special_tokens

            if is_v01_checkpoint and tokenizer_has_closing_tags:
                raise ValueError(
                    "Incompatible vocabulary: Attempted to load a CycloneLM v0.1 checkpoint into an incompatible "
                    "v0.2 vocabulary that includes closing special tokens (e.g. </tool_call>). Retrain the model on the "
                    "realigned v0.2 vocabulary or use the matching v0.1 tokenizer."
                )

        self.checkpoint_path = ckpt_path

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate text from a given prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CycloneLMEngine is not initialized. Call load_model() first.")

        start_time = time.time()

        # Tokenize prompt
        encoded = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device)
        prompt_len = input_ids.size(1)

        # Map stop tokens to IDs
        stop_ids = []
        if stop_tokens:
            for st in stop_tokens:
                if st in self.tokenizer.token_to_id:
                    stop_ids.append(self.tokenizer.token_to_id[st])

        # Autoregressive generation
        out_tokens = model_generate(
            model=self.model,
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_id,
            stop_token_ids=stop_ids,
        )

        all_ids = out_tokens[0].tolist()
        new_ids = all_ids[prompt_len:]

        full_text = self.tokenizer.decode(all_ids, skip_special_tokens=False)
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        clean_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "clean_text": clean_text,
            "full_text": full_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_ids),
            "total_tokens": len(all_ids),
            "latency_ms": latency_ms,
            "tokens_per_sec": (len(new_ids) / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0,
        }

    def health(self) -> Dict[str, Any]:
        """Return operational health status, resource utilization, and model metadata."""
        if self.model is None:
            return {
                "status": "uninitialized",
                "model_name": "CycloneLM v0.1",
                "ready": False,
                "device": str(self.device),
            }

        param_count = self.model.count_parameters()
        return {
            "status": "ready",
            "model_name": "CycloneLM v0.1",
            "ready": True,
            "parameters": param_count,
            "device": str(self.device),
            "vocab_size": self.config.vocab_size if self.config else None,
            "max_seq_len": self.config.max_seq_len if self.config else None,
            "checkpoint": self.checkpoint_metadata,
        }
