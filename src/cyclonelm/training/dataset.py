"""Dataset and DataLoader utilities for CycloneLM training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer


class CycloneDataset(Dataset):
    """PyTorch Dataset for causal language model next-token prediction.

    Segments token sequences into windows of length max_seq_len with shifted targets.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        max_seq_len: int = 128,
        stride: Optional[int] = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        total_tokens = len(token_ids)
        if total_tokens <= max_seq_len:
            # Pad short sequences
            pad_len = (max_seq_len + 1) - total_tokens
            padded = list(token_ids) + [0] * pad_len
            x = torch.tensor(padded[:max_seq_len], dtype=torch.long)
            y = torch.tensor(padded[1 : max_seq_len + 1], dtype=torch.long)
            mask = torch.tensor([1] * (total_tokens - 1) + [0] * (pad_len + 1), dtype=torch.long)[:max_seq_len]
            self.samples.append((x, y, mask))
        else:
            for i in range(0, total_tokens - max_seq_len, self.stride):
                chunk = token_ids[i : i + max_seq_len + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                mask = torch.ones(max_seq_len, dtype=torch.long)
                self.samples.append((x, y, mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, y, mask = self.samples[idx]
        return {
            "input_ids": x,
            "targets": y,
            "attention_mask": mask,
        }


def load_corpus_tokens(
    filepath: str | Path,
    tokenizer: BPETokenizer,
    add_eos_per_line: bool = True,
) -> List[int]:
    """Read a text file and encode into a flattened token ID stream."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    token_ids: List[int] = []
    for line in lines:
        enc = tokenizer.encode(line, add_bos=False, add_eos=add_eos_per_line)
        token_ids.extend(enc["input_ids"])

    return token_ids


def create_dataloaders(
    train_tokens: Sequence[int],
    val_tokens: Sequence[int],
    max_seq_len: int = 128,
    batch_size: int = 8,
    stride: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders."""
    train_dataset = CycloneDataset(train_tokens, max_seq_len=max_seq_len, stride=stride)
    val_dataset = CycloneDataset(val_tokens, max_seq_len=max_seq_len, stride=stride)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


class InstructionSFTDataset(Dataset):
    """Supervised Fine-Tuning Dataset with prompt loss masking.

    Each example is formatted as:
      <user>{instruction}</user><assistant>{response}</assistant><eos>
    Prompt tokens (<user>...<assistant>) have their target set to ignore_index (-100),
    so loss is strictly computed on predicting the assistant's response.
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: BPETokenizer,
        max_seq_len: int = 128,
        ignore_index: int = -100,
    ) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for ex in examples:
            inst = ex.get("instruction", "").strip()
            resp = ex.get("response", "").strip()
            if not inst or not resp:
                continue

            prompt_str = f"<user>{inst}</user><assistant>"
            resp_str = f"{resp}</assistant>"

            p_enc = tokenizer.encode(prompt_str, add_bos=False, add_eos=False)["input_ids"]
            r_enc = tokenizer.encode(resp_str, add_bos=False, add_eos=True)["input_ids"]

            seq_ids = p_enc + r_enc
            if len(seq_ids) > max_seq_len + 1:
                seq_ids = seq_ids[: max_seq_len + 1]

            x_ids = seq_ids[:-1]
            y_ids = seq_ids[1:]

            # Mask out prompt positions in targets
            prompt_len = len(p_enc)
            mask_boundary = min(len(y_ids), max(0, prompt_len - 1))
            y_masked = [ignore_index] * mask_boundary + y_ids[mask_boundary:]

            # Pad
            curr_len = len(x_ids)
            pad_len = max(0, max_seq_len - curr_len)
            x_padded = x_ids[:max_seq_len] + [tokenizer.pad_id] * pad_len
            y_padded = y_masked[:max_seq_len] + [ignore_index] * pad_len
            attn_mask = [1] * min(curr_len, max_seq_len) + [0] * pad_len

            self.samples.append((
                torch.tensor(x_padded, dtype=torch.long),
                torch.tensor(y_padded, dtype=torch.long),
                torch.tensor(attn_mask, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, y, mask = self.samples[idx]
        return {
            "input_ids": x,
            "targets": y,
            "attention_mask": mask,
        }


def create_sft_dataloaders(
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    tokenizer: BPETokenizer,
    max_seq_len: int = 128,
    batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders for SFT."""
    train_dataset = InstructionSFTDataset(train_examples, tokenizer, max_seq_len=max_seq_len)
    val_dataset = InstructionSFTDataset(val_examples, tokenizer, max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader
