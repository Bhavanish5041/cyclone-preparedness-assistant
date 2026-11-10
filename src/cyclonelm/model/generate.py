"""Autoregressive text generation routines for CycloneLM."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.cyclonelm.model.transformer import CycloneLM


@torch.no_grad()
def generate(
    model: CycloneLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    stop_token_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from a prompt tensor.

    Args:
        model: Trained CycloneLM model in eval mode.
        prompt_ids: LongTensor of shape (batch_size, prompt_len) or (prompt_len,).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (>0.0 for stochastic, 0.0 for greedy).
        top_k: Optional top-k filtering threshold.
        repetition_penalty: Penalty factor (>1.0 reduces token repetition).
        eos_token_id: Optional token ID indicating end-of-sequence.
        stop_token_ids: Optional sequence of token IDs that immediately halt generation.

    Returns:
        LongTensor containing the prompt concatenated with newly generated tokens.
    """
    model.eval()

    if prompt_ids.dim() == 1:
        tokens = prompt_ids.unsqueeze(0)
    else:
        tokens = prompt_ids.clone()

    stop_set = set(stop_token_ids or [])
    if eos_token_id is not None:
        stop_set.add(eos_token_id)

    max_ctx = model.config.max_seq_len

    for _ in range(max_new_tokens):
        # Truncate to context window if needed
        idx_cond = tokens if tokens.size(1) <= max_ctx else tokens[:, -max_ctx:]

        outputs = model(idx_cond)
        # Take logits at the final time step: (B, vocab_size)
        logits = outputs.logits[:, -1, :].clone()

        if repetition_penalty != 1.0:
            for b in range(tokens.size(0)):
                for token_id in set(tokens[b].tolist()):
                    val = logits[b, token_id].item()
                    if val > 0:
                        logits[b, token_id] = val / repetition_penalty
                    else:
                        logits[b, token_id] = val * repetition_penalty

        if temperature <= 0.0:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled_logits = logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat((tokens, next_token), dim=1)

        # Check stopping criteria (single batch check)
        if stop_set and next_token.item() in stop_set:
            break

    return tokens
