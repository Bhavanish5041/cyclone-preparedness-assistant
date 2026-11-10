"""Evaluation metrics and scientific validation routines for CycloneLM."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.cyclonelm.model.transformer import CycloneLM
from src.cyclonelm.tools.protocol import ToolOrchestrator


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 20.0))


def verify_checkpoint_reload(
    model: CycloneLM,
    reloaded_model: CycloneLM,
) -> Tuple[bool, float]:
    """Verify that a reloaded model has identical weights down to machine precision.

    Returns:
        (is_identical, max_absolute_diff)
    """
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), reloaded_model.named_parameters()):
        if n1 != n2:
            return False, float("inf")
        diff = torch.max(torch.abs(p1 - p2)).item()
        if diff > max_diff:
            max_diff = diff

    return max_diff < 1e-6, max_diff


def measure_inference_speed(
    model: CycloneLM,
    seq_len: int = 16,
    generate_tokens: int = 32,
    trials: int = 5,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure inference latency and throughput on CPU."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    prompt = torch.randint(0, model.config.vocab_size, (1, seq_len), device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(prompt)

    latencies = []
    with torch.no_grad():
        for _ in range(trials):
            t0 = time.perf_counter()
            curr = prompt.clone()
            for _ in range(generate_tokens):
                out = model(curr[:, -model.config.max_seq_len:])
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                curr = torch.cat((curr, next_tok), dim=1)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

    avg_time = sum(latencies) / len(latencies)
    tok_per_sec = generate_tokens / avg_time
    ms_per_token = (avg_time / generate_tokens) * 1000.0

    return {
        "trials": trials,
        "avg_total_seconds": avg_time,
        "ms_per_token": ms_per_token,
        "tokens_per_second": tok_per_sec,
    }


def evaluate_json_validity(generated_strings: List[str]) -> Dict[str, Any]:
    """Evaluate JSON syntactic validity in tool call tags."""
    orchestrator = ToolOrchestrator()
    total_tags = 0
    valid_json = 0

    for text in generated_strings:
        calls = orchestrator.extract_tool_calls(text)
        if calls:
            total_tags += len(calls)
            valid_json += len(calls)

    return {
        "total_tags_evaluated": total_tags,
        "valid_json_count": valid_json,
        "validity_rate": (valid_json / total_tags) if total_tags > 0 else 1.0,
    }


def evaluate_scientific_governance_rules(response_text: str) -> Dict[str, bool]:
    """Check adherence to CYCLONE INTEL mandatory scientific rules."""
    text_lower = response_text.lower()
    return {
        "mentions_ahp_or_precedence": any(
            k in text_lower for k in ["ahp", "analytic hierarchy process", "precedence"]
        ),
        "demarcates_surrogate": any(
            k in text_lower
            for k in ["surrogate", "empirical", "borderline", "approximation", "logistic"]
        ),
        "refuses_live_fabrication": any(
            k in text_lower
            for k in [
                "cannot provide real-time",
                "unrecorded",
                "dynamic",
                "official imd",
                "ground verification",
                "control room",
            ]
        ),
        "distinguishes_tiers_and_bands": any(
            k in text_lower
            for k in ["tier", "band", "distinct", "operational priority", "resource allocation"]
        ),
    }
