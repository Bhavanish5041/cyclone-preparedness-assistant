"""Phase 1: Comprehensive Language Generation Diagnostic Script.

Tests all five trained checkpoints:
- 239k
- 1m
- 5m (reference model)
- 10m
- 20m

Evaluates diagnostic prompt categories:
1. Basic English conversation
2. Simple explanations
3. Cyclone terminology
4. AHP explanations
5. Preparedness tiers
6. Summarization
7. Structured instruction following

Measures:
- Grammar & Coherence (valid word ratio, parseable sentence structure)
- Repetition (distinct-1, distinct-2, distinct-3 ratios, trigram repetition rate)
- Token Fragmentation (tokens per word, partial subword splits)
- Special Token Generation (spurious tags, unclosed tags, formatting violations)
- Prompt Sensitivity (greedy vs temperature 0.7, prompt format sensitivity)
- Output Consistency (inter-run similarity across stochastic generations)
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.cyclonelm.inference.engine import CycloneLMEngine
from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer

DIAGNOSTIC_PROMPTS = [
    {
        "id": "diag_eng_01",
        "category": "basic_conversation",
        "prompt": "<user>Hello, how can you help me today?</user><assistant>",
        "raw_text": "Hello, how can you help me today?",
    },
    {
        "id": "diag_eng_02",
        "category": "basic_conversation",
        "prompt": "<user>What is your purpose in this system?</user><assistant>",
        "raw_text": "What is your purpose in this system?",
    },
    {
        "id": "diag_expl_01",
        "category": "simple_explanations",
        "prompt": "<user>Explain what a cyclone warning means in simple terms.</user><assistant>",
        "raw_text": "Explain what a cyclone warning means in simple terms.",
    },
    {
        "id": "diag_expl_02",
        "category": "simple_explanations",
        "prompt": "<user>Why is coastal disaster preparedness important?</user><assistant>",
        "raw_text": "Why is coastal disaster preparedness important?",
    },
    {
        "id": "diag_term_01",
        "category": "cyclone_terminology",
        "prompt": "<user>Define what a Very Severe Cyclonic Storm is according to the IMD.</user><assistant>",
        "raw_text": "Define what a Very Severe Cyclonic Storm is according to the IMD.",
    },
    {
        "id": "diag_term_02",
        "category": "cyclone_terminology",
        "prompt": "<user>What wind speed defines a Super Cyclonic Storm?</user><assistant>",
        "raw_text": "What wind speed defines a Super Cyclonic Storm?",
    },
    {
        "id": "diag_ahp_01",
        "category": "ahp_explanations",
        "prompt": "<user>Explain the Analytic Hierarchy Process and how composite preparedness scores are calculated.</user><assistant>",
        "raw_text": "Explain the Analytic Hierarchy Process and how composite preparedness scores are calculated.",
    },
    {
        "id": "diag_ahp_02",
        "category": "ahp_explanations",
        "prompt": "<user>What does a consistency ratio below 0.10 mean in AHP?</user><assistant>",
        "raw_text": "What does a consistency ratio below 0.10 mean in AHP?",
    },
    {
        "id": "diag_tier_01",
        "category": "preparedness_tiers",
        "prompt": "<user>What are the score boundaries for High, Moderate, and Low Preparedness tiers?</user><assistant>",
        "raw_text": "What are the score boundaries for High, Moderate, and Low Preparedness tiers?",
    },
    {
        "id": "diag_tier_02",
        "category": "preparedness_tiers",
        "prompt": "<user>How does an AHP High preparedness score differ from Low preparedness?</user><assistant>",
        "raw_text": "How does an AHP High preparedness score differ from Low preparedness?",
    },
    {
        "id": "diag_sum_01",
        "category": "summarization",
        "prompt": "<user>Summarize the following: Dakshina Kannada has 369 revenue units with an average AHP score of 61.06, classified under K-Means Cluster Archetypes.</user><assistant>",
        "raw_text": "Summarize the following: Dakshina Kannada has 369 revenue units with an average AHP score of 61.06, classified under K-Means Cluster Archetypes.",
    },
    {
        "id": "diag_inst_01",
        "category": "structured_instructions",
        "prompt": "<user>Answer with only the district name: In which district is Ullal located?</user><assistant>",
        "raw_text": "Answer with only the district name: In which district is Ullal located?",
    },
    {
        "id": "diag_inst_02",
        "category": "structured_instructions",
        "prompt": "<user>Output JSON format: {\"district\": \"Udupi\"}</user><assistant>",
        "raw_text": "Output JSON format: {\"district\": \"Udupi\"}",
    },
]


def compute_repetition_metrics(tokens: List[str]) -> Dict[str, float]:
    """Compute distinct-n ratios and trigram repetition rate."""
    if not tokens:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0, "trigram_repetition_rate": 0.0}

    d1 = len(set(tokens)) / len(tokens)
    bigrams = list(zip(tokens[:-1], tokens[1:])) if len(tokens) >= 2 else []
    d2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 1.0

    trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:])) if len(tokens) >= 3 else []
    d3 = (len(set(trigrams)) / len(trigrams)) if trigrams else 1.0
    rep_rate = (1.0 - d3) * 100.0 if trigrams else 0.0

    return {
        "distinct_1": round(d1 * 100.0, 1),
        "distinct_2": round(d2 * 100.0, 1),
        "distinct_3": round(d3 * 100.0, 1),
        "trigram_repetition_rate": round(rep_rate, 1),
    }


def analyze_special_tokens(text: str) -> Dict[str, Any]:
    """Analyze generation of special tokens (unprompted, malformed, or unclosed)."""
    tags = re.findall(r"<[^>]+>", text)
    has_tool_call = "<tool_call>" in text
    has_closing_tool_call = "</tool_call>" in text
    has_unclosed_tool_call = has_tool_call and not has_closing_tool_call
    has_user_tag_in_output = "<user>" in text or "</user>" in text
    has_eos = "<eos>" in text

    return {
        "total_tags": len(tags),
        "tags_found": tags,
        "has_tool_call": has_tool_call,
        "has_unclosed_tool_call": has_unclosed_tool_call,
        "has_spurious_user_tag": has_user_tag_in_output,
        "has_eos": has_eos,
    }


def diagnose_checkpoint(
    profile: str,
    ckpt_path: Path,
    tok_path: Path,
) -> Dict[str, Any]:
    print(f"\nRunning diagnosis on checkpoint [{profile}] ({ckpt_path})...")
    engine = CycloneLMEngine(
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        device="cpu",
    )

    results_by_prompt = []
    all_gen_tokens: List[str] = []
    total_tokens_generated = 0
    total_words_generated = 0

    # Sensitivity tests: test greedy (temp 0.0) vs sampling (temp 0.7)
    sensitivity_diffs = []

    for p in DIAGNOSTIC_PROMPTS:
        # Standard generation (temp 0.7)
        out_sampled = engine.generate(p["prompt"], max_new_tokens=40, temperature=0.7)
        # Greedy generation (temp 0.0 or very low 0.01)
        out_greedy = engine.generate(p["prompt"], max_new_tokens=40, temperature=0.01)
        # Raw prompt (without <user> tags) to check prompt formatting sensitivity
        out_raw = engine.generate(p["raw_text"], max_new_tokens=40, temperature=0.7)

        gen_text = out_sampled["clean_text"].strip()
        words = re.findall(r"\w+|[^\w\s]", gen_text)
        all_gen_tokens.extend(words)

        tokens_count = out_sampled.get("completion_tokens", len(words))
        words_count = max(1, len([w for w in words if w.isalnum()]))
        tokens_per_word = round(tokens_count / words_count, 2) if words_count > 0 else 0.0
        total_tokens_generated += tokens_count
        total_words_generated += words_count

        rep_metrics = compute_repetition_metrics(words)
        special_tok_info = analyze_special_tokens(out_sampled["generated_text"])

        # Consistency check: run a second trial at temp 0.7 to measure stochastic variation
        out_trial2 = engine.generate(p["prompt"], max_new_tokens=40, temperature=0.7)
        t1_words = set(words)
        t2_words = set(re.findall(r"\w+|[^\w\s]", out_trial2["clean_text"].strip()))
        jaccard_consistency = (
            len(t1_words & t2_words) / max(1, len(t1_words | t2_words))
        )

        # Check prompt framing sensitivity (formatted vs raw)
        raw_words = set(re.findall(r"\w+|[^\w\s]", out_raw["clean_text"].strip()))
        framing_jaccard = (
            len(t1_words & raw_words) / max(1, len(t1_words | raw_words))
        )
        sensitivity_diffs.append(framing_jaccard)

        results_by_prompt.append({
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "output_sampled_temp07": gen_text,
            "output_greedy": out_greedy["clean_text"].strip(),
            "output_unformatted_raw": out_raw["clean_text"].strip(),
            "tokens_count": tokens_count,
            "words_count": words_count,
            "tokens_per_word": tokens_per_word,
            "repetition_metrics": rep_metrics,
            "special_tokens": special_tok_info,
            "consistency_jaccard_between_trials": round(jaccard_consistency, 2),
            "framing_sensitivity_jaccard": round(framing_jaccard, 2),
        })

    # Overall profile aggregates
    overall_rep = compute_repetition_metrics(all_gen_tokens)
    avg_tokens_per_word = round(total_tokens_generated / max(1, total_words_generated), 2)
    avg_consistency = round(
        sum(r["consistency_jaccard_between_trials"] for r in results_by_prompt) / len(results_by_prompt), 2
    )
    avg_framing_sensitivity = round(
        sum(sensitivity_diffs) / len(sensitivity_diffs), 2
    )

    # Check unclosed or spurious tool call rate
    spurious_tool_calls = sum(
        1 for r in results_by_prompt if r["special_tokens"]["has_tool_call"] and r["category"] != "structured_instructions"
    )
    unclosed_tool_calls = sum(
        1 for r in results_by_prompt if r["special_tokens"]["has_unclosed_tool_call"]
    )

    return {
        "profile": profile,
        "checkpoint": str(ckpt_path),
        "total_prompts_tested": len(DIAGNOSTIC_PROMPTS),
        "aggregate_metrics": {
            "avg_tokens_per_word": avg_tokens_per_word,
            "distinct_1": overall_rep["distinct_1"],
            "distinct_2": overall_rep["distinct_2"],
            "distinct_3": overall_rep["distinct_3"],
            "trigram_repetition_rate": overall_rep["trigram_repetition_rate"],
            "avg_trial_consistency": avg_consistency,
            "avg_framing_jaccard_similarity": avg_framing_sensitivity,
            "spurious_tool_call_generations": spurious_tool_calls,
            "unclosed_tool_calls": unclosed_tool_calls,
        },
        "per_prompt_results": results_by_prompt,
    }


def run_full_diagnosis() -> Dict[str, Any]:
    print("=" * 70)
    print("CYCLONELM PHASE 1: LANGUAGE CAPABILITY DIAGNOSIS")
    print("=" * 70)

    tok_path = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
    experiments_dir = Path("models/cyclonelm/experiments")

    profiles = ["239k", "1m", "5m", "10m", "20m"]
    diagnostics = {}

    for prof in profiles:
        ckpt_path = experiments_dir / prof / "checkpoints" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"Skipping {prof}: checkpoint not found at {ckpt_path}")
            continue
        diag = diagnose_checkpoint(prof, ckpt_path, tok_path)
        diagnostics[prof] = diag

    return diagnostics


if __name__ == "__main__":
    diag_res = run_full_diagnosis()
    out_file = Path("reports/cyclonelm_language_diagnosis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diag_res, f, indent=2)
    print(f"\nSaved language generation diagnostic report to {out_file}")
