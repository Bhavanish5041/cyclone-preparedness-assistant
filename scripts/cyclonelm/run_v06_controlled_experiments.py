"""CycloneLM v0.6: Controlled Language Capability Improvement Experiments.

Orchestrates controlled experiments on the 5.30M architecture:
1. Baseline Reference: 239K (from Phase 5)
2. Baseline Reference: 5.30M (from Phase 5)
3. Controlled Exp 1: 5M_SFT (Instruction-focused training with prompt loss masking)
4. Controlled Exp 2: 5M_Tuned_LR (Calibrated learning rate 5e-4 and extended training steps)
5. Decoding Experiments: Greedy vs Sampling vs Repetition Penalty (1.15) on 5.30M

Measures and compares:
- Test Perplexity & Loss
- Exact Match Accuracy
- Semantic Correctness Accuracy (factually verified, zero hallucination)
- Tool Selection Accuracy (exact JSON tool call vs semantic tool intent)
- Argument Validity (Pydantic schema validation)
- Evidence Grounding
- Repetition Rate & Trigram Diversity
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.cyclonelm.config.model_config import CycloneLMConfig, TrainingConfig
from src.cyclonelm.config.scaling_profiles import get_profile_config
from src.cyclonelm.evaluation.scaling_evaluator import ScalingEvaluator
from src.cyclonelm.evaluation.semantic_benchmark import SemanticBenchmarkRunner
from src.cyclonelm.inference.engine import CycloneLMEngine
from src.cyclonelm.model.transformer import CycloneLM
from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer
from src.cyclonelm.training.dataset import create_sft_dataloaders
from src.cyclonelm.training.trainer import CycloneTrainer, set_seed


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line.strip()))
    return rows


def load_sft_examples() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load instruction and tool-use examples for SFT with loss masking."""
    v02_dir = Path("data/cyclonelm/v0.2")
    train_inst = read_jsonl(v02_dir / "instructions" / "train.jsonl")
    val_inst = read_jsonl(v02_dir / "instructions" / "val.jsonl")
    train_tool = read_jsonl(v02_dir / "tool_use" / "train.jsonl")
    val_tool = read_jsonl(v02_dir / "tool_use" / "val.jsonl")

    train_all = train_inst + train_tool
    val_all = val_inst + val_tool
    return train_all, val_all


def train_sft_experiment(
    experiment_name: str,
    output_dir: Path,
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-4,
    max_steps: int = 250,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 70)
    print(f"TRAINING CONTROLLED SFT EXPERIMENT: [{experiment_name}]")
    print(f"  - Model: 5.30M (d=256, L=6, H=8, d_ff=1024)")
    print(f"  - Learning Rate: {learning_rate} (min: {min_learning_rate})")
    print(f"  - Max Steps: {max_steps} | Seed: {seed}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tok_path = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
    tokenizer = BPETokenizer.load(tok_path)

    cfg = get_profile_config("5m")
    cfg.vocab_size = tokenizer.vocab_size

    train_exs, val_exs = load_sft_examples()
    print(f"Loaded SFT dataset: {len(train_exs)} train pairs, {len(val_exs)} val pairs")

    train_loader, val_loader = create_sft_dataloaders(
        train_examples=train_exs,
        val_examples=val_exs,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        batch_size=8,
    )

    train_cfg = TrainingConfig(
        batch_size=8,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        weight_decay=0.01,
        warmup_steps=30,
        max_steps=max_steps,
        epochs=30,
        grad_clip=1.0,
        grad_accum_steps=1,
        device="cpu",
        seed=seed,
        log_interval=25,
        eval_interval=25,
        save_interval=50,
        checkpoint_dir=str(ckpt_dir),
    )

    model = CycloneLM(cfg)
    trainer = CycloneTrainer(
        model=model,
        train_config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    summary = trainer.train()

    # Save summary and config
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    cfg.to_yaml(output_dir / "config.yaml")

    return summary


def evaluate_checkpoint_all_tiers(
    ckpt_path: Path,
    tok_path: Path,
    name: str,
    training_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print(f"\nEvaluating [{name}] on Held-Out Benchmark across All Tiers...")
    engine = CycloneLMEngine(checkpoint_path=ckpt_path, tokenizer_path=tok_path, device="cpu")

    # 1. Standard Scaling Evaluator
    scaling_eval = ScalingEvaluator()
    std_results = scaling_eval.evaluate_model(engine, training_summary=training_summary)

    # 2. Refined Semantic Benchmark (with calibrated decoding: temp=0.3, rep_penalty=1.15)
    semantic_runner = SemanticBenchmarkRunner()
    semantic_results = semantic_runner.evaluate_engine(
        engine,
        temperature=0.3,
        repetition_penalty=1.15,
        max_new_tokens=40,
    )

    # 3. Decoding comparison on 3 sample prompts
    decoding_samples = {}
    test_prompts = [
        ("what_is_ahp", "<user>What is the primary role of the Analytic Hierarchy Process in CYCLONE INTEL?</user><assistant>"),
        ("lookup_tallur", "<user>Search for village Tallur in the preparedness database.</user><assistant>"),
        ("live_refusal", "<user>What is the current live wind speed right now in Mangalore?</user><assistant>"),
    ]
    for pid, ptext in test_prompts:
        decoding_samples[pid] = {
            "greedy": engine.generate(ptext, max_new_tokens=35, temperature=0.0, repetition_penalty=1.0)["clean_text"],
            "raw_sampling": engine.generate(ptext, max_new_tokens=35, temperature=0.7, repetition_penalty=1.0)["clean_text"],
            "calibrated": engine.generate(ptext, max_new_tokens=35, temperature=0.3, repetition_penalty=1.15)["clean_text"],
        }

    return {
        "name": name,
        "checkpoint": str(ckpt_path),
        "standard_metrics": {
            "test_loss": std_results["language_quality"]["test_loss"],
            "test_perplexity": std_results["language_quality"]["test_perplexity"],
            "val_loss": std_results["language_quality"]["validation_loss"],
            "val_perplexity": std_results["language_quality"]["validation_perplexity"],
            "domain_accuracy": std_results["domain_quality"]["overall_domain_accuracy"],
            "tool_selection_accuracy": std_results["tool_use_quality"]["tool_selection_accuracy"],
            "argument_validity": std_results["tool_use_quality"]["argument_validity"],
            "grounding_accuracy": std_results["tool_use_quality"]["grounding_accuracy"],
            "ms_per_token": std_results["inference_performance"]["ms_per_token"],
        },
        "semantic_metrics": semantic_results["evaluation_summary"],
        "decoding_comparison": decoding_samples,
    }


def run_all_v06_experiments() -> Dict[str, Any]:
    print("=" * 70)
    print("CYCLONELM v0.6: CONTROLLED IMPROVEMENT EXPERIMENT PIPELINE")
    print("=" * 70)

    tok_path = Path("models/cyclonelm/v0.2/tokenizer/vocab.json")
    exp_base = Path("models/cyclonelm/experiments")

    # 1. Evaluate Reference Baselines
    print("\n--- Step 1: Evaluating Reference Baselines (239K and 5M) ---")
    results = {}

    ckpt_239k = exp_base / "239k" / "checkpoints" / "best_model.pt"
    if ckpt_239k.exists():
        with open(exp_base / "239k" / "training_summary.json") as f:
            sum_239k = json.load(f)
        results["239k_baseline"] = evaluate_checkpoint_all_tiers(
            ckpt_239k, tok_path, "239k_baseline", sum_239k
        )

    ckpt_5m = exp_base / "5m" / "checkpoints" / "best_model.pt"
    if ckpt_5m.exists():
        with open(exp_base / "5m" / "training_summary.json") as f:
            sum_5m = json.load(f)
        results["5m_baseline"] = evaluate_checkpoint_all_tiers(
            ckpt_5m, tok_path, "5m_baseline", sum_5m
        )

    # 2. Controlled Experiment 1: 5M with Instruction Loss Masking (SFT)
    print("\n--- Step 2: Running Controlled Exp 1 (5M_SFT with Loss Masking) ---")
    dir_sft = exp_base / "5m_sft"
    best_sft = dir_sft / "checkpoints" / "best_model.pt"
    if not best_sft.exists():
        sum_sft = train_sft_experiment(
            experiment_name="5m_sft",
            output_dir=dir_sft,
            learning_rate=1e-3,
            max_steps=250,
            seed=42,
        )
    else:
        with open(dir_sft / "training_summary.json") as f:
            sum_sft = json.load(f)

    results["5m_sft"] = evaluate_checkpoint_all_tiers(
        best_sft, tok_path, "5m_sft", sum_sft
    )

    # 3. Controlled Experiment 2: 5M with Calibrated Learning Rate & Extended Duration
    print("\n--- Step 3: Running Controlled Exp 2 (5M_Tuned_LR) ---")
    dir_tuned = exp_base / "5m_tuned_lr"
    best_tuned = dir_tuned / "checkpoints" / "best_model.pt"
    if not best_tuned.exists():
        sum_tuned = train_sft_experiment(
            experiment_name="5m_tuned_lr",
            output_dir=dir_tuned,
            learning_rate=5e-4,
            min_learning_rate=5e-5,
            max_steps=350,
            seed=42,
        )
    else:
        with open(dir_tuned / "training_summary.json") as f:
            sum_tuned = json.load(f)

    results["5m_tuned_lr"] = evaluate_checkpoint_all_tiers(
        best_tuned, tok_path, "5m_tuned_lr", sum_tuned
    )

    # Save Consolidated v0.6 Results
    out_file = Path("reports/cyclonelm_v06_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved v0.6 controlled results to: {out_file}")

    return results


if __name__ == "__main__":
    run_all_v06_experiments()
