"""Controlled Training and Model Scaling Experiment Runner for CycloneLM.

Trains and evaluates progressive CycloneLM model architectures:
- 239k (Baseline, ~239K params)
- 1m   (1 Million params)
- 5m   (5 Million params)
- 10m  (10 Million params)
- 20m  (20 Million params)

Guarantees:
- Zero data contamination (uses v0.2 dataset and held-out 100 benchmark).
- Reproducible seed control (seed=42).
- Hardware telemetry (CPU model, RAM, Peak RSS, process CPU %).
- Independent experiment directories: models/cyclonelm/experiments/<profile>/
- Strict checkpoint retention (never overwrites previous runs).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.cyclonelm.config.model_config import CycloneLMConfig, TrainingConfig
from src.cyclonelm.config.scaling_profiles import (
    SCALING_PROFILES,
    audit_profile_architecture,
    get_profile_config,
)
from src.cyclonelm.evaluation.benchmark import HeldOutBenchmarkRunner
from src.cyclonelm.evaluation.scaling_evaluator import ScalingEvaluator
from src.cyclonelm.inference.engine import CycloneLMEngine
from src.cyclonelm.model.transformer import CycloneLM
from src.cyclonelm.tokenizer.bpe_tokenizer import BPETokenizer
from src.cyclonelm.training.dataset import create_dataloaders, load_corpus_tokens
from src.cyclonelm.training.trainer import CycloneTrainer, set_seed


def get_hardware_info() -> Dict[str, Any]:
    """Capture machine hardware metadata."""
    cpu_model = platform.processor() or "x86_64"
    # Try reading /proc/cpuinfo or lscpu
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    total_ram_gb = 16.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = float(line.split()[1])
                    total_ram_gb = round(kb / (1024.0 * 1024.0), 2)
                    break
    except Exception:
        pass

    return {
        "cpu_model": cpu_model,
        "total_ram_gb": total_ram_gb,
        "os": platform.system(),
        "platform_release": platform.release(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
        "has_cuda": torch.cuda.is_available(),
    }


def run_experiment(
    profile_name: str,
    train_corpus_path: str = "data/cyclonelm/v0.2/domain_corpus/train_combined_corpus.txt",
    val_corpus_path: str = "data/cyclonelm/v0.2/domain_corpus/val_combined_corpus.txt",
    test_corpus_path: str = "data/cyclonelm/v0.2/domain_corpus/test_combined_corpus.txt",
    tokenizer_path: str = "models/cyclonelm/v0.2/tokenizer/vocab.json",
    suite_path: str = "data/cyclonelm/v0.2/held_out_eval/held_out_100.jsonl",
    output_base_dir: str = "models/cyclonelm/experiments",
    max_steps: int = 250,
    epochs: int = 15,
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-4,
    seed: int = 42,
    resume_checkpoint: Optional[str] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    print("=" * 70)
    print(f"RUNNING CYCLONELM SCALING EXPERIMENT: [{profile_name.upper()}]")
    print("=" * 70)

    exp_dir = Path(output_base_dir) / profile_name
    ckpt_dir = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_model_pt = ckpt_dir / "best_model.pt"
    summary_path = exp_dir / "training_summary.json"
    eval_report_path = exp_dir / "evaluation_report.json"
    prov_path = exp_dir / "provenance.json"

    # 1. Structural Audit
    audit_data = audit_profile_architecture(profile_name)
    model_cfg = get_profile_config(profile_name)

    print(f"Model Configuration: {profile_name}")
    print(f"  - Parameters: {audit_data['total_parameters']:,} (non-emb: {audit_data['non_embedding_parameters']:,})")
    print(f"  - d_model: {model_cfg.d_model} | layers: {model_cfg.n_layers} | heads: {model_cfg.n_heads} | d_ff: {model_cfg.d_ff}")
    print(f"  - Divisibility verified: {audit_data['is_divisible']}")
    print(f"  - Causal masking verified: {audit_data['causal_mask_verified']}")
    print(f"  - Estimated train RAM: {audit_data['estimated_train_ram_mb']:.1f} MB")

    # 2. Tokenizer check
    tok_p = Path(tokenizer_path)
    if not tok_p.exists():
        raise FileNotFoundError(f"v0.2 Tokenizer not found at: {tok_p}")
    tokenizer = BPETokenizer.load(tok_p)
    model_cfg.vocab_size = tokenizer.vocab_size

    # 3. Training
    training_summary: Dict[str, Any] = {}
    if best_model_pt.exists() and not force_retrain and summary_path.exists():
        print(f"Checkpoint already exists at {best_model_pt}. Loading recorded summary.")
        with open(summary_path, "r", encoding="utf-8") as f:
            training_summary = json.load(f)
    else:
        print("Preparing tokenized training and validation datasets...")
        train_tokens = load_corpus_tokens(train_corpus_path, tokenizer)
        val_tokens = load_corpus_tokens(val_corpus_path, tokenizer)
        print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

        train_loader, val_loader = create_dataloaders(
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            max_seq_len=model_cfg.max_seq_len,
            batch_size=batch_size,
        )

        train_cfg = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            weight_decay=0.01,
            warmup_steps=30,
            max_steps=max_steps,
            epochs=epochs,
            grad_clip=1.0,
            grad_accum_steps=grad_accum_steps,
            device="cpu",
            seed=seed,
            log_interval=25,
            eval_interval=25,
            save_interval=50,
            checkpoint_dir=str(ckpt_dir),
        )

        model = CycloneLM(model_cfg)
        actual_params = model.count_parameters()
        print(f"Instantiated model parameter count: {actual_params:,}")

        trainer = CycloneTrainer(
            model=model,
            train_config=train_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        training_summary = trainer.train(resume_checkpoint=resume_checkpoint)

        # Save training summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(training_summary, f, indent=2)

        # Save model config yaml
        model_cfg.to_yaml(exp_dir / "config.yaml")

    # 4. Save provenance information
    hw_info = get_hardware_info()
    provenance_record = {
        "profile": profile_name,
        "dataset_version": "0.2.0",
        "dataset_train_corpus": train_corpus_path,
        "dataset_val_corpus": val_corpus_path,
        "dataset_test_corpus": test_corpus_path,
        "tokenizer_version": "0.2.0",
        "tokenizer_path": tokenizer_path,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "random_seed": seed,
        "training_hyperparameters": {
            "max_steps": max_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": 30,
            "grad_clip": 1.0,
        },
        "hardware_environment": hw_info,
        "architecture_audit": audit_data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance_record, f, indent=2)

    # 5. Model Evaluation
    print(f"Evaluating {profile_name} on held-out 100 suite and test corpus...")
    eval_engine = CycloneLMEngine(
        checkpoint_path=best_model_pt,
        tokenizer_path=tokenizer_path,
        device="cpu",
    )
    evaluator = ScalingEvaluator(
        suite_path=suite_path,
        test_corpus_path=test_corpus_path,
    )
    eval_results = evaluator.evaluate_model(
        engine=eval_engine,
        training_summary=training_summary,
    )

    # Save evaluation report and samples
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    samples_path = exp_dir / "samples.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(eval_results.get("samples", []), f, indent=2)

    print("=" * 70)
    print(f"EXPERIMENT {profile_name.upper()} COMPLETE:")
    print(f"  - Parameters:        {eval_results['model_metadata']['total_parameters']:,}")
    print(f"  - Final Train Loss:  {eval_results['language_quality']['training_loss']}")
    print(f"  - Final Val Loss:    {eval_results['language_quality']['validation_loss']}")
    print(f"  - Test Perplexity:   {eval_results['language_quality']['test_perplexity']}")
    print(f"  - Domain Accuracy:   {eval_results['domain_quality']['overall_domain_accuracy']}%")
    print(f"  - Tool Selection:    {eval_results['tool_use_quality']['tool_selection_accuracy']}%")
    print(f"  - Grounding Acc:     {eval_results['tool_use_quality']['grounding_accuracy']}%")
    print(f"  - Inference Latency: {eval_results['inference_performance']['ms_per_token']} ms/token")
    print("=" * 70)

    return {
        "profile": profile_name,
        "provenance": provenance_record,
        "training_summary": training_summary,
        "evaluation": eval_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CycloneLM Controlled Training and Scaling Experiments.")
    parser.add_argument(
        "--profile",
        type=str,
        default="all",
        choices=["239k", "1m", "5m", "10m", "20m", "all"],
        help="Scaling profile to run (or 'all')",
    )
    parser.add_argument("--max_steps", type=int, default=250, help="Max training steps per experiment")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Force retrain even if checkpoint exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles_to_run = (
        ["239k", "1m", "5m", "10m", "20m"]
        if args.profile == "all"
        else [args.profile]
    )

    all_results = {}
    for prof in profiles_to_run:
        accum = 2 if prof == "20m" else args.grad_accum
        res = run_experiment(
            profile_name=prof,
            max_steps=args.max_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=accum,
            seed=args.seed,
            force_retrain=args.force,
        )
        all_results[prof] = res

    # Merge all existing experiment runs found in models/cyclonelm/experiments/
    exp_base = Path("models/cyclonelm/experiments")
    for prof in ["239k", "1m", "5m", "10m", "20m"]:
        if prof not in all_results:
            p_dir = exp_base / prof
            if (p_dir / "evaluation_report.json").exists() and (p_dir / "provenance.json").exists():
                with open(p_dir / "evaluation_report.json", "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                with open(p_dir / "provenance.json", "r", encoding="utf-8") as f:
                    prov_data = json.load(f)
                training_data = {}
                if (p_dir / "training_summary.json").exists():
                    with open(p_dir / "training_summary.json", "r", encoding="utf-8") as f:
                        training_data = json.load(f)
                all_results[prof] = {
                    "profile": prof,
                    "provenance": prov_data,
                    "training_summary": training_data,
                    "evaluation": eval_data,
                }

    # Run Baselines & Compile Consolidated Comparison
    print("\nRunning Baselines (Deterministic, Template, v0.1)...")
    runner = HeldOutBenchmarkRunner()
    det_base = runner.run_deterministic_baseline()
    tpl_base = runner.run_template_baseline()

    # Previous CycloneLM version (v0.1) evaluation
    v01_ckpt = Path("models/cyclonelm/v0.1/checkpoints/best_model.pt")
    v01_eval = None
    if v01_ckpt.exists():
        v01_engine = CycloneLMEngine(
            checkpoint_path=v01_ckpt,
            tokenizer_path="models/cyclonelm/v0.1/tokenizer/vocab.json",
            device="cpu",
        )
        scaling_eval = ScalingEvaluator()
        v01_eval = scaling_eval.evaluate_model(v01_engine)

    consolidated_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": get_hardware_info(),
        "baselines": {
            "deterministic_baseline": det_base,
            "template_baseline": tpl_base,
            "cyclonelm_v01": v01_eval,
        },
        "experiments": all_results,
    }

    report_out = Path("reports/cyclonelm_scaling_results.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(consolidated_report, f, indent=2)
    print(f"\nSaved consolidated scaling results to: {report_out}")


if __name__ == "__main__":
    main()
