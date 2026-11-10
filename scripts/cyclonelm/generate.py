"""Text generation CLI script for CycloneLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cyclonelm.inference.engine import CycloneLMEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text using trained CycloneLM.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/cyclonelm/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/cyclonelm/tokenizer/vocab.json",
        help="Path to tokenizer vocab JSON",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tropical cyclones in Coastal Karnataka require",
        help="Prompt text for autoregressive completion",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = CycloneLMEngine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )

    print("=" * 60)
    print("CYCLONELM TEXT GENERATION")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Temp: {args.temperature} | Top-k: {args.top_k} | Max tokens: {args.max_new_tokens}")
    print("=" * 60)

    result = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n--- RAW GENERATION ---")
    print(result["full_text"])
    print("\n--- CLEAN GENERATION ---")
    print(result["prompt"] + result["clean_text"])
    print(f"\nCompletion tokens: {result['completion_tokens']} | Latency: {result['latency_ms']:.2f}ms | Throughput: {result['tokens_per_sec']:.1f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
