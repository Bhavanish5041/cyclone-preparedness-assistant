"""Evaluation command script for CycloneLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cyclonelm.evaluation.benchmark import CycloneLMBenchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CycloneLM across architecture, inference, and governance.")
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
        "--output_report",
        type=str,
        default="reports/cyclonelm_evaluation_report.json",
        help="Destination JSON path for evaluation report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("CYCLONELM COMPREHENSIVE BENCHMARK EVALUATION")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer:  {args.tokenizer}")
    print("=" * 60)

    benchmark = CycloneLMBenchmark(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )

    report = benchmark.run_all_suites()

    # Save report
    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- BENCHMARK RESULTS SUMMARY ---")
    print(f"Parameters:              {report['model_metadata']['total_parameters']:,} (< 1M: {report['model_metadata']['within_one_million_limit']})")
    print(f"Checkpoint size:         {report['checkpoint_verification']['checkpoint_size_kb']:.1f} KB")
    print(f"Weight reload identical: {report['checkpoint_verification']['weights_identical']} (max diff: {report['checkpoint_verification']['max_absolute_weight_diff']:.2e})")
    print(f"CPU Throughput:          {report['inference_performance']['tokens_per_second']:.1f} tokens/sec ({report['inference_performance']['ms_per_token']:.2f} ms/token)")
    print(f"Tools registered:        {report['tool_protocol']['tools_registered_count']}")
    print(f"Lookup Community:        {'SUCCESS' if report['tool_protocol']['lookup_community_success'] else 'FAILED'}")
    print(f"Compare Districts:       {'SUCCESS' if report['tool_protocol']['compare_districts_success'] else 'FAILED'}")
    print(f"Duplicate call blocked:  {'SUCCESS' if report['tool_protocol']['duplicate_calls_detected'] else 'FAILED'}")
    print(f"Governance adherence:    {'PASSED' if report['scientific_governance']['all_rules_covered'] else 'FAILED'}")
    print(f"Peak RSS Memory:         {report['memory_usage']['peak_rss_mb']:.1f} MB")
    print(f"Report written to:       {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
