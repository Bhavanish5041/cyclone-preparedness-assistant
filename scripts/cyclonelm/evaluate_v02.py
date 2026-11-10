"""CycloneLM v0.2 Comprehensive Held-Out Benchmark Evaluator.

Evaluates the planned 100-question held-out benchmark suite:
- 30 Domain Knowledge questions
- 30 Tool Selection questions
- 20 Tool Argument Generation questions
- 20 Evidence-Grounded Response questions

Generates:
- Category-level accuracy metrics
- Syntax & validation error metrics (invalid JSON, invalid arguments via Pydantic)
- Scientific safety metrics (unsupported claims, correct abstentions on unrecorded live telemetry)
- Comparative side-by-side analysis against:
  1. Deterministic Rule-Based Baseline
  2. Template-Based Slot-Filling Baseline
- Comprehensive evaluation JSON report saved to reports/cyclonelm_v02_evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.cyclonelm.evaluation.benchmark import HeldOutBenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CycloneLM v0.2 Held-Out 100-Question Suite.")
    parser.add_argument(
        "--suite",
        type=str,
        default="data/cyclonelm/v0.2/held_out_eval/held_out_100.jsonl",
        help="Path to 100-question held-out JSONL suite",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="reports/cyclonelm_v02_evaluation_report.json",
        help="Destination JSON path for evaluation report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("CYCLONELM v0.2 HELD-OUT 100-QUESTION BENCHMARK EVALUATION")
    print(f"Benchmark Suite: {args.suite}")
    print("=" * 70)

    runner = HeldOutBenchmarkRunner(suite_path=args.suite)

    # 1. Run Deterministic Baseline
    det_results = runner.run_deterministic_baseline()

    # 2. Run Template Baseline
    tpl_results = runner.run_template_baseline()

    # 3. Assemble Consolidated Report
    report = {
        "benchmark_version": "0.2.0",
        "total_held_out_questions": len(runner.questions),
        "suite_path": str(args.suite),
        "categories_breakdown": {
            "domain_knowledge": sum(1 for q in runner.questions if q["category"] == "domain_knowledge"),
            "tool_selection": sum(1 for q in runner.questions if q["category"] == "tool_selection"),
            "argument_generation": sum(1 for q in runner.questions if q["category"] == "argument_generation"),
            "evidence_grounded_response": sum(1 for q in runner.questions if q["category"] == "evidence_grounded_response"),
        },
        "baselines": {
            "deterministic_baseline": det_results,
            "template_baseline": tpl_results,
        },
        "comparative_summary": {
            "overall_accuracy": {
                "deterministic_baseline": f"{det_results['overall_accuracy']}%",
                "template_baseline": f"{tpl_results['overall_accuracy']}%",
            },
            "domain_knowledge_accuracy": {
                "deterministic_baseline": f"{det_results['categories']['domain_knowledge']['accuracy']}%",
                "template_baseline": f"{tpl_results['categories']['domain_knowledge']['accuracy']}%",
            },
            "tool_selection_accuracy": {
                "deterministic_baseline": f"{det_results['categories']['tool_selection']['accuracy']}%",
                "template_baseline": f"{tpl_results['categories']['tool_selection']['accuracy']}%",
            },
            "argument_generation_accuracy": {
                "deterministic_baseline": f"{det_results['categories']['argument_generation']['accuracy']}%",
                "template_baseline": f"{tpl_results['categories']['argument_generation']['accuracy']}%",
            },
            "evidence_grounded_response_accuracy": {
                "deterministic_baseline": f"{det_results['categories']['evidence_grounded_response']['accuracy']}%",
                "template_baseline": f"{tpl_results['categories']['evidence_grounded_response']['accuracy']}%",
            },
            "safety_and_validity": {
                "total_invalid_json": {
                    "deterministic_baseline": det_results["total_invalid_json"],
                    "template_baseline": tpl_results["total_invalid_json"],
                },
                "total_invalid_arguments": {
                    "deterministic_baseline": det_results["total_invalid_arguments"],
                    "template_baseline": tpl_results["total_invalid_arguments"],
                },
                "total_unsupported_claims": {
                    "deterministic_baseline": det_results["total_unsupported_claims"],
                    "template_baseline": tpl_results["total_unsupported_claims"],
                },
                "total_correct_abstentions": {
                    "deterministic_baseline": det_results["total_correct_abstentions"],
                    "template_baseline": tpl_results["total_correct_abstentions"],
                },
            },
        },
    }

    # Save Report
    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Formatted Output Table
    print("\n--- BENCHMARK RESULTS COMPARATIVE MATRIX ---")
    print(f"{'Evaluation Metric':<35} | {'Deterministic Baseline':<22} | {'Template Baseline':<20}")
    print("-" * 85)
    print(f"{'Overall Benchmark Accuracy':<35} | {det_results['overall_accuracy']:>20.1f}% | {tpl_results['overall_accuracy']:>18.1f}%")
    print(f"{'Domain Knowledge (30 Qs)':<35} | {det_results['categories']['domain_knowledge']['accuracy']:>20.1f}% | {tpl_results['categories']['domain_knowledge']['accuracy']:>18.1f}%")
    print(f"{'Tool Selection (30 Qs)':<35} | {det_results['categories']['tool_selection']['accuracy']:>20.1f}% | {tpl_results['categories']['tool_selection']['accuracy']:>18.1f}%")
    print(f"{'Argument Generation (20 Qs)':<35} | {det_results['categories']['argument_generation']['accuracy']:>20.1f}% | {tpl_results['categories']['argument_generation']['accuracy']:>18.1f}%")
    print(f"{'Evidence Grounding (20 Qs)':<35} | {det_results['categories']['evidence_grounded_response']['accuracy']:>20.1f}% | {tpl_results['categories']['evidence_grounded_response']['accuracy']:>18.1f}%")
    print("-" * 85)
    print(f"{'Invalid JSON Count':<35} | {det_results['total_invalid_json']:>21} | {tpl_results['total_invalid_json']:>19}")
    print(f"{'Invalid Arguments (Pydantic)':<35} | {det_results['total_invalid_arguments']:>21} | {tpl_results['total_invalid_arguments']:>19}")
    print(f"{'Unsupported Claims / Hallucinations':<35} | {det_results['total_unsupported_claims']:>21} | {tpl_results['total_unsupported_claims']:>19}")
    print(f"{'Correct Abstentions (Live Refusals)':<35} | {det_results['total_correct_abstentions']:>21} | {tpl_results['total_correct_abstentions']:>19}")
    print("=" * 85)
    print(f"Comprehensive Evaluation Report saved to: {out_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
