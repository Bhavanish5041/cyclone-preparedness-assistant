#!/usr/bin/env python3
"""Run automated model quality evaluation pipeline (Stage 3 CI Gate).

Executes ModelQualityPipeline computing:
- Accuracy, precision, recall per class
- Macro F1 and weighted F1
- Confusion matrix
- Low-tier recall
- AHP surrogate concordance
- Prediction failure rate
- Schema validity
- Assistant tool intent accuracy
- Tool argument validity
- Grounded factual correctness
- Unsupported claim frequency
- Refusal precision and recall
- Latency and resource profiling

Generates:
- reports/model_quality_report.json
- reports/model_quality_summary.md

Exits with 0 if all quality gates pass, 1 otherwise.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.model_quality_pipeline import ModelQualityPipeline


def main():
    print("=" * 70)
    print("CYCLONE INTEL: Automated Model Quality Evaluation (Stage 3)")
    print("=" * 70)

    pipeline = ModelQualityPipeline()
    report = pipeline.evaluate_all()

    status = report.get("pipeline_status")
    print(f"\nEvaluation Status: {status}")
    print(f"Total Samples Evaluated: {report.get('total_samples_evaluated')}")

    print("\nQuality Gate Summary:")
    all_passed = True
    for gate_name, gate in report.get("quality_gates", {}).items():
        res = "PASS ✅" if gate["passed"] else "FAIL ❌"
        if not gate["passed"]:
            all_passed = False
        print(f"  {gate_name:<30}: {gate['measured']} (threshold {gate['threshold']}) -> {res}")

    reports_dir = PROJECT_ROOT / "reports"
    print(f"\nArtifacts generated:")
    print(f"  JSON Report: {reports_dir / 'model_quality_report.json'}")
    print(f"  Markdown Summary: {reports_dir / 'model_quality_summary.md'}")

    if all_passed and status == "PASSED":
        print("\nAll Stage 3 model quality gates PASSED successfully.")
        sys.exit(0)
    else:
        print("\nOne or more Stage 3 quality gates FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
