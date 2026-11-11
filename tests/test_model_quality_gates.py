"""Automated quality gate regression checks for CYCLONE INTEL.

Verifies:
- All 11 model quality gates pass against documented acceptance baselines.
- Scientific invariance is preserved (deterministic AHP scores never violated).
- Surrogate model tier concordance exceeds 85%.
- Hallucination and unsupported claim frequency remains strictly 0.0%.
- Dynamic operational telemetry refusal precision and recall are 100%.
"""

import json
from pathlib import Path
import pytest

from src.evaluation.model_quality_pipeline import (
    ACCEPTANCE_THRESHOLDS,
    ModelQualityPipeline,
)


@pytest.fixture(scope="module")
def quality_evaluation_report():
    """Run model quality pipeline and return report."""
    pipeline = ModelQualityPipeline()
    report = pipeline.evaluate_all()
    return report


def test_quality_pipeline_status_passed(quality_evaluation_report):
    """Ensure the entire model quality pipeline achieves PASSED status."""
    assert quality_evaluation_report["pipeline_status"] == "PASSED"


def test_intent_accuracy_threshold(quality_evaluation_report):
    """Ensure intent accuracy meets or exceeds threshold."""
    gate = quality_evaluation_report["quality_gates"]["intent_accuracy"]
    assert gate["passed"] is True
    assert gate["measured"] >= ACCEPTANCE_THRESHOLDS["intent_accuracy"]


def test_intent_macro_f1_threshold(quality_evaluation_report):
    """Ensure intent macro F1 meets or exceeds threshold."""
    gate = quality_evaluation_report["quality_gates"]["intent_macro_f1"]
    assert gate["passed"] is True
    assert gate["measured"] >= ACCEPTANCE_THRESHOLDS["intent_macro_f1"]


def test_argument_correctness_threshold(quality_evaluation_report):
    """Ensure tool argument correctness meets or exceeds threshold."""
    gate = quality_evaluation_report["quality_gates"]["argument_correctness"]
    assert gate["passed"] is True
    assert gate["measured"] >= ACCEPTANCE_THRESHOLDS["argument_correctness"]


def test_entity_resolution_accuracy(quality_evaluation_report):
    """Ensure entity resolution accuracy meets or exceeds threshold."""
    gate = quality_evaluation_report["quality_gates"]["entity_resolution_accuracy"]
    assert gate["passed"] is True
    assert gate["measured"] >= ACCEPTANCE_THRESHOLDS["entity_resolution_accuracy"]


def test_multi_intent_completeness(quality_evaluation_report):
    """Ensure multi-intent requests are completely handled."""
    gate = quality_evaluation_report["quality_gates"]["multi_intent_completeness"]
    assert gate["passed"] is True
    assert gate["measured"] >= ACCEPTANCE_THRESHOLDS["multi_intent_completeness"]


def test_grounded_factual_correctness(quality_evaluation_report):
    """Ensure deterministic AHP composite scores and tiers achieve 100% fidelity."""
    gate = quality_evaluation_report["quality_gates"]["grounded_factual_correctness"]
    assert gate["passed"] is True
    assert gate["measured"] == 1.0


def test_unsupported_claim_frequency(quality_evaluation_report):
    """Ensure unsupported claim and hallucination rate is strictly zero."""
    gate = quality_evaluation_report["quality_gates"]["unsupported_claim_frequency"]
    assert gate["passed"] is True
    assert gate["measured"] == 0.0


def test_refusal_precision_and_recall(quality_evaluation_report):
    """Ensure dynamic telemetry requests are refused with 100% precision and recall."""
    prec_gate = quality_evaluation_report["quality_gates"]["refusal_precision"]
    rec_gate = quality_evaluation_report["quality_gates"]["refusal_recall"]
    assert prec_gate["passed"] is True
    assert rec_gate["passed"] is True
    assert prec_gate["measured"] == 1.0
    assert rec_gate["measured"] == 1.0


def test_surrogate_concordance_rate(quality_evaluation_report):
    """Ensure surrogate model tier concordance exceeds 85%."""
    gate = quality_evaluation_report["quality_gates"]["surrogate_concordance_rate"]
    assert gate["passed"] is True
    assert gate["measured"] >= ACCEPTANCE_THRESHOLDS["surrogate_concordance_rate"]


def test_ahp_score_invariance_gate(quality_evaluation_report):
    """Ensure deterministic AHP composite scores and priority bands achieve 100% invariance."""
    gate = quality_evaluation_report["quality_gates"]["ahp_score_invariance"]
    assert gate["passed"] is True
    assert gate["measured"] == 1.0


def test_no_ml_downgrade_of_ahp_low_gate(quality_evaluation_report):
    """Ensure ML surrogate predictions never downgrade authoritative AHP Low classifications."""
    gate = quality_evaluation_report["quality_gates"]["no_ml_downgrade_of_ahp_low"]
    assert gate["passed"] is True
    assert gate["measured"] == 1.0


def test_latency_performance_limits(quality_evaluation_report):
    """Ensure mean deterministic pipeline latency remains under 250ms on CPU."""
    mean_lat = quality_evaluation_report["latency_metrics"]["mean_ms"]
    assert mean_lat < ACCEPTANCE_THRESHOLDS["max_avg_latency_ms"]
