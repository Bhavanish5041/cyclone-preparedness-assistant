"""Unit tests for CycloneLM evaluation metrics and scientific rules."""

import math
from pathlib import Path

import pytest
import torch

from src.cyclonelm.config.model_config import CycloneLMConfig
from src.cyclonelm.evaluation.metrics import (
    compute_perplexity,
    evaluate_scientific_governance_rules,
    measure_inference_speed,
    verify_checkpoint_reload,
)
from src.cyclonelm.model.transformer import CycloneLM


def test_perplexity_calculation():
    loss = 2.0
    ppl = compute_perplexity(loss)
    assert math.isclose(ppl, math.exp(2.0), rel_tol=1e-5)


def test_checkpoint_reload_identity():
    cfg = CycloneLMConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=16)
    m1 = CycloneLM(cfg)
    m2 = CycloneLM(cfg)
    m2.load_state_dict(m1.state_dict())

    identical, max_diff = verify_checkpoint_reload(m1, m2)
    assert identical is True
    assert max_diff < 1e-6


def test_scientific_governance_adherence_checker():
    compliant_text = (
        "AHP PRECEDENCE MANDATE: The deterministic AHP Preparedness tier maintains absolute operational precedence. "
        "Logistic Regression is an empirical surrogate model. We cannot provide real-time weather observations. "
        "Operational Priority Bands allocate emergency resources, which are distinct from AHP tiers."
    )
    result = evaluate_scientific_governance_rules(compliant_text)
    assert result["mentions_ahp_or_precedence"] is True
    assert result["demarcates_surrogate"] is True
    assert result["refuses_live_fabrication"] is True
    assert result["distinguishes_tiers_and_bands"] is True


def test_inference_speed_benchmark():
    cfg = CycloneLMConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=32)
    model = CycloneLM(cfg)
    perf = measure_inference_speed(model, seq_len=8, generate_tokens=8, trials=2, device="cpu")
    assert perf["tokens_per_second"] > 0.0
    assert perf["ms_per_token"] > 0.0


def test_held_out_100_suite_benchmark():
    from src.cyclonelm.evaluation.benchmark import HeldOutBenchmarkRunner

    suite_path = Path("data/cyclonelm/v0.2/held_out_eval/held_out_100.jsonl")
    assert suite_path.exists(), f"Held out suite missing: {suite_path}"

    runner = HeldOutBenchmarkRunner(suite_path=suite_path)
    assert len(runner.questions) == 100

    det_res = runner.run_deterministic_baseline()
    assert det_res["total_questions"] == 100
    assert det_res["overall_accuracy"] >= 95.0
    assert det_res["categories"]["domain_knowledge"]["correct_abstentions"] == 3
    assert det_res["total_invalid_arguments"] == 0

    tpl_res = runner.run_template_baseline()
    assert tpl_res["total_questions"] == 100
    assert tpl_res["overall_accuracy"] < det_res["overall_accuracy"]


def test_dataset_zero_leakage_invariants():
    import json

    stats_path = Path("data/cyclonelm/v0.2/dataset_statistics.json")
    assert stats_path.exists()
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    audit = stats["overlap_audit"]
    assert audit["zero_leakage_verified"] is True
    assert audit["train_val_overlap"] == 0
    assert audit["train_test_overlap"] == 0
    assert audit["train_heldout_overlap"] == 0
    assert audit["val_heldout_overlap"] == 0


def test_execute_all_seven_deterministic_tools():
    from src.cyclonelm.tools.protocol import ToolCallRequest, ToolOrchestrator

    orchestrator = ToolOrchestrator()
    tools = [
        ("lookup_community", {"query": "Tallur"}),
        ("analyze_infrastructure_and_exposure", {"identifier": "Tallur"}),
        ("explain_preparedness_assessment", {"identifier": "Tallur"}),
        ("get_intervention_recommendations", {"identifier": "Tallur"}),
        ("compare_districts", {"district_a": "Dakshina Kannada", "district_b": "Udupi"}),
        ("filter_and_rank_communities", {"district": "Dakshina Kannada", "limit": 2}),
        ("explain_vulnerability_archetype", {"archetype_id": 0}),
    ]

    for tool_name, params in tools:
        rec = orchestrator.execute_tool_call(ToolCallRequest(tool=tool_name, parameters=params))
        assert rec.success is True, f"Tool {tool_name} failed: {rec.error}"
        assert rec.data is not None

