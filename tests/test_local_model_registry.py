"""Unit tests for the local Git-compatible Model Registry and CLI."""

import json
from pathlib import Path
import pytest

from src.models.registry.local_registry import (
    DEFAULT_REGISTRY_PATH,
    LocalModelRegistry,
    compute_file_sha256,
)


@pytest.fixture
def registry():
    """Create LocalModelRegistry instance."""
    return LocalModelRegistry()


def test_registry_manifest_exists():
    """Verify that default model_registry.json exists on disk."""
    assert DEFAULT_REGISTRY_PATH.exists()


def test_list_models_contains_all_three_families(registry):
    """Verify registry tracks Logistic Regression, K-Means, and Qwen."""
    models = registry.list_models()
    model_ids = {m["model_identifier"] for m in models}

    assert "village-preparedness-classifier" in model_ids
    assert "coastal-vulnerability-kmeans" in model_ids
    assert "qwen-orchestration-llm" in model_ids


def test_validate_all_artifacts(registry):
    """Verify all model artifact files exist and match their SHA256 hashes."""
    results = registry.validate_artifacts()
    assert len(results) >= 5

    for res in results:
        assert res["status"] == "VALID", f"Artifact check failed: {res}"


def test_get_model_details(registry):
    """Verify metadata attributes for each model."""
    # 1. Logistic Regression
    lr = registry.get_model("village-preparedness-classifier")
    assert lr["framework"].startswith("scikit-learn")
    assert lr["status"] == "production"
    assert lr["input_schema"]["feature_count"] == 26
    assert lr["evaluation_metrics"]["test_accuracy"] >= 0.90
    assert "methodology_note" in lr

    # 2. K-Means
    km = registry.get_model("coastal-vulnerability-kmeans")
    assert km["status"] == "production"
    assert km["input_schema"]["dimensions"] == 4
    assert len(km["output_schema"]["clusters"]) == 4

    # 3. Qwen
    qwen = registry.get_model("qwen-orchestration-llm")
    assert qwen["status"] == "production"
    assert qwen["artifact_files"][0]["model_tag"] == "qwen2.5:0.5b-instruct"
    assert qwen["evaluation_metrics"]["intent_accuracy"] == 1.0


def test_get_nonexistent_model_raises_keyerror(registry):
    """Verify KeyError is raised when querying invalid model ID."""
    with pytest.raises(KeyError):
        registry.get_model("nonexistent-model-xyz")


def test_promotion_and_rollback_lifecycle(tmp_path):
    """Verify promotion and rollback lifecycle on an isolated registry copy."""
    test_manifest_path = tmp_path / "model_registry_test.json"
    with open(DEFAULT_REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Add a candidate version
    data["models"]["village-preparedness-classifier"]["versions"]["v2.0.0-candidate"] = {
        "version": "v2.0.0-candidate",
        "status": "candidate",
        "history": [],
    }
    with open(test_manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    test_reg = LocalModelRegistry(registry_path=test_manifest_path)

    # 1. Promote v2.0.0-candidate to production
    test_reg.promote_model(
        model_id="village-preparedness-classifier",
        version="v2.0.0-candidate",
        target_status="production",
        author="test_suite",
        note="Candidate passed quality checks",
    )

    updated = test_reg.get_model("village-preparedness-classifier")
    assert updated["version"] == "v2.0.0-candidate"
    assert updated["status"] == "production"

    # Previous version should be demoted from production to validated
    old_v = test_reg.get_model("village-preparedness-classifier", version="v1.0.0")
    assert old_v["status"] == "validated"

    # 2. Rollback to v1.0.0
    test_reg.rollback_model(
        model_id="village-preparedness-classifier",
        target_version="v1.0.0",
        author="test_suite",
        note="Emergency rollback to v1.0.0",
    )

    restored = test_reg.get_model("village-preparedness-classifier")
    assert restored["version"] == "v1.0.0"
    assert restored["status"] == "production"
