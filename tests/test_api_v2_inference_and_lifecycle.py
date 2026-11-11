"""Integration tests for FastAPI v2 inference, health/readiness, request tracking, and registry endpoints."""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_root_health_and_ready():
    """Verify root health and readiness probes return 200 OK with valid schemas."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["total_units"] == 1879

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    data_ready = res_ready.json()
    assert data_ready["ready"] is True
    assert data_ready["dataset_loaded"] is True
    assert data_ready["model_loaded"] is True


def test_api_health_readiness_probe():
    """Verify /api/health and /api/health/ready endpoints."""
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["model_loaded"] is True

    res_ready = client.get("/api/health/ready")
    assert res_ready.status_code == 200
    data = res_ready.json()
    assert data["ready"] is True
    assert data["checks"]["dataset_loaded"] is True
    assert data["checks"]["surrogate_model_loaded"] is True


def test_request_id_middleware_and_process_time_headers():
    """Verify X-Request-ID and X-Process-Time headers are set and propagated."""
    # Scenario A: Client provides custom Request ID
    custom_id = "test-custom-req-12345"
    res1 = client.get("/health", headers={"X-Request-ID": custom_id})
    assert res1.status_code == 200
    assert res1.headers.get("X-Request-ID") == custom_id
    assert "X-Process-Time" in res1.headers

    # Scenario B: Server generates unique Request ID
    res2 = client.get("/health")
    assert res2.status_code == 200
    assert res2.headers.get("X-Request-ID", "").startswith("req_")
    assert "X-Process-Time" in res2.headers


def test_model_registry_endpoints():
    """Verify /api/models/registry catalog and item detail endpoints."""
    # 1. List registered models
    res_list = client.get("/api/models/registry")
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert data_list["artifact_verification_status"] == "ALL_VERIFIED"
    assert len(data_list["models"]) >= 3

    model_ids = {m["model_identifier"] for m in data_list["models"]}
    assert "village-preparedness-classifier" in model_ids
    assert "coastal-vulnerability-kmeans" in model_ids
    assert "qwen-orchestration-llm" in model_ids

    # 2. Get details for village-preparedness-classifier
    res_detail = client.get("/api/models/registry/village-preparedness-classifier")
    assert res_detail.status_code == 200
    data_detail = res_detail.json()
    assert data_detail["status"] == "production"
    assert data_detail["framework"].startswith("scikit-learn")
    assert data_detail["input_schema"]["feature_count"] == 26

    # 3. 404 for nonexistent model
    res_404 = client.get("/api/models/registry/nonexistent-model")
    assert res_404.status_code == 404


def test_model_predict_by_area_id():
    """Verify surrogate inference by village name with deterministic AHP ground-truth."""
    payload = {"area_id": "Tallur"}
    res = client.post("/api/models/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["model_name"] == "village-preparedness-classifier"
    assert data["village_name"] == "Tallur"
    assert data["district_name"] == "Udupi"
    assert data["predicted_tier"] in ["High", "Medium", "Low"]
    assert "High" in data["probabilities"]
    assert "operational_precedence_note" in data

    # Verify deterministic AHP metrics match ground truth
    assert data["deterministic_ahp_tier"] == "High"
    assert round(data["deterministic_ahp_score"], 2) == 74.34


def test_model_predict_with_raw_features():
    """Verify surrogate inference when supplying raw 26-feature dictionary."""
    from src.features.inference_features import CANONICAL_26_FEATURES

    dummy_features = {f: 0.0 for f in CANONICAL_26_FEATURES}
    dummy_features["has_pucca_road"] = 1.0
    dummy_features["has_major_district_road"] = 1.0
    dummy_features["bus_connectivity"] = 1.0

    payload = {"features": dummy_features}
    res = client.post("/api/models/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["predicted_tier"] in ["High", "Medium", "Low"]
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert data["confidence_score"] > 0.0


def test_model_predict_validation_errors():
    """Verify 422 errors when neither or invalid parameters are provided."""
    # 1. Neither area_id nor features
    res1 = client.post("/api/models/predict", json={})
    assert res1.status_code == 422

    # 2. Unknown village name
    res2 = client.post("/api/models/predict", json={"area_id": "NonExistentVillage123"})
    assert res2.status_code == 404

    # 3. Incomplete features dictionary
    res3 = client.post("/api/models/predict", json={"features": {"dist_coast_km": 1.2}})
    assert res3.status_code == 422


def test_model_predict_ahp_low_precedence_mandate():
    """Verify that AHP Low classification enforces operational precedence note and non-downgrade."""
    payload = {"area_id": "Bappalgundi"}
    res = client.post("/api/models/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["village_name"] == "Bappalgundi"
    assert data["deterministic_ahp_tier"] == "Low"
    assert round(data["deterministic_ahp_score"], 2) == 20.14
    assert "AHP" in data["operational_precedence_note"]
    assert "precedence" in data["operational_precedence_note"].lower()
