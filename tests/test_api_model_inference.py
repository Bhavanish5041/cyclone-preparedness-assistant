"""Unit and integration tests for FastAPI Model Inference Integration (Phase 2.4).

Ensures:
- Faithful 9-feature engineering reproduction matching modelling_dataset.parquet to machine precision.
- Robust input validation (rejection of missing columns, NaNs, Infs, non-numeric types).
- Safe single-time model loading in FastAPI lifespan from W&B registered model cache.
- Clear startup failure when configured W&B model artifact cannot be retrieved.
- Active artifact version, schema status, and provenance exposure in /health and /api/v1/model/health.
- Validated inference via area_id and ad-hoc feature payloads (/api/v1/model/predict).
- Probability simplex adherence (sum == 1.0) and boundary-case detection.
- Absolute operational precedence of deterministic AHP baseline (no downgrading of Low tier).
- Zero regression in existing decision-support and prioritization endpoints.
"""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.features.inference_features import (
    CANONICAL_26_FEATURES,
    NINE_ENGINEERED_FEATURES,
    SEVENTEEN_BASE_FEATURES,
    engineer_inference_features,
)
from src.models.load_registered_model import WandbRetrievalError, load_registered_model

MODELLING_PATH = Path("/home/bhavanish/cyclone-preparedness/data/processed/modelling_dataset.parquet")
UNIFIED_V2_PATH = Path("data/processed/unified_village_preparedness_v2.parquet")
METADATA_PATH = Path("models/logistic_regression_metadata.json")


@pytest.fixture(scope="module")
def df_modelling():
    assert MODELLING_PATH.exists(), f"Missing modelling dataset at {MODELLING_PATH}"
    return pd.read_parquet(MODELLING_PATH)


@pytest.fixture(scope="module")
def df_unified():
    assert UNIFIED_V2_PATH.exists(), f"Missing unified dataset at {UNIFIED_V2_PATH}"
    return pd.read_parquet(UNIFIED_V2_PATH)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient instance managing app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# 1. FEATURE ENGINEERING EQUIVALENCE TESTS
# =============================================================================

def test_feature_engineering_equivalence_all_1879_villages(df_modelling, df_unified):
    """Verify feature engineering reproduces all 26 features across all 1,879 villages to <= 1e-12."""
    # Align unified v2 with modelling dataset by area_id
    aligned_v2 = df_unified.set_index("area_id").loc[df_modelling["area_id"]].reset_index()

    X_eng = engineer_inference_features(aligned_v2)
    assert X_eng.shape == (1879, 26)
    assert list(X_eng.columns) == CANONICAL_26_FEATURES

    # Verify each feature matches modelling_dataset.parquet
    max_discrepancies = {}
    for feat in CANONICAL_26_FEATURES:
        max_diff = float(np.max(np.abs(X_eng[feat] - df_modelling[feat])))
        max_discrepancies[feat] = max_diff
        assert max_diff <= 1e-12, f"Feature '{feat}' discrepancy {max_diff:.2e} exceeds tolerance 1e-12"


def test_feature_engineering_from_single_village_dict(df_unified):
    """Verify feature engineering on a single-record dictionary returns 1-row DataFrame."""
    single_dict = df_unified.iloc[0].to_dict()
    X_single = engineer_inference_features(single_dict)

    assert isinstance(X_single, pd.DataFrame)
    assert X_single.shape == (1, 26)
    assert list(X_single.columns) == CANONICAL_26_FEATURES
    assert not X_single.isna().any().any()
    assert np.isfinite(X_single.to_numpy()).all()


def test_feature_engineering_with_precomputed_26_features(df_modelling):
    """Verify input already containing the 26 canonical features is preserved and aligned."""
    sample_26 = df_modelling[CANONICAL_26_FEATURES].iloc[:10].copy()
    # Scramble column order
    scrambled_cols = list(reversed(CANONICAL_26_FEATURES))
    sample_scrambled = sample_26[scrambled_cols].copy()

    X_out = engineer_inference_features(sample_scrambled)
    assert list(X_out.columns) == CANONICAL_26_FEATURES
    np.testing.assert_allclose(X_out.to_numpy(), sample_26.to_numpy(), atol=1e-12)


# =============================================================================
# 2. FEATURE VALIDATION & CORRUPTED INPUT HANDLING
# =============================================================================

def test_feature_engineering_missing_raw_indicator_raises_error(df_unified):
    """Verify ValueError when required raw indicator (e.g. population) is missing."""
    bad_dict = df_unified.iloc[0].to_dict()
    del bad_dict["population"]

    with pytest.raises(ValueError, match="Cannot engineer model features: input is missing 1 required column"):
        engineer_inference_features(bad_dict)


def test_feature_engineering_missing_base_feature_raises_error(df_unified):
    """Verify ValueError when required base feature (e.g. dist_coast_km) is missing."""
    bad_dict = df_unified.iloc[0].to_dict()
    del bad_dict["dist_coast_km"]

    with pytest.raises(ValueError, match="Cannot engineer model features: input is missing"):
        engineer_inference_features(bad_dict)


def test_feature_engineering_nan_value_raises_error(df_unified):
    """Verify ValueError when NaN is present in input."""
    bad_dict = df_unified.iloc[0].to_dict()
    bad_dict["elevation_mean"] = np.nan

    with pytest.raises(ValueError, match="Missing \\(NaN/null\\) values detected"):
        engineer_inference_features(bad_dict)


def test_feature_engineering_inf_value_raises_error(df_unified):
    """Verify ValueError when +/-Inf is present in input."""
    bad_dict = df_unified.iloc[0].to_dict()
    bad_dict["elevation_mean"] = np.inf

    with pytest.raises(ValueError, match="Infinite \\(\\+/-Inf\\) values detected"):
        engineer_inference_features(bad_dict)


def test_feature_engineering_non_numeric_type_raises_error(df_unified):
    """Verify ValueError when non-numeric string is passed in a numeric feature."""
    bad_dict = df_unified.iloc[0].to_dict()
    bad_dict["elevation_mean"] = "NOT_A_NUMBER"

    with pytest.raises(ValueError, match="Non-numeric features detected"):
        engineer_inference_features(bad_dict)


def test_feature_engineering_empty_input_raises_error():
    """Verify ValueError on empty input."""
    with pytest.raises(ValueError, match="Input feature data is empty"):
        engineer_inference_features({})


# =============================================================================
# 3. FASTAPI LIFESPAN & STARTUP ERROR HANDLING
# =============================================================================

def test_startup_successful_model_loading(client: TestClient):
    """Verify model container is loaded and attached to app state during lifespan."""
    assert hasattr(app.state, "model_container")
    assert app.state.model_container is not None
    assert app.state.model_container.artifact_version == "v0"
    assert len(app.state.model_container.features) == 26


def test_startup_failed_registry_retrieval_raises_runtime_error():
    """Verify startup fails clearly when configured W&B model artifact cannot be retrieved."""
    original_container = getattr(app.state, "model_container", None)
    try:
        with patch("src.api.main.load_registered_model", side_effect=WandbRetrievalError("Failed to fetch W&B artifact")):
            with pytest.raises(RuntimeError, match="FastAPI startup aborted: failed to load required W&B model artifact"):
                with TestClient(app):
                    pass
    finally:
        # Restore container for subsequent tests
        app.state.model_container = original_container or load_registered_model("village-preparedness-classifier:v0")


# =============================================================================
# 4. HEALTH & READINESS ENDPOINTS
# =============================================================================

def test_health_check_endpoint_reports_model_status(client: TestClient):
    """Verify /health endpoint reports model loaded status, artifact version, and schema status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "healthy"
    assert data["dataset_loaded"] is True
    assert data["model_loaded"] is True
    assert data["model_artifact_name"] == "village-preparedness-classifier"
    assert data["model_artifact_version"] == "v0"
    assert data["model_feature_schema_status"] == "VALID_26_FEATURES"
    assert "REPRODUCIBLE RECONSTRUCTION" in data["model_provenance"]

    # Security check: ensure no filesystem paths, tokens, or API keys are exposed
    body_text = resp.text
    assert "WANDB_API_KEY" not in body_text
    assert "/home/" not in body_text


def test_model_health_endpoint_details(client: TestClient):
    """Verify dedicated /api/v1/model/health endpoint."""
    resp = client.get("/api/v1/model/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["artifact_name"] == "village-preparedness-classifier"
    assert data["artifact_version"] == "v0"
    assert data["features_count"] == 26
    assert data["feature_schema_status"] == "VALID_26_FEATURES"
    assert data["target_classes"] == ["High", "Low", "Medium"]
    assert data["target_nature"] == "AHP_CAPACITY_TIER_SURROGATE"


# =============================================================================
# 5. INFERENCE ENDPOINT: AREA IDENTIFIER LOOKUP
# =============================================================================

def test_predict_by_area_id_success(client: TestClient):
    """Verify inference by area_id (Panjikal village)."""
    # Panjikal village: SHRID / area_id = '11-29-575-05562-617533', PC11 ID = '617533'
    payload = {"area_id": "11-29-575-05562-617533"}
    resp = client.post("/api/v1/model/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["area_id"] == "11-29-575-05562-617533"
    assert data["village_name"] == "Panjikal"
    assert data["district_name"] == "Dakshina Kannada"
    assert data["predicted_preparedness_tier"] == "High"

    probs = data["class_probabilities"]
    assert pytest.approx(probs["prob_high"] + probs["prob_low"] + probs["prob_medium"], abs=1e-5) == 1.0
    assert probs["prob_high"] > 0.99
    assert data["probability_margin"] > 0.95
    assert data["is_boundary_case"] is False

    # AHP Baseline Context
    assert data["deterministic_ahp_tier"] == "High"
    assert data["ahp_agreement_status"] == "FULL_AGREEMENT"

    # Lineage and Disclaimer
    assert data["model_artifact_name"] == "village-preparedness-classifier"
    assert data["model_artifact_version"] == "v0"
    assert data["target_nature"] == "AHP_CAPACITY_TIER_SURROGATE"
    assert "OPERATIONAL NOTICE" in data["disclaimer"]
    assert "Deterministic AHP assessments maintain absolute operational precedence" in data["disclaimer"]


def test_predict_by_pc11_id_success(client: TestClient):
    """Verify inference by 6-digit PC11 ID (Bellare village)."""
    payload = {"area_id": "617756"}  # Bellare
    resp = client.post("/api/v1/model/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["village_name"] == "Bellare"
    assert data["predicted_preparedness_tier"] == "High"
    probs = data["class_probabilities"]
    assert pytest.approx(probs["prob_high"] + probs["prob_low"] + probs["prob_medium"], abs=1e-5) == 1.0


def test_predict_by_area_id_not_found(client: TestClient):
    """Verify 404 response when querying an unknown area identifier."""
    payload = {"area_id": "nonexistent_village_id_99999"}
    resp = client.post("/api/v1/model/predict", json=payload)
    assert resp.status_code == 404
    assert "not found in unified preparedness dataset" in resp.json()["detail"]


# =============================================================================
# 6. INFERENCE ENDPOINT: FEATURE PAYLOAD
# =============================================================================

def test_predict_by_precomputed_26_features_payload(client: TestClient, df_modelling):
    """Verify inference by supplying all 26 canonical features directly."""
    sample_26 = df_modelling[CANONICAL_26_FEATURES].iloc[0].to_dict()
    payload = {"features": sample_26}

    resp = client.post("/api/v1/model/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["area_id"] is None
    assert data["predicted_preparedness_tier"] in ["High", "Low", "Medium"]
    probs = data["class_probabilities"]
    assert pytest.approx(probs["prob_high"] + probs["prob_low"] + probs["prob_medium"], abs=1e-5) == 1.0
    assert data["model_artifact_version"] == "v0"


def test_predict_by_raw_indicators_payload(client: TestClient, df_unified):
    """Verify inference by supplying raw village indicators (automatic 9-feature engineering)."""
    raw_sample = df_unified.iloc[5].to_dict()
    # Filter to only necessary raw fields to ensure no precomputed 9 features are present
    raw_keys = set(SEVENTEEN_BASE_FEATURES).union(
        ["population", "pop_density_sqkm", "dist_nearest_shelter_km", "dist_nearest_hospital_km",
         "shelter_capacity_nearby", "cvi_integrated", "dist_nearest_gauge_km"]
    )
    raw_payload = {k: float(raw_sample[k]) for k in raw_keys}

    resp = client.post("/api/v1/model/predict", json={"features": raw_payload})
    assert resp.status_code == 200
    data = resp.json()

    assert data["predicted_preparedness_tier"] in ["High", "Low", "Medium"]
    probs = data["class_probabilities"]
    assert pytest.approx(probs["prob_high"] + probs["prob_low"] + probs["prob_medium"], abs=1e-5) == 1.0


def test_predict_empty_payload_validation_422(client: TestClient):
    """Verify 422 error when neither area_id nor features is provided."""
    resp = client.post("/api/v1/model/predict", json={})
    assert resp.status_code == 422
    assert "Either 'area_id' or 'features' dictionary must be provided" in resp.text


def test_predict_invalid_feature_nan_returns_422(client: TestClient, df_modelling):
    """Verify 422 error when features contain NaN."""
    sample_26 = df_modelling[CANONICAL_26_FEATURES].iloc[0].to_dict()
    sample_26["elevation_mean"] = None  # serialized as null/NaN

    resp = client.post("/api/v1/model/predict", json={"features": sample_26})
    assert resp.status_code == 422
    assert "elevation_mean" in resp.text


# =============================================================================
# 7. BOUNDARY CASE DETECTION & AMBIGUITY FLAGS
# =============================================================================

def test_predict_boundary_case_ambiguity_flag(client: TestClient, df_unified):
    """Verify is_boundary_case flag is correctly computed when probability margin is narrow."""
    # Find a village in unified dataset where margin < 0.10 or max_prob < 0.60
    probs = df_unified[["prob_high", "prob_low", "prob_medium"]].to_numpy()
    sorted_p = np.sort(probs, axis=1)
    boundary_mask = (sorted_p[:, -1] - sorted_p[:, -2] < 0.10) | (sorted_p[:, -1] < 0.60)
    assert boundary_mask.any(), "Expected at least one boundary case village in dataset"

    boundary_row = df_unified[boundary_mask].iloc[0]
    resp = client.post("/api/v1/model/predict", json={"area_id": str(boundary_row["area_id"])})
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_boundary_case"] is True
    # Verify definition: margin < 0.10 OR max_prob < 0.60
    assert data["probability_margin"] < 0.10 or data["max_probability"] < 0.60


# =============================================================================
# 8. AHP PRECEDENCE & DISCORDANCE SAFEGUARDS
# =============================================================================

def test_ahp_precedence_on_low_tier_village(client: TestClient, df_unified):
    """Verify that when an AHP Low-tier village is queried, ML cannot downgrade priority."""
    # Find an AHP Low tier village
    low_villages = df_unified[df_unified["preparedness_tier"] == "Low"]
    assert not low_villages.empty

    low_row = low_villages.iloc[0]
    resp = client.post("/api/v1/model/predict", json={"area_id": str(low_row["area_id"])})
    assert resp.status_code == 200
    data = resp.json()

    assert data["deterministic_ahp_tier"] == "Low"

    # If ML predicted Medium or High, verify precedence mandate note is attached
    if data["predicted_preparedness_tier"] != "Low":
        assert data["ahp_agreement_status"] == "BORDERLINE_DISAGREEMENT"
        assert "AHP PRECEDENCE MANDATE" in data["ahp_precedence_note"]
        assert "ML model prediction cannot downgrade emergency priority" in data["ahp_precedence_note"]


def test_decision_support_unit_evaluation_preserves_band1_critical(client: TestClient, df_unified):
    """Verify existing decision support endpoint preserves BAND_1_CRITICAL for Low AHP tier."""
    low_villages = df_unified[df_unified["preparedness_tier"] == "Low"]
    low_id = str(low_villages.iloc[0]["pc11_id"])

    resp = client.get(f"/api/v1/decision-support/units/{low_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["preparedness_tier"] == "Low"
    assert data["provisional_priority_band"] == "BAND_1_CRITICAL"


# =============================================================================
# 9. EXISTING ENDPOINTS REGRESSION TEST
# =============================================================================

def test_existing_endpoints_unaffected(client: TestClient):
    """Verify all existing decision support, scenario, and archetype endpoints remain intact."""
    # 1. /api/v1/units/search
    search_resp = client.get("/api/v1/units/search?query=Tallur")
    assert search_resp.status_code == 200
    assert len(search_resp.json()) > 0

    # 2. /api/v1/prioritization/scenario
    scenario_resp = client.post(
        "/api/v1/prioritization/scenario",
        json={"weights": {"healthcare": 0.4, "demographic": 0.2, "infrastructure": 0.2, "coastal": 0.2}},
    )
    assert scenario_resp.status_code == 200

    # 3. /api/v1/archetypes
    archetypes_resp = client.get("/api/v1/archetypes")
    assert archetypes_resp.status_code == 200
    assert len(archetypes_resp.json()["archetypes"]) == 4

    # 4. /api/v1/decision-support/summary
    summary_resp = client.get("/api/v1/decision-support/summary")
    assert summary_resp.status_code == 200
    assert summary_resp.json()["total_units"] == 1879
