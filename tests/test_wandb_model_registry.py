"""Unit and integration tests for Weights & Biases Model Registry and Versioning (Phase 2.3).

Ensures:
- Reproducible artifact packaging with the 7 required model and documentation files.
- Sanitization of bulky split index data from registered metadata.
- Strict feature schema preservation (26 features, canonical ordering, categorical/numerical grouping).
- Evaluation summary and operational alerts preservation.
- Model card and limitations documentation containing clear AHP-tier surrogate boundaries.
- Strict version specification enforcement (rejection of unversioned addresses).
- Strict prohibition of 'production' alias assignment prior to operational governance sign-off.
- Resilient error handling for network, authentication, and artifact corruption failures.
- Robust input schema validation (missing columns, NaNs, Infs, non-numeric types).
- Invariance to arbitrary input column ordering.
- Complete prediction and probability equivalence with the local baseline model across held-out test data.
- Offline execution capability with zero dependency on active internet connection or live credentials.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.load_registered_model import (
    AHP_TARGET_MEANING,
    CATEGORICAL_FEATURES,
    DEFAULT_ARTIFACT_NAME,
    TARGET_CLASSES,
    RegisteredModelContainer,
    WandbRetrievalError,
    build_artifact_staging_package,
    load_registered_model,
)
from scripts.register_model_wandb import register_model_artifact

# File paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = LOCAL_MODEL_DIR / "logistic_regression_tier.joblib"
ENCODER_PATH = LOCAL_MODEL_DIR / "label_encoder.joblib"
METADATA_PATH = LOCAL_MODEL_DIR / "logistic_regression_metadata.json"
MODELLING_PATH = PROJECT_ROOT / "data" / "processed" / "modelling_dataset.parquet"


@pytest.fixture(scope="module")
def metadata():
    assert METADATA_PATH.exists(), f"Missing metadata at {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def df_modelling():
    assert MODELLING_PATH.exists(), f"Missing dataset at {MODELLING_PATH}"
    return pd.read_parquet(MODELLING_PATH)


@pytest.fixture(scope="module")
def test_split_data(metadata, df_modelling):
    test_indices = metadata.get("split_indices", {}).get("test", [])
    assert len(test_indices) == 282, f"Expected 282 test indices, found {len(test_indices)}"
    features = metadata["features"]
    X_test = df_modelling.iloc[test_indices][features].copy()
    y_test = df_modelling.iloc[test_indices]["preparedness_tier"].copy()
    return X_test, y_test


@pytest.fixture(scope="module")
def staged_package(tmp_path_factory):
    staging_dir = tmp_path_factory.mktemp("staged_artifact")
    build_artifact_staging_package(model_dir=LOCAL_MODEL_DIR, staging_dir=staging_dir)
    return staging_dir


# ==============================================================================
# 1. ARTIFACT PACKAGING & SANITIZATION TESTS
# ==============================================================================

def test_build_artifact_staging_package_contents(staged_package):
    """Verify all 7 standardized files exist in the staged package."""
    expected_files = [
        "logistic_regression_tier.joblib",
        "label_encoder.joblib",
        "logistic_regression_metadata.json",
        "feature_schema.json",
        "evaluation_summary.json",
        "model_card.md",
        "intended_use_and_limitations.md",
    ]
    for filename in expected_files:
        staged_file = staged_package / filename
        assert staged_file.exists(), f"Missing expected file in bundle: {filename}"
        assert staged_file.stat().st_size > 0, f"File {filename} is empty"


def test_build_artifact_staging_package_sanitizes_split_indices(staged_package, metadata):
    """Verify bulky split_indices are removed from metadata to keep artifact lean."""
    # Source metadata has split_indices
    assert "split_indices" in metadata

    staged_meta_path = staged_package / "logistic_regression_metadata.json"
    with open(staged_meta_path, "r", encoding="utf-8") as f:
        staged_meta = json.load(f)

    assert "split_indices" not in staged_meta, "Bulky split_indices should be stripped from registered metadata"
    assert staged_meta["features"] == metadata["features"]
    assert staged_meta["model_type"] == metadata["model_type"]


def test_build_artifact_staging_package_missing_source_files(tmp_path):
    """Verify FileNotFoundError is raised if any primary model file is missing."""
    empty_src = tmp_path / "empty_models"
    empty_src.mkdir()
    target_dir = tmp_path / "staged"

    with pytest.raises(FileNotFoundError, match="Missing required model pipeline"):
        build_artifact_staging_package(model_dir=empty_src, staging_dir=target_dir)


# ==============================================================================
# 2. FEATURE SCHEMA INTEGRITY TESTS
# ==============================================================================

def test_feature_schema_integrity(staged_package, metadata):
    """Verify 26 features, canonical ordering, and categorical/numerical feature sets."""
    schema_path = staged_package / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["features_count"] == 26
    assert schema["feature_order_is_strict"] is True
    assert schema["feature_names_canonical_order"] == metadata["features"]
    assert schema["categorical_features"] == CATEGORICAL_FEATURES
    assert len(schema["numerical_features"]) == 23
    assert set(schema["categorical_features"]).union(set(schema["numerical_features"])) == set(metadata["features"])

    target = schema["target"]
    assert target["name"] == "preparedness_tier"
    assert target["classes"] == TARGET_CLASSES
    assert target["class_encoding"] == {"High": 0, "Low": 1, "Medium": 2}


# ==============================================================================
# 3. EVALUATION SUMMARY & OPERATIONAL ALERTS TESTS
# ==============================================================================

def test_evaluation_summary_metrics(staged_package):
    """Verify performance metrics and low-tier recall operational alert."""
    eval_path = staged_package / "evaluation_summary.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_summary = json.load(f)

    assert eval_summary["sample_size_test"] == 282
    overall = eval_summary["overall_performance"]
    assert pytest.approx(overall["accuracy"], abs=1e-3) == 0.9433
    assert pytest.approx(overall["macro_f1"], abs=1e-3) == 0.9060

    per_class = eval_summary["per_class_performance"]
    assert pytest.approx(per_class["Low"]["recall"], abs=1e-3) == 0.7391

    alerts = eval_summary["operational_alerts"]
    assert any("73.91%" in alert for alert in alerts)
    assert any("Low Preparedness Tier recall" in alert for alert in alerts)


# ==============================================================================
# 4. MODEL CARD & LIMITATIONS OPERATIONAL BOUNDARY TESTS
# ==============================================================================

def test_model_card_and_limitations_operational_language(staged_package):
    """Verify mandatory operational boundary notices in documentation artifacts."""
    model_card = (staged_package / "model_card.md").read_text(encoding="utf-8")
    limitations = (staged_package / "intended_use_and_limitations.md").read_text(encoding="utf-8")

    # Model Card checks
    assert "AHP-derived village preparedness tiers" in model_card
    assert "Deterministic AHP assessments hold operational precedence" in model_card
    assert "NOT physical cyclone-risk or hazard prediction" in model_card

    # Limitations checks
    assert "AHP-Tier Surrogate (Not Casualty/Hazard Risk)" in limitations
    assert "Deterministic AHP Precedence" in limitations
    assert "Low-Tier Recall Caveat (73.91%)" in limitations
    assert "Boundary Case Ambiguity" in limitations


# ==============================================================================
# 5. STRICT VERSION SELECTION & ALIAS SAFETY TESTS
# ==============================================================================

def test_load_registered_model_rejects_unversioned_address():
    """Verify rejection of artifact address without explicit version or alias tag."""
    with pytest.raises(ValueError, match="lacks an explicit version or alias tag"):
        load_registered_model("village-preparedness-classifier")

    with pytest.raises(ValueError, match="non-empty string"):
        load_registered_model("")

    with pytest.raises(ValueError, match="non-empty string"):
        load_registered_model(None)  # type: ignore


def test_register_model_artifact_forbids_production_alias():
    """Verify assigning 'production' alias is strictly forbidden in Phase 2.3."""
    with pytest.raises(ValueError, match="Forbidden alias 'production'"):
        register_model_artifact(aliases=["production"])

    with pytest.raises(ValueError, match="Forbidden alias 'production'"):
        register_model_artifact(aliases=["candidate", "production"])


# ==============================================================================
# 6. LOCAL CACHE LOADING & ARTIFACT RECOVERY TESTS
# ==============================================================================

def test_load_registered_model_from_cache(staged_package, tmp_path):
    """Verify loading from local cached directory without network access."""
    # Setup cache directory matching expected format
    cache_dir = tmp_path / "test_cache"
    model_address = "bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:candidate"
    cached_dest = cache_dir / model_address.replace("/", "_").replace(":", "_")
    cached_dest.mkdir(parents=True)

    # Copy staged files to cache destination
    for item in staged_package.iterdir():
        if item.is_file():
            (cached_dest / item.name).write_bytes(item.read_bytes())

    container = load_registered_model(
        artifact_address=model_address,
        cache_dir=cache_dir,
    )

    assert isinstance(container, RegisteredModelContainer)
    assert len(container.features) == 26
    assert container.classes_ == ["High", "Low", "Medium"]
    assert container.artifact_version == "candidate"
    assert container.artifact_source == model_address


def test_load_registered_model_corrupted_artifact_error(tmp_path):
    """Verify ValueError is raised if downloaded artifact is missing required files."""
    cache_dir = tmp_path / "corrupt_cache"
    model_address = "test-entity/test-project/test-model:v1"

    # Mock wandb artifact download that creates an incomplete set of files
    mock_artifact = MagicMock()
    mock_artifact.version = "v1"

    def fake_download(root):
        dest = Path(root)
        dest.mkdir(parents=True, exist_ok=True)
        # Omit label_encoder.joblib intentionally
        (dest / "logistic_regression_tier.joblib").write_bytes(b"dummy_data")
        (dest / "logistic_regression_metadata.json").write_text("{}", encoding="utf-8")

    mock_artifact.download.side_effect = fake_download
    mock_api = MagicMock()
    mock_api.artifact.return_value = mock_artifact

    with pytest.raises(ValueError, match="Corrupted or incomplete artifact"):
        load_registered_model(
            artifact_address=model_address,
            cache_dir=cache_dir,
            api=mock_api,
        )


def test_load_registered_model_corrupted_metadata_json(staged_package, tmp_path):
    """Verify ValueError is raised if metadata JSON is malformed."""
    cache_dir = tmp_path / "bad_meta_cache"
    model_address = "test-entity/test-project/test-model:v2"
    cached_dest = cache_dir / model_address.replace("/", "_").replace(":", "_")
    cached_dest.mkdir(parents=True)

    (cached_dest / "logistic_regression_tier.joblib").write_bytes(
        (staged_package / "logistic_regression_tier.joblib").read_bytes()
    )
    (cached_dest / "label_encoder.joblib").write_bytes(
        (staged_package / "label_encoder.joblib").read_bytes()
    )
    (cached_dest / "logistic_regression_metadata.json").write_text("NOT_VALID_JSON{{{", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse artifact metadata JSON"):
        load_registered_model(
            artifact_address=model_address,
            cache_dir=cache_dir,
        )


# ==============================================================================
# 7. W&B API FAILURE & NETWORK RESILIENCE TESTS
# ==============================================================================

def test_wandb_api_initialization_failure():
    """Verify WandbRetrievalError is raised when W&B Api client initialization fails."""
    with patch("wandb.Api", side_effect=Exception("WANDB_API_KEY not found / not authenticated")):
        with pytest.raises(WandbRetrievalError, match="Failed to initialize W&B API client"):
            load_registered_model(
                artifact_address="village-preparedness-classifier:v0",
                cache_dir=Path("/nonexistent/cache/dir"),
            )


def test_wandb_artifact_download_network_failure(tmp_path):
    """Verify WandbRetrievalError is raised when downloading artifact fails."""
    mock_api = MagicMock()
    mock_api.artifact.side_effect = Exception("HTTP 503: Service Unavailable / Connection refused")

    with pytest.raises(WandbRetrievalError, match="Failed to retrieve registered model artifact"):
        load_registered_model(
            artifact_address="bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:v0",
            cache_dir=tmp_path / "network_fail_cache",
            api=mock_api,
        )


# ==============================================================================
# 8. INPUT SCHEMA VALIDATION & COLUMN ALIGNMENT TESTS
# ==============================================================================

@pytest.fixture(scope="module")
def valid_container(staged_package, tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("valid_container_cache")
    model_address = "bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:candidate"
    cached_dest = cache_dir / model_address.replace("/", "_").replace(":", "_")
    cached_dest.mkdir(parents=True)
    for item in staged_package.iterdir():
        if item.is_file():
            (cached_dest / item.name).write_bytes(item.read_bytes())

    return load_registered_model(artifact_address=model_address, cache_dir=cache_dir)


def test_container_rejects_non_dataframe(valid_container):
    """Verify non-DataFrame input is rejected."""
    with pytest.raises(ValueError, match="Input X must be a pandas DataFrame"):
        valid_container.predict([[1.0] * 26])  # type: ignore


def test_container_rejects_missing_features(valid_container, test_split_data):
    """Verify missing features are detected and reported."""
    X_test, _ = test_split_data
    # Drop two real features from container's schema
    dropped = [valid_container.features[0], valid_container.features[5]]
    X_incomplete = X_test.drop(columns=dropped)

    with pytest.raises(ValueError, match="missing 2 required model features"):
        valid_container.predict(X_incomplete)


def test_container_rejects_non_numeric_types(valid_container, test_split_data):
    """Verify non-numeric column types are rejected."""
    X_test, _ = test_split_data
    X_invalid = X_test.copy()
    X_invalid["bus_connectivity"] = X_invalid["bus_connectivity"].astype(str) + "_str"

    with pytest.raises(ValueError, match="Non-numeric features detected"):
        valid_container.predict(X_invalid)


def test_container_rejects_nans(valid_container, test_split_data):
    """Verify NaN values are caught prior to inference."""
    X_test, _ = test_split_data
    X_invalid = X_test.copy()
    X_invalid.iloc[0, 3] = np.nan

    with pytest.raises(ValueError, match="Missing \\(NaN\\) values detected"):
        valid_container.predict(X_invalid)


def test_container_rejects_infinities(valid_container, test_split_data):
    """Verify infinite (+/-Inf) values are caught prior to inference."""
    X_test, _ = test_split_data
    X_invalid = X_test.copy()
    X_invalid.iloc[0, 5] = np.inf

    with pytest.raises(ValueError, match="Infinite \\(\\+/-Inf\\) values detected"):
        valid_container.predict(X_invalid)


def test_container_reordering_invariance(valid_container, test_split_data):
    """Verify container correctly re-aligns columns regardless of input column order."""
    X_test, _ = test_split_data
    # Reverse column order
    reversed_cols = list(reversed(X_test.columns))
    X_reversed = X_test[reversed_cols].copy()

    aligned = valid_container.validate_and_align_input(X_reversed)
    assert list(aligned.columns) == valid_container.features

    preds_original = valid_container.predict(X_test)
    preds_reversed = valid_container.predict(X_reversed)
    np.testing.assert_array_equal(preds_original, preds_reversed)


# ==============================================================================
# 9. INFERENCE CONTRACTS & PROBABILITY NORMALIZATION TESTS
# ==============================================================================

def test_container_predict_contract(valid_container, test_split_data):
    """Verify predict returns 1D array of string tier labels."""
    X_test, _ = test_split_data
    preds = valid_container.predict(X_test.iloc[:20])

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (20,)
    assert set(preds).issubset(set(TARGET_CLASSES))


def test_container_predict_proba_simplex(valid_container, test_split_data):
    """Verify predict_proba returns normalized probabilities strictly summing to 1.0."""
    X_test, _ = test_split_data
    probs_dict = valid_container.predict_proba(X_test)

    assert set(probs_dict.keys()) == {"prob_high", "prob_low", "prob_medium"}
    p_high = probs_dict["prob_high"]
    p_low = probs_dict["prob_low"]
    p_med = probs_dict["prob_medium"]

    assert len(p_high) == len(X_test)
    assert (p_high >= 0.0).all() and (p_high <= 1.0).all()
    assert (p_low >= 0.0).all() and (p_low <= 1.0).all()
    assert (p_med >= 0.0).all() and (p_med <= 1.0).all()

    # Simplex constraint check: sum of probabilities == 1.0
    row_sums = p_high + p_low + p_med
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_container_predict_detailed_contract(valid_container, test_split_data):
    """Verify predict_detailed produces rich DataFrame with boundary case detection."""
    X_test, _ = test_split_data
    df_detailed = valid_container.predict_detailed(X_test.iloc[:50])

    expected_cols = [
        "predicted_preparedness_tier",
        "prob_high",
        "prob_low",
        "prob_medium",
        "max_probability",
        "probability_margin",
        "is_boundary_case",
        "model_artifact_version",
        "retrieved_from",
        "target_nature",
    ]
    for col in expected_cols:
        assert col in df_detailed.columns, f"Missing expected column in detailed output: {col}"

    assert (df_detailed["target_nature"] == "AHP_CAPACITY_TIER_SURROGATE").all()
    assert df_detailed["is_boundary_case"].dtype == bool
    assert (df_detailed["probability_margin"] >= 0.0).all()


# ==============================================================================
# 10. PREDICTION EQUIVALENCE WITH LOCAL MODEL TESTS
# ==============================================================================

def test_prediction_equivalence_with_local_baseline(valid_container, test_split_data):
    """Verify 100% prediction match and exact probability equivalence on held-out test split."""
    X_test, _ = test_split_data

    # Direct local model inference
    local_pipeline = joblib.load(PIPELINE_PATH)
    local_encoder = joblib.load(ENCODER_PATH)

    local_preds = local_encoder.inverse_transform(local_pipeline.predict(X_test))
    local_probs = local_pipeline.predict_proba(X_test)

    # Registered container inference
    container_preds = valid_container.predict(X_test)
    container_probs_dict = valid_container.predict_proba(X_test)

    # 1. Exact match on predicted class labels
    np.testing.assert_array_equal(local_preds, container_preds)

    # 2. Map local probability columns by class name
    clf = local_pipeline.named_steps["model"]
    for col_idx, class_val in enumerate(clf.classes_):
        class_name = local_encoder.inverse_transform([class_val])[0].lower()
        key = f"prob_{class_name}"
        container_p = container_probs_dict[key]
        local_p = local_probs[:, col_idx]

        # Exact match within floating-point tolerance
        np.testing.assert_allclose(container_p, local_p, atol=1e-6)


# ==============================================================================
# 11. OFFLINE REGISTRATION WORKFLOW TESTS (MOCKED)
# ==============================================================================

def test_register_model_artifact_offline_mocked(tmp_path):
    """Verify register_model_artifact executes cleanly in offline mode with mocked W&B."""
    mock_run = MagicMock()
    mock_run.id = "mock_test_run_123"
    mock_run.url = "https://wandb.ai/mock/run"
    mock_run.log_artifact = MagicMock()
    mock_run.link_artifact = MagicMock()
    mock_run.finish = MagicMock()

    with patch("wandb.init", return_value=mock_run):
        result = register_model_artifact(
            project="test-project",
            entity="test-entity",
            artifact_name="test-classifier",
            collection_name="test-collection",
            model_dir=LOCAL_MODEL_DIR,
            mode="offline",
            aliases=["candidate", "validated"],
        )

        assert result["status"] == "success"
        assert result["run_id"] == "mock_test_run_123"
        assert result["aliases"] == ["candidate", "validated"]
        assert result["metrics"]["test_accuracy"] == pytest.approx(0.9433, abs=1e-3)
        assert mock_run.log_artifact.called
        # In offline mode, registry linking is gracefully skipped
        assert not mock_run.link_artifact.called
        assert result["registry_linked"] is False
        assert mock_run.finish.called


def test_register_model_artifact_online_mocked(tmp_path):
    """Verify register_model_artifact executes registry linking in online mode with mocked W&B."""
    mock_run = MagicMock()
    mock_run.id = "mock_online_run_456"
    mock_run.url = "https://wandb.ai/mock/online/run"
    mock_run.log_artifact = MagicMock()
    mock_run.link_artifact = MagicMock()
    mock_run.finish = MagicMock()

    with patch("src.tracking.wandb_logger.is_wandb_authenticated", return_value=True), \
         patch("scripts.register_model_wandb.is_wandb_authenticated", return_value=True):
        with patch("wandb.init", return_value=mock_run):
            result = register_model_artifact(
                project="test-project",
                entity="test-entity",
                artifact_name="test-classifier",
                collection_name="test-collection",
                model_dir=LOCAL_MODEL_DIR,
                mode="online",
                aliases=["candidate", "validated"],
            )

            assert result["status"] == "success"
            assert result["run_id"] == "mock_online_run_456"
            assert result["registry_linked"] is True
            assert mock_run.log_artifact.called
            assert mock_run.link_artifact.called
            assert mock_run.finish.called
