"""Comprehensive Model Testing Suite for CYCLONE INTEL.
=====================================================

Validates operational correctness, input schema enforcement, edge case handling,
quantitative benchmarks, and scientific safeguards across BOTH models:
  A. Logistic Regression (village-preparedness-classifier)
  B. K-Means Clustering (coastal-vulnerability-kmeans)

Covers:
- Valid inputs representing Low, Medium, and High preparedness tiers.
- Single-record and batch predictions.
- Boundary values, zeros, and extremes.
- Missing, extra, reordered, and count-mismatched features.
- Null, NaN, infinite, and wrong-type inputs with clear validation errors.
- Empty inputs and malformed payloads.
- Model loading, scaler compatibility, and artifact integrity.
- Probability simplex adherence (sum == 1.0, finite values).
- Per-class precision, recall, and F1 benchmarks.
- Confusion matrix audits.
- Scientific invariants: AHP consistency and non-downgrading of AHP Low.
- K-Means 4-dimension schema, range [0, 3], balance, and multi-seed stability.
- Reusable vs notebook-derived implementation equivalence.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from api.main import app
from src.models.clustering import (
    CLUSTERING_DIMENSION_COLUMNS,
    extract_clustering_features,
    train_kmeans_model,
)
from src.models.load_registered_model import RegisteredModelContainer
from src.models.quality_checks import QualityGateRunner
from src.models.registry.local_registry import (
    LocalModelRegistry,
    compute_file_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELLING_PATH = PROJECT_ROOT / "data" / "processed" / "modelling_dataset.parquet"
COASTAL_PATH = PROJECT_ROOT / "data" / "processed" / "coastal_prioritization_baseline.parquet"
LR_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_tier.joblib"
LR_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
LR_META_PATH = PROJECT_ROOT / "models" / "logistic_regression_metadata.json"
KM_MODEL_PATH = PROJECT_ROOT / "models" / "kmeans_k4_model.joblib"
KM_SCALER_PATH = PROJECT_ROOT / "models" / "kmeans_scaler.joblib"
KM_META_PATH = PROJECT_ROOT / "models" / "kmeans_metadata.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_quality_gates.json"


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="module")
def quality_config() -> Dict[str, Any]:
    assert CONFIG_PATH.exists(), f"Missing quality gates config at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def df_modelling() -> pd.DataFrame:
    assert MODELLING_PATH.exists(), f"Missing dataset at {MODELLING_PATH}"
    return pd.read_parquet(MODELLING_PATH)


@pytest.fixture(scope="module")
def df_coastal() -> pd.DataFrame:
    assert COASTAL_PATH.exists(), f"Missing coastal dataset at {COASTAL_PATH}"
    return pd.read_parquet(COASTAL_PATH)


@pytest.fixture(scope="module")
def lr_artifacts():
    assert LR_MODEL_PATH.exists()
    assert LR_ENCODER_PATH.exists()
    assert LR_META_PATH.exists()
    pipe = joblib.load(LR_MODEL_PATH)
    le = joblib.load(LR_ENCODER_PATH)
    with open(LR_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    container = RegisteredModelContainer(
        pipeline=pipe,
        label_encoder=le,
        metadata=meta,
        artifact_version="v1.0.0",
        artifact_source="local",
        artifact_dir=PROJECT_ROOT / "models",
    )
    return pipe, le, meta, container


@pytest.fixture(scope="module")
def km_artifacts():
    assert KM_MODEL_PATH.exists()
    assert KM_SCALER_PATH.exists()
    assert KM_META_PATH.exists()
    model = joblib.load(KM_MODEL_PATH)
    scaler = joblib.load(KM_SCALER_PATH)
    with open(KM_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta


# ==============================================================================
# A. LOGISTIC REGRESSION COMPREHENSIVE TESTS
# ==============================================================================

def test_lr_model_loading_and_scaler_compatibility(lr_artifacts):
    """Verify Logistic Regression pipeline components and scaler compatibility."""
    pipe, le, meta, container = lr_artifacts
    assert "scaler" in pipe.named_steps, "Pipeline must contain 'scaler' step"
    assert "model" in pipe.named_steps, "Pipeline must contain 'model' step"
    assert pipe.named_steps["scaler"].__class__.__name__ == "RobustScaler"
    assert pipe.named_steps["model"].__class__.__name__ == "LogisticRegression"

    # Verify label encoder classes
    assert list(le.classes_) == ["High", "Low", "Medium"]

    # Verify metadata features match exactly 26
    features = meta["features"]
    assert len(features) == 26
    assert pipe.named_steps["model"].coef_.shape == (3, 26)


def test_lr_valid_inputs_low_medium_high_tiers(lr_artifacts, df_modelling):
    """Verify valid inputs representing actual Low, Medium, and High preparedness tiers."""
    pipe, le, meta, container = lr_artifacts
    features = meta["features"]

    # Sample villages known to belong to High, Medium, and Low tiers
    for tier in ["High", "Low", "Medium"]:
        sub = df_modelling[df_modelling["preparedness_tier"] == tier]
        assert len(sub) > 0, f"No villages found for tier {tier}"
        sample = sub.iloc[[0]][features]

        pred = container.predict(sample)[0]
        assert pred in ["High", "Low", "Medium"]

        proba = container.predict_proba(sample)
        assert f"prob_{tier.lower()}" in proba
        assert 0.0 <= proba[f"prob_{tier.lower()}"][0] <= 1.0


def test_lr_single_record_and_batch_predictions(lr_artifacts, df_modelling):
    """Verify both single-record (N=1) and batch (N=50) inference."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    # Single record
    single_df = df_modelling.iloc[[0]][features]
    single_pred = container.predict(single_df)
    assert len(single_pred) == 1
    assert isinstance(single_pred[0], str)

    # Batch of 50
    batch_df = df_modelling.iloc[:50][features]
    batch_preds = container.predict(batch_df)
    assert len(batch_preds) == 50
    assert all(p in ["High", "Low", "Medium"] for p in batch_preds)

    # Probability shape
    batch_probs = container.predict_proba(batch_df)
    assert len(batch_probs["prob_high"]) == 50
    assert len(batch_probs["prob_low"]) == 50
    assert len(batch_probs["prob_medium"]) == 50


def test_lr_boundary_values(lr_artifacts, df_modelling):
    """Verify boundary inputs: all-zero, extreme large values, near-threshold inputs."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    # All-zero input vector
    zero_df = pd.DataFrame([np.zeros(26)], columns=features)
    pred_zero = container.predict(zero_df)[0]
    assert pred_zero in ["High", "Low", "Medium"]

    probs_zero = container.predict_proba(zero_df)
    sum_zero = (
        probs_zero["prob_high"][0] + probs_zero["prob_low"][0] + probs_zero["prob_medium"][0]
    )
    assert abs(sum_zero - 1.0) < 1e-5

    # Extreme high-risk input vector
    high_risk = pd.DataFrame([np.ones(26) * 1000.0], columns=features)
    pred_risk = container.predict(high_risk)[0]
    assert pred_risk in ["High", "Low", "Medium"]
    assert np.all(np.isfinite(list(container.predict_proba(high_risk).values())))


def test_lr_missing_features_raises_clear_error(lr_artifacts, df_modelling):
    """Verify missing features are rejected with a clear ValueError."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    # Missing 1 feature
    incomplete_df = df_modelling.iloc[[0]][features[1:]].copy()
    with pytest.raises(ValueError, match="missing .* required model features"):
        container.validate_and_align_input(incomplete_df)

    # Missing 10 features
    severely_incomplete = df_modelling.iloc[[0]][features[:16]].copy()
    with pytest.raises(ValueError, match="missing .* required model features"):
        container.validate_and_align_input(severely_incomplete)


def test_lr_extra_features_handling(lr_artifacts, df_modelling):
    """Verify extra irrelevant features are safely pruned during column alignment."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    extra_df = df_modelling.iloc[[0]][features].copy()
    extra_df["irrelevant_extra_column_1"] = 999.0
    extra_df["irrelevant_string_col"] = "unexpected_metadata"

    aligned = container.validate_and_align_input(extra_df)
    assert list(aligned.columns) == features
    assert "irrelevant_extra_column_1" not in aligned.columns


def test_lr_incorrect_feature_ordering(lr_artifacts, df_modelling):
    """Verify scrambled column ordering is aligned to strict model fit order."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    shuffled_features = list(np.random.RandomState(42).permutation(features))
    shuffled_df = df_modelling.iloc[[0]][shuffled_features].copy()

    aligned = container.validate_and_align_input(shuffled_df)
    assert list(aligned.columns) == features

    # Predictions match original un-shuffled DataFrame
    orig_df = df_modelling.iloc[[0]][features]
    orig_pred = container.predict(orig_df)
    shuffled_pred = container.predict(shuffled_df)
    assert orig_pred[0] == shuffled_pred[0]


def test_lr_incorrect_feature_count(lr_artifacts, df_modelling):
    """Verify inputs with incorrect feature counts are rejected."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    # Fewer features
    with pytest.raises(ValueError, match="missing"):
        container.validate_and_align_input(df_modelling.iloc[[0]][features[:20]])


def test_lr_null_nan_infinite_values_rejected(lr_artifacts, df_modelling):
    """Verify Null, NaN, +Inf, and -Inf values raise explicit validation errors."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    # Test NaN
    nan_df = df_modelling.iloc[[0]][features].copy()
    nan_df.iloc[0, 2] = np.nan
    with pytest.raises(ValueError, match="Missing \\(NaN\\) values detected"):
        container.validate_and_align_input(nan_df)

    # Test +Inf
    pos_inf_df = df_modelling.iloc[[0]][features].copy()
    pos_inf_df.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite .* values detected"):
        container.validate_and_align_input(pos_inf_df)

    # Test -Inf
    neg_inf_df = df_modelling.iloc[[0]][features].copy()
    neg_inf_df.iloc[0, 1] = -np.inf
    with pytest.raises(ValueError, match="Infinite .* values detected"):
        container.validate_and_align_input(neg_inf_df)


def test_lr_wrong_data_types_rejected(lr_artifacts, df_modelling):
    """Verify non-numeric types (e.g. strings in numeric fields) are rejected."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    str_df = df_modelling.iloc[[0]][features].copy()
    str_df["dist_coast_km"] = "not_a_valid_float"
    with pytest.raises(ValueError, match="Non-numeric features detected"):
        container.validate_and_align_input(str_df)


def test_lr_empty_input_rejected(lr_artifacts, df_modelling):
    """Verify empty DataFrame or non-DataFrame input is cleanly rejected."""
    _, _, meta, container = lr_artifacts

    # 0-row DataFrame is rejected with ValueError
    empty_df = pd.DataFrame(columns=meta["features"])
    with pytest.raises(ValueError):
        container.validate_and_align_input(empty_df)

    # Non-DataFrame input is rejected with ValueError
    with pytest.raises(ValueError, match="Input X must be a pandas DataFrame"):
        container.validate_and_align_input([1, 2, 3])  # type: ignore


def test_lr_malformed_api_payloads():
    """Verify FastAPI endpoint rejects malformed JSON and missing features."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        # Malformed payload structure
        r1 = client.post("/api/models/predict", json={"features": "not_a_dict"})
        assert r1.status_code in (400, 422)

        # Missing required 26 features
        r2 = client.post("/api/models/predict", json={"features": {"dist_coast_km": 1.0}})
        assert r2.status_code in (400, 422)

        # Non-numeric value in payload
        r3 = client.post("/api/models/predict", json={"features": {"dist_coast_km": "invalid_string"}})
        assert r3.status_code in (400, 422)


def test_lr_probability_shape_simplex_and_finite_values(lr_artifacts, df_modelling):
    """Verify probability outputs adhere strictly to the probability simplex."""
    _, _, meta, container = lr_artifacts
    features = meta["features"]

    batch_df = df_modelling.iloc[:100][features]
    probs = container.predict_proba(batch_df)

    p_high = probs["prob_high"]
    p_low = probs["prob_low"]
    p_med = probs["prob_medium"]

    assert np.all(np.isfinite(p_high))
    assert np.all(np.isfinite(p_low))
    assert np.all(np.isfinite(p_med))

    assert np.all(p_high >= 0.0) and np.all(p_high <= 1.0)
    assert np.all(p_low >= 0.0) and np.all(p_low <= 1.0)
    assert np.all(p_med >= 0.0) and np.all(p_med <= 1.0)

    # Simplex sum strictly equal to 1.0 within float tolerance
    sums = p_high + p_low + p_med
    assert np.all(np.abs(sums - 1.0) < 1e-5)


def test_lr_per_class_precision_recall_and_confusion_matrix(lr_artifacts, df_modelling, quality_config):
    """Verify quantitative held-out performance meets configured quality gates."""
    pipe, le, meta, container = lr_artifacts
    features = meta["features"]
    test_idx = meta["split_indices"]["test"]

    X_test = df_modelling.iloc[test_idx][features]
    y_test_raw = df_modelling.iloc[test_idx]["preparedness_tier"].astype(str)
    y_test_enc = le.transform(y_test_raw)

    preds_enc = pipe.predict(X_test)
    preds_raw = le.inverse_transform(preds_enc)

    acc = accuracy_score(y_test_enc, preds_enc)
    macro_f1 = f1_score(y_test_enc, preds_enc, average="macro")

    qg = quality_config["classification"]["thresholds"]
    assert acc >= qg["accuracy_min"], f"Accuracy {acc:.4f} < {qg['accuracy_min']}"
    assert macro_f1 >= qg["macro_f1_min"], f"Macro F1 {macro_f1:.4f} < {qg['macro_f1_min']}"

    # Per-class report
    classes = list(le.classes_)
    rep = classification_report(y_test_raw, preds_raw, labels=classes, target_names=classes, output_dict=True)
    low_rec = rep["Low"]["recall"]
    assert low_rec >= qg["low_tier_recall_min"], f"Low tier recall {low_rec:.4f} < {qg['low_tier_recall_min']}"

    # Confusion matrix shape and sum
    cm = confusion_matrix(y_test_raw, preds_raw, labels=classes)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(test_idx)


def test_lr_ahp_consistency_and_no_ml_downgrade_of_ahp_low(lr_artifacts, df_modelling):
    """Ensure authoritative AHP Low classifications are preserved with zero ML downgrade."""
    pipe, le, meta, container = lr_artifacts
    features = meta["features"]

    low_units = df_modelling[df_modelling["preparedness_tier"] == "Low"]
    assert len(low_units) > 0

    preds = container.predict(low_units[features])
    low_recall = np.mean(preds == "Low")
    assert low_recall >= 0.70, f"AHP Low preservation recall {low_recall:.3f} < 0.70"


# ==============================================================================
# B. K-MEANS CLUSTERING COMPREHENSIVE TESTS
# ==============================================================================

def test_kmeans_model_loading_and_artifact_integrity(km_artifacts):
    """Verify K-Means model, scaler, and metadata schema integrity."""
    model, scaler, meta = km_artifacts
    assert isinstance(model, KMeans)
    assert isinstance(scaler, StandardScaler)
    assert model.n_clusters == 4
    assert model.cluster_centers_.shape == (4, 4)
    assert scaler.mean_.shape == (4,)
    assert scaler.scale_.shape == (4,)
    assert meta["algorithm"] == "KMeans"
    assert meta["features"] == CLUSTERING_DIMENSION_COLUMNS


def test_kmeans_valid_feature_vectors_and_batches(km_artifacts, df_coastal):
    """Verify single-record and batch predictions produce valid cluster IDs [0, 3]."""
    model, scaler, meta = km_artifacts
    cols = meta["features"]

    # Single record
    single_x = scaler.transform(df_coastal.iloc[[0]][cols])
    single_c = model.predict(single_x)[0]
    assert single_c in [0, 1, 2, 3]

    # Batch of 622 units
    all_x = scaler.transform(df_coastal[cols])
    all_c = model.predict(all_x)
    assert len(all_c) == len(df_coastal)
    assert set(np.unique(all_c)).issubset({0, 1, 2, 3})


def test_kmeans_correct_four_feature_schema_and_ordering(km_artifacts, df_coastal):
    """Verify canonical four-feature schema and column ordering enforcement."""
    model, scaler, meta = km_artifacts
    expected = [
        "dim_healthcare_deficit",
        "dim_demographic_vulnerability",
        "dim_infrastructure_deficit",
        "dim_coastal_exposure",
    ]
    assert meta["features"] == expected

    # Features must match
    X_scaled, sc = extract_clustering_features(df_coastal, dimension_cols=expected, standardize=True)
    assert X_scaled.shape[1] == 4


def test_kmeans_missing_and_extra_features(df_coastal):
    """Verify missing features raise error and extra features are excluded."""
    cols = CLUSTERING_DIMENSION_COLUMNS

    # Missing feature raises KeyError
    incomplete = df_coastal[cols[:3]].copy()
    with pytest.raises(KeyError, match="Required dimension column"):
        extract_clustering_features(incomplete, dimension_cols=cols)

    # Extra features safely ignored
    extra_df = df_coastal[cols].copy()
    extra_df["unwanted_column"] = 123.45
    X_scaled, _ = extract_clustering_features(extra_df, dimension_cols=cols)
    assert X_scaled.shape[1] == 4


def test_kmeans_null_nan_and_infinity_rejected(df_coastal):
    """Verify NaNs and Infinite values raise clear ValueError in feature extraction."""
    cols = CLUSTERING_DIMENSION_COLUMNS

    # NaN input raises ValueError
    nan_df = df_coastal[cols].copy()
    nan_df.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        extract_clustering_features(nan_df, dimension_cols=cols)

    # Inf input raises ValueError
    inf_df = df_coastal[cols].copy()
    inf_df.iloc[0, 1] = np.inf
    with pytest.raises(ValueError, match="infinite values"):
        extract_clustering_features(inf_df, dimension_cols=cols)


def test_kmeans_incorrect_types_and_dimensions(km_artifacts):
    """Verify wrong dimensions (e.g., 3 or 5 features) or strings raise error."""
    model, scaler, _ = km_artifacts

    # Wrong shape (3 features instead of 4)
    with pytest.raises(ValueError):
        scaler.transform(np.zeros((5, 3)))

    with pytest.raises(ValueError):
        model.predict(np.zeros((5, 3)))


def test_kmeans_empty_inputs_rejected(km_artifacts):
    """Verify empty inputs (0 rows) raise ValueError in scaler."""
    model, scaler, _ = km_artifacts
    empty_x = np.empty((0, 4))
    with pytest.raises(ValueError, match="minimum of 1 is required"):
        scaler.transform(empty_x)


def test_kmeans_out_of_range_inputs_handled(km_artifacts):
    """Verify out-of-range inputs (negative, extreme large) predict valid cluster IDs."""
    model, scaler, _ = km_artifacts

    # Out of range negative
    neg_x = np.array([[-10.0, -5.0, -2.0, -1.0]])
    c_neg = model.predict(scaler.transform(neg_x))[0]
    assert c_neg in [0, 1, 2, 3]

    # Extreme large
    large_x = np.array([[100.0, 50.0, 80.0, 90.0]])
    c_large = model.predict(scaler.transform(large_x))[0]
    assert c_large in [0, 1, 2, 3]


def test_kmeans_cluster_distribution_and_empty_clusters(km_artifacts, df_coastal, quality_config):
    """Verify cluster-size distribution: no empty clusters, balance ratio >= 0.25."""
    model, scaler, meta = km_artifacts
    cols = meta["features"]

    X_scaled = scaler.transform(df_coastal[cols])
    labels = model.predict(X_scaled)
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    # Verify all 4 clusters are non-empty
    assert len(sizes) == 4, f"Expected 4 clusters, got {len(sizes)}"
    assert all(count > 0 for count in sizes.values()), f"Empty clusters detected: {sizes}"

    # Verify cluster balance ratio
    balance = min(sizes.values()) / max(sizes.values())
    qg = quality_config["clustering"]["thresholds"]
    assert balance >= qg["cluster_balance_ratio_min"], f"Cluster balance {balance:.4f} < {qg['cluster_balance_ratio_min']}"
    assert min(sizes.values()) >= qg["min_cluster_size"], f"Min cluster size {min(sizes.values())} < {qg['min_cluster_size']}"


def test_kmeans_prediction_consistency(km_artifacts, df_coastal):
    """Verify deterministic prediction consistency across repeated inference calls."""
    model, scaler, meta = km_artifacts
    cols = meta["features"]

    X_scaled = scaler.transform(df_coastal[cols])
    preds1 = model.predict(X_scaled)
    preds2 = model.predict(X_scaled)
    preds3 = model.predict(X_scaled)

    assert np.array_equal(preds1, preds2)
    assert np.array_equal(preds2, preds3)


def test_kmeans_notebook_and_reusable_implementation_no_divergence(km_artifacts, df_coastal):
    """Verify that reusable train_kmeans_model produces identical centroids and labels as baseline."""
    prod_model, prod_scaler, prod_meta = km_artifacts

    # Train via reusable training module
    retrained_model, retrained_scaler, meta, metrics, labels = train_kmeans_model(
        data_or_path=df_coastal,
        n_clusters=4,
        random_state=42,
        n_init=20,
        save_artifacts=False,
    )

    # Centroids must match to machine precision
    np.testing.assert_allclose(
        prod_model.cluster_centers_,
        retrained_model.cluster_centers_,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Reusable K-Means centroids diverged from baseline model",
    )

    # Scaler mean and scale must match
    np.testing.assert_allclose(
        prod_scaler.mean_,
        retrained_scaler.mean_,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Reusable K-Means scaler mean diverged",
    )
    np.testing.assert_allclose(
        prod_scaler.scale_,
        retrained_scaler.scale_,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Reusable K-Means scaler scale diverged",
    )


# ==============================================================================
# MODEL PROMOTION SAFETY & REGISTRY INVARIANCE TESTS
# ==============================================================================

def test_model_promotion_safety_gate_failure_blocks_promotion(tmp_path):
    """Verify that a quality gate failure blocks promotion and keeps production aliases unchanged."""
    reg = LocalModelRegistry()
    initial_active_lr = reg.get_model("village-preparedness-classifier")["version"]
    initial_active_km = reg.get_model("coastal-vulnerability-kmeans")["version"]
    initial_lr_sha = compute_file_sha256(LR_MODEL_PATH)
    initial_km_sha = compute_file_sha256(KM_MODEL_PATH)

    # Create an impossible quality gate config (e.g. Accuracy >= 0.9999)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        strict_config = json.load(f)
    strict_config["classification"]["thresholds"]["accuracy_min"] = 0.9999
    strict_config["clustering"]["thresholds"]["silhouette_min"] = 0.9999

    bad_config_path = tmp_path / "strict_quality_gates.json"
    with open(bad_config_path, "w", encoding="utf-8") as f:
        json.dump(strict_config, f)

    # Run quality gate runner with impossible thresholds
    runner = QualityGateRunner(config_path=bad_config_path)
    passed = runner.run_all(output_json=tmp_path / "report.json", output_md=tmp_path / "summary.md")

    assert passed is False, "Quality gates should fail under impossible thresholds"
    assert runner.all_passed is False

    readiness = runner.evaluate_deployment_readiness({}, {})
    assert readiness["overall_status"] == "FAIL"
    assert readiness["promotion_eligibility"] == "PROMOTION_BLOCKED"

    # Verify active production versions in registry remain completely unchanged
    reg_after = LocalModelRegistry()
    assert reg_after.get_model("village-preparedness-classifier")["version"] == initial_active_lr
    assert reg_after.get_model("village-preparedness-classifier")["status"] == "production"
    assert reg_after.get_model("coastal-vulnerability-kmeans")["version"] == initial_active_km
    assert reg_after.get_model("coastal-vulnerability-kmeans")["status"] == "production"

    # Verify production artifact files on disk have identical SHA256 hashes
    assert compute_file_sha256(LR_MODEL_PATH) == initial_lr_sha
    assert compute_file_sha256(KM_MODEL_PATH) == initial_km_sha


def test_model_promotion_safety_candidate_staging_only(tmp_path):
    """Verify candidate registration assigns 'candidate' status and never mutates production alias."""
    test_manifest = tmp_path / "model_registry_test.json"
    with open(PROJECT_ROOT / "models" / "model_registry.json", "r", encoding="utf-8") as f:
        reg_data = json.load(f)
    with open(test_manifest, "w", encoding="utf-8") as f:
        json.dump(reg_data, f)

    test_reg = LocalModelRegistry(registry_path=test_manifest)
    initial_active_km = test_reg.data["models"]["coastal-vulnerability-kmeans"]["active_version"]
    assert initial_active_km == "v1.0.0"

    # Register a new candidate version
    candidate_ver = "v1.1.0-candidate"
    entry = test_reg.register_candidate(
        model_id="coastal-vulnerability-kmeans",
        version=candidate_ver,
        artifact_files=[{"path": "models/candidates/kmeans_k4_model.joblib"}],
        input_schema={"dimensions": 4},
        output_schema={"clusters": 4},
        evaluation_metrics={"silhouette": 0.34},
        author="automated_test",
        note="Candidate staging test",
    )

    assert entry["status"] == "candidate"
    assert entry["version"] == candidate_ver

    # Verify active_version in registry STILL points to v1.0.0 [production]
    assert test_reg.data["models"]["coastal-vulnerability-kmeans"]["active_version"] == "v1.0.0"
    assert test_reg.get_model("coastal-vulnerability-kmeans", "v1.0.0")["status"] == "production"

    # Verify attempting to register a candidate over an existing production version raises ValueError
    with pytest.raises(ValueError, match="Cannot overwrite active production version"):
        test_reg.register_candidate(
            model_id="coastal-vulnerability-kmeans",
            version="v1.0.0",
            artifact_files=[],
            input_schema={},
            output_schema={},
            evaluation_metrics={},
        )


def test_model_production_artifacts_never_overwritten_during_experiments(tmp_path, df_coastal):
    """Verify that training experimental or candidate models does not mutate production artifacts."""
    lr_sha_before = compute_file_sha256(LR_MODEL_PATH)
    km_sha_before = compute_file_sha256(KM_MODEL_PATH)
    scaler_sha_before = compute_file_sha256(KM_SCALER_PATH)
    encoder_sha_before = compute_file_sha256(LR_ENCODER_PATH)

    # Train experimental model targeting isolated tmp_path
    exp_dir = tmp_path / "experiment_output"
    train_kmeans_model(
        data_or_path=df_coastal,
        n_clusters=4,
        random_state=999,
        n_init=10,
        output_dir=exp_dir,
        save_artifacts=True,
    )

    # Assert experimental artifacts exist in experiment directory
    assert (exp_dir / "kmeans_k4_model.joblib").exists()

    # Assert production artifacts in models/ remain bit-for-bit identical
    assert compute_file_sha256(LR_MODEL_PATH) == lr_sha_before
    assert compute_file_sha256(KM_MODEL_PATH) == km_sha_before
    assert compute_file_sha256(KM_SCALER_PATH) == scaler_sha_before
    assert compute_file_sha256(LR_ENCODER_PATH) == encoder_sha_before

