"""Automated tests for Milestone 2B: Model reproduction, serialization, and probability inference."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "logistic_regression_tier.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METADATA_PATH = MODEL_DIR / "logistic_regression_metadata.json"
PROBABILITIES_PATH = PROJECT_ROOT / "data" / "processed" / "village_preparedness_probabilities.parquet"
MODELLING_PATH = PROJECT_ROOT / "data" / "processed" / "modelling_dataset.parquet"


@pytest.fixture(scope="module")
def pipeline():
    assert PIPELINE_PATH.exists(), f"Missing pipeline artifact at {PIPELINE_PATH}"
    return joblib.load(PIPELINE_PATH)


@pytest.fixture(scope="module")
def encoder():
    assert ENCODER_PATH.exists(), f"Missing encoder artifact at {ENCODER_PATH}"
    return joblib.load(ENCODER_PATH)


@pytest.fixture(scope="module")
def metadata():
    assert METADATA_PATH.exists(), f"Missing metadata file at {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def df_probs():
    assert PROBABILITIES_PATH.exists(), f"Missing probabilities dataset at {PROBABILITIES_PATH}"
    return pd.read_parquet(PROBABILITIES_PATH)


@pytest.fixture(scope="module")
def df_model():
    assert MODELLING_PATH.exists(), f"Missing modelling dataset at {MODELLING_PATH}"
    return pd.read_parquet(MODELLING_PATH)


def test_pipeline_structure(pipeline):
    """Verify that the pipeline contains both RobustScaler and LogisticRegression."""
    assert "scaler" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert hasattr(pipeline.named_steps["scaler"], "transform")
    assert hasattr(pipeline.named_steps["model"], "predict_proba")


def test_label_encoder_mapping(encoder):
    """Verify exact alphabetical class mapping of LabelEncoder."""
    classes = list(encoder.classes_)
    assert classes == ["High", "Low", "Medium"]
    assert encoder.transform(["High"])[0] == 0
    assert encoder.transform(["Low"])[0] == 1
    assert encoder.transform(["Medium"])[0] == 2


def test_metadata_fields_and_integrity(metadata):
    """Verify that metadata JSON includes all required auditing fields."""
    required_keys = [
        "model_artifact",
        "model_nature",
        "model_type",
        "scikit_learn_version",
        "features",
        "preprocessing",
        "hyperparameters",
        "split_configuration",
        "target",
        "validation_results",
        "split_indices",
    ]
    for key in required_keys:
        assert key in metadata, f"Metadata missing key: {key}"

    assert len(metadata["features"]) == 26
    assert metadata["hyperparameters"]["C"] == 1.0
    assert metadata["hyperparameters"]["solver"] == "lbfgs"
    assert metadata["split_configuration"]["train_size"] == 1315
    assert metadata["split_configuration"]["val_size"] == 282
    assert metadata["split_configuration"]["test_size"] == 282


def test_reproduced_metrics_match_baseline(metadata):
    """Assert reproduced metrics match recorded 70/15/15 test metrics with exact tolerance."""
    test_metrics = metadata["validation_results"]["test_metrics"]
    benchmark = metadata["validation_results"]["recorded_benchmark_metrics"]

    np.testing.assert_allclose(test_metrics["accuracy"], benchmark["test_accuracy"], atol=1e-4)
    np.testing.assert_allclose(test_metrics["macro_f1"], benchmark["test_macro_f1"], atol=1e-4)
    np.testing.assert_allclose(test_metrics["weighted_f1"], benchmark["test_weighted_f1"], atol=1e-4)


def test_probability_dataset_shape_and_keys(df_probs, df_model):
    """Verify probabilities dataset row count, uniqueness, and key alignment."""
    assert len(df_probs) == 1879
    assert df_probs["shrid2"].nunique() == 1879
    assert df_probs["shrid2"].isna().sum() == 0
    assert (df_probs["shrid2"] == df_model["area_id"]).all()


def test_probability_numerical_validity(df_probs):
    """Verify that probabilities are finite, in [0, 1], and sum to 1.0."""
    for tier in ["prob_low", "prob_medium", "prob_high"]:
        assert df_probs[tier].isna().sum() == 0
        assert np.isfinite(df_probs[tier]).all()
        assert (df_probs[tier] >= 0.0).all()
        assert (df_probs[tier] <= 1.0).all()

    # Sum to 1 within rounding tolerance
    sums = df_probs["prob_low"] + df_probs["prob_medium"] + df_probs["prob_high"]
    np.testing.assert_allclose(sums, 1.0, atol=1e-3)


def test_predicted_tiers_match_probability_argmax(df_probs, encoder):
    """Verify that predicted_preparedness_tier matches argmax of class probabilities."""
    for _, row in df_probs.head(100).iterrows():
        p_high = row["prob_high"]
        p_low = row["prob_low"]
        p_med = row["prob_medium"]
        max_tier = "High" if (p_high >= p_low and p_high >= p_med) else ("Low" if (p_low >= p_med) else "Medium")
        assert row["predicted_preparedness_tier"] == max_tier


def test_prediction_provenance_accounting(df_probs):
    """Verify exact in-sample vs out-of-sample partition counts."""
    counts = df_probs["prediction_provenance"].value_counts().to_dict()
    assert counts.get("TRAIN_IN_SAMPLE") == 1315
    assert counts.get("VAL_OUT_OF_SAMPLE") == 282
    assert counts.get("TEST_OUT_OF_SAMPLE") == 282
    assert sum(counts.values()) == 1879
