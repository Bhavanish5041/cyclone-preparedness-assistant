"""Model Loading and Inference Operational Smoke Test Script.
=========================================================

Verifies operational readiness and runtime functionality in CI/CD (Job 3):
  1. Model artifact files exist on disk.
  2. Model objects load successfully into memory.
  3. Preprocessing scalers (RobustScaler / StandardScaler) load and bind.
  4. Expected feature schemas are present and validated.
  5. One valid inference executes successfully for each model.
  6. Invalid inference inputs are rejected correctly with expected exceptions.
  7. Predicted output values and probabilities are strictly valid and finite.
  8. Operational functionality demonstrated without depending on external services.

Exit code 0 = All operational smoke tests PASSED.
Exit code 1 = Operational failure detected.
"""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LR_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_tier.joblib"
LR_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
LR_META_PATH = PROJECT_ROOT / "models" / "logistic_regression_metadata.json"
KM_MODEL_PATH = PROJECT_ROOT / "models" / "kmeans_k4_model.joblib"
KM_SCALER_PATH = PROJECT_ROOT / "models" / "kmeans_scaler.joblib"
KM_META_PATH = PROJECT_ROOT / "models" / "kmeans_metadata.json"

FAIL = False


def log_check(name: str, passed: bool, detail: str = "") -> None:
    global FAIL
    icon = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {status:4s} | {name}" + (f" — {detail}" if detail else ""))
    if not passed:
        FAIL = True


def test_logistic_regression_smoke() -> None:
    print("\n" + "=" * 65)
    print("OPERATIONAL SMOKE TEST: LOGISTIC REGRESSION SURROGATE")
    print("=" * 65)

    # 1. Artifact presence
    log_check("Artifact file exists: logistic_regression_tier.joblib", LR_MODEL_PATH.exists())
    log_check("Encoder file exists: label_encoder.joblib", LR_ENCODER_PATH.exists())
    log_check("Metadata file exists: logistic_regression_metadata.json", LR_META_PATH.exists())

    if not (LR_MODEL_PATH.exists() and LR_ENCODER_PATH.exists() and LR_META_PATH.exists()):
        return

    # 2. Loading
    try:
        pipeline = joblib.load(LR_MODEL_PATH)
        le = joblib.load(LR_ENCODER_PATH)
        with open(LR_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        log_check("Pipeline and encoder load successfully into memory", True)
    except Exception as e:
        log_check("Pipeline and encoder load into memory", False, str(e))
        return

    # 3. Scaler binding & feature schema
    has_scaler = "scaler" in pipeline.named_steps
    scaler_type = pipeline.named_steps["scaler"].__class__.__name__ if has_scaler else None
    is_valid_scaler = has_scaler and scaler_type == "RobustScaler"
    log_check("Correct scaler loaded with model (RobustScaler)", is_valid_scaler, f"scaler={scaler_type}")

    features = meta.get("features", [])
    log_check("Expected 26-feature schema present in metadata", len(features) == 26, f"count={len(features)}")

    classes = list(le.classes_)
    log_check("Target classes complete (High, Low, Medium)", set(classes) == {"High", "Low", "Medium"}, f"classes={classes}")

    # 4. Valid inference
    try:
        sample_x = pd.DataFrame([np.zeros(26)], columns=features)
        raw_pred = pipeline.predict(sample_x)[0]
        label = le.inverse_transform([raw_pred])[0]
        log_check("Valid inference succeeds", label in classes, f"predicted_tier={label}")

        probs = pipeline.predict_proba(sample_x)[0]
        finite = bool(np.all(np.isfinite(probs)))
        prob_sum = float(np.sum(probs))
        log_check("Output probabilities finite and sum to 1.0", finite and abs(prob_sum - 1.0) < 1e-5, f"sum={prob_sum:.6f}")
    except Exception as e:
        log_check("Valid inference succeeds", False, str(e))

    # 5. Invalid inference rejected
    try:
        incomplete_x = pd.DataFrame([np.zeros(10)], columns=features[:10])
        pipeline.predict(incomplete_x)
        log_check("Invalid feature schema rejected", False, "Failed to reject 10-feature input")
    except ValueError:
        log_check("Invalid inference input rejected correctly (ValueError)", True, "Incomplete feature schema caught")
    except Exception as e:
        log_check("Invalid inference input rejected correctly", True, f"Caught: {type(e).__name__}")


def test_kmeans_smoke() -> None:
    print("\n" + "=" * 65)
    print("OPERATIONAL SMOKE TEST: K-MEANS VULNERABILITY CLUSTERING")
    print("=" * 65)

    # 1. Artifact presence
    log_check("Artifact file exists: kmeans_k4_model.joblib", KM_MODEL_PATH.exists())
    log_check("Scaler file exists: kmeans_scaler.joblib", KM_SCALER_PATH.exists())
    log_check("Metadata file exists: kmeans_metadata.json", KM_META_PATH.exists())

    if not (KM_MODEL_PATH.exists() and KM_SCALER_PATH.exists() and KM_META_PATH.exists()):
        return

    # 2. Loading
    try:
        km_model = joblib.load(KM_MODEL_PATH)
        scaler = joblib.load(KM_SCALER_PATH)
        with open(KM_META_PATH, "r", encoding="utf-8") as f:
            km_meta = json.load(f)
        log_check("K-Means model and scaler load successfully", True)
    except Exception as e:
        log_check("K-Means model and scaler load", False, str(e))
        return

    # 3. Scaler binding & feature schema
    log_check("Model n_clusters is exactly 4", getattr(km_model, "n_clusters", None) == 4, f"k={km_model.n_clusters}")
    log_check("Cluster centers shape is (4, 4)", getattr(km_model, "cluster_centers_", np.zeros((1, 1))).shape == (4, 4))
    log_check("Scaler mean shape matches 4 dimensions", getattr(scaler, "mean_", np.zeros(1)).shape == (4,))

    km_features = km_meta.get("features", [])
    log_check("Expected 4-feature schema present in metadata", len(km_features) == 4, f"features={km_features}")

    # 4. Valid inference
    try:
        sample_x = np.array([[0.3, 0.4, 0.5, 0.6]])
        scaled_x = scaler.transform(sample_x)
        c_id = int(km_model.predict(scaled_x)[0])
        log_check("Valid clustering inference succeeds", c_id in [0, 1, 2, 3], f"cluster_id={c_id}")
    except Exception as e:
        log_check("Valid clustering inference succeeds", False, str(e))

    # 5. Invalid inference rejected
    try:
        bad_dim_x = np.array([[0.5, 0.5]])  # 2 features instead of 4
        scaler.transform(bad_dim_x)
        log_check("Invalid feature count rejected", False, "Failed to reject 2-feature vector")
    except ValueError:
        log_check("Invalid feature dimension rejected correctly (ValueError)", True, "2-feature input caught")
    except Exception as e:
        log_check("Invalid feature dimension rejected correctly", True, f"Caught: {type(e).__name__}")


def main():
    print("=" * 65)
    print("CYCLONE INTEL — CI/CD MODEL ARTIFACT OPERATIONAL SMOKE TESTS")
    print("=" * 65)

    test_logistic_regression_smoke()
    test_kmeans_smoke()

    print("\n" + "=" * 65)
    if FAIL:
        print("RESULT: OPERATIONAL SMOKE TEST FAILED — See details above.")
        print("=" * 65)
        sys.exit(1)
    else:
        print("RESULT: ALL OPERATIONAL SMOKE TESTS PASSED [MODELS READY]")
        print("=" * 65)
        sys.exit(0)


if __name__ == "__main__":
    main()
