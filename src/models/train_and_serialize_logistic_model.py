"""Milestone 2B: Controlled reproduction, validation, and serialization of the Preparedness Logistic Regression Model.

Reconstructs the teammate's documented 70/15/15 stratified experiment using:
- Input dataset: cyclone-preparedness/data/processed/modelling_dataset.parquet
- Exact 26 MODEL_FEATURES
- Pipeline: RobustScaler -> LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=42)
- Split: train_test_split(test_size=0.15, val_size=0.15, random_state=42, stratify=y_cls_enc)

Validates test accuracy == 0.9433 and macro F1 == 0.9060 before saving artifacts.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("serialize_logistic_model")

# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELLING_PATH = Path(os.environ.get("CYCLONE_DATA_DIR", PROJECT_ROOT / "data" / "processed")) / "modelling_dataset.parquet"
DEFAULT_MODEL_DIR = Path(os.environ.get("CYCLONE_MODEL_DIR", PROJECT_ROOT / "models"))
PIPELINE_OUTPUT_PATH = DEFAULT_MODEL_DIR / "logistic_regression_tier.joblib"
ENCODER_OUTPUT_PATH = DEFAULT_MODEL_DIR / "label_encoder.joblib"
METADATA_OUTPUT_PATH = DEFAULT_MODEL_DIR / "logistic_regression_metadata.json"

# Exact 26 features verified in Milestone 2A audit
MODEL_FEATURES: List[str] = [
    "dist_coast_km",
    "elevation_mean",
    "avg_household_size",
    "sc_share",
    "st_share",
    "vulnerable_pop_share",
    "road_connectivity_score",
    "drinking_water_score",
    "n_hc_10km",
    "n_phc_chc_10km",
    "n_gauges_20km",
    "n_shelters_10km",
    "cvi_physical",
    "cvi_social",
    "has_pucca_road",
    "has_major_district_road",
    "bus_connectivity",
    "log1p_population",
    "log1p_pop_density_sqkm",
    "log1p_dist_nearest_shelter_km",
    "log1p_dist_nearest_hospital_km",
    "coastal_risk_score",
    "evacuation_sufficiency",
    "vulnerability_exposure",
    "infra_composite",
    "monitoring_coverage",
]

# Benchmark metrics from cyclone-preparedness/data/processed/model_results.csv (split_70_15_15)
EXPECTED_TEST_ACCURACY = 0.943262
EXPECTED_TEST_MACRO_F1 = 0.906023
EXPECTED_TEST_WEIGHTED_F1 = 0.942247


def make_70_15_15_splits(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the exact 70/15/15 stratified train/val/test split from 02_Baseline_Models.ipynb.

    Args:
        X: Feature DataFrame of shape (1879, 26).
        y_enc: Encoded target vector of shape (1879,).
        random_state: Random seed (42).

    Returns:
        Tuple of (idx_train, idx_val, idx_test, y_train, y_val, y_test).
    """
    indices = np.arange(len(X))
    idx_tmp, idx_test, y_tmp, y_test = train_test_split(
        indices, y_enc, test_size=0.15, stratify=y_enc, random_state=random_state
    )
    val_frac = 0.15 / (1.0 - 0.15)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=random_state
    )
    return idx_train, idx_val, idx_test, y_train, y_val, y_test


def train_validate_and_serialize(
    modelling_path: Path = DEFAULT_MODELLING_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> Dict[str, Any]:
    """Execute reproducible model training, metric verification, and serialization."""
    logger.info(f"Loading modeling dataset from {modelling_path}...")
    df = pd.read_parquet(modelling_path)

    # 1. Feature verification
    for feat in MODEL_FEATURES:
        if feat not in df.columns:
            raise KeyError(f"Required model feature '{feat}' missing from modelling dataset")
    if len(MODEL_FEATURES) != 26:
        raise ValueError(f"Expected exactly 26 features, got {len(MODEL_FEATURES)}")

    X = df[MODEL_FEATURES].copy()
    if X.isna().any().any():
        raise ValueError("Missing values detected in feature matrix X")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite values detected in feature matrix X")

    # 2. Target encoding
    y_raw = df["preparedness_tier"].astype(str).copy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    class_mapping = {str(cls_name): int(idx) for idx, cls_name in enumerate(le.classes_)}
    logger.info(f"Class mapping: {class_mapping}")

    # 3. Splits reproduction
    idx_train, idx_val, idx_test, y_train, y_val, y_test = make_70_15_15_splits(X, y_enc, random_state=42)
    logger.info(f"Splits sizes: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

    X_train = X.iloc[idx_train]
    X_val = X.iloc[idx_val]
    X_test = X.iloc[idx_test]

    # 4. Pipeline construction
    # Note: scikit-learn >= 1.8 automatically uses multinomial cross-entropy with solver='lbfgs'
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )

    logger.info("Fitting RobustScaler + LogisticRegression on 70% train split (1,315 villages)...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate splits
    def compute_metrics(x_sub: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        y_p = pipeline.predict(x_sub)
        return {
            "accuracy": float(accuracy_score(y_true, y_p)),
            "macro_f1": float(f1_score(y_true, y_p, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_p, average="weighted")),
        }

    train_metrics = compute_metrics(X_train, y_train)
    val_metrics = compute_metrics(X_val, y_val)
    test_metrics = compute_metrics(X_test, y_test)

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Val metrics:   {val_metrics}")
    logger.info(f"Test metrics:  {test_metrics}")

    # 6. Verify metrics parity against recorded results
    tol = 1e-4
    acc_diff = abs(test_metrics["accuracy"] - EXPECTED_TEST_ACCURACY)
    f1_diff = abs(test_metrics["macro_f1"] - EXPECTED_TEST_MACRO_F1)
    wtd_diff = abs(test_metrics["weighted_f1"] - EXPECTED_TEST_WEIGHTED_F1)

    logger.info(f"Metric deltas: Accuracy Δ={acc_diff:.6f}, Macro F1 Δ={f1_diff:.6f}, Weighted F1 Δ={wtd_diff:.6f}")
    if acc_diff > tol or f1_diff > tol or wtd_diff > tol:
        raise RuntimeError(
            f"Reproduction verification failed! Reproduced metrics differ from recorded baseline:\n"
            f"Test Accuracy: {test_metrics['accuracy']:.6f} vs Expected {EXPECTED_TEST_ACCURACY}\n"
            f"Test Macro F1: {test_metrics['macro_f1']:.6f} vs Expected {EXPECTED_TEST_MACRO_F1}"
        )
    logger.info("VERIFICATION PASSED: Reproduced metrics match recorded 70/15/15 baseline with 100% parity.")

    # 7. Detailed test metrics breakdown
    pred_test = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, pred_test).tolist()
    cr = classification_report(y_test, pred_test, target_names=list(le.classes_), output_dict=True)

    # 8. Serialize pipeline and encoder
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = model_dir / "logistic_regression_tier.joblib"
    encoder_path = model_dir / "label_encoder.joblib"
    metadata_path = model_dir / "logistic_regression_metadata.json"

    logger.info(f"Serializing fitted pipeline to {joblib_path}...")
    joblib.dump(pipeline, joblib_path)

    logger.info(f"Serializing label encoder to {encoder_path}...")
    joblib.dump(le, encoder_path)

    # 9. Structure comprehensive metadata
    metadata = {
        "model_artifact": "logistic_regression_tier.joblib",
        "model_nature": (
            "REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction "
            "of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. "
            "It is NOT the original unpersisted in-memory estimator."
        ),
        "model_type": "sklearn.pipeline.Pipeline (RobustScaler + LogisticRegression)",
        "framework": "scikit-learn",
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_dataset_source": str(modelling_path),
        "total_dataset_rows": len(df),
        "n_features": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "feature_order_is_strict": True,
        "preprocessing": {
            "scaler": "sklearn.preprocessing.RobustScaler",
            "with_centering": True,
            "with_scaling": True,
            "quantile_range": [25.0, 75.0],
        },
        "hyperparameters": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
            "random_state": 42,
            "multi_class": "multinomial (automatic in scikit-learn >= 1.8)",
        },
        "split_configuration": {
            "split_name": "split_70_15_15",
            "strategy": "Stratified train_test_split",
            "random_state": 42,
            "train_size": len(idx_train),
            "val_size": len(idx_val),
            "test_size": len(idx_test),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "target": {
            "target_variable": "preparedness_tier",
            "target_nature": "Empirical AHP/MCDA capacity tier (NOT calibrated physical hazard/casualty risk)",
            "classes": list(le.classes_),
            "class_encoding": class_mapping,
            "probability_column_mapping": {
                f"probs[:, {idx}]": f"prob_{cls_name.lower()}"
                for idx, cls_name in enumerate(le.classes_)
            },
        },
        "validation_results": {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "recorded_benchmark_metrics": {
                "test_accuracy": EXPECTED_TEST_ACCURACY,
                "test_macro_f1": EXPECTED_TEST_MACRO_F1,
                "test_weighted_f1": EXPECTED_TEST_WEIGHTED_F1,
            },
            "metric_discrepancy": {
                "accuracy_delta": acc_diff,
                "macro_f1_delta": f1_diff,
                "weighted_f1_delta": wtd_diff,
            },
            "test_confusion_matrix": {
                "classes": list(le.classes_),
                "matrix": cm,
            },
            "test_classification_report": cr,
        },
        "split_indices": {
            "train": idx_train.tolist(),
            "val": idx_val.tolist(),
            "test": idx_test.tolist(),
        },
    }

    logger.info(f"Writing metadata to {metadata_path}...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model reproduction, validation, and serialization successfully completed.")
    return metadata


def main() -> None:
    train_validate_and_serialize()


if __name__ == "__main__":
    main()
