"""Milestone 2B: Offline probability inference and prediction provenance generator.

Loads the validated serialized pipeline (models/logistic_regression_tier.joblib)
and label encoder (models/label_encoder.joblib), executes inference on the verified
26-feature matrix, explicitly maps probability columns using estimator classes_,
and tags each prediction with in-sample vs out-of-sample provenance.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict_preparedness_probs")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CYCLONE_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
MODEL_DIR = Path(os.environ.get("CYCLONE_MODEL_DIR", PROJECT_ROOT / "models"))

DEFAULT_MODELLING_PATH = DATA_DIR / "modelling_dataset.parquet"
DEFAULT_MODEL_PATH = MODEL_DIR / "logistic_regression_tier.joblib"
DEFAULT_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "logistic_regression_metadata.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "village_preparedness_probabilities.parquet"

PROBABILITY_DISCLAIMER = (
    "DISCLAIMER: Predicted class probabilities (prob_low, prob_medium, prob_high) represent "
    "multinomial logistic regression likelihoods modeling structural preparedness tiers derived from "
    "the AHP/MCDA framework. They are NOT empirically calibrated disaster-risk probabilities, "
    "storm-surge damage forecasts, or physical casualty estimates."
)


def generate_preparedness_probabilities(
    modelling_path: Path = DEFAULT_MODELLING_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    encoder_path: Path = DEFAULT_ENCODER_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Generate and validate class probabilities with full prediction provenance.

    Args:
        modelling_path: Path to modelling_dataset.parquet.
        model_path: Path to serialized Pipeline joblib.
        encoder_path: Path to serialized LabelEncoder joblib.
        metadata_path: Path to model metadata JSON.
        output_path: Path to write village_preparedness_probabilities.parquet.
        save_csv: Whether to write companion CSV.

    Returns:
        DataFrame containing predicted tiers, probabilities, and provenance.
    """
    logger.info(f"Loading pipeline from {model_path}...")
    pipeline = joblib.load(model_path)

    logger.info(f"Loading label encoder from {encoder_path}...")
    le = joblib.load(encoder_path)

    logger.info(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names: List[str] = meta["features"]
    split_indices: Dict[str, List[int]] = meta["split_indices"]

    logger.info(f"Loading feature dataset from {modelling_path}...")
    df = pd.read_parquet(modelling_path)

    # 1. Feature verification and extraction
    X = df[feature_names].copy()
    if X.isna().any().any():
        raise ValueError("Missing values found in feature matrix X")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite values found in feature matrix X")

    # 2. Probability inference and prediction
    logger.info(f"Executing predict() and predict_proba() across {len(X)} villages...")
    raw_preds = pipeline.predict(X)
    raw_probs = pipeline.predict_proba(X)

    # 3. Explicit probability column mapping via model.classes_ and LabelEncoder
    model_classes = pipeline.named_steps["model"].classes_
    logger.info(f"Model internal classes_: {model_classes}")
    logger.info(f"LabelEncoder classes_: {list(le.classes_)}")

    # Map each class string to its corresponding probability column index
    prob_cols = {}
    for col_idx, class_val in enumerate(model_classes):
        class_name = le.inverse_transform([class_val])[0]
        prob_cols[class_name.lower()] = raw_probs[:, col_idx]

    # Validate required class names exist
    for tier in ["low", "medium", "high"]:
        if tier not in prob_cols:
            raise KeyError(f"Expected class probability '{tier}' not found in model outputs: {list(prob_cols.keys())}")

    # 4. Numerical validation of probabilities
    for tier, prob_arr in prob_cols.items():
        if not np.isfinite(prob_arr).all():
            raise ValueError(f"Non-finite probabilities detected for tier '{tier}'")
        if (prob_arr < 0.0).any() or (prob_arr > 1.0).any():
            raise ValueError(f"Probabilities out of [0, 1] range for tier '{tier}'")

    prob_sums = prob_cols["low"] + prob_cols["medium"] + prob_cols["high"]
    if not np.allclose(prob_sums, 1.0, atol=1e-5):
        raise ValueError(f"Probability simplex violation! Sums range from {prob_sums.min():.6f} to {prob_sums.max():.6f}")

    # 5. Inverse transform predictions to canonical string labels
    predicted_tiers = le.inverse_transform(raw_preds)

    # 6. Prediction Provenance Tracking (In-Sample vs Out-of-Sample)
    train_set = set(split_indices["train"])
    val_set = set(split_indices["val"])
    test_set = set(split_indices["test"])

    provenance = []
    for idx in range(len(df)):
        if idx in train_set:
            provenance.append("TRAIN_IN_SAMPLE")
        elif idx in val_set:
            provenance.append("VAL_OUT_OF_SAMPLE")
        elif idx in test_set:
            provenance.append("TEST_OUT_OF_SAMPLE")
        else:
            provenance.append("UNTRACKED_SPLIT")

    # 7. Construct Output DataFrame
    results_df = pd.DataFrame(
        {
            "shrid2": df["area_id"],
            "area_id": df["area_id"],
            "predicted_preparedness_tier": predicted_tiers,
            "prob_low": np.round(prob_cols["low"], 4),
            "prob_medium": np.round(prob_cols["medium"], 4),
            "prob_high": np.round(prob_cols["high"], 4),
            "prediction_provenance": provenance,
            "actual_preparedness_tier": df["preparedness_tier"],
            "probability_disclaimer": PROBABILITY_DISCLAIMER,
        }
    )

    # Provenance summary log
    prov_counts = results_df["prediction_provenance"].value_counts().to_dict()
    logger.info(f"Prediction provenance breakdown: {prov_counts}")

    # In-sample vs out-of-sample accuracy check
    train_acc = (results_df.loc[results_df["prediction_provenance"] == "TRAIN_IN_SAMPLE", "predicted_preparedness_tier"] == results_df.loc[results_df["prediction_provenance"] == "TRAIN_IN_SAMPLE", "actual_preparedness_tier"]).mean()
    val_acc = (results_df.loc[results_df["prediction_provenance"] == "VAL_OUT_OF_SAMPLE", "predicted_preparedness_tier"] == results_df.loc[results_df["prediction_provenance"] == "VAL_OUT_OF_SAMPLE", "actual_preparedness_tier"]).mean()
    test_acc = (results_df.loc[results_df["prediction_provenance"] == "TEST_OUT_OF_SAMPLE", "predicted_preparedness_tier"] == results_df.loc[results_df["prediction_provenance"] == "TEST_OUT_OF_SAMPLE", "actual_preparedness_tier"]).mean()
    overall_acc = (results_df["predicted_preparedness_tier"] == results_df["actual_preparedness_tier"]).mean()

    logger.info(f"Accuracy audit: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}, Overall={overall_acc:.4f}")

    # 8. Persist Parquet and CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing probabilities dataset ({len(results_df)} rows) to {output_path}...")
    results_df.to_parquet(output_path, index=False)

    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        logger.info(f"Writing companion CSV to {csv_path}...")
        results_df.to_csv(csv_path, index=False)

    logger.info("Probability inference and provenance generation completed successfully.")
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Preparedness Class Probabilities.")
    parser.add_argument(
        "--modelling-path",
        type=Path,
        default=DEFAULT_MODELLING_PATH,
        help="Path to modelling_dataset.parquet",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to serialized pipeline joblib",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output probabilities parquet",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing companion CSV copy",
    )
    args = parser.parse_args()

    generate_preparedness_probabilities(
        modelling_path=args.modelling_path,
        model_path=args.model_path,
        output_path=args.output_path,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
