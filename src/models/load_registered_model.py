"""W&B Model Registry retrieval and registered inference module.

Provides production retrieval, validation, and inference wrapping for registered
Logistic Regression preparedness-tier classifier artifacts.

Design Principles:
- Strictly enforces explicit version/alias specification (disallows silent fallback to arbitrary latest).
- Validates 26-feature schema, ordering, and finite numeric values prior to inference.
- Provides probability mapping strictly mapped via model.classes_ and LabelEncoder.
- Handles network, authentication, and artifact corruption errors gracefully.
- Operates read-only on downloaded or local cached artifacts without modifying local model files.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import wandb

logger = logging.getLogger(__name__)

# Default canonical settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROJECT = "cyclone-preparedness-assistant"
DEFAULT_ARTIFACT_NAME = "village-preparedness-classifier"
DEFAULT_MODEL_DIR = Path(os.environ.get("CYCLONE_MODEL_DIR", PROJECT_ROOT / "models"))
DEFAULT_CACHE_DIR = Path(os.environ.get("WANDB_CACHE_DIR", PROJECT_ROOT / ".wandb_artifacts_cache"))

# Canonical feature groups
CATEGORICAL_FEATURES = ["has_pucca_road", "has_major_district_road", "bus_connectivity"]
TARGET_CLASSES = ["High", "Low", "Medium"]
AHP_TARGET_MEANING = (
    "AHP-derived village preparedness tiers (High, Medium, Low) reflecting composite physical, "
    "social, and infrastructural readiness. Deterministic AHP assessments serve as the baseline ground truth. "
    "Logistic Regression outputs provide calibrated tier probabilities for granular boundary case detection, "
    "NOT physical cyclone-risk or hazard prediction."
)


class WandbRetrievalError(RuntimeError):
    """Raised when artifact retrieval from W&B fails due to network, auth, or resolution errors."""

    pass


class RegisteredModelContainer:
    """Inference wrapper around a verified, retrieved W&B model artifact."""

    def __init__(
        self,
        pipeline: Any,
        label_encoder: Any,
        metadata: Dict[str, Any],
        artifact_version: str,
        artifact_source: str,
        artifact_dir: Path,
        feature_schema: Optional[Dict[str, Any]] = None,
    ):
        self.pipeline = pipeline
        self.label_encoder = label_encoder
        self.metadata = metadata
        self.artifact_version = artifact_version
        self.artifact_source = artifact_source
        self.artifact_dir = Path(artifact_dir)
        self.feature_schema = feature_schema or {}

        # Extract expected features
        self.features: List[str] = self.metadata.get("features", [])
        if len(self.features) != 26:
            raise ValueError(f"Expected 26 features in metadata, found {len(self.features)}")

        # Verify classes
        self.classes_: List[str] = list(self.label_encoder.classes_)
        for expected_cls in TARGET_CLASSES:
            if expected_cls not in self.classes_:
                raise ValueError(f"Expected target class '{expected_cls}' missing in label encoder: {self.classes_}")

        # Map internal estimator class values to string class names
        clf = self.pipeline.named_steps["model"]
        self.estimator_classes_ = clf.classes_
        self.class_index_map: Dict[str, int] = {}
        for col_idx, class_val in enumerate(self.estimator_classes_):
            class_name = self.label_encoder.inverse_transform([class_val])[0]
            self.class_index_map[class_name] = col_idx

    def validate_and_align_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input schema and align columns to exact 26-feature fit order.

        Args:
            X: Input DataFrame containing village features.

        Returns:
            DataFrame containing exactly the 26 features in strict fit order.

        Raises:
            ValueError: If required features are missing, non-numeric, or contain NaNs/Infs.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Input X must be a pandas DataFrame, got {type(X)}")

        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing {len(missing)} required model features: {missing[:5]}. "
                "Ensure candidate feature engineering has been applied prior to inference."
            )

        X_aligned = X[self.features].copy()

        # Check dtypes, NaNs, and Infs
        invalid_types = []
        nan_cols = []
        inf_cols = []

        for col in self.features:
            series = X_aligned[col]
            if not pd.api.types.is_numeric_dtype(series):
                invalid_types.append(col)
            n_nan = int(series.isna().sum())
            if n_nan > 0:
                nan_cols.append((col, n_nan))
            if pd.api.types.is_numeric_dtype(series):
                n_inf = int(np.isinf(series.to_numpy(dtype=float, na_value=0.0)).sum())
                if n_inf > 0:
                    inf_cols.append((col, n_inf))

        errors = []
        if invalid_types:
            errors.append(f"Non-numeric features detected: {invalid_types[:3]}")
        if nan_cols:
            errors.append(f"Missing (NaN) values detected: {nan_cols[:3]}")
        if inf_cols:
            errors.append(f"Infinite (+/-Inf) values detected: {inf_cols[:3]}")

        if errors:
            raise ValueError("Input schema validation failed: " + "; ".join(errors))

        return X_aligned

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict preparedness tier strings ('High', 'Low', 'Medium').

        Args:
            X: Input feature DataFrame.

        Returns:
            1D numpy array of predicted tier strings.
        """
        X_aligned = self.validate_and_align_input(X)
        raw_preds = self.pipeline.predict(X_aligned)
        return self.label_encoder.inverse_transform(raw_preds)

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute calibrated class probabilities strictly mapped to tier names.

        Args:
            X: Input feature DataFrame.

        Returns:
            Dictionary with keys 'prob_high', 'prob_low', 'prob_medium'.
        """
        X_aligned = self.validate_and_align_input(X)
        raw_probs = self.pipeline.predict_proba(X_aligned)

        # Normalize to strictly sum to 1.0 (prevent float precision rounding warnings)
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm_probs = raw_probs / row_sums

        results = {}
        for cls in TARGET_CLASSES:
            col_idx = self.class_index_map[cls]
            results[f"prob_{cls.lower()}"] = norm_probs[:, col_idx]

        return results

    def predict_detailed(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute predicted tiers, class probabilities, and boundary-case ambiguity flags.

        Args:
            X: Input feature DataFrame.

        Returns:
            DataFrame containing predictions, probabilities, margin, and boundary case flags.
        """
        X_aligned = self.validate_and_align_input(X)
        preds = self.predict(X_aligned)
        probs_dict = self.predict_proba(X_aligned)

        p_high = probs_dict["prob_high"]
        p_low = probs_dict["prob_low"]
        p_med = probs_dict["prob_medium"]

        prob_matrix = np.column_stack([p_high, p_low, p_med])
        sorted_probs = np.sort(prob_matrix, axis=1)
        max_prob = sorted_probs[:, -1]
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        is_boundary = (margin < 0.10) | (max_prob < 0.60)

        df_out = pd.DataFrame(
            {
                "predicted_preparedness_tier": preds,
                "prob_high": p_high,
                "prob_low": p_low,
                "prob_medium": p_med,
                "max_probability": max_prob,
                "probability_margin": margin,
                "is_boundary_case": is_boundary,
                "model_artifact_version": self.artifact_version,
                "retrieved_from": self.artifact_source,
                "target_nature": "AHP_CAPACITY_TIER_SURROGATE",
            },
            index=X.index,
        )
        return df_out


def build_artifact_staging_package(
    model_dir: Optional[Union[str, Path]] = None,
    staging_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Prepare the standardized 7-file model artifact bundle for W&B registration.

    Files staged:
    1. logistic_regression_tier.joblib
    2. label_encoder.joblib
    3. logistic_regression_metadata.json (sanitized: split indices removed)
    4. feature_schema.json
    5. evaluation_summary.json
    6. model_card.md
    7. intended_use_and_limitations.md

    Args:
        model_dir: Source directory containing local models (defaults to models/).
        staging_dir: Directory where bundle files will be staged.

    Returns:
        Path to the populated staging directory.
    """
    src_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    target_dir = Path(staging_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model pipeline
    pipeline_src = src_dir / "logistic_regression_tier.joblib"
    if not pipeline_src.exists():
        raise FileNotFoundError(f"Missing required model pipeline at {pipeline_src}")
    shutil.copy2(pipeline_src, target_dir / "logistic_regression_tier.joblib")

    # 2. Label encoder
    encoder_src = src_dir / "label_encoder.joblib"
    if not encoder_src.exists():
        raise FileNotFoundError(f"Missing required label encoder at {encoder_src}")
    shutil.copy2(encoder_src, target_dir / "label_encoder.joblib")

    # 3. Model metadata (sanitized of bulky raw split indices)
    meta_src = src_dir / "logistic_regression_metadata.json"
    if not meta_src.exists():
        raise FileNotFoundError(f"Missing required model metadata at {meta_src}")
    with open(meta_src, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    sanitized_meta = dict(meta_data)
    # Remove raw split index arrays if present to prevent uploading unnecessary row index data
    if "split_indices" in sanitized_meta:
        del sanitized_meta["split_indices"]

    with open(target_dir / "logistic_regression_metadata.json", "w", encoding="utf-8") as f:
        json.dump(sanitized_meta, f, indent=2)

    # 4. Feature schema JSON
    features = meta_data["features"]
    schema_dict = {
        "features_count": len(features),
        "feature_order_is_strict": True,
        "categorical_features": [f for f in CATEGORICAL_FEATURES if f in features],
        "numerical_features": [f for f in features if f not in CATEGORICAL_FEATURES],
        "feature_names_canonical_order": features,
        "target": {
            "name": "preparedness_tier",
            "classes": TARGET_CLASSES,
            "class_encoding": {"High": 0, "Low": 1, "Medium": 2},
            "description": AHP_TARGET_MEANING,
        },
    }
    with open(target_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2)

    # 5. Model evaluation summary
    val_results = meta_data.get("validation_results", {})
    test_metrics = val_results.get("test_metrics", {})
    clf_report = val_results.get("test_classification_report", {})
    cm_dict = val_results.get("test_confusion_matrix", {})

    eval_summary = {
        "evaluation_timestamp_utc": meta_data.get("creation_timestamp_utc"),
        "model_artifact": "logistic_regression_tier.joblib",
        "sample_size_test": meta_data.get("split_configuration", {}).get("test_size", 282),
        "overall_performance": {
            "accuracy": test_metrics.get("accuracy"),
            "macro_f1": test_metrics.get("macro_f1"),
            "weighted_f1": test_metrics.get("weighted_f1"),
        },
        "per_class_performance": {
            cls: {
                "precision": clf_report.get(cls, {}).get("precision"),
                "recall": clf_report.get(cls, {}).get("recall"),
                "f1_score": clf_report.get(cls, {}).get("f1-score"),
                "support": clf_report.get(cls, {}).get("support"),
            }
            for cls in TARGET_CLASSES
            if cls in clf_report
        },
        "confusion_matrix": cm_dict,
        "operational_alerts": [
            "Low Preparedness Tier recall is 73.91% (below provisional 80% engineering threshold). "
            "6 of 23 Low-tier villages misclassified as Medium. Vulnerable villages warrant manual field verification."
        ],
    }
    with open(target_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # 6. Model card
    model_card_content = f"""# Model Card: Village Preparedness Tier Classifier

## Model Details
- **Model Name:** Village Preparedness Tier Classifier
- **Model Artifact:** `logistic_regression_tier.joblib`
- **Architecture:** Scikit-learn Pipeline (RobustScaler + multinomial LogisticRegression)
- **Framework:** scikit-learn {meta_data.get('scikit_learn_version', '1.9.1')}
- **Date Registered:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
- **Provenance:** Reproducible reconstruction of teammate's documented 70/15/15 experiment (`notebooks/02_Baseline_Models.ipynb`).

## Intended Use
- **Primary Use:** Decision support for disaster preparedness authorities in Coastal Karnataka.
- **Classification Target:** Village preparedness capacity tiers (`High`, `Medium`, `Low`).
- **Target Definition:** {AHP_TARGET_MEANING}
- **Deterministic Precedence:** Deterministic AHP assessments hold operational precedence over model predictions.

## Features & Schema
- **Input Dimension:** Exactly 26 engineered spatial, infrastructural, demographic, and hazard features.
- **Strict Ordering:** Feature columns must match fit order exactly.
- **Categorical Indicators:** `has_pucca_road`, `has_major_district_road`, `bus_connectivity`.

## Evaluation Summary (Held-Out Test Split, N=282)
- **Overall Accuracy:** {test_metrics.get('accuracy', 0.9433):.4f} (94.33%)
- **Macro F1-Score:** {test_metrics.get('macro_f1', 0.9060):.4f}
- **High Preparedness Recall:** {clf_report.get('High', {}).get('recall', 0.9444):.4f}
- **Medium Preparedness Recall:** {clf_report.get('Medium', {}).get('recall', 0.9704):.4f}
- **Low Preparedness Recall:** {clf_report.get('Low', {}).get('recall', 0.7391):.4f} (Safety Alert: 6/23 Low-tier villages predicted as Medium)

## Ethical & Safety Considerations
- **Human-in-the-Loop:** Predictions serve as secondary verification and boundary-case detection; they do NOT replace local ground-truth administrative judgment.
- **Vulnerability Sensitivity:** Low-tier villages must receive safety-buffer inspections even if classified as Medium.
"""
    with open(target_dir / "model_card.md", "w", encoding="utf-8") as f:
        f.write(model_card_content.strip() + "\n")

    # 7. Intended use and limitations
    limitations_content = f"""# Intended Use and Model Limitations

## Operational Scope
This model is developed for **decision-support assistance** across the coastal districts of Karnataka (Dakshina Kannada, Udupi, Uttara Kannada).

## Core Limitations
1. **AHP-Tier Surrogate (Not Casualty/Hazard Risk):**
   The model predicts preparedness capacity bands derived from Analytical Hierarchy Process (AHP) scoring. It is **NOT** a calibrated physical cyclone-risk or casualty forecast.

2. **Deterministic AHP Precedence:**
   In any case of conflict between deterministic AHP scoring and Logistic Regression predictions, the deterministic AHP score remains authoritative.

3. **Low-Tier Recall Caveat (73.91%):**
   On held-out testing, Low-tier recall dropped to 73.91% (17 correct, 6 misclassified as Medium). In disaster relief, under-estimating vulnerability is severe. All villages with Low AHP tier or Low predicted probability > 20% should receive priority inspection.

4. **Boundary Case Ambiguity:**
   Villages with top probability margin < 10% or maximum probability < 60% are flagged as boundary cases and require manual review.

5. **Static Geographic Baseline:**
   Trained on 2011 census demographics and infrastructure baseline. Does not reflect infrastructure improvements constructed after baseline collection.
"""
    with open(target_dir / "intended_use_and_limitations.md", "w", encoding="utf-8") as f:
        f.write(limitations_content.strip() + "\n")

    return target_dir


def load_registered_model(
    artifact_address: str,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    api: Optional[Any] = None,
) -> RegisteredModelContainer:
    """Retrieve an explicitly specified W&B model artifact version and instantiate an inference container.

    Args:
        artifact_address: Explicit artifact identifier with version or alias tag.
            Examples:
            - "village-preparedness-classifier:v0"
            - "village-preparedness-classifier:candidate"
            - "bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:v0"
            - "wandb-registry-model/village-preparedness-classifier:candidate"
        entity: Optional W&B entity.
        project: Optional W&B project.
        cache_dir: Optional local directory to store/retrieve downloaded artifact files.
        api: Optional wandb.Api instance (useful for mocking and custom setups).

    Returns:
        RegisteredModelContainer configured for validated inference.

    Raises:
        ValueError: If artifact_address lacks an explicit version/alias tag, or if downloaded files are corrupted.
        WandbRetrievalError: If authentication or network communication fails.
    """
    if not artifact_address or not isinstance(artifact_address, str):
        raise ValueError("artifact_address must be a non-empty string")

    # Strict version check: Disallow retrieving arbitrary latest without explicit version tag
    if ":" not in artifact_address:
        raise ValueError(
            f"Artifact address '{artifact_address}' lacks an explicit version or alias tag. "
            "To prevent silent drift and maintain auditability, you must specify a version (e.g. ':v0' or ':candidate')."
        )

    # Resolve full qualified path if necessary
    wandb_api = api
    default_entity = getattr(wandb_api, "default_entity", None) if wandb_api else None
    eff_entity = entity or default_entity or os.environ.get("WANDB_ENTITY") or "bhavanishreddy69-na"
    eff_project = project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT

    full_path = artifact_address.strip()
    if not full_path.startswith("wandb-registry-") and "/" not in full_path:
        full_path = f"{eff_entity}/{eff_project}/{full_path}"

    base_cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    target_cache_dir = base_cache / full_path.replace("/", "_").replace(":", "_")

    # Check if complete valid artifact files already exist in local cache
    cached_pipeline = target_cache_dir / "logistic_regression_tier.joblib"
    cached_encoder = target_cache_dir / "label_encoder.joblib"
    cached_meta = target_cache_dir / "logistic_regression_metadata.json"

    if cached_pipeline.exists() and cached_encoder.exists() and cached_meta.exists():
        logger.info(f"Using locally cached artifact from {target_cache_dir}")
        download_dir = target_cache_dir
        version_label = artifact_address.split(":")[-1]
    else:
        # Initialize W&B API only if we need to retrieve from remote
        if wandb_api is None:
            try:
                wandb_api = wandb.Api()
            except Exception as e:
                raise WandbRetrievalError(
                    f"Failed to initialize W&B API client and artifact was not found in cache. "
                    f"Ensure you are authenticated with `wandb login` or WANDB_API_KEY is set: {e}"
                )

        logger.info(f"Retrieving model artifact '{full_path}' from W&B...")
        try:
            artifact = wandb_api.artifact(full_path, type="model")
            target_cache_dir.mkdir(parents=True, exist_ok=True)
            artifact.download(root=str(target_cache_dir))
            download_dir = target_cache_dir
            version_label = getattr(artifact, "version", artifact_address.split(":")[-1])
        except Exception as e:
            raise WandbRetrievalError(
                f"Failed to retrieve registered model artifact '{full_path}' from W&B: {e}"
            )

    # Verify required files exist
    pipeline_file = download_dir / "logistic_regression_tier.joblib"
    encoder_file = download_dir / "label_encoder.joblib"
    meta_file = download_dir / "logistic_regression_metadata.json"
    schema_file = download_dir / "feature_schema.json"

    for req_file in [pipeline_file, encoder_file, meta_file]:
        if not req_file.exists():
            raise ValueError(
                f"Corrupted or incomplete artifact: missing required model file '{req_file.name}' in {download_dir}"
            )

    # Load metadata
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse artifact metadata JSON from {meta_file}: {e}")

    feature_schema = None
    if schema_file.exists():
        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                feature_schema = json.load(f)
        except Exception:
            pass

    # Load serialized model objects
    try:
        pipeline = joblib.load(pipeline_file)
        label_encoder = joblib.load(encoder_file)
    except Exception as e:
        raise ValueError(f"Failed to unpickle model artifact components: {e}")

    # Validate that pipeline matches metadata expectations
    expected_features = metadata.get("features", [])
    if len(expected_features) != 26:
        raise ValueError(f"Metadata features count is {len(expected_features)}, expected 26")

    scaler = pipeline.named_steps.get("scaler")
    clf = pipeline.named_steps.get("model")
    if scaler is None or clf is None:
        raise ValueError("Pipeline structure invalid: missing 'scaler' or 'model' named steps")

    if hasattr(clf, "coef_") and clf.coef_.shape[1] != 26:
        raise ValueError(f"Classifier coefficients shape {clf.coef_.shape} does not match 26 features")

    logger.info(f"Successfully loaded and verified registered model artifact: {full_path}")
    return RegisteredModelContainer(
        pipeline=pipeline,
        label_encoder=label_encoder,
        metadata=metadata,
        artifact_version=version_label,
        artifact_source=full_path,
        artifact_dir=download_dir,
        feature_schema=feature_schema,
    )
