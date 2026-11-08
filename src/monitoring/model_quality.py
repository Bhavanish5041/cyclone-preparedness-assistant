"""Model quality monitoring, schema validation, and drift auditing module.

Implements Phase 2 production monitoring requirements:
- Loading model metadata and provenance tracking.
- Strict inference input schema validation (feature presence, strict ordering, data types, nulls, infs).
- Comprehensive model quality evaluation (accuracy, precision, recall, F1, confusion matrix,
  class imbalance, Brier score, log loss, probability distributions, boundary cases,
  and AHP agreement consistency).
- Statistical data drift auditing (Kolmogorov-Smirnov tests, Wasserstein distance, Population Stability Index).
- Prediction distribution drift auditing (probability simplex shifts, argmax class distribution divergence).
- Evidently AI integration bridge and fallback reporting.
- Structured Pydantic report generation with timestamps, artifact versions, and alerts.

NOTE: This module strictly isolates monitoring and auditing from prediction logic.
It does NOT modify any AHP formulas, K-Means archetypes, or priority-band rules.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Canonical paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = Path(os.environ.get("CYCLONE_MODEL_DIR", PROJECT_ROOT / "models"))
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "logistic_regression_metadata.json"
DEFAULT_PIPELINE_PATH = DEFAULT_MODEL_DIR / "logistic_regression_tier.joblib"
DEFAULT_ENCODER_PATH = DEFAULT_MODEL_DIR / "label_encoder.joblib"
DEFAULT_REPORTS_DIR = Path(os.environ.get("MONITORING_REPORTS_DIR", PROJECT_ROOT / "reports/model_monitoring"))

# Canonical feature groups and target definition
CATEGORICAL_FEATURES = ["has_pucca_road", "has_major_district_road", "bus_connectivity"]
TARGET_CLASSES = ["High", "Low", "Medium"]
AHP_TARGET_MEANING = (
    "AHP-derived village preparedness tiers (High, Medium, Low) reflecting composite physical, "
    "social, and infrastructural readiness. Deterministic AHP assessments serve as the baseline ground truth. "
    "Logistic Regression outputs provide calibrated tier probabilities for granular boundary case detection, "
    "NOT physical cyclone-risk or hazard prediction."
)

# Optional Evidently AI imports with dual legacy/modern resolution
try:
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset, TargetDriftPreset
    from evidently.pipeline.column_mapping import ColumnMapping
    EVIDENTLY_AVAILABLE = True
    EVIDENTLY_IMPORT_ERROR = None
except ImportError:
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset, TargetDriftPreset
        from evidently.legacy.pipeline.column_mapping import ColumnMapping
        EVIDENTLY_AVAILABLE = True
        EVIDENTLY_IMPORT_ERROR = None
    except ImportError as _e:
        Report = None  # type: ignore
        DataQualityPreset = None  # type: ignore
        DataDriftPreset = None  # type: ignore
        ClassificationPreset = None  # type: ignore
        TargetDriftPreset = None  # type: ignore
        ColumnMapping = None  # type: ignore
        EVIDENTLY_AVAILABLE = False
        EVIDENTLY_IMPORT_ERROR = str(_e)


# =============================================================================
# Structured Pydantic Schemas for Monitoring Outputs
# =============================================================================

class SchemaValidationResult(BaseModel):
    """Result of inference input schema validation."""

    is_valid: bool = Field(..., description="Whether input strictly conforms to model schema")
    total_records: int = Field(..., description="Number of evaluated input records")
    expected_feature_count: int = Field(26, description="Number of expected features")
    received_feature_count: int = Field(..., description="Number of features present in input")
    missing_features: List[str] = Field(default_factory=list, description="Required features missing from input")
    extra_features: List[str] = Field(default_factory=list, description="Unrecognized extra columns")
    order_matches_strictly: bool = Field(..., description="Whether column sequence exactly matches training order")
    invalid_dtypes: Dict[str, str] = Field(default_factory=dict, description="Columns with non-numeric data types")
    null_counts: Dict[str, int] = Field(default_factory=dict, description="Counts of NaN/None values per column")
    infinite_counts: Dict[str, int] = Field(default_factory=dict, description="Counts of +/-Inf values per column")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    errors: List[str] = Field(default_factory=list, description="Fatal validation errors")


class ModelQualityMetrics(BaseModel):
    """Empirical quality metrics evaluated against ground-truth or reference labels."""

    sample_size: int = Field(..., description="Evaluation sample size")
    accuracy: float = Field(..., description="Overall classification accuracy")
    macro_precision: float = Field(..., description="Unweighted mean precision across classes")
    weighted_precision: float = Field(..., description="Support-weighted precision across classes")
    macro_recall: float = Field(..., description="Unweighted mean recall across classes")
    weighted_recall: float = Field(..., description="Support-weighted recall across classes")
    macro_f1: float = Field(..., description="Unweighted mean F1 score across classes")
    weighted_f1: float = Field(..., description="Support-weighted F1 score across classes")
    per_class_metrics: Dict[str, Dict[str, float]] = Field(
        ..., description="Precision, recall, f1-score, and support per class"
    )
    confusion_matrix: Dict[str, Any] = Field(
        ..., description="Confusion matrix labels and 2D integer count matrix"
    )
    class_balance: Dict[str, float] = Field(
        ..., description="Fraction of records belonging to each class in evaluation sample"
    )
    brier_scores: Optional[Dict[str, float]] = Field(
        None, description="Brier score loss per class (lower is better, 0.0 is perfect)"
    )
    log_loss_value: Optional[float] = Field(
        None, description="Multinomial cross-entropy log loss (lower is better)"
    )
    boundary_case_count: Optional[int] = Field(
        None, description="Number of boundary cases (prob margin < 0.10 or max prob < 0.60)"
    )
    boundary_case_ratio: Optional[float] = Field(
        None, description="Ratio of boundary cases to total sample size"
    )
    ahp_agreement_rate: Optional[float] = Field(
        None, description="Proportion of cases where predicted tier agrees with AHP tier"
    )
    disagreement_counts: Optional[Dict[str, int]] = Field(
        None, description="Breakdown of specific off-diagonal disagreement transitions"
    )


class FeatureDriftResult(BaseModel):
    """Distribution drift result for a single candidate feature."""

    feature_name: str
    test_method: str = Field("two_sample_kolmogorov_smirnov", description="Statistical test used")
    statistic: float = Field(..., description="KS test statistic D")
    p_value: float = Field(..., description="Two-tailed asymptotic p-value")
    wasserstein_distance: float = Field(..., description="First Wasserstein / Earth Mover's Distance")
    drift_detected: bool = Field(..., description="Whether p-value is below significance threshold")
    significance_level: float = Field(0.05, description="Significance threshold alpha")


class DataDriftReport(BaseModel):
    """Aggregate dataset-level feature drift report."""

    total_features_evaluated: int
    drifted_features_count: int
    drift_share: float = Field(..., description="Proportion of features displaying significant drift")
    dataset_drift_detected: bool = Field(..., description="True if drift_share exceeds drift_threshold")
    drift_threshold: float = Field(0.33, description="Dataset-level drift alert threshold (default: 33%)")
    per_feature_drift: Dict[str, FeatureDriftResult]


class PredictionDriftReport(BaseModel):
    """Model output and probability prediction drift report."""

    sample_size_reference: int
    sample_size_current: int
    probability_drift: Dict[str, Dict[str, float]] = Field(
        ..., description="KS statistic, p-value, and Wasserstein distance per class probability column"
    )
    class_distribution_reference: Dict[str, float] = Field(
        ..., description="Reference split predicted class share"
    )
    class_distribution_current: Dict[str, float] = Field(
        ..., description="Current split predicted class share"
    )
    population_stability_index: Dict[str, float] = Field(
        ..., description="PSI per class probability distribution"
    )
    prediction_drift_detected: bool = Field(
        ..., description="Whether any class probability or output distribution exhibits significant shift"
    )


class EvidentlyIntegrationStatus(BaseModel):
    """Status and configuration for Evidently AI integration."""

    installed: bool
    version: Optional[str] = None
    status_message: str
    supported_presets: List[str] = Field(
        default_factory=lambda: [
            "DataQualityPreset",
            "DataDriftPreset",
            "ClassificationPreset",
            "TargetDriftPreset",
        ]
    )
    proposed_pip_dependency: str = "evidently==0.7.23"


class EvidentlyReportResult(BaseModel):
    """Result and metadata for an individual Evidently report."""

    report_type: str = Field(..., description="Report type: data_quality, feature_drift, prediction_drift, or classification_performance")
    status: str = Field(..., description="'success' or 'failed'")
    html_path: Optional[str] = Field(None, description="Path to generated HTML report file")
    json_path: Optional[str] = Field(None, description="Path to generated JSON report file")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_description: Optional[str] = Field(None, description="Clear description of evaluation target")
    summary_metrics: Dict[str, Any] = Field(default_factory=dict, description="Key summary metrics extracted from report")


class EvidentlySuiteResult(BaseModel):
    """Container for multi-report Evidently monitoring execution."""

    evidently_version: str
    status: str = Field(..., description="'success', 'partial', or 'failed'")
    output_directory: str
    reports: Dict[str, EvidentlyReportResult] = Field(default_factory=dict)
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ComprehensiveMonitoringReport(BaseModel):
    """Full structured monitoring output for deployment pipelines and dashboards."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_artifact: str = "logistic_regression_tier.joblib"
    model_version_metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_validation: SchemaValidationResult
    quality_metrics: Optional[ModelQualityMetrics] = None
    data_drift: Optional[DataDriftReport] = None
    prediction_drift: Optional[PredictionDriftReport] = None
    evidently_status: EvidentlyIntegrationStatus
    active_alerts: List[str] = Field(default_factory=list)
    evidently_suite: Optional[EvidentlySuiteResult] = Field(
        None, description="Detailed Evidently AI suite results when run_evidently is True"
    )


class EvidentlyExecutionError(RuntimeError):
    """Raised when Evidently report execution fails or Evidently is unavailable."""

    pass


# =============================================================================
# Helper Functions & Core Implementations
# =============================================================================

def load_model_metadata(metadata_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate the serialized logistic regression metadata JSON.

    Args:
        metadata_path: Optional path to JSON metadata file. Defaults to models/logistic_regression_metadata.json.

    Returns:
        Dictionary containing verified model provenance and hyperparameters.
    """
    path = metadata_path or DEFAULT_METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Basic structural check
    required_keys = ["model_artifact", "features", "target", "validation_results"]
    for key in required_keys:
        if key not in meta:
            raise KeyError(f"Metadata file {path} missing required auditing field: {key}")

    return meta


class FeatureSchemaValidator:
    """Validates inference input schemas against strict model specifications."""

    def __init__(self, metadata_path: Optional[Path] = None):
        self.metadata = load_model_metadata(metadata_path)
        self.expected_features: List[str] = self.metadata["features"]

    def validate(self, df: pd.DataFrame, strict_order: bool = True) -> SchemaValidationResult:
        """Validate input DataFrame feature names, ordering, nulls, and types.

        Args:
            df: Input feature DataFrame.
            strict_order: Whether column sequence must match training feature order exactly.

        Returns:
            SchemaValidationResult summarizing issues found.
        """
        received_cols = list(df.columns)
        missing = [f for f in self.expected_features if f not in received_cols]
        extra = [c for c in received_cols if c not in self.expected_features]

        # Check strict ordering among present expected features
        order_matches = True
        if not missing and strict_order:
            order_matches = received_cols[: len(self.expected_features)] == self.expected_features

        # Check data types, nulls, infs on expected features present
        invalid_dtypes: Dict[str, str] = {}
        null_counts: Dict[str, int] = {}
        inf_counts: Dict[str, int] = {}
        warnings: List[str] = []
        errors: List[str] = []

        if missing:
            errors.append(
                f"Missing {len(missing)} required model features: {missing}. "
                "Ensure candidate feature engineering has been applied before inference."
            )

        if extra:
            warnings.append(
                f"Input contains {len(extra)} extra unmodeled columns (e.g. {extra[:3]}). "
                "These will be ignored if strict extraction is used."
            )

        for feat in self.expected_features:
            if feat in df.columns:
                col_series = df[feat]
                # Type check: must be numeric
                if not pd.api.types.is_numeric_dtype(col_series):
                    invalid_dtypes[feat] = str(col_series.dtype)
                    errors.append(f"Feature '{feat}' has non-numeric dtype '{col_series.dtype}'")

                # Null check
                n_nulls = int(col_series.isna().sum())
                if n_nulls > 0:
                    null_counts[feat] = n_nulls
                    errors.append(f"Feature '{feat}' contains {n_nulls} missing (NaN/None) values")

                # Infinite check
                if pd.api.types.is_numeric_dtype(col_series):
                    n_infs = int(np.isinf(col_series.to_numpy(dtype=float, na_value=0.0)).sum())
                    if n_infs > 0:
                        inf_counts[feat] = n_infs
                        errors.append(f"Feature '{feat}' contains {n_infs} infinite (+/-Inf) values")

        if not order_matches and not missing:
            warnings.append(
                "Feature column sequence differs from fit order. Scikit-learn Pipeline "
                "will raise an exception unless columns are reordered."
            )

        is_valid = len(errors) == 0

        return SchemaValidationResult(
            is_valid=is_valid,
            total_records=len(df),
            expected_feature_count=len(self.expected_features),
            received_feature_count=len(received_cols),
            missing_features=missing,
            extra_features=extra,
            order_matches_strictly=order_matches,
            invalid_dtypes=invalid_dtypes,
            null_counts=null_counts,
            infinite_counts=inf_counts,
            warnings=warnings,
            errors=errors,
        )

    def extract_and_align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and reorder expected 26 features strictly in fit order.

        Args:
            df: Input DataFrame containing at least the 26 expected features.

        Returns:
            DataFrame with exactly 26 columns in canonical order.

        Raises:
            ValueError: If schema validation fails with fatal errors.
        """
        val_result = self.validate(df, strict_order=False)
        if not val_result.is_valid:
            raise ValueError(
                f"Cannot extract and align features: Input schema validation failed with {len(val_result.errors)} errors: "
                + "; ".join(val_result.errors[:3])
            )

        return df[self.expected_features].copy()


def validate_input_schema(
    df: pd.DataFrame,
    metadata_path: Optional[Path] = None,
    strict_order: bool = True,
) -> SchemaValidationResult:
    """Convenience function to validate an input DataFrame against model schema."""
    validator = FeatureSchemaValidator(metadata_path)
    return validator.validate(df, strict_order=strict_order)


class ModelQualityEvaluator:
    """Evaluates classification performance, calibration, boundary cases, and AHP consistency."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["High", "Low", "Medium"]

    def evaluate(
        self,
        y_true: Union[pd.Series, np.ndarray, List[str]],
        y_pred: Union[pd.Series, np.ndarray, List[str]],
        y_probs: Optional[np.ndarray] = None,
    ) -> ModelQualityMetrics:
        """Compute comprehensive performance and quality metrics.

        Args:
            y_true: Ground truth or reference AHP tiers.
            y_pred: Model predicted tiers.
            y_probs: Optional Nx3 matrix of predicted class probabilities aligned to sorted class_names.

        Returns:
            ModelQualityMetrics model.
        """
        y_t = np.array(y_true).astype(str)
        y_p = np.array(y_pred).astype(str)

        if len(y_t) != len(y_p):
            raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}")

        sample_size = len(y_t)
        if sample_size == 0:
            raise ValueError("Cannot evaluate model quality on empty dataset")

        acc = float(accuracy_score(y_t, y_p))
        macro_p = float(precision_score(y_t, y_p, labels=self.class_names, average="macro", zero_division=0))
        wtd_p = float(precision_score(y_t, y_p, labels=self.class_names, average="weighted", zero_division=0))
        macro_r = float(recall_score(y_t, y_p, labels=self.class_names, average="macro", zero_division=0))
        wtd_r = float(recall_score(y_t, y_p, labels=self.class_names, average="weighted", zero_division=0))
        macro_f1 = float(f1_score(y_t, y_p, labels=self.class_names, average="macro", zero_division=0))
        wtd_f1 = float(f1_score(y_t, y_p, labels=self.class_names, average="weighted", zero_division=0))

        # Per-class report
        clf_dict = classification_report(
            y_t, y_p, labels=self.class_names, output_dict=True, zero_division=0
        )
        per_class = {}
        for cls in self.class_names:
            if cls in clf_dict:
                per_class[cls] = {
                    "precision": float(clf_dict[cls]["precision"]),
                    "recall": float(clf_dict[cls]["recall"]),
                    "f1_score": float(clf_dict[cls]["f1-score"]),
                    "support": float(clf_dict[cls]["support"]),
                }

        # Confusion Matrix
        cm = confusion_matrix(y_t, y_p, labels=self.class_names).tolist()
        cm_data = {"labels": self.class_names, "matrix": cm}

        # Class balance
        val_counts = pd.Series(y_t).value_counts(normalize=True).to_dict()
        class_balance = {cls: float(val_counts.get(cls, 0.0)) for cls in self.class_names}

        # Probability metrics (Brier, Log Loss, Boundary cases)
        brier_dict: Optional[Dict[str, float]] = None
        log_loss_val: Optional[float] = None
        boundary_count: Optional[int] = None
        boundary_ratio: Optional[float] = None

        if y_probs is not None:
            if y_probs.shape != (sample_size, len(self.class_names)):
                raise ValueError(
                    f"y_probs shape {y_probs.shape} does not match ({sample_size}, {len(self.class_names)})"
                )

            # Ensure probabilities strictly sum to 1.0 to eliminate float rounding warnings in sklearn.metrics.log_loss
            y_sums = y_probs.sum(axis=1, keepdims=True)
            y_sums[y_sums == 0] = 1.0
            probs_clean = y_probs / y_sums

            # Map class strings to integers for log_loss
            cls_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
            y_t_idx = np.array([cls_to_idx.get(c, 0) for c in y_t])

            try:
                log_loss_val = float(log_loss(y_t_idx, probs_clean, labels=list(range(len(self.class_names)))))
            except Exception as e:
                logger.warning(f"Failed to compute log loss: {e}")

            brier_dict = {}
            for i, cls in enumerate(self.class_names):
                binary_true = (y_t == cls).astype(int)
                brier_dict[cls] = float(brier_score_loss(binary_true, probs_clean[:, i]))

            # Boundary cases: probability margin < 0.10 or max probability < 0.60
            sorted_probs = np.sort(probs_clean, axis=1)
            max_prob = sorted_probs[:, -1]
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            is_boundary = (margin < 0.10) | (max_prob < 0.60)
            boundary_count = int(is_boundary.sum())
            boundary_ratio = float(boundary_count / sample_size)

        # Disagreement breakdown
        disagreements = y_t != y_p
        disagreement_counts = {}
        for i, t_val in enumerate(y_t):
            p_val = y_p[i]
            if t_val != p_val:
                key = f"{t_val}_predicted_as_{p_val}"
                disagreement_counts[key] = disagreement_counts.get(key, 0) + 1

        return ModelQualityMetrics(
            sample_size=sample_size,
            accuracy=acc,
            macro_precision=macro_p,
            weighted_precision=wtd_p,
            macro_recall=macro_r,
            weighted_recall=wtd_r,
            macro_f1=macro_f1,
            weighted_f1=wtd_f1,
            per_class_metrics=per_class,
            confusion_matrix=cm_data,
            class_balance=class_balance,
            brier_scores=brier_dict,
            log_loss_value=log_loss_val,
            boundary_case_count=boundary_count,
            boundary_case_ratio=boundary_ratio,
            ahp_agreement_rate=acc,
            disagreement_counts=disagreement_counts,
        )


def evaluate_model_quality(
    y_true: Union[pd.Series, np.ndarray, List[str]],
    y_pred: Union[pd.Series, np.ndarray, List[str]],
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> ModelQualityMetrics:
    """Convenience wrapper to evaluate model classification and calibration quality."""
    evaluator = ModelQualityEvaluator(class_names=class_names)
    return evaluator.evaluate(y_true=y_true, y_pred=y_pred, y_probs=y_probs)


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and current continuous distributions.

    PSI Interpretation:
    - PSI < 0.10: No significant distribution change.
    - 0.10 <= PSI < 0.25: Moderate shift; warrants monitoring.
    - PSI >= 0.25: Significant distribution shift; action required.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Determine quantile bins based on reference data
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    # Ensure strictly increasing edges to avoid empty zero-width bins
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 3:
        # Fallback to linear binning if quantiles collapse
        unique_edges = np.linspace(min(ref.min(), cur.min()) - 1e-5, max(ref.max(), cur.max()) + 1e-5, num_bins + 1)

    ref_counts, _ = np.histogram(ref, bins=unique_edges)
    cur_counts, _ = np.histogram(cur, bins=unique_edges)

    # Convert to fractions with Laplace epsilon smoothing to avoid div-by-zero
    eps = 1e-4
    ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(cur) + eps * len(cur_counts))

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi_val))


def compute_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    alpha: float = 0.05,
    drift_share_threshold: float = 0.33,
) -> DataDriftReport:
    """Compute feature-level and dataset-level distribution drift using Kolmogorov-Smirnov and Wasserstein distance.

    Args:
        reference_df: Reference baseline dataset (e.g. 70% training split).
        current_df: Current comparison dataset (e.g. held-out test split, regional subset, or inference batch).
        features: Optional list of numeric features to audit. Defaults to all common numeric columns.
        alpha: Statistical significance threshold (default: 0.05).
        drift_share_threshold: Fraction of drifting features to trigger dataset-level alert (default: 0.33).

    Returns:
        DataDriftReport with per-feature statistics.
    """
    if features is None:
        common_cols = [c for c in reference_df.columns if c in current_df.columns]
        features = [c for c in common_cols if pd.api.types.is_numeric_dtype(reference_df[c])]

    per_feature: Dict[str, FeatureDriftResult] = {}
    drifted_count = 0

    for feat in features:
        ref_vals = reference_df[feat].dropna().to_numpy(dtype=float)
        cur_vals = current_df[feat].dropna().to_numpy(dtype=float)

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_res = ks_2samp(ref_vals, cur_vals)
        w_dist = float(wasserstein_distance(ref_vals, cur_vals))
        p_val = float(ks_res.pvalue)
        stat = float(ks_res.statistic)

        is_drift = bool(p_val < alpha)
        if is_drift:
            drifted_count += 1

        per_feature[feat] = FeatureDriftResult(
            feature_name=feat,
            test_method="two_sample_kolmogorov_smirnov",
            statistic=stat,
            p_value=p_val,
            wasserstein_distance=w_dist,
            drift_detected=is_drift,
            significance_level=alpha,
        )

    total_eval = len(per_feature)
    drift_share = float(drifted_count / total_eval) if total_eval > 0 else 0.0
    dataset_drift = drift_share >= drift_share_threshold

    return DataDriftReport(
        total_features_evaluated=total_eval,
        drifted_features_count=drifted_count,
        drift_share=drift_share,
        dataset_drift_detected=dataset_drift,
        drift_threshold=drift_share_threshold,
        per_feature_drift=per_feature,
    )


def compute_prediction_drift(
    reference_probs: np.ndarray,
    current_probs: np.ndarray,
    reference_preds: Optional[np.ndarray] = None,
    current_preds: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> PredictionDriftReport:
    """Compute prediction score and probability distribution drift across splits.

    Args:
        reference_probs: Nx3 matrix of reference split predicted probabilities.
        current_probs: Mx3 matrix of current split predicted probabilities.
        reference_preds: Optional array of reference predicted tier strings.
        current_preds: Optional array of current predicted tier strings.
        class_names: List of class strings ['High', 'Low', 'Medium'].
        alpha: Statistical significance threshold.

    Returns:
        PredictionDriftReport.
    """
    classes = class_names or ["High", "Low", "Medium"]
    if reference_probs.shape[1] != len(classes) or current_probs.shape[1] != len(classes):
        raise ValueError(f"Probability matrices must have shape (N, {len(classes)})")

    prob_drift: Dict[str, Dict[str, float]] = {}
    psi_dict: Dict[str, float] = {}
    any_drift = False

    for idx, cls in enumerate(classes):
        ref_col = reference_probs[:, idx]
        cur_col = current_probs[:, idx]

        ks_res = ks_2samp(ref_col, cur_col)
        w_dist = float(wasserstein_distance(ref_col, cur_col))
        psi_val = compute_psi(ref_col, cur_col)

        p_val = float(ks_res.pvalue)
        stat = float(ks_res.statistic)
        drift_flag = p_val < alpha or psi_val >= 0.25
        if drift_flag:
            any_drift = True

        prob_drift[cls] = {
            "ks_statistic": stat,
            "p_value": p_val,
            "wasserstein_distance": w_dist,
            "psi": psi_val,
            "drift_detected": float(drift_flag),
        }
        psi_dict[cls] = psi_val

    # Class distribution shares
    if reference_preds is not None:
        ref_shares = pd.Series(reference_preds).value_counts(normalize=True).to_dict()
    else:
        ref_argmax = np.argmax(reference_probs, axis=1)
        ref_shares = pd.Series([classes[i] for i in ref_argmax]).value_counts(normalize=True).to_dict()

    if current_preds is not None:
        cur_shares = pd.Series(current_preds).value_counts(normalize=True).to_dict()
    else:
        cur_argmax = np.argmax(current_probs, axis=1)
        cur_shares = pd.Series([classes[i] for i in cur_argmax]).value_counts(normalize=True).to_dict()

    ref_dist = {cls: float(ref_shares.get(cls, 0.0)) for cls in classes}
    cur_dist = {cls: float(cur_shares.get(cls, 0.0)) for cls in classes}

    return PredictionDriftReport(
        sample_size_reference=len(reference_probs),
        sample_size_current=len(current_probs),
        probability_drift=prob_drift,
        class_distribution_reference=ref_dist,
        class_distribution_current=cur_dist,
        population_stability_index=psi_dict,
        prediction_drift_detected=any_drift,
    )


def check_evidently_availability() -> EvidentlyIntegrationStatus:
    """Inspect whether Evidently AI is installed in the current environment."""
    if not EVIDENTLY_AVAILABLE:
        return EvidentlyIntegrationStatus(
            installed=False,
            version=None,
            status_message=(
                f"Evidently AI is NOT available: {EVIDENTLY_IMPORT_ERROR}. "
                "The built-in SciPy/NumPy statistical monitoring engine is currently active as fallback."
            ),
        )

    try:
        import evidently  # type: ignore

        version = getattr(evidently, "__version__", "unknown")
        return EvidentlyIntegrationStatus(
            installed=True,
            version=version,
            status_message=f"Evidently AI version {version} is installed and available for execution.",
        )
    except Exception as e:
        return EvidentlyIntegrationStatus(
            installed=False,
            version=None,
            status_message=f"Error inspecting Evidently installation: {e}",
        )


class EvidentlyReportGenerator:
    """Modular Evidently AI report generator for production and offline model monitoring.

    Generates:
    - Data Quality Report: missing values, invalid values, column stats, distributions.
    - Feature Drift Report: 26 model features across reference (training) and comparison (test) splits.
    - Prediction Drift Report: predicted class distribution and probability simplex shifts.
    - Classification Performance Report: accuracy, precision, recall, F1, confusion matrix vs AHP labels.

    Note on Target:
    The classification target strictly represents AHP-derived village preparedness tiers (High, Medium, Low),
    reflecting composite physical, social, and infrastructural readiness. It is NOT physical cyclone-risk or
    hazard prediction. Deterministic AHP assessments serve as the baseline ground truth.
    """

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.metadata = load_model_metadata(self.metadata_path)
        self.expected_features: List[str] = self.metadata["features"]
        self.target_classes: List[str] = self.metadata["target"]["classes"]
        self.categorical_features = [f for f in CATEGORICAL_FEATURES if f in self.expected_features]
        self.numerical_features = [f for f in self.expected_features if f not in self.categorical_features]
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_REPORTS_DIR

    def _ensure_evidently(self) -> None:
        if not EVIDENTLY_AVAILABLE or Report is None:
            raise EvidentlyExecutionError(
                f"Evidently AI is not available in the active environment: {EVIDENTLY_IMPORT_ERROR}"
            )

    def _save_report(
        self,
        report: Any,
        output_prefix: str,
        save_html: bool,
        save_json: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        html_path = None
        json_path = None

        if save_html:
            hp = self.output_dir / f"{output_prefix}.html"
            report.save_html(str(hp))
            html_path = str(hp.resolve())

        if save_json:
            jp = self.output_dir / f"{output_prefix}.json"
            report.save_json(str(jp))
            json_path = str(jp.resolve())

        return html_path, json_path

    def generate_data_quality_report(
        self,
        current_features: pd.DataFrame,
        reference_features: Optional[pd.DataFrame] = None,
        save_html: bool = True,
        save_json: bool = True,
        output_prefix: str = "evidently_data_quality",
    ) -> Tuple[Any, EvidentlyReportResult]:
        """Generate Evidently Data Quality report assessing missing values, stats, and types."""
        self._ensure_evidently()

        if len(current_features) == 0:
            raise ValueError("Cannot generate Data Quality report on empty dataset")

        missing_cur = [f for f in self.expected_features if f not in current_features.columns]
        if missing_cur:
            raise ValueError(f"Input features missing {len(missing_cur)} required columns: {missing_cur[:5]}")

        cur_df = current_features[self.expected_features].copy()
        ref_df = None
        if reference_features is not None:
            if len(reference_features) == 0:
                raise ValueError("Reference features dataset is empty")
            missing_ref = [f for f in self.expected_features if f not in reference_features.columns]
            if missing_ref:
                raise ValueError(f"Reference features missing {len(missing_ref)} required columns: {missing_ref[:5]}")
            ref_df = reference_features[self.expected_features].copy()

        col_mapping = ColumnMapping(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)

        html_p, json_p = self._save_report(report, output_prefix, save_html, save_json)

        # Extract summary metrics
        summary: Dict[str, Any] = {}
        for m in report.as_dict().get("metrics", []):
            if m.get("metric") == "DatasetSummaryMetric":
                res = m.get("result", {}).get("current", {})
                summary["number_of_rows"] = res.get("number_of_rows")
                summary["number_of_columns"] = res.get("number_of_columns")
                summary["number_of_missing_values"] = res.get("number_of_missing_values")
                summary["number_of_empty_rows"] = res.get("number_of_empty_rows")
                summary["number_of_empty_columns"] = res.get("number_of_empty_columns")
                summary["number_of_constant_columns"] = res.get("number_of_constant_columns")

        result = EvidentlyReportResult(
            report_type="data_quality",
            status="success",
            html_path=html_p,
            json_path=json_p,
            target_description="Input feature distributions, missing values, and column-level summary statistics",
            summary_metrics=summary,
        )
        return report, result

    def generate_feature_drift_report(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        save_html: bool = True,
        save_json: bool = True,
        output_prefix: str = "evidently_feature_drift",
    ) -> Tuple[Any, EvidentlyReportResult]:
        """Generate Evidently Feature Drift report comparing 26 model features."""
        self._ensure_evidently()

        if len(reference_features) == 0 or len(current_features) == 0:
            raise ValueError("Datasets for feature drift evaluation must not be empty")

        missing_ref = [f for f in self.expected_features if f not in reference_features.columns]
        missing_cur = [f for f in self.expected_features if f not in current_features.columns]
        if missing_ref:
            raise ValueError(f"Reference features missing {len(missing_ref)} required columns: {missing_ref[:5]}")
        if missing_cur:
            raise ValueError(f"Current features missing {len(missing_cur)} required columns: {missing_cur[:5]}")

        ref_df = reference_features[self.expected_features].copy()
        cur_df = current_features[self.expected_features].copy()

        col_mapping = ColumnMapping(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)

        html_p, json_p = self._save_report(report, output_prefix, save_html, save_json)

        summary: Dict[str, Any] = {}
        for m in report.as_dict().get("metrics", []):
            if m.get("metric") == "DatasetDriftMetric":
                res = m.get("result", {})
                summary["number_of_columns"] = res.get("number_of_columns")
                summary["number_of_drifted_columns"] = res.get("number_of_drifted_columns")
                summary["share_of_drifted_columns"] = res.get("share_of_drifted_columns")
                summary["dataset_drift"] = res.get("dataset_drift")

        result = EvidentlyReportResult(
            report_type="feature_drift",
            status="success",
            html_path=html_p,
            json_path=json_p,
            target_description="Distribution drift across 26 model features (offline training vs test comparison)",
            summary_metrics=summary,
        )
        return report, result

    def generate_prediction_drift_report(
        self,
        reference_preds: Union[pd.Series, np.ndarray, List[str]],
        current_preds: Union[pd.Series, np.ndarray, List[str]],
        reference_probs: Optional[np.ndarray] = None,
        current_probs: Optional[np.ndarray] = None,
        reference_labels: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
        current_labels: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
        save_html: bool = True,
        save_json: bool = True,
        output_prefix: str = "evidently_prediction_drift",
    ) -> Tuple[Any, EvidentlyReportResult]:
        """Generate Evidently Prediction Drift report assessing predicted class and probability shifts."""
        self._ensure_evidently()

        ref_p = list(reference_preds)
        cur_p = list(current_preds)
        if len(ref_p) == 0 or len(cur_p) == 0:
            raise ValueError("Prediction datasets must not be empty")

        ref_data: Dict[str, Any] = {"prediction": ref_p}
        cur_data: Dict[str, Any] = {"prediction": cur_p}
        prob_cols: List[str] = []

        if reference_probs is not None and current_probs is not None:
            if reference_probs.shape[1] != len(self.target_classes) or current_probs.shape[1] != len(self.target_classes):
                raise ValueError(
                    f"Probability matrices must have shape (N, {len(self.target_classes)})"
                )
            for idx, cls in enumerate(self.target_classes):
                col_name = f"prob_{cls.lower()}"
                prob_cols.append(col_name)
                ref_data[col_name] = reference_probs[:, idx]
                cur_data[col_name] = current_probs[:, idx]

        cat_cols = ["prediction"]
        if reference_labels is not None and current_labels is not None:
            ref_data["target"] = list(reference_labels)
            cur_data["target"] = list(current_labels)
            cat_cols.append("target")

        ref_df = pd.DataFrame(ref_data)
        cur_df = pd.DataFrame(cur_data)

        col_mapping = ColumnMapping(
            prediction="prediction",
            target="target" if "target" in ref_df.columns else None,
            categorical_features=cat_cols,
            numerical_features=prob_cols if prob_cols else None,
        )

        metrics: List[Any] = [DataDriftPreset()]
        if "target" in ref_df.columns:
            metrics.append(TargetDriftPreset())

        report = Report(metrics=metrics)
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)

        html_p, json_p = self._save_report(report, output_prefix, save_html, save_json)

        summary: Dict[str, Any] = {}
        for m in report.as_dict().get("metrics", []):
            if m.get("metric") == "DatasetDriftMetric":
                res = m.get("result", {})
                summary["number_of_columns"] = res.get("number_of_columns")
                summary["number_of_drifted_columns"] = res.get("number_of_drifted_columns")
                summary["share_of_drifted_columns"] = res.get("share_of_drifted_columns")
                summary["dataset_drift"] = res.get("dataset_drift")

        result = EvidentlyReportResult(
            report_type="prediction_drift",
            status="success",
            html_path=html_p,
            json_path=json_p,
            target_description="Predicted tier class frequencies and calibrated probability distributions",
            summary_metrics=summary,
        )
        return report, result

    def generate_classification_performance_report(
        self,
        current_labels: Union[pd.Series, np.ndarray, List[str]],
        current_preds: Union[pd.Series, np.ndarray, List[str]],
        current_probs: Optional[np.ndarray] = None,
        reference_labels: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
        reference_preds: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
        reference_probs: Optional[np.ndarray] = None,
        save_html: bool = True,
        save_json: bool = True,
        output_prefix: str = "evidently_classification_performance",
    ) -> Tuple[Any, EvidentlyReportResult]:
        """Generate Evidently Classification Performance report against verified AHP ground-truth labels."""
        self._ensure_evidently()

        cur_l = list(current_labels)
        cur_p = list(current_preds)
        if len(cur_l) != len(cur_p):
            raise ValueError(f"Current labels and predictions length mismatch: {len(cur_l)} vs {len(cur_p)}")
        if len(cur_l) == 0:
            raise ValueError("Evaluation dataset must not be empty")

        cur_data: Dict[str, Any] = {"target": cur_l, "prediction": cur_p}
        ref_data: Optional[Dict[str, Any]] = None

        if reference_labels is not None and reference_preds is not None:
            ref_l = list(reference_labels)
            ref_p = list(reference_preds)
            if len(ref_l) != len(ref_p):
                raise ValueError(f"Reference labels and predictions length mismatch: {len(ref_l)} vs {len(ref_p)}")
            ref_data = {"target": ref_l, "prediction": ref_p}

        prob_classes: Optional[List[str]] = None
        if current_probs is not None:
            if current_probs.shape[1] != len(self.target_classes):
                raise ValueError(f"current_probs shape {current_probs.shape} does not match classes {self.target_classes}")
            # Ensure probabilities strictly sum to 1.0 to eliminate float rounding artifacts in downstream sklearn metrics
            cur_sums = current_probs.sum(axis=1, keepdims=True)
            cur_sums[cur_sums == 0] = 1.0
            cur_norm_probs = current_probs / cur_sums

            prob_classes = list(self.target_classes)
            for idx, cls in enumerate(self.target_classes):
                cur_data[cls] = cur_norm_probs[:, idx]

        if ref_data is not None and reference_probs is not None:
            if reference_probs.shape[1] != len(self.target_classes):
                raise ValueError(f"reference_probs shape {reference_probs.shape} does not match classes {self.target_classes}")
            ref_sums = reference_probs.sum(axis=1, keepdims=True)
            ref_sums[ref_sums == 0] = 1.0
            ref_norm_probs = reference_probs / ref_sums
            for idx, cls in enumerate(self.target_classes):
                ref_data[cls] = ref_norm_probs[:, idx]

        cur_df = pd.DataFrame(cur_data)
        ref_df = pd.DataFrame(ref_data) if ref_data is not None else None

        col_mapping = ColumnMapping(
            target="target",
            prediction=prob_classes if prob_classes else "prediction",
            target_names=None,
        )

        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)

        html_p, json_p = self._save_report(report, output_prefix, save_html, save_json)

        summary: Dict[str, Any] = {}
        for m in report.as_dict().get("metrics", []):
            if m.get("metric") == "ClassificationQualityMetric":
                res = m.get("result", {}).get("current", {})
                summary["accuracy"] = res.get("accuracy")
                summary["precision"] = res.get("precision")
                summary["recall"] = res.get("recall")
                summary["f1"] = res.get("f1")

        result = EvidentlyReportResult(
            report_type="classification_performance",
            status="success",
            html_path=html_p,
            json_path=json_p,
            target_description=AHP_TARGET_MEANING,
            summary_metrics=summary,
        )
        return report, result

    def run_suite(
        self,
        current_features: pd.DataFrame,
        reference_features: Optional[pd.DataFrame] = None,
        current_labels: Optional[pd.Series] = None,
        current_preds: Optional[pd.Series] = None,
        current_probs: Optional[np.ndarray] = None,
        reference_labels: Optional[pd.Series] = None,
        reference_preds: Optional[pd.Series] = None,
        reference_probs: Optional[np.ndarray] = None,
        save_html: bool = True,
        save_json: bool = True,
    ) -> EvidentlySuiteResult:
        """Run all applicable Evidently reports and compile a comprehensive suite result."""
        self._ensure_evidently()
        reports_dict: Dict[str, EvidentlyReportResult] = {}
        overall_status = "success"

        # 1. Data Quality
        try:
            _, dq_res = self.generate_data_quality_report(
                current_features=current_features,
                reference_features=reference_features,
                save_html=save_html,
                save_json=save_json,
            )
            reports_dict["data_quality"] = dq_res
        except Exception as e:
            logger.error(f"Evidently Data Quality report generation failed: {e}")
            overall_status = "partial" if reports_dict else "failed"
            reports_dict["data_quality"] = EvidentlyReportResult(
                report_type="data_quality",
                status="failed",
                error_message=str(e),
                target_description="Input feature distributions and missing values",
            )

        # 2. Feature Drift (requires reference features)
        if reference_features is not None:
            try:
                _, fd_res = self.generate_feature_drift_report(
                    reference_features=reference_features,
                    current_features=current_features,
                    save_html=save_html,
                    save_json=save_json,
                )
                reports_dict["feature_drift"] = fd_res
            except Exception as e:
                logger.error(f"Evidently Feature Drift report generation failed: {e}")
                overall_status = "partial"
                reports_dict["feature_drift"] = EvidentlyReportResult(
                    report_type="feature_drift",
                    status="failed",
                    error_message=str(e),
                    target_description="Feature distribution drift (26 features)",
                )

        # 3. Prediction Drift (requires reference & current preds)
        if reference_preds is not None and current_preds is not None:
            try:
                _, pd_res = self.generate_prediction_drift_report(
                    reference_preds=reference_preds,
                    current_preds=current_preds,
                    reference_probs=reference_probs,
                    current_probs=current_probs,
                    reference_labels=reference_labels,
                    current_labels=current_labels,
                    save_html=save_html,
                    save_json=save_json,
                )
                reports_dict["prediction_drift"] = pd_res
            except Exception as e:
                logger.error(f"Evidently Prediction Drift report generation failed: {e}")
                overall_status = "partial"
                reports_dict["prediction_drift"] = EvidentlyReportResult(
                    report_type="prediction_drift",
                    status="failed",
                    error_message=str(e),
                    target_description="Prediction distribution and probability drift",
                )

        # 4. Classification Performance (requires labels and preds)
        if current_labels is not None and current_preds is not None:
            try:
                _, cp_res = self.generate_classification_performance_report(
                    current_labels=current_labels,
                    current_preds=current_preds,
                    current_probs=current_probs,
                    reference_labels=reference_labels,
                    reference_preds=reference_preds,
                    reference_probs=reference_probs,
                    save_html=save_html,
                    save_json=save_json,
                )
                reports_dict["classification_performance"] = cp_res
            except Exception as e:
                logger.error(f"Evidently Classification Performance report generation failed: {e}")
                overall_status = "partial"
                reports_dict["classification_performance"] = EvidentlyReportResult(
                    report_type="classification_performance",
                    status="failed",
                    error_message=str(e),
                    target_description=AHP_TARGET_MEANING,
                )

        import evidently
        v = getattr(evidently, "__version__", "unknown")

        return EvidentlySuiteResult(
            evidently_version=v,
            status=overall_status,
            output_directory=str(self.output_dir.resolve()),
            reports=reports_dict,
        )


class ModelQualityMonitor:
    """Integrated monitoring coordinator for model quality, schema validation, and drift auditing."""

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        encoder_path: Optional[Path] = None,
    ):
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.metadata = load_model_metadata(self.metadata_path)
        self.features: List[str] = self.metadata["features"]
        self.target_classes: List[str] = self.metadata["target"]["classes"]
        self.validator = FeatureSchemaValidator(self.metadata_path)
        self.evaluator = ModelQualityEvaluator(class_names=self.target_classes)

    def run_monitoring_cycle(
        self,
        current_features: pd.DataFrame,
        reference_features: Optional[pd.DataFrame] = None,
        current_labels: Optional[pd.Series] = None,
        current_preds: Optional[pd.Series] = None,
        current_probs: Optional[np.ndarray] = None,
        reference_probs: Optional[np.ndarray] = None,
        reference_preds: Optional[pd.Series] = None,
        reference_labels: Optional[pd.Series] = None,
        run_evidently: bool = False,
        evidently_output_dir: Optional[Union[str, Path]] = None,
    ) -> ComprehensiveMonitoringReport:
        """Execute a complete model monitoring cycle on incoming and reference data.

        Args:
            current_features: Feature DataFrame for the current evaluation period or inference batch.
            reference_features: Optional reference feature DataFrame (e.g. training split).
            current_labels: Optional ground-truth or reference AHP tiers for current data.
            current_preds: Optional model predicted tiers for current data.
            current_probs: Optional predicted class probability matrix for current data.
            reference_probs: Optional predicted class probability matrix for reference data.
            reference_preds: Optional predicted tiers for reference data.
            reference_labels: Optional ground-truth or reference AHP tiers for reference data.
            run_evidently: Whether to execute the Evidently AI monitoring suite.
            evidently_output_dir: Directory where Evidently HTML and JSON reports will be saved.

        Returns:
            ComprehensiveMonitoringReport containing validation, metrics, drift, and alerts.
        """
        alerts: List[str] = []

        # 1. Schema Validation
        schema_val = self.validator.validate(current_features)
        if not schema_val.is_valid:
            alerts.append(f"CRITICAL: Schema validation failed with {len(schema_val.errors)} fatal errors.")

        # 2. Performance Quality Metrics (if ground truth and predictions are available)
        # Note: Performance evaluation requires verified AHP ground-truth labels.
        # Provisional engineering thresholds: Low-tier recall < 80%, Boundary cases > 15%.
        quality_metrics: Optional[ModelQualityMetrics] = None
        if current_labels is not None and current_preds is not None:
            quality_metrics = self.evaluator.evaluate(
                y_true=current_labels, y_pred=current_preds, y_probs=current_probs
            )
            low_recall = quality_metrics.per_class_metrics.get("Low", {}).get("recall", 1.0)
            if low_recall < 0.80:
                alerts.append(
                    f"WARNING: Low Preparedness Tier recall dropped to {low_recall:.2%}. "
                    "Risk of under-estimating highly vulnerable villages!"
                )
            if quality_metrics.boundary_case_ratio and quality_metrics.boundary_case_ratio > 0.15:
                alerts.append(
                    f"NOTICE: Elevated boundary case proportion ({quality_metrics.boundary_case_ratio:.1%}). "
                    "Higher likelihood of tier ambiguity."
                )

        # 3. Data Drift (if reference features provided and schema valid)
        # Provisional engineering threshold: Dataset drift alerted if > 33% of features drift.
        data_drift: Optional[DataDriftReport] = None
        if reference_features is not None and schema_val.is_valid:
            ref_aligned = self.validator.extract_and_align(reference_features)
            cur_aligned = self.validator.extract_and_align(current_features)
            data_drift = compute_data_drift(ref_aligned, cur_aligned, features=self.features)
            if data_drift.dataset_drift_detected:
                alerts.append(
                    f"DATA DRIFT ALERT: {data_drift.drifted_features_count} / {data_drift.total_features_evaluated} "
                    f"features ({data_drift.drift_share:.1%}) display significant distribution shift."
                )

        # 4. Prediction Drift (if probabilities provided for both)
        pred_drift: Optional[PredictionDriftReport] = None
        if reference_probs is not None and current_probs is not None:
            pred_drift = compute_prediction_drift(
                reference_probs=reference_probs,
                current_probs=current_probs,
                reference_preds=reference_preds.to_numpy() if reference_preds is not None else None,
                current_preds=current_preds.to_numpy() if current_preds is not None else None,
                class_names=self.target_classes,
            )
            if pred_drift.prediction_drift_detected:
                alerts.append(
                    "PREDICTION DRIFT ALERT: Model output probabilities or predicted class distributions "
                    "have diverged significantly from baseline."
                )

        # 5. Evidently status
        evidently_status = check_evidently_availability()

        # 6. Optional Evidently Suite Execution
        evidently_suite: Optional[EvidentlySuiteResult] = None
        if run_evidently:
            if not EVIDENTLY_AVAILABLE:
                raise EvidentlyExecutionError(
                    f"Evidently AI is not available in the active environment: {EVIDENTLY_IMPORT_ERROR}"
                )
            generator = EvidentlyReportGenerator(
                metadata_path=self.metadata_path,
                output_dir=evidently_output_dir,
            )
            evidently_suite = generator.run_suite(
                current_features=current_features,
                reference_features=reference_features,
                current_labels=current_labels,
                current_preds=current_preds,
                current_probs=current_probs,
                reference_labels=reference_labels,
                reference_preds=reference_preds,
                reference_probs=reference_probs,
            )
            if evidently_suite.status == "failed":
                alerts.append("CRITICAL: Evidently monitoring suite failed to generate reports.")
            elif evidently_suite.status == "partial":
                alerts.append("WARNING: One or more Evidently monitoring reports failed during suite execution.")

        return ComprehensiveMonitoringReport(
            report_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_artifact=self.metadata.get("model_artifact", "logistic_regression_tier.joblib"),
            model_version_metadata={
                "creation_timestamp_utc": self.metadata.get("creation_timestamp_utc"),
                "scikit_learn_version": self.metadata.get("scikit_learn_version"),
                "python_version": self.metadata.get("python_version"),
                "model_nature": self.metadata.get("model_nature"),
            },
            schema_validation=schema_val,
            quality_metrics=quality_metrics,
            data_drift=data_drift,
            prediction_drift=pred_drift,
            evidently_status=evidently_status,
            active_alerts=alerts,
            evidently_suite=evidently_suite,
        )
