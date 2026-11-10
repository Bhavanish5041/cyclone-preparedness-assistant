"""Pydantic schemas for analytical models, AHP distributions, and clusters."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PreparednessAnalyticsResponse(BaseModel):
    """Analytics on AHP preparedness scores and domain indicators."""
    total_units: int
    tier_distribution: Dict[str, int]
    score_summary: Dict[str, float]  # mean, median, min, max, std
    domain_averages: Dict[str, float]  # medical, shelters, infrastructure, early_warning, socioeconomic
    district_averages: Dict[str, Dict[str, float]]
    disclaimer: str


class PriorityBandsAnalyticsResponse(BaseModel):
    """Analytics on 4 operational priority bands and deficit triggers."""
    total_units: int
    band_distribution: Dict[str, int]
    trigger_counts: Dict[str, int]
    district_band_crosstab: Dict[str, Dict[str, int]]
    disclaimer: str


class ClusterDetail(BaseModel):
    """Cluster profile definition."""
    cluster_id: int
    label: str
    count: int
    district_distribution: Dict[str, int]
    primary_intervention_need: str
    detailed_description: str
    dimension_means: Dict[str, float]


class ClustersAnalyticsResponse(BaseModel):
    """Analytics on K-Means vulnerability archetypes and UK fallback."""
    total_units: int
    assigned_count: int
    unassigned_count: int
    clusters: List[ClusterDetail]
    disclaimer: str


class ModelMetrics(BaseModel):
    """Accuracy and F1 metrics."""
    accuracy: float
    macro_f1: float
    weighted_f1: float


class ConfusionMatrixData(BaseModel):
    """Multiclass confusion matrix."""
    classes: List[str]
    matrix: List[List[int]]


class ModelPerformanceResponse(BaseModel):
    """Logistic regression surrogate model performance and metadata."""
    model_name: str
    model_type: str
    framework: str
    features_count: int
    features: List[str]
    split_info: Dict[str, Any]
    train_metrics: ModelMetrics
    val_metrics: ModelMetrics
    test_metrics: ModelMetrics
    test_confusion_matrix: ConfusionMatrixData
    test_classification_report: Dict[str, Any]
    concordance_rate_overall: float
    disclaimer: str
