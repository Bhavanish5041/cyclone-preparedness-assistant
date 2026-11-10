"""Analytics and model intelligence endpoints."""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.analytics import (
    ClusterDetail,
    ClustersAnalyticsResponse,
    ConfusionMatrixData,
    ModelMetrics,
    ModelPerformanceResponse,
    PreparednessAnalyticsResponse,
    PriorityBandsAnalyticsResponse,
)
from src.assistant.decision_support import DecisionSupportService
from src.integration.decision_matrix import (
    COASTAL_PROXIMITY_THRESHOLD_KM,
    DECISION_SUPPORT_DISCLAIMER,
    MEDICAL_DEFICIT_THRESHOLD,
    SHELTER_DEFICIT_THRESHOLD,
)
from src.models.clustering import ARCHETYPE_DISCLAIMER

router = APIRouter(prefix="/analytics", tags=["Analytics & Models"])

MODEL_METADATA_PATH = Path("models/logistic_regression_metadata.json")


@router.get("/preparedness", response_model=PreparednessAnalyticsResponse, summary="Authoritative AHP preparedness distribution")
def get_preparedness_analytics(
    service: DecisionSupportService = Depends(get_service),
) -> PreparednessAnalyticsResponse:
    """Return distribution of AHP composite scores and dimension metrics across all 1,879 units."""
    df = service.ranked_df
    t_counts = df["preparedness_tier"].astype(str).str.capitalize().value_counts().to_dict()

    score_series = df["preparedness_score"].dropna()
    score_summary = {
        "mean": round(float(score_series.mean()), 2),
        "median": round(float(score_series.median()), 2),
        "min": round(float(score_series.min()), 2),
        "max": round(float(score_series.max()), 2),
        "std": round(float(score_series.std()), 2),
    }

    domain_averages = {
        "medical": round(float(df["score_medical"].mean()), 2) if "score_medical" in df.columns else 0.0,
        "shelters": round(float(df["score_shelters"].mean()), 2) if "score_shelters" in df.columns else 0.0,
        "infrastructure": round(float(df["score_infrastructure"].mean()), 2) if "score_infrastructure" in df.columns else 0.0,
        "early_warning": round(float(df["score_early_warning"].mean()), 2) if "score_early_warning" in df.columns else 0.0,
        "socioeconomic": round(float(df["score_socioeconomic"].mean()), 2) if "score_socioeconomic" in df.columns else 0.0,
    }

    district_averages = {}
    for d_name in sorted(df["district_name"].unique()):
        sub = df[df["district_name"] == d_name]
        district_averages[str(d_name)] = {
            "mean_score": round(float(sub["preparedness_score"].mean()), 2),
            "medical": round(float(sub["score_medical"].mean()), 2) if "score_medical" in sub.columns else 0.0,
            "shelters": round(float(sub["score_shelters"].mean()), 2) if "score_shelters" in sub.columns else 0.0,
            "infrastructure": round(float(sub["score_infrastructure"].mean()), 2) if "score_infrastructure" in sub.columns else 0.0,
            "early_warning": round(float(sub["score_early_warning"].mean()), 2) if "score_early_warning" in sub.columns else 0.0,
            "socioeconomic": round(float(sub["score_socioeconomic"].mean()), 2) if "score_socioeconomic" in sub.columns else 0.0,
        }

    return PreparednessAnalyticsResponse(
        total_units=len(df),
        tier_distribution={
            "High": int(t_counts.get("High", 0)),
            "Medium": int(t_counts.get("Medium", 0)),
            "Low": int(t_counts.get("Low", 0)),
        },
        score_summary=score_summary,
        domain_averages=domain_averages,
        district_averages=district_averages,
        disclaimer=DECISION_SUPPORT_DISCLAIMER,
    )


@router.get("/priority-bands", response_model=PriorityBandsAnalyticsResponse, summary="Operational priority band distributions and deficit triggers")
def get_priority_bands_analytics(
    service: DecisionSupportService = Depends(get_service),
) -> PriorityBandsAnalyticsResponse:
    """Return priority band allocations and sectoral deficit trigger counts."""
    df = service.ranked_df
    b_counts = df["priority_band"].value_counts().to_dict()

    t_coast = (df["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM).sum()
    t_shelter = (df["score_shelters"] < SHELTER_DEFICIT_THRESHOLD).sum() if "score_shelters" in df.columns else 0
    t_med = (df["score_medical"] < MEDICAL_DEFICIT_THRESHOLD).sum() if "score_medical" in df.columns else 0

    crosstab_dict = {}
    for d_name in sorted(df["district_name"].unique()):
        sub = df[df["district_name"] == d_name]
        d_b = sub["priority_band"].value_counts().to_dict()
        crosstab_dict[str(d_name)] = {
            "BAND_1_CRITICAL": int(d_b.get("BAND_1_CRITICAL", 0)),
            "BAND_2_SPECIALIZED": int(d_b.get("BAND_2_SPECIALIZED", 0)),
            "BAND_3_MODERATE": int(d_b.get("BAND_3_MODERATE", 0)),
            "BAND_4_ADEQUATE": int(d_b.get("BAND_4_ADEQUATE", 0)),
        }

    return PriorityBandsAnalyticsResponse(
        total_units=len(df),
        band_distribution={
            "BAND_1_CRITICAL": int(b_counts.get("BAND_1_CRITICAL", 0)),
            "BAND_2_SPECIALIZED": int(b_counts.get("BAND_2_SPECIALIZED", 0)),
            "BAND_3_MODERATE": int(b_counts.get("BAND_3_MODERATE", 0)),
            "BAND_4_ADEQUATE": int(b_counts.get("BAND_4_ADEQUATE", 0)),
        },
        trigger_counts={
            "coastal_proximity_under_5km": int(t_coast),
            "shelter_deficit_under_20": int(t_shelter),
            "medical_deficit_under_35": int(t_med),
        },
        district_band_crosstab=crosstab_dict,
        disclaimer=DECISION_SUPPORT_DISCLAIMER,
    )


@router.get("/clusters", response_model=ClustersAnalyticsResponse, summary="K-Means vulnerability archetypes and UK fallback catalog")
def get_clusters_analytics(
    service: DecisionSupportService = Depends(get_service),
) -> ClustersAnalyticsResponse:
    """Return descriptive profiles and distributions for the 4 vulnerability archetypes."""
    catalog = service.get_archetype_catalog()
    df = service.ranked_df

    cluster_list = []
    for arch in catalog.archetypes:
        cluster_list.append(
            ClusterDetail(
                cluster_id=arch.archetype_id,
                label=arch.short_label,
                count=arch.unit_count,
                district_distribution={
                    "Dakshina Kannada": arch.count_dakshina_kannada,
                    "Udupi": arch.count_udupi,
                    "Uttara Kannada": 0,
                },
                primary_intervention_need=arch.primary_intervention_need,
                detailed_description=arch.detailed_description,
                dimension_means=arch.dimension_means,
            )
        )

    assigned_count = int(df["vulnerability_archetype_id"].notna().sum())
    unassigned_count = int(df["vulnerability_archetype_id"].isna().sum())

    return ClustersAnalyticsResponse(
        total_units=len(df),
        assigned_count=assigned_count,
        unassigned_count=unassigned_count,
        clusters=cluster_list,
        disclaimer=ARCHETYPE_DISCLAIMER,
    )


@router.get("/model-performance", response_model=ModelPerformanceResponse, summary="Surrogate Logistic Regression evaluation metrics and confusion matrix")
def get_model_performance(
    service: DecisionSupportService = Depends(get_service),
) -> ModelPerformanceResponse:
    """Return model evaluation metrics, test confusion matrix, and feature schema."""
    if not MODEL_METADATA_PATH.exists():
        raise RuntimeError("Model metadata file not found.")

    with open(MODEL_METADATA_PATH, "r") as f:
        meta = json.load(f)

    v_res = meta.get("validation_results", {})
    t_cm = v_res.get("test_confusion_matrix", {"classes": ["High", "Low", "Medium"], "matrix": []})

    train_m = v_res.get("train_metrics", {})
    val_m = v_res.get("val_metrics", {})
    test_m = v_res.get("test_metrics", {})

    # Calculate overall concordance between AHP tier and predicted tier in dataset
    df = service.ranked_df
    concordance = 0.0
    if "predicted_preparedness_tier" in df.columns:
        match_count = (df["preparedness_tier"].astype(str).str.lower() == df["predicted_preparedness_tier"].astype(str).str.lower()).sum()
        concordance = round(float(match_count / len(df) * 100), 2)

    return ModelPerformanceResponse(
        model_name="village-preparedness-classifier",
        model_type=meta.get("model_type", "RobustScaler + LogisticRegression"),
        framework=meta.get("framework", "scikit-learn"),
        features_count=meta.get("n_features", 26),
        features=meta.get("features", []),
        split_info=meta.get("split_configuration", {}),
        train_metrics=ModelMetrics(
            accuracy=round(float(train_m.get("accuracy", 0.0)), 4),
            macro_f1=round(float(train_m.get("macro_f1", 0.0)), 4),
            weighted_f1=round(float(train_m.get("weighted_f1", 0.0)), 4),
        ),
        val_metrics=ModelMetrics(
            accuracy=round(float(val_m.get("accuracy", 0.0)), 4),
            macro_f1=round(float(val_m.get("macro_f1", 0.0)), 4),
            weighted_f1=round(float(val_m.get("weighted_f1", 0.0)), 4),
        ),
        test_metrics=ModelMetrics(
            accuracy=round(float(test_m.get("accuracy", 0.0)), 4),
            macro_f1=round(float(test_m.get("macro_f1", 0.0)), 4),
            weighted_f1=round(float(test_m.get("weighted_f1", 0.0)), 4),
        ),
        test_confusion_matrix=ConfusionMatrixData(
            classes=t_cm.get("classes", ["High", "Low", "Medium"]),
            matrix=t_cm.get("matrix", []),
        ),
        test_classification_report=v_res.get("test_classification_report", {}),
        concordance_rate_overall=concordance,
        disclaimer=(
            "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate "
            "approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated "
            "physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. "
            "Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
        ),
    )
