"""Dashboard overview endpoints."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.dashboard import DashboardOverviewResponse, DistrictOverviewItem
from src.assistant.decision_support import DecisionSupportService
from src.integration.decision_matrix import DECISION_SUPPORT_DISCLAIMER

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/overview", response_model=DashboardOverviewResponse, summary="Regional command-center executive summary")
def get_dashboard_overview(
    service: DecisionSupportService = Depends(get_service),
) -> DashboardOverviewResponse:
    """Return macroscopic summary across all 1,879 administrative units."""
    df = service.ranked_df
    total_units = len(df)

    # Priority Band counts
    b_counts = df["priority_band"].value_counts().to_dict()
    band_counts = {
        "BAND_1_CRITICAL": int(b_counts.get("BAND_1_CRITICAL", 0)),
        "BAND_2_SPECIALIZED": int(b_counts.get("BAND_2_SPECIALIZED", 0)),
        "BAND_3_MODERATE": int(b_counts.get("BAND_3_MODERATE", 0)),
        "BAND_4_ADEQUATE": int(b_counts.get("BAND_4_ADEQUATE", 0)),
    }

    # Preparedness Tier counts
    t_counts = df["preparedness_tier"].astype(str).str.capitalize().value_counts().to_dict()
    tier_counts = {
        "High": int(t_counts.get("High", 0)),
        "Medium": int(t_counts.get("Medium", 0)),
        "Low": int(t_counts.get("Low", 0)),
    }

    # Archetype cluster counts
    c_counts = df["vulnerability_archetype_label"].fillna("Unassigned (Uttara Kannada Fallback)").value_counts().to_dict()
    cluster_counts = {str(k): int(v) for k, v in c_counts.items()}

    # District breakdowns
    district_items = []
    for d_name in sorted(df["district_name"].unique()):
        sub_df = df[df["district_name"] == d_name]
        d_b_counts = sub_df["priority_band"].value_counts().to_dict()
        d_t_counts = sub_df["preparedness_tier"].astype(str).str.capitalize().value_counts().to_dict()

        district_items.append(
            DistrictOverviewItem(
                district_name=str(d_name),
                total_units=len(sub_df),
                mean_preparedness_score=round(float(sub_df["preparedness_score"].mean()), 2),
                band_counts={
                    "BAND_1_CRITICAL": int(d_b_counts.get("BAND_1_CRITICAL", 0)),
                    "BAND_2_SPECIALIZED": int(d_b_counts.get("BAND_2_SPECIALIZED", 0)),
                    "BAND_3_MODERATE": int(d_b_counts.get("BAND_3_MODERATE", 0)),
                    "BAND_4_ADEQUATE": int(d_b_counts.get("BAND_4_ADEQUATE", 0)),
                },
                tier_counts={
                    "High": int(d_t_counts.get("High", 0)),
                    "Medium": int(d_t_counts.get("Medium", 0)),
                    "Low": int(d_t_counts.get("Low", 0)),
                },
                mean_dist_nearest_shelter_km=round(float(sub_df["dist_nearest_shelter_km"].mean()), 2),
                mean_dist_nearest_hospital_km=round(float(sub_df["dist_nearest_hospital_km"].mean()), 2),
                mean_dist_coast_km=round(float(sub_df["dist_coast_km"].mean()), 2),
                critical_units_count=int(d_b_counts.get("BAND_1_CRITICAL", 0)),
            )
        )

    # Infrastructure highlights
    infra_highlights = {
        "total_population": int(df["population"].sum()) if "population" in df.columns else 0,
        "total_households": int(df["households"].sum()) if "households" in df.columns else 0,
        "mean_dist_coast_km": round(float(df["dist_coast_km"].mean()), 2),
        "mean_dist_nearest_shelter_km": round(float(df["dist_nearest_shelter_km"].mean()), 2),
        "mean_dist_nearest_hospital_km": round(float(df["dist_nearest_hospital_km"].mean()), 2),
        "mean_dist_nearest_phc_km": round(float(df["dist_nearest_phc_km"].mean()), 2) if "dist_nearest_phc_km" in df.columns else 0.0,
        "pucca_road_pct": round(float(df["has_pucca_road"].mean() * 100), 1) if "has_pucca_road" in df.columns else 0.0,
        "coastal_units_under_5km": int((df["dist_coast_km"] < 5.0).sum()),
    }

    return DashboardOverviewResponse(
        total_units=total_units,
        districts_count=len(district_items),
        assigned_clusters_count=int(df["vulnerability_archetype_id"].notna().sum()),
        unassigned_clusters_count=int(df["vulnerability_archetype_id"].isna().sum()),
        priority_band_counts=band_counts,
        preparedness_tier_counts=tier_counts,
        cluster_counts=cluster_counts,
        districts=district_items,
        infrastructure_highlights=infra_highlights,
        data_disclaimer=DECISION_SUPPORT_DISCLAIMER,
    )
