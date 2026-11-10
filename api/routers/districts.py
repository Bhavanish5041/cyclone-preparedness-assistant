"""District intelligence endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Path, status

from api.dependencies import get_service
from api.schemas.districts import DistrictListResponse, DistrictSummary, SubdistrictSummary
from src.assistant.decision_support import DecisionSupportService

router = APIRouter(prefix="/districts", tags=["Districts"])


def _build_district_summary(df, d_name: str, include_subdistricts: bool = False) -> DistrictSummary:
    sub_df = df[df["district_name"] == d_name]
    if sub_df.empty:
        raise KeyError(f"District '{d_name}' not found.")

    b_counts = sub_df["priority_band"].value_counts().to_dict()
    band_counts = {
        "BAND_1_CRITICAL": int(b_counts.get("BAND_1_CRITICAL", 0)),
        "BAND_2_SPECIALIZED": int(b_counts.get("BAND_2_SPECIALIZED", 0)),
        "BAND_3_MODERATE": int(b_counts.get("BAND_3_MODERATE", 0)),
        "BAND_4_ADEQUATE": int(b_counts.get("BAND_4_ADEQUATE", 0)),
    }

    t_counts = sub_df["preparedness_tier"].astype(str).str.capitalize().value_counts().to_dict()
    tier_counts = {
        "High": int(t_counts.get("High", 0)),
        "Medium": int(t_counts.get("Medium", 0)),
        "Low": int(t_counts.get("Low", 0)),
    }

    c_counts = sub_df["vulnerability_archetype_label"].fillna("Unassigned (Uttara Kannada Fallback)").value_counts().to_dict()
    cluster_counts = {str(k): int(v) for k, v in c_counts.items()}

    subdistricts = None
    if include_subdistricts and "subdistrict_name" in sub_df.columns:
        subdistricts = []
        for s_name in sorted(sub_df["subdistrict_name"].dropna().unique()):
            s_df = sub_df[sub_df["subdistrict_name"] == s_name]
            s_b_counts = s_df["priority_band"].value_counts().to_dict()
            subdistricts.append(
                SubdistrictSummary(
                    subdistrict_name=str(s_name),
                    total_units=len(s_df),
                    mean_preparedness_score=round(float(s_df["preparedness_score"].mean()), 2),
                    band_counts={
                        "BAND_1_CRITICAL": int(s_b_counts.get("BAND_1_CRITICAL", 0)),
                        "BAND_2_SPECIALIZED": int(s_b_counts.get("BAND_2_SPECIALIZED", 0)),
                        "BAND_3_MODERATE": int(s_b_counts.get("BAND_3_MODERATE", 0)),
                        "BAND_4_ADEQUATE": int(s_b_counts.get("BAND_4_ADEQUATE", 0)),
                    },
                    critical_units_count=int(s_b_counts.get("BAND_1_CRITICAL", 0)),
                )
            )

    d_code = str(sub_df["district_code"].iloc[0]) if "district_code" in sub_df.columns else None

    return DistrictSummary(
        district_name=d_name,
        district_code=d_code,
        total_units=len(sub_df),
        mean_preparedness_score=round(float(sub_df["preparedness_score"].mean()), 2),
        min_preparedness_score=round(float(sub_df["preparedness_score"].min()), 2),
        max_preparedness_score=round(float(sub_df["preparedness_score"].max()), 2),
        band_counts=band_counts,
        tier_counts=tier_counts,
        cluster_counts=cluster_counts,
        mean_dist_coast_km=round(float(sub_df["dist_coast_km"].mean()), 2),
        mean_dist_nearest_shelter_km=round(float(sub_df["dist_nearest_shelter_km"].mean()), 2),
        mean_dist_nearest_hospital_km=round(float(sub_df["dist_nearest_hospital_km"].mean()), 2),
        mean_dist_nearest_phc_km=round(float(sub_df["dist_nearest_phc_km"].mean()), 2) if "dist_nearest_phc_km" in sub_df.columns else 0.0,
        pct_pucca_road=round(float(sub_df["has_pucca_road"].mean() * 100), 1) if "has_pucca_road" in sub_df.columns else 0.0,
        pct_drinking_water_secure=round(float((sub_df["drinking_water_score"] > 0.5).mean() * 100), 1) if "drinking_water_score" in sub_df.columns else 0.0,
        total_population=int(sub_df["population"].sum()) if "population" in sub_df.columns else 0,
        total_households=int(sub_df["households"].sum()) if "households" in sub_df.columns else 0,
        mean_vulnerable_pop_share=round(float(sub_df["vulnerable_pop_share"].mean()), 4) if "vulnerable_pop_share" in sub_df.columns else 0.0,
        subdistricts=subdistricts,
    )


@router.get("", response_model=DistrictListResponse, summary="List all 3 coastal districts and comparative profiles")
def get_districts(
    service: DecisionSupportService = Depends(get_service),
) -> DistrictListResponse:
    """Return comparative summaries for Dakshina Kannada, Udupi, and Uttara Kannada."""
    df = service.ranked_df
    districts = []
    for d_name in sorted(df["district_name"].unique()):
        districts.append(_build_district_summary(df, str(d_name), include_subdistricts=False))
    return DistrictListResponse(total_districts=len(districts), districts=districts)


@router.get("/{district_id}", response_model=DistrictSummary, summary="Retrieve granular district intelligence dossier")
def get_district_detail(
    district_id: str = Path(..., description="District name or slug (e.g. 'udupi', 'dakshina-kannada', 'uttara-kannada')"),
    service: DecisionSupportService = Depends(get_service),
) -> DistrictSummary:
    """Return detailed metrics, subdistricts/taluks, and vulnerability distributions for a district."""
    df = service.ranked_df
    clean_id = district_id.strip().lower().replace("-", " ")
    match_name = None
    for d_name in df["district_name"].unique():
        if d_name.lower() == clean_id or d_name.lower().replace(" ", "-") == district_id.strip().lower():
            match_name = d_name
            break

    if not match_name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"District '{district_id}' not found. Available districts: 'Dakshina Kannada', 'Udupi', 'Uttara Kannada'.",
        )

    return _build_district_summary(df, str(match_name), include_subdistricts=True)
