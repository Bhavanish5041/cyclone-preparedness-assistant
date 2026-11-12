"""Communities explorer and individual unit inspection endpoints."""

from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_service
from api.schemas.communities import CommunitiesListResponse, CommunityDetailResponse, CommunityListItem
from src.assistant.decision_support import DecisionSupportService
from src.integration.decision_matrix import DECISION_SUPPORT_DISCLAIMER

router = APIRouter(prefix="/communities", tags=["Communities"])


@router.get("", response_model=CommunitiesListResponse, summary="Search, filter, and paginate communities")
def list_communities(
    search: Optional[str] = Query(None, description="Search by village name, PC11 ID, or SHRID"),
    district: Optional[str] = Query(None, description="Filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada')"),
    tier: Optional[str] = Query(None, description="Filter by AHP preparedness tier ('High', 'Medium', 'Low')"),
    priority_band: Optional[str] = Query(None, description="Filter by priority band ('BAND_1_CRITICAL', 'BAND_2_SPECIALIZED', 'BAND_3_MODERATE', 'BAND_4_ADEQUATE')"),
    archetype_id: Optional[int] = Query(None, description="Filter by archetype ID (0, 1, 2, 3 or -1 for unassigned)"),
    sort_by: str = Query("priority_rank", description="Field to sort by ('priority_rank', 'preparedness_score', 'dist_coast_km', 'population', 'village_name')"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order ('asc' or 'desc')"),
    page: int = Query(1, ge=1, description="Page number starting from 1"),
    limit: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    service: DecisionSupportService = Depends(get_service),
) -> CommunitiesListResponse:
    """Return paginated list of communities with multi-criteria filtering."""
    df = service.ranked_df.copy()

    # Apply search filter
    if search and search.strip():
        q = search.strip().lower()
        mask = (
            df["village_name"].astype(str).str.lower().str.contains(q)
            | df["pc11_id"].astype(str).str.lower().str.contains(q)
            | df["shrid2"].astype(str).str.lower().str.contains(q)
        )
        if "subdistrict_name" in df.columns:
            mask = mask | df["subdistrict_name"].astype(str).str.lower().str.contains(q)
        df = df[mask]

    # District filter
    if district and district.strip():
        df = df[df["district_name"].str.lower() == district.strip().lower()]

    # Preparedness Tier filter
    if tier and tier.strip():
        df = df[df["preparedness_tier"].astype(str).str.lower() == tier.strip().lower()]

    # Priority Band filter
    if priority_band and priority_band.strip():
        df = df[df["priority_band"].astype(str).str.upper() == priority_band.strip().upper()]

    # Archetype ID filter
    if archetype_id is not None:
        if archetype_id == -1:
            df = df[df["vulnerability_archetype_id"].isna()]
        else:
            df = df[df["vulnerability_archetype_id"] == archetype_id]

    # Sorting
    ascending = (sort_order.lower() == "asc")
    sort_column_map = {
        "priority_rank": "overall_priority_rank",
        "preparedness_score": "preparedness_score",
        "dist_coast_km": "dist_coast_km",
        "population": "population",
        "village_name": "village_name",
        "band_rank": "band_rank",
    }
    target_col = sort_column_map.get(sort_by, "overall_priority_rank")
    if target_col in df.columns:
        df = df.sort_values(target_col, ascending=ascending, na_position="last")

    total = len(df)
    total_pages = max(1, (total + limit - 1) // limit)
    offset = (page - 1) * limit
    page_df = df.iloc[offset : offset + limit]

    items = []
    for _, row in page_df.iterrows():
        raw_unit_type = row.get("unit_type")
        unit_type = (
            "VILLAGE"
            if pd.isna(raw_unit_type) or str(raw_unit_type).strip().lower() in ("nan", "none", "")
            else str(raw_unit_type).strip()
        )
        items.append(
            CommunityListItem(
                area_id=str(row["area_id"]),
                pc11_id=str(row["pc11_id"]),
                shrid2=str(row["shrid2"]),
                village_name=str(row["village_name"]),
                district_name=str(row["district_name"]),
                subdistrict_name=str(row["subdistrict_name"]) if pd.notna(row.get("subdistrict_name")) else None,
                unit_type=unit_type,
                lat=round(float(row["lat"]), 6),
                lon=round(float(row["lon"]), 6),
                preparedness_score=round(float(row["preparedness_score"]), 2),
                preparedness_tier=str(row["preparedness_tier"]),
                priority_band=str(row["priority_band"]),
                priority_rank=int(row["overall_priority_rank"]),
                band_rank=int(row["band_rank"]),
                vulnerability_archetype_id=int(row["vulnerability_archetype_id"]) if pd.notna(row.get("vulnerability_archetype_id")) else None,
                vulnerability_archetype_label=str(row["vulnerability_archetype_label"]) if pd.notna(row.get("vulnerability_archetype_label")) else None,
                population=int(row["population"]) if pd.notna(row.get("population")) else None,
                dist_coast_km=round(float(row["dist_coast_km"]), 2) if pd.notna(row.get("dist_coast_km")) else None,
                dist_nearest_shelter_km=round(float(row["dist_nearest_shelter_km"]), 2) if pd.notna(row.get("dist_nearest_shelter_km")) else None,
                dist_nearest_hospital_km=round(float(row["dist_nearest_hospital_km"]), 2) if pd.notna(row.get("dist_nearest_hospital_km")) else None,
            )
        )

    return CommunitiesListResponse(
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        items=items,
    )


@router.get("/{community_id}", response_model=CommunityDetailResponse, summary="Retrieve granular community inspection dossier")
def get_community_detail(
    community_id: str,
    service: DecisionSupportService = Depends(get_service),
) -> CommunityDetailResponse:
    """Retrieve complete multidimensional dossier, ML predictions, indicators, and advisories."""
    df = service.ranked_df
    clean_id = community_id.strip()

    # Match by pc11_id, shrid2, area_id, or village_name
    match = df[
        (df["pc11_id"].astype(str) == clean_id)
        | (df["shrid2"].astype(str) == clean_id)
        | (df["area_id"].astype(str) == clean_id)
        | (df["village_name"].astype(str).str.lower() == clean_id.lower())
    ]

    if match.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Community '{community_id}' not found in study area.",
        )

    row = match.iloc[0]

    # Evaluate using decision support engine for advisories
    try:
        eval_res = service.evaluate_decision_unit(str(row["pc11_id"]))
        advisories = [f"{adv.advisory_focus}: {adv.operational_recommendation}" for adv in eval_res.recommended_advisories]
        triggers = [adv.directive_id for adv in eval_res.recommended_advisories]
        rationale = eval_res.priority_explanation
    except Exception:
        advisories = ["Standard emergency response protocol."]
        triggers = []
        rationale = "Evaluation determined from deterministic baseline."

    # AHP vs ML agreement & precedence
    pred_tier = str(row.get("predicted_preparedness_tier", "N/A"))
    ahp_tier = str(row["preparedness_tier"])
    ahp_agreement = "FULL_AGREEMENT" if pred_tier.lower() == ahp_tier.lower() else "BORDERLINE_DISAGREEMENT"
    ahp_note = None
    if ahp_tier.lower() == "low" and pred_tier.lower() != "low":
        ahp_note = (
            "AHP PRECEDENCE MANDATE: Authoritative AHP capacity tier is 'Low'. "
            "Per operational safety rules, ML model prediction cannot downgrade emergency priority."
        )

    provenance = {
        "census_vintage": 2011,
        "healthcare_source": "MoHFW/NHM (2015-19)",
        "coastline_source": "Natural Earth 10m (v5.1.1, 2022) in UTM Zone 43N",
        "shelter_source": "NDMA / District Disaster Management Registry",
        "operational_limitations": (
            "Facility coordinates represent administrative registration, NOT confirmed active doctor staffing, "
            "functioning ICU beds, or backup generator fuel status during cyclonic power failure. "
            "Census statistics represent the 2011 decennial baseline."
        ),
    }

    raw_unit_type = row.get("unit_type")
    unit_type = (
        "VILLAGE"
        if pd.isna(raw_unit_type) or str(raw_unit_type).strip().lower() in ("nan", "none", "")
        else str(raw_unit_type).strip()
    )

    return CommunityDetailResponse(
        area_id=str(row["area_id"]),
        pc11_id=str(row["pc11_id"]),
        shrid2=str(row["shrid2"]),
        village_name=str(row["village_name"]),
        district_name=str(row["district_name"]),
        subdistrict_name=str(row["subdistrict_name"]) if pd.notna(row.get("subdistrict_name")) else None,
        unit_type=unit_type,
        lat=round(float(row["lat"]), 6),
        lon=round(float(row["lon"]), 6),
        preparedness_score=round(float(row["preparedness_score"]), 2),
        preparedness_tier=ahp_tier,
        score_medical=round(float(row.get("score_medical", 0.0)), 2),
        score_shelters=round(float(row.get("score_shelters", 0.0)), 2),
        score_infrastructure=round(float(row.get("score_infrastructure", 0.0)), 2),
        score_early_warning=round(float(row.get("score_early_warning", 0.0)), 2),
        score_socioeconomic=round(float(row.get("score_socioeconomic", 0.0)), 2),
        priority_band=str(row["priority_band"]),
        priority_rank=int(row["overall_priority_rank"]),
        band_rank=int(row["band_rank"]),
        triggers_activated=triggers,
        primary_deficit_rationale=rationale,
        actionable_advisories=advisories,
        predicted_preparedness_tier=pred_tier if pd.notna(pred_tier) else None,
        prob_high=round(float(row["prob_high"]), 4) if pd.notna(row.get("prob_high")) else None,
        prob_medium=round(float(row["prob_medium"]), 4) if pd.notna(row.get("prob_medium")) else None,
        prob_low=round(float(row["prob_low"]), 4) if pd.notna(row.get("prob_low")) else None,
        probability_margin=round(float(abs(row.get("prob_high", 0) - row.get("prob_low", 0))), 4) if pd.notna(row.get("prob_high")) and pd.notna(row.get("prob_low")) else None,
        is_boundary_case=bool(row.get("prob_high", 0) > 0.35 and row.get("prob_medium", 0) > 0.35),
        ahp_agreement=ahp_agreement,
        ahp_precedence_note=ahp_note,
        ml_disclaimer=DECISION_SUPPORT_DISCLAIMER,
        vulnerability_archetype_id=int(row["vulnerability_archetype_id"]) if pd.notna(row.get("vulnerability_archetype_id")) else None,
        vulnerability_archetype_label=str(row["vulnerability_archetype_label"]) if pd.notna(row.get("vulnerability_archetype_label")) else None,
        cluster_assignment_status=str(row.get("cluster_assignment_status", "ASSIGNED" if pd.notna(row.get("vulnerability_archetype_id")) else "UNASSIGNED_FALLBACK")),
        archetype_intervention_need=str(row.get("archetype_intervention_need", "Standard multi-sector mitigation protocol.")) if pd.notna(row.get("archetype_intervention_need")) else None,
        population=int(row["population"]) if pd.notna(row.get("population")) else 0,
        households=int(row["households"]) if pd.notna(row.get("households")) else 0,
        pop_density_sqkm=round(float(row.get("pop_density_sqkm", 0.0)), 1),
        vulnerable_pop_share=round(float(row.get("vulnerable_pop_share", 0.0)), 4),
        sc_share=round(float(row.get("sc_share", 0.0)), 4),
        st_share=round(float(row.get("st_share", 0.0)), 4),
        dist_coast_km=round(float(row.get("dist_coast_km", 0.0)), 2),
        elevation_mean=round(float(row.get("elevation_mean", 0.0)), 1),
        dist_nearest_shelter_km=round(float(row.get("dist_nearest_shelter_km", 0.0)), 2),
        n_shelters_10km=int(row.get("n_shelters_10km", 0)),
        dist_nearest_hospital_km=round(float(row.get("dist_nearest_hospital_km", 0.0)), 2),
        dist_nearest_phc_km=round(float(row.get("dist_nearest_phc_km", 0.0)), 2),
        has_phc_or_chc=bool(row.get("has_phc_or_chc", False)),
        road_connectivity_score=round(float(row.get("road_connectivity_score", 0.0)), 2),
        drinking_water_score=round(float(row.get("drinking_water_score", 0.0)), 2),
        has_pucca_road=bool(row.get("has_pucca_road", False)),
        bus_connectivity=bool(row.get("bus_connectivity", False)),
        tauktae_affected=bool(row.get("tauktae_affected", False)) if pd.notna(row.get("tauktae_affected")) else None,
        provenance_and_limitations=provenance,
    )
