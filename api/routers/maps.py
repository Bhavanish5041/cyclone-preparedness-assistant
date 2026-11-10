"""Geospatial maps endpoints."""

from typing import List, Optional
import pandas as pd
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_service
from api.schemas.maps import MapCommunitiesResponse, MapCommunityPoint
from src.assistant.decision_support import DecisionSupportService

router = APIRouter(prefix="/maps", tags=["Maps & Geospatial"])


@router.get("/communities", response_model=MapCommunitiesResponse, summary="Retrieve spatial points for interactive map")
def get_map_communities(
    district: Optional[str] = Query(None, description="Filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada')"),
    tier: Optional[str] = Query(None, description="Filter by preparedness tier ('High', 'Medium', 'Low')"),
    priority_band: Optional[str] = Query(None, description="Filter by priority band ('BAND_1_CRITICAL', etc.)"),
    archetype_id: Optional[int] = Query(None, description="Filter by archetype ID (0, 1, 2, 3 or -1 for unassigned)"),
    service: DecisionSupportService = Depends(get_service),
) -> MapCommunitiesResponse:
    """Return lightweight geospatial coordinate records with priority attributes for mapping."""
    df = service.ranked_df.copy()

    if district and district.strip():
        df = df[df["district_name"].str.lower() == district.strip().lower()]

    if tier and tier.strip():
        df = df[df["preparedness_tier"].astype(str).str.lower() == tier.strip().lower()]

    if priority_band and priority_band.strip():
        df = df[df["priority_band"].astype(str).str.upper() == priority_band.strip().upper()]

    if archetype_id is not None:
        if archetype_id == -1:
            df = df[df["vulnerability_archetype_id"].isna()]
        else:
            df = df[df["vulnerability_archetype_id"] == archetype_id]

    points: List[MapCommunityPoint] = []
    for _, row in df.iterrows():
        points.append(
            MapCommunityPoint(
                area_id=str(row["area_id"]),
                pc11_id=str(row["pc11_id"]),
                name=str(row["village_name"]),
                district=str(row["district_name"]),
                subdistrict=str(row["subdistrict_name"]) if pd.notna(row.get("subdistrict_name")) else None,
                lat=round(float(row["lat"]), 6),
                lon=round(float(row["lon"]), 6),
                score=round(float(row["preparedness_score"]), 2),
                tier=str(row["preparedness_tier"]),
                band=str(row["priority_band"]),
                priority_rank=int(row["overall_priority_rank"]),
                archetype_id=int(row["vulnerability_archetype_id"]) if pd.notna(row.get("vulnerability_archetype_id")) else None,
                dist_coast_km=round(float(row["dist_coast_km"]), 2) if pd.notna(row.get("dist_coast_km")) else None,
                dist_nearest_shelter_km=round(float(row["dist_nearest_shelter_km"]), 2) if pd.notna(row.get("dist_nearest_shelter_km")) else None,
                dist_nearest_hospital_km=round(float(row["dist_nearest_hospital_km"]), 2) if pd.notna(row.get("dist_nearest_hospital_km")) else None,
                population=int(row["population"]) if pd.notna(row.get("population")) else None,
            )
        )

    return MapCommunitiesResponse(total=len(points), points=points)
