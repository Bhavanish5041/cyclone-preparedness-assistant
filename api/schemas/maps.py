"""Pydantic schemas for geospatial maps endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MapCommunityPoint(BaseModel):
    """Spatial point for Leaflet/MapLibre mapping."""
    area_id: str
    pc11_id: str
    name: str
    district: str
    subdistrict: Optional[str] = None
    lat: float
    lon: float
    score: float
    tier: str
    band: str
    priority_rank: int
    archetype_id: Optional[int] = None
    dist_coast_km: Optional[float] = None
    dist_nearest_shelter_km: Optional[float] = None
    dist_nearest_hospital_km: Optional[float] = None
    population: Optional[int] = None


class MapCommunitiesResponse(BaseModel):
    """Geo-data payload for all or filtered communities."""
    total: int
    points: List[MapCommunityPoint]
