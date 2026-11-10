"""Pydantic schemas for district intelligence and comparative metrics."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SubdistrictSummary(BaseModel):
    """Taluk / subdistrict summary within a district."""
    subdistrict_name: str
    total_units: int
    mean_preparedness_score: float
    band_counts: Dict[str, int]
    critical_units_count: int


class DistrictSummary(BaseModel):
    """Comprehensive district summary model."""
    district_name: str
    district_code: Optional[str] = None
    total_units: int
    mean_preparedness_score: float
    min_preparedness_score: float
    max_preparedness_score: float
    band_counts: Dict[str, int]
    tier_counts: Dict[str, int]
    cluster_counts: Dict[str, int]
    mean_dist_coast_km: float
    mean_dist_nearest_shelter_km: float
    mean_dist_nearest_hospital_km: float
    mean_dist_nearest_phc_km: float
    pct_pucca_road: float
    pct_drinking_water_secure: float
    total_population: int
    total_households: int
    mean_vulnerable_pop_share: float
    subdistricts: Optional[List[SubdistrictSummary]] = None


class DistrictListResponse(BaseModel):
    """List of all 3 coastal districts."""
    total_districts: int = 3
    districts: List[DistrictSummary]
