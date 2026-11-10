"""Pydantic schemas for dashboard overview and regional executive metrics."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DistrictOverviewItem(BaseModel):
    """District summary in dashboard overview."""
    district_name: str
    total_units: int
    mean_preparedness_score: float
    band_counts: Dict[str, int]
    tier_counts: Dict[str, int]
    mean_dist_nearest_shelter_km: float
    mean_dist_nearest_hospital_km: float
    mean_dist_coast_km: float
    critical_units_count: int


class DashboardOverviewResponse(BaseModel):
    """Authoritative regional dashboard overview across all 1,879 administrative units."""
    total_units: int = Field(default=1879, description="Total revenue villages and census towns")
    districts_count: int = Field(default=3, description="Count of coastal Karnataka districts")
    assigned_clusters_count: int = Field(default=622, description="Units assigned to K-Means archetypes (DK & Udupi)")
    unassigned_clusters_count: int = Field(default=1257, description="Units in Uttara Kannada using fallback")
    priority_band_counts: Dict[str, int] = Field(
        ...,
        description="Authoritative priority band counts (BAND_1_CRITICAL, BAND_2_SPECIALIZED, BAND_3_MODERATE, BAND_4_ADEQUATE)"
    )
    preparedness_tier_counts: Dict[str, int] = Field(
        ...,
        description="Deterministic AHP capacity tier counts (High, Medium, Low)"
    )
    cluster_counts: Dict[str, int] = Field(
        ...,
        description="Vulnerability cluster counts"
    )
    districts: List[DistrictOverviewItem] = Field(
        ...,
        description="Per-district aggregations"
    )
    infrastructure_highlights: Dict[str, Any] = Field(
        ...,
        description="Aggregated infrastructure numbers (population, shelters, healthcare)"
    )
    data_disclaimer: str = Field(
        ...,
        description="Scientific governance and vintage disclaimer"
    )
