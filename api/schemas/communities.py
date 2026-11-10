"""Pydantic schemas for community explorer and individual dossier inspection."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CommunityListItem(BaseModel):
    """Compact community item for table list and filtering."""
    area_id: str
    pc11_id: str
    shrid2: str
    village_name: str
    district_name: str
    subdistrict_name: Optional[str] = None
    unit_type: str = "VILLAGE"
    lat: float
    lon: float
    preparedness_score: float
    preparedness_tier: str
    priority_band: str
    priority_rank: int
    band_rank: int
    vulnerability_archetype_id: Optional[int] = None
    vulnerability_archetype_label: Optional[str] = None
    population: Optional[int] = None
    dist_coast_km: Optional[float] = None
    dist_nearest_shelter_km: Optional[float] = None
    dist_nearest_hospital_km: Optional[float] = None


class CommunitiesListResponse(BaseModel):
    """Paginated list of communities."""
    total: int
    page: int
    limit: int
    total_pages: int
    items: List[CommunityListItem]


class CommunityDetailResponse(BaseModel):
    """Comprehensive administrative unit inspection dossier."""
    # Identification
    area_id: str
    pc11_id: str
    shrid2: str
    village_name: str
    district_name: str
    subdistrict_name: Optional[str] = None
    unit_type: str = "VILLAGE"
    lat: float
    lon: float

    # Deterministic AHP
    preparedness_score: float
    preparedness_tier: str
    score_medical: float
    score_shelters: float
    score_infrastructure: float
    score_early_warning: float
    score_socioeconomic: float

    # Operational Prioritization
    priority_band: str
    priority_rank: int
    band_rank: int
    triggers_activated: List[str]
    primary_deficit_rationale: str
    actionable_advisories: List[str]

    # Surrogate ML Classification
    predicted_preparedness_tier: Optional[str] = None
    prob_high: Optional[float] = None
    prob_medium: Optional[float] = None
    prob_low: Optional[float] = None
    probability_margin: Optional[float] = None
    is_boundary_case: Optional[bool] = None
    ahp_agreement: Optional[str] = None
    ahp_precedence_note: Optional[str] = None
    ml_disclaimer: str

    # Vulnerability Archetype
    vulnerability_archetype_id: Optional[int] = None
    vulnerability_archetype_label: Optional[str] = None
    cluster_assignment_status: str
    archetype_intervention_need: Optional[str] = None

    # Granular Spatial & Census Indicators
    population: int
    households: int
    pop_density_sqkm: float
    vulnerable_pop_share: float
    sc_share: float
    st_share: float
    dist_coast_km: float
    elevation_mean: float
    dist_nearest_shelter_km: float
    n_shelters_10km: int
    dist_nearest_hospital_km: float
    dist_nearest_phc_km: float
    has_phc_or_chc: bool
    road_connectivity_score: float
    drinking_water_score: float
    has_pucca_road: bool
    bus_connectivity: bool
    tauktae_affected: Optional[bool] = None

    # Provenance
    provenance_and_limitations: Dict[str, Any]
