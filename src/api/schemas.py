"""Pydantic request and response schemas for Cyclone Preparedness Decision Support API.

Defines strict type contracts, validation rules, and documentation for:
- Health and system readiness checks
- Geographic unit search and granular detail inspection
- Prioritization weighting configurations and dynamic recomputation
- Multi-scenario sensitivity and rank divergence comparison
- Discovered vulnerability archetypes and operational intervention summaries
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class HealthCheckResponse(BaseModel):
    """System health check and dataset inventory status."""

    status: str = Field("healthy", description="Service health state")
    dataset_loaded: bool = Field(..., description="Whether archetype dataset is loaded in memory")
    total_units: int = Field(..., description="Total administrative units in spatial coverage")
    total_villages: int = Field(..., description="Count of rural revenue villages (563 in study area)")
    total_census_towns: int = Field(..., description="Count of statutory/Census towns (59 in study area)")
    districts: List[str] = Field(..., description="Covered coastal districts")
    archetypes_count: int = Field(4, description="Number of discovered vulnerability archetypes")
    version: str = Field("1.0.0", description="API contract version")


class WeightConfiguration(BaseModel):
    """Configurable weights for multidimensional prioritization heuristic."""

    healthcare: float = Field(0.25, ge=0.0, description="Weight for healthcare deficit dimension")
    demographic: float = Field(0.25, ge=0.0, description="Weight for demographic vulnerability dimension")
    infrastructure: float = Field(0.25, ge=0.0, description="Weight for infrastructure deficit dimension")
    coastal: float = Field(0.25, ge=0.0, description="Weight for coastal hazard proximity dimension")
    auto_normalize: bool = Field(
        True,
        description="If True, automatically normalizes weights to sum to 1.0 (e.g. from UI sliders [0..100]). "
        "If False, rejects configurations whose sum is not close to 1.0.",
    )
    include_population_scale: bool = Field(
        False,
        description="Whether to incorporate log-transformed population scale into demographic score alongside SC/ST ratio.",
    )

    @model_validator(mode="after")
    def validate_and_normalize_weights(self) -> "WeightConfiguration":
        """Validate non-negativity and handle normalization."""
        raw_sum = self.healthcare + self.demographic + self.infrastructure + self.coastal
        if raw_sum <= 0.0:
            raise ValueError("Total weight sum must be strictly positive (cannot all be zero).")

        if not self.auto_normalize and abs(raw_sum - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0 when auto_normalize is False, got {raw_sum:.4f}")

        return self

    def get_normalized_weights(self) -> Dict[str, float]:
        """Return normalized dictionary of weights summing exactly to 1.0."""
        total = self.healthcare + self.demographic + self.infrastructure + self.coastal
        return {
            "healthcare": round(self.healthcare / total, 6),
            "demographic": round(self.demographic / total, 6),
            "infrastructure": round(self.infrastructure / total, 6),
            "coastal": round(self.coastal / total, 6),
        }


class UnitSummary(BaseModel):
    """Compact summary of an administrative unit."""

    shrid2: str = Field(..., description="SHRUG secondary polygon identifier")
    pc11_id: str = Field(..., description="Census of India 2011 Primary Key")
    unit_name: str = Field(..., description="Village or Census Town official name")
    unit_type: str = Field(..., description="'VILLAGE' or 'TOWN'")
    district_name: str = Field(..., description="Dakshina Kannada or Udupi")
    priority_score: float = Field(..., description="Current priority score [0.0 - 1.0]")
    priority_rank: int = Field(..., description="Current rank (1 = highest priority need)")
    vulnerability_archetype_id: int = Field(..., description="Discovered archetype cluster ID (0..3)")
    vulnerability_archetype_label: str = Field(..., description="Descriptive archetype label")


class UnitDetailResponse(BaseModel):
    """Granular inspection record for a single administrative unit."""

    shrid2: str
    pc11_id: str
    unit_name: str
    unit_type: str
    district_name: str
    latitude: float
    longitude: float

    # The 4 Dimension Scores (0.0 to 1.0)
    dimension_scores: Dict[str, float] = Field(
        ...,
        description="Four independent normalized dimension scores (healthcare, demographic, infrastructure, coastal)",
    )

    # Raw Supporting Proxies
    raw_indicators: Dict[str, Any] = Field(
        ...,
        description="Raw physical indicators (distances in km, population counts, facility names)",
    )

    # Current Prioritization Ranking
    priority_score: float
    priority_rank: int

    # Discovered Archetype Profile
    archetype: Dict[str, Any] = Field(
        ...,
        description="Archetype assignment, descriptive profile, and recommended intervention guidance",
    )

    # Provenance, Vintages, & Operational Disclaimers
    provenance_and_limitations: Dict[str, Any] = Field(
        ...,
        description="Vintages (Census 2011, MoHFW 2015-19, NE 10m 2022) and explicit operational caveats",
    )


class PrioritizationScenarioRequest(BaseModel):
    """Request payload for custom prioritization weighting scenario."""

    scenario_name: str = Field("custom_scenario", description="Descriptive scenario name")
    weights: WeightConfiguration = Field(default_factory=WeightConfiguration)
    top_n: int = Field(20, ge=1, le=622, description="Number of top units to return")
    district_filter: Optional[str] = Field(
        None, description="Optional filter by district ('Dakshina Kannada' or 'Udupi')"
    )


class PrioritizationScenarioResponse(BaseModel):
    """Response payload for custom prioritization weighting scenario."""

    scenario_name: str
    weights_applied: Dict[str, float]
    include_population_scale: bool
    total_units_evaluated: int
    score_statistics: Dict[str, float]
    top_units: List[UnitSummary]
    disclaimer: str


class ScenarioComparisonRequest(BaseModel):
    """Request payload to compare rankings between two weighting scenarios."""

    scenario_a_name: str = Field("Scenario A (e.g. Equal Weights)")
    scenario_a_weights: WeightConfiguration = Field(default_factory=WeightConfiguration)

    scenario_b_name: str = Field("Scenario B (e.g. Hazard Focused)")
    scenario_b_weights: WeightConfiguration = Field(
        default_factory=lambda: WeightConfiguration(healthcare=0.15, demographic=0.15, infrastructure=0.20, coastal=0.50)
    )

    top_divergent_count: int = Field(15, ge=1, le=100, description="Number of top rank-shifting units to report")


class RankDivergenceItem(BaseModel):
    """Details of a single unit whose rank shifted significantly between scenarios."""

    shrid2: str
    unit_name: str
    district_name: str
    vulnerability_archetype_label: str
    rank_scenario_a: int
    rank_scenario_b: int
    rank_shift: int = Field(..., description="rank_a - rank_b (positive = higher priority in B)")
    direction: str = Field(..., description="'Surged in Priority' or 'Dropped in Priority'")


class ScenarioComparisonResponse(BaseModel):
    """Response payload comparing two prioritization scenarios."""

    scenario_a_name: str
    scenario_b_name: str
    spearman_rank_correlation: float
    kendall_tau_correlation: float
    mean_absolute_rank_shift: float
    max_rank_shift: int
    top_divergent_units: List[RankDivergenceItem]
    methodological_note: str


class ArchetypeSummary(BaseModel):
    """Descriptive and statistical summary of a single vulnerability archetype."""

    archetype_id: int
    short_label: str
    detailed_description: str
    primary_intervention_need: str
    unit_count: int
    percentage_of_total: float
    count_dakshina_kannada: int
    count_udupi: int
    dimension_means: Dict[str, float]
    raw_indicator_means: Dict[str, float]


class ArchetypeCatalogResponse(BaseModel):
    """Catalog of all discovered vulnerability archetypes."""

    total_archetypes: int
    archetypes: List[ArchetypeSummary]
    clustering_methodology: str
    clustering_disclaimer: str
