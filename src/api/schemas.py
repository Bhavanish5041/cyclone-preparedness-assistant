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
    """System health check, dataset inventory status, and model readiness."""

    status: str = Field("healthy", description="Service health state")
    dataset_loaded: bool = Field(..., description="Whether archetype dataset is loaded in memory")
    total_units: int = Field(..., description="Total administrative units in spatial coverage")
    total_villages: int = Field(..., description="Count of rural revenue villages (563 in study area)")
    total_census_towns: int = Field(..., description="Count of statutory/Census towns (59 in study area)")
    districts: List[str] = Field(..., description="Covered coastal districts")
    archetypes_count: int = Field(4, description="Number of discovered vulnerability archetypes")
    version: str = Field("1.0.0", description="API contract version")

    # Phase 2.4 Model Readiness Fields
    model_loaded: bool = Field(False, description="Whether the W&B registered model is loaded and ready for inference")
    model_artifact_name: Optional[str] = Field(None, description="Registered W&B artifact name")
    model_artifact_version: Optional[str] = Field(None, description="Exact active artifact version or alias")
    model_feature_schema_status: Optional[str] = Field(None, description="Feature schema validation status")
    model_provenance: Optional[str] = Field(None, description="Model lineage and provenance statement")


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


# =============================================================================
# Milestone 3B: Integrated Decision-Support Engine Schemas
# =============================================================================


class InterventionItem(BaseModel):
    """Specific evidence-grounded advisory planning recommendation."""

    directive_id: str = Field(..., description="Unique advisory code (e.g., ADV_1_HOSPITAL_ISOLATION)")
    advisory_focus: str = Field(..., description="Thematic sector (e.g., Healthcare Transit Isolation)")
    operational_recommendation: str = Field(
        ..., description="Actionable advisory verification checklist for local authorities"
    )
    lead_coordinating_agency: str = Field(
        ..., description="Recommended coordinating authority (e.g., Taluk Health Officer, Gram Panchayat)"
    )
    advisory_type: str = Field(
        ..., description="'INDIVIDUAL_EVIDENCE' (unit condition met) or 'ARCHETYPE_CONTEXTUAL' (cluster context)"
    )
    eligibility_rule: str = Field(..., description="Boolean formula evaluated on unit indicators")
    supporting_evidence: Dict[str, Any] = Field(
        ..., description="Actual observed column values for this specific unit justifying the advisory"
    )


class DecisionSupportEvaluation(BaseModel):
    """Complete decision-support evaluation for an administrative unit."""

    # 1. Geographic and Administrative Identification
    shrid2: str = Field(..., description="Canonical SHRUG primary identifier")
    pc11_id: str = Field(..., description="Census of India 2011 identifier")
    village_name: str = Field(..., description="Official revenue village or town name")
    district_name: str = Field(..., description="Dakshina Kannada, Udupi, or Uttara Kannada")
    subdistrict_name: str = Field(..., description="Taluk / Subdistrict name")
    unit_type: Optional[str] = Field(None, description="VILLAGE, TOWN, or None (1,257 nulls in UK)")
    lat: float = Field(..., description="Representative point latitude (WGS84)")
    lon: float = Field(..., description="Representative point longitude (WGS84)")

    # 2. Lens 1: Deterministic AHP Preparedness Baseline
    preparedness_score: float = Field(..., ge=0.0, le=100.0, description="Deterministic AHP index [0.0 - 100.0]")
    preparedness_tier: str = Field(..., description="Deterministic AHP baseline tier: Low, Medium, or High")
    domain_scores: Dict[str, float] = Field(
        ...,
        description="AHP standardized domain scores: score_medical, score_shelters, score_infrastructure, score_early_warning, score_socioeconomic",
    )

    # 3. Lens 2: Statistical Classification Stability (Multinomial Logistic Regression)
    predicted_preparedness_tier: str = Field(..., description="Model-predicted tier (Logistic Regression)")
    class_probabilities: Dict[str, float] = Field(
        ..., description="Uncalibrated probabilities: prob_low, prob_medium, prob_high"
    )
    max_class_probability: float = Field(..., ge=0.3333, le=1.0, description="Highest class probability")
    probability_margin: float = Field(..., ge=0.0, le=1.0, description="Margin: highest minus second-highest class")
    prediction_provenance: str = Field(
        ..., description="Data split: TRAIN_IN_SAMPLE (1,315), VAL_OUT_OF_SAMPLE (282), or TEST_OUT_OF_SAMPLE (282)"
    )
    model_agreement_status: str = Field(
        ..., description="FULL_AGREEMENT (1,792) or BORDERLINE_DISAGREEMENT (87)"
    )
    is_boundary_case: bool = Field(
        ..., description="True if probability_margin < 0.10 or max_class_probability < 0.60 (provisional flag)"
    )

    # 4. Lens 3: Spatial Vulnerability Clustering (K-Means)
    vulnerability_archetype_id: Optional[int] = Field(None, description="0, 1, 2, 3, or None (1,257 nulls in UK)")
    vulnerability_archetype_label: Optional[str] = Field(None, description="Archetype label or None (1,257 nulls in UK)")
    cluster_assignment_status: str = Field(
        ..., description="'ASSIGNED' (DK & Udupi, N=622) or 'UNASSIGNED_OUT_OF_STUDY_AREA' (UK, N=1,257)"
    )

    # 5. Synthesized Decision Support Outputs
    provisional_priority_band: str = Field(
        ..., description="BAND_1_CRITICAL, BAND_2_SPECIALIZED, BAND_3_MODERATE, or BAND_4_ADEQUATE"
    )
    band_rank: int = Field(..., ge=1, le=1879, description="Within-band deterministic rank (1 = highest priority)")
    overall_priority_rank: int = Field(
        ..., ge=1, le=1879, description="Overall deterministic priority rank (1 = highest intervention priority)"
    )
    priority_explanation: str = Field(..., description="Transparent textual synthesis of priority factors")
    recommended_advisories: List[InterventionItem] = Field(..., description="List of evidence-grounded advisories")

    # 6. Data Integrity and Advisory Planning Disclaimers
    has_unverified_indicators: bool = Field(False, description="True if any required indicator was null/missing")
    advisory_disclaimer: str = Field(
        ..., description="Advisory disclaimer reinforcing decision-support nature of recommendations"
    )


class RegionalDecisionSummaryResponse(BaseModel):
    """Regional summary of operational priority bands, trigger counts, and model agreement."""

    total_units: int = Field(1879, description="Total administrative units in regional coverage")
    band_counts: Dict[str, int] = Field(..., description="Count of units per priority band")
    district_band_breakdown: Dict[str, Dict[str, int]] = Field(..., description="Priority bands by district")
    model_agreement_counts: Dict[str, int] = Field(..., description="FULL_AGREEMENT vs BORDERLINE_DISAGREEMENT")
    provenance_agreement: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Agreement metrics separated by train (in-sample) vs validation/test (out-of-sample) splits",
    )
    uttara_kannada_trigger_breakdown: Dict[str, int] = Field(..., description="Trigger counts for unassigned UK units")
    methodological_disclaimer: str = Field(..., description="Methodological disclaimer on advisory planning nature")


class DecisionEvaluationListResponse(BaseModel):
    """Paginated list of decision-support evaluations."""

    total_evaluated: int = Field(..., description="Total units evaluated in dataset (1,879)")
    total_matching: int = Field(..., description="Total units matching query filters")
    limit: int = Field(..., description="Page size limit")
    offset: int = Field(..., description="Offset index")
    units: List[DecisionSupportEvaluation] = Field(..., description="Matching evaluated units")


# =============================================================================
# Phase 2.4: Registered Model Inference Schemas
# =============================================================================

class ModelPredictionRequest(BaseModel):
    """Request payload for model inference."""

    area_id: Optional[str] = Field(
        None,
        description="Area identifier (SHRID, PC11 ID, or village name) to load from unified dataset",
        examples=["11-29-575-05562-617533", "617533", "Panjikal"],
    )
    features: Optional[Dict[str, float]] = Field(
        None,
        description="Direct dictionary of feature values (either 26 canonical features or raw indicators for engineering)",
    )

    @model_validator(mode="after")
    def validate_request_payload(self) -> "ModelPredictionRequest":
        if not self.area_id and not self.features:
            raise ValueError("Either 'area_id' or 'features' dictionary must be provided.")
        return self


class ModelPredictionResponse(BaseModel):
    """Calibrated model prediction and diagnostic response payload."""

    area_id: Optional[str] = Field(None, description="Queried or resolved area identifier")
    village_name: Optional[str] = Field(None, description="Village name if resolved")
    district_name: Optional[str] = Field(None, description="District name if resolved")

    # Model Predictions & Calibrated Probabilities
    predicted_preparedness_tier: str = Field(..., description="Model-predicted tier: High, Low, or Medium")
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Calibrated tier probabilities strictly summing to 1.0 (prob_high, prob_low, prob_medium)",
    )
    probability_margin: float = Field(..., description="Margin: highest minus second-highest class probability")
    max_probability: float = Field(..., description="Maximum class probability")
    is_boundary_case: bool = Field(..., description="True if margin < 0.10 or max_probability < 0.60")

    # Authoritative Baseline AHP Context
    deterministic_ahp_tier: Optional[str] = Field(None, description="Authoritative baseline AHP tier if known")
    deterministic_ahp_score: Optional[float] = Field(None, description="Authoritative baseline AHP score if known")
    ahp_agreement_status: Optional[str] = Field(
        None,
        description="FULL_AGREEMENT (model tier == AHP tier) or BORDERLINE_DISAGREEMENT (model tier != AHP tier)",
    )
    ahp_precedence_note: Optional[str] = Field(
        None,
        description="Operational precedence note reinforcing that deterministic AHP classification cannot be downgraded",
    )

    # Model Artifact & Lineage
    model_artifact_name: str = Field(..., description="Registered W&B model artifact name")
    model_artifact_version: str = Field(..., description="Active W&B artifact version or alias")
    model_provenance: str = Field(..., description="Model lineage and provenance statement")
    target_nature: str = Field("AHP_CAPACITY_TIER_SURROGATE", description="Operational nature of prediction target")
    disclaimer: str = Field(..., description="Mandatory operational disclaimer clarifying surrogate capacity nature")


class ModelHealthResponse(BaseModel):
    """Status and metadata of the active registered W&B model artifact."""

    status: str = Field("healthy", description="Model service health state")
    model_loaded: bool = Field(..., description="Whether the registered model container is active")
    artifact_name: str = Field(..., description="Registered W&B artifact name")
    artifact_version: str = Field(..., description="Exact active artifact version or alias")
    artifact_source: str = Field(..., description="Qualified artifact source identifier")
    features_count: int = Field(26, description="Expected feature vector dimension")
    feature_schema_status: str = Field(..., description="Validation status of the 26 canonical features")
    target_classes: List[str] = Field(..., description="Target preparedness classes: High, Low, Medium")
    model_provenance: str = Field(..., description="Lineage and reconstruction provenance")
    target_nature: str = Field("AHP_CAPACITY_TIER_SURROGATE", description="Operational nature of target")

