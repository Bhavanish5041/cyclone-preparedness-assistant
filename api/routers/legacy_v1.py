"""Backward-compatibility router for v1 REST endpoints.

Provides compatibility with:
- Streamlit decision-support dashboard in REST mode (DecisionSupportApiClient)
- Legacy v1 API clients and integration test suites
- Decoupled v1 endpoint contracts

All analytical formulations remain strictly encapsulated within DecisionSupportService
and ToolCallingOrchestrator, maintaining separation of concerns and deterministic precedence.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from api.dependencies import (
    get_model,
    get_orchestrator,
    get_service,
)
from src.api.schemas import (
    ArchetypeCatalogResponse,
    DecisionEvaluationListResponse,
    DecisionSupportEvaluation,
    ModelHealthResponse,
    ModelPredictionRequest,
    ModelPredictionResponse,
    PrioritizationScenarioRequest,
    PrioritizationScenarioResponse,
    RegionalDecisionSummaryResponse,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
    UnitDetailResponse,
    UnitSummary,
)
from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.orchestrator import ToolCallingOrchestrator
from src.assistant.llm.schemas import ChatRequest, ChatResponse
from src.features.inference_features import engineer_inference_features
from src.models.load_registered_model import RegisteredModelContainer
from src.monitoring.prometheus_metrics import (
    record_inference_error,
    record_inference_success,
)

logger = logging.getLogger("api_legacy_v1")

router = APIRouter(prefix="/api/v1", tags=["Legacy Compatibility (v1)"])

MODEL_PREDICTION_DISCLAIMER = (
    "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate "
    "approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated "
    "physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. "
    "Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
)


# =============================================================================
# Geographic Units & Inspection
# =============================================================================


@router.get(
    "/units/search",
    response_model=List[UnitSummary],
    summary="Search administrative units by name, Census ID, or SHRID",
)
def search_units(
    query: str = Query(
        ...,
        min_length=1,
        description="Search term matching village/town name, Census PC11 ID, or SHRID",
        examples=["Tallur", "608720", "11-29-569"],
    ),
    district: Optional[str] = Query(
        None,
        description="Optional filter by district name ('Dakshina Kannada' or 'Udupi')",
        examples=["Udupi", "Dakshina Kannada"],
    ),
    limit: int = Query(
        15,
        ge=1,
        le=100,
        description="Maximum number of matching units to return",
    ),
    service: DecisionSupportService = Depends(get_service),
) -> List[UnitSummary]:
    """Search and filter administrative units with case-insensitive matching."""
    clean_q = query.strip()
    if not clean_q:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Search query cannot be empty or solely whitespace.",
        )

    valid_districts = {"dakshina kannada", "udupi", "uttara kannada"}
    if district and district.strip().lower() not in valid_districts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid district '{district}'. Allowed values: 'Dakshina Kannada', 'Udupi', or 'Uttara Kannada'.",
        )

    return service.search_units(query=clean_q, district_filter=district, limit=limit)


@router.get(
    "/units/{identifier}",
    response_model=UnitDetailResponse,
    summary="Retrieve granular indicators, dimension scores, and archetype guidance for a unit",
)
def get_unit_detail(
    identifier: str = Path(
        ...,
        min_length=1,
        description="Census 2011 PC11 ID (e.g. '608720'), SHRID (e.g. '11-29-569-05522-608720'), or exact unit name",
        examples=["608720", "Tallur"],
    ),
    service: DecisionSupportService = Depends(get_service),
) -> UnitDetailResponse:
    """Retrieve full multidimensional inspection record for an administrative unit."""
    try:
        return service.get_unit_detail(identifier)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Administrative unit '{identifier}' not found in coastal Karnataka study area.",
        )


# =============================================================================
# Prioritization Scenarios & Archetypes
# =============================================================================


@router.post(
    "/prioritization/scenario",
    response_model=PrioritizationScenarioResponse,
    summary="Compute priority scores and deterministic rankings under custom weights",
)
def compute_prioritization_scenario(
    request: PrioritizationScenarioRequest,
    service: DecisionSupportService = Depends(get_service),
) -> PrioritizationScenarioResponse:
    """Dynamically recompute multidimensional prioritization scores under custom weights."""
    valid_districts = {"dakshina kannada", "udupi", "uttara kannada"}
    if request.district_filter and request.district_filter.strip().lower() not in valid_districts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid district filter '{request.district_filter}'. Allowed: 'Dakshina Kannada', 'Udupi', 'Uttara Kannada'.",
        )

    return service.compute_prioritization_scenario(request)


@router.post(
    "/prioritization/compare",
    response_model=ScenarioComparisonResponse,
    summary="Compare rankings between two weighting scenarios and analyze rank divergence",
)
def compare_prioritization_scenarios(
    request: ScenarioComparisonRequest,
    service: DecisionSupportService = Depends(get_service),
) -> ScenarioComparisonResponse:
    """Evaluate Spearman rank correlation and top diverging units between two policy weighting schemes."""
    return service.compare_scenarios(request)


@router.get(
    "/archetypes",
    response_model=ArchetypeCatalogResponse,
    summary="Retrieve catalog of all four discovered vulnerability archetypes and intervention guidance",
)
def get_archetype_catalog(
    service: DecisionSupportService = Depends(get_service),
) -> ArchetypeCatalogResponse:
    """Return descriptive profiles, district distributions, and intervention checklists for all archetypes."""
    return service.get_archetype_catalog()


# =============================================================================
# Integrated Decision-Support Endpoints
# =============================================================================


@router.get(
    "/decision-support/summary",
    response_model=RegionalDecisionSummaryResponse,
    summary="Retrieve regional summary of priority bands, trigger counts, and model concordance",
)
def get_regional_decision_summary(
    service: DecisionSupportService = Depends(get_service),
) -> RegionalDecisionSummaryResponse:
    """Return regional aggregation across all 1,879 records in the study area."""
    return service.get_regional_decision_summary()


@router.get(
    "/decision-support/units/{identifier}",
    response_model=DecisionSupportEvaluation,
    summary="Retrieve full decision-support evaluation and grounded advisories for an administrative unit",
)
def evaluate_decision_unit(
    identifier: str = Path(
        ...,
        min_length=1,
        description="Census PC11 ID (e.g. '608720'), SHRID (e.g. '11-29-569-05522-608720'), or village name",
        examples=["608720", "Tallur", "Bappalgundi"],
    ),
    service: DecisionSupportService = Depends(get_service),
) -> DecisionSupportEvaluation:
    """Evaluate a unit's priority band, within-band rank, model stability, and grounded advisories."""
    try:
        return service.evaluate_decision_unit(identifier)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Administrative unit '{identifier}' not found in unified preparedness dataset.",
        )


@router.get(
    "/decision-support/units",
    response_model=DecisionEvaluationListResponse,
    summary="Retrieve paginated decision evaluations with optional district and priority band filtering",
)
def get_decision_units(
    district: Optional[str] = Query(
        None,
        description="Optional filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada')",
        examples=["Uttara Kannada", "Dakshina Kannada", "Udupi"],
    ),
    priority_band: Optional[str] = Query(
        None,
        description="Optional filter by priority band ('BAND_1_CRITICAL', 'BAND_2_SPECIALIZED', 'BAND_3_MODERATE', 'BAND_4_ADEQUATE')",
        examples=["BAND_1_CRITICAL", "BAND_2_SPECIALIZED"],
    ),
    limit: int = Query(
        50,
        ge=1,
        le=1879,
        description="Maximum number of units to return per page",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Offset index for pagination",
    ),
    service: DecisionSupportService = Depends(get_service),
) -> DecisionEvaluationListResponse:
    """Retrieve filtered, paginated decision evaluations ranked deterministically by priority."""
    valid_districts = {"dakshina kannada", "udupi", "uttara kannada"}
    if district and district.strip().lower() not in valid_districts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid district '{district}'. Allowed: 'Dakshina Kannada', 'Udupi', 'Uttara Kannada'.",
        )

    valid_bands = {"BAND_1_CRITICAL", "BAND_2_SPECIALIZED", "BAND_3_MODERATE", "BAND_4_ADEQUATE"}
    if priority_band and priority_band.strip().upper() not in valid_bands:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid priority_band '{priority_band}'. Allowed: {sorted(valid_bands)}.",
        )

    return service.get_decision_units(
        district=district,
        priority_band=priority_band,
        limit=limit,
        offset=offset,
    )


# =============================================================================
# Registered Model Inference & Diagnostics
# =============================================================================


@router.get(
    "/model/health",
    response_model=ModelHealthResponse,
    summary="Check registered model health, active artifact version, and schema integrity",
)
def get_model_health(
    model: Optional[RegisteredModelContainer] = Depends(get_model),
) -> ModelHealthResponse:
    """Return active artifact version, canonical feature schema status, and model provenance."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Surrogate ML model container is not loaded.",
        )
    return ModelHealthResponse(
        status="healthy",
        model_loaded=True,
        artifact_name="village-preparedness-classifier",
        artifact_version=model.artifact_version,
        artifact_source=model.artifact_source,
        features_count=len(model.features),
        feature_schema_status=f"VALID_{len(model.features)}_FEATURES",
        target_classes=model.classes_,
        model_provenance=model.metadata.get("model_nature", "REPRODUCIBLE RECONSTRUCTION"),
        target_nature="AHP_CAPACITY_TIER_SURROGATE",
    )


@router.post(
    "/model/predict",
    response_model=ModelPredictionResponse,
    summary="Generate village preparedness tier prediction using registered model",
)
def predict_model_tier_v1(
    request: ModelPredictionRequest,
    service: DecisionSupportService = Depends(get_service),
    model: Optional[RegisteredModelContainer] = Depends(get_model),
) -> ModelPredictionResponse:
    """Generate village preparedness tier surrogate prediction and boundary case analysis."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Surrogate ML model container is not loaded.",
        )

    area_id = None
    village_name = None
    district_name = None
    ahp_tier = None
    ahp_score = None
    ahp_agreement = None
    ahp_precedence_note = None

    if request.features is not None:
        try:
            X_aligned = engineer_inference_features(request.features)
        except ValueError as e:
            record_inference_error("validation_error")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(e),
            )

        if request.area_id:
            clean_id = request.area_id.strip()
            df = service.unified_df
            match = df[
                (df["area_id"] == clean_id)
                | (df["pc11_id"] == clean_id)
                | (df["shrid2"] == clean_id)
                | (df["village_name"].str.lower() == clean_id.lower())
            ]
            if not match.empty:
                row = match.iloc[0]
                area_id = str(row["area_id"])
                village_name = str(row["village_name"])
                district_name = str(row["district_name"])
                ahp_tier = str(row["preparedness_tier"])
                ahp_score = float(row["preparedness_score"])
            else:
                area_id = clean_id
    else:
        clean_id = request.area_id.strip()  # type: ignore[union-attr]
        df = service.unified_df
        match = df[
            (df["area_id"] == clean_id)
            | (df["pc11_id"] == clean_id)
            | (df["shrid2"] == clean_id)
            | (df["village_name"].str.lower() == clean_id.lower())
        ]
        if match.empty:
            record_inference_error("not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Administrative unit '{request.area_id}' not found in unified preparedness dataset.",
            )

        row = match.iloc[0]
        area_id = str(row["area_id"])
        village_name = str(row["village_name"])
        district_name = str(row["district_name"])
        ahp_tier = str(row["preparedness_tier"])
        ahp_score = float(row["preparedness_score"])

        try:
            X_aligned = engineer_inference_features(row.to_dict())
        except ValueError as e:
            record_inference_error("validation_error")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Feature extraction failed for administrative unit '{clean_id}': {e}",
            )

    inf_start = time.perf_counter()
    try:
        preds_df = model.predict_detailed(X_aligned)
    except Exception as e:
        record_inference_error("runtime_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {e}",
        )
    inf_duration = time.perf_counter() - inf_start

    res_row = preds_df.iloc[0]
    pred_tier = str(res_row["predicted_preparedness_tier"])
    prob_high = round(float(res_row["prob_high"]), 6)
    prob_low = round(float(res_row["prob_low"]), 6)
    prob_medium = round(float(res_row["prob_medium"]), 6)
    prob_margin = round(float(res_row["probability_margin"]), 6)
    max_prob = round(float(res_row["max_probability"]), 6)
    is_boundary = bool(res_row["is_boundary_case"])

    if ahp_tier is not None:
        if pred_tier.lower() == ahp_tier.lower():
            ahp_agreement = "FULL_AGREEMENT"
        else:
            ahp_agreement = "BORDERLINE_DISAGREEMENT"

        if ahp_tier.lower() == "low" and pred_tier.lower() != "low":
            ahp_precedence_note = (
                "AHP PRECEDENCE MANDATE: Authoritative AHP capacity tier is 'Low'. "
                "Per operational safety rules, ML model prediction cannot downgrade emergency priority."
            )
        elif ahp_agreement == "BORDERLINE_DISAGREEMENT":
            ahp_precedence_note = (
                f"DISCORDANCE NOTICE: Model predicts '{pred_tier}', differing from baseline AHP '{ahp_tier}'. "
                "Deterministic AHP assessment holds operational precedence; flagged for administrative review."
            )

    record_inference_success(
        duration_seconds=inf_duration,
        predicted_tier=pred_tier,
        is_boundary=is_boundary,
        model_version=model.artifact_version,
        ahp_tier=ahp_tier,
        agreement_status=ahp_agreement,
    )

    return ModelPredictionResponse(
        area_id=area_id,
        village_name=village_name,
        district_name=district_name,
        predicted_preparedness_tier=pred_tier,
        class_probabilities={
            "prob_high": prob_high,
            "prob_low": prob_low,
            "prob_medium": prob_medium,
        },
        probability_margin=prob_margin,
        max_probability=max_prob,
        is_boundary_case=is_boundary,
        deterministic_ahp_tier=ahp_tier,
        deterministic_ahp_score=ahp_score,
        ahp_agreement_status=ahp_agreement,
        ahp_precedence_note=ahp_precedence_note,
        model_artifact_name="village-preparedness-classifier",
        model_artifact_version=model.artifact_version,
        model_provenance=model.metadata.get("model_nature", "REPRODUCIBLE RECONSTRUCTION"),
        target_nature="AHP_CAPACITY_TIER_SURROGATE",
        disclaimer=MODEL_PREDICTION_DISCLAIMER,
    )


# =============================================================================
# Conversational Assistant
# =============================================================================


@router.post(
    "/assistant/chat",
    response_model=ChatResponse,
    summary="Conversational decision support assistant with deterministic tool grounding",
)
async def chat_with_assistant_v1(
    request: ChatRequest,
    orchestrator: ToolCallingOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Execute conversational turn with tool-calling orchestrator."""
    return await orchestrator.process_chat(request)
