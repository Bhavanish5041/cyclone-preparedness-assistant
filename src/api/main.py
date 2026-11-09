"""FastAPI application for Cyclone Preparedness Decision Support.

Exposes RESTful endpoints for:
1. System health, data readiness, and inventory status
2. Geographic unit autocomplete search across 622 administrative units
3. Granular unit indicator, dimension score, and archetype inspection
4. Dynamic weight-based prioritization recomputation
5. Multi-scenario sensitivity and rank divergence comparison
6. Discovered vulnerability archetype catalog and operational intervention guidance

Adheres strictly to separation of concerns: route handlers remain thin,
while analytical formulations are encapsulated in DecisionSupportService.
"""

from contextlib import asynccontextmanager
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.schemas import (
    ArchetypeCatalogResponse,
    DecisionEvaluationListResponse,
    DecisionSupportEvaluation,
    HealthCheckResponse,
    ModelHealthResponse,
    ModelPredictionRequest,
    ModelPredictionResponse,
    PrioritizationScenarioRequest,
    PrioritizationScenarioResponse,
    RankDivergenceItem,
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
from src.models.load_registered_model import RegisteredModelContainer, load_registered_model
from src.monitoring.prometheus_metrics import (
    get_metrics_response,
    get_normalized_endpoint,
    record_http_request,
    record_inference_error,
    record_inference_success,
    set_model_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("api_main")

MODEL_PREDICTION_DISCLAIMER = (
    "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate "
    "approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated "
    "physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. "
    "Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for safe, single-time service initialization."""
    logger.info("Initializing DecisionSupportService during application startup...")
    app.state.service = DecisionSupportService()
    logger.info("DecisionSupportService successfully initialized and cached in app.state.")
    app.state.orchestrator = ToolCallingOrchestrator(service=app.state.service)
    logger.info("ToolCallingOrchestrator successfully initialized and cached in app.state.")

    # Phase 2.4: Load W&B registered model artifact once during startup
    artifact_address = os.environ.get("WANDB_MODEL_ARTIFACT", "village-preparedness-classifier:v0")
    logger.info(f"Loading W&B registered model artifact: '{artifact_address}'...")
    try:
        model_container = load_registered_model(artifact_address=artifact_address)
        app.state.model_container = model_container
        set_model_status(
            loaded=True,
            artifact_name="village-preparedness-classifier",
            artifact_version=model_container.artifact_version,
        )
        logger.info(
            f"Successfully loaded and verified W&B model artifact: {model_container.artifact_source} "
            f"({model_container.artifact_version})"
        )
    except Exception as e:
        logger.error(f"Failed to load configured W&B model artifact '{artifact_address}': {e}")
        app.state.model_container = None
        set_model_status(loaded=False)
        raise RuntimeError(
            f"FastAPI startup aborted: failed to load required W&B model artifact '{artifact_address}': {e}"
        )

    yield
    logger.info("FastAPI application shutdown complete.")


app = FastAPI(
    title="Cyclone Preparedness Decision Support API",
    description=(
        "Evidence-based spatial decision support backend for coastal Karnataka (Dakshina Kannada and Udupi). "
        "Provides RESTful access to multidimensional preparedness dimension scores, dynamic policy scenario "
        "recomputation, sensitivity rank stability analysis, and unsupervised vulnerability archetype profiles."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for local Streamlit dashboard and development tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """Track HTTP request count and duration across all endpoints with bounded cardinality."""
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start_time
        endpoint = get_normalized_endpoint(request)
        record_http_request(
            method=request.method,
            endpoint=endpoint,
            status_code=status_code,
            duration_seconds=duration,
        )


def get_service(request: Request) -> DecisionSupportService:
    """Dependency injection provider for DecisionSupportService."""
    if not hasattr(request.app.state, "service") or request.app.state.service is None:
        logger.info("Creating on-demand DecisionSupportService instance...")
        request.app.state.service = DecisionSupportService()
    return request.app.state.service


def get_model(request: Request) -> RegisteredModelContainer:
    """Dependency injection provider for registered model container."""
    container = getattr(request.app.state, "model_container", None)
    if container is None:
        # Fallback to on-demand load if lifespan was bypassed in testing fixtures
        artifact_address = os.environ.get("WANDB_MODEL_ARTIFACT", "village-preparedness-classifier:v0")
        try:
            container = load_registered_model(artifact_address=artifact_address)
            request.app.state.model_container = container
            set_model_status(
                loaded=True,
                artifact_name="village-preparedness-classifier",
                artifact_version=container.artifact_version,
            )
        except Exception as e:
            record_inference_error("model_unavailable")
            set_model_status(loaded=False)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Registered model artifact is unavailable: {e}",
            )
    return container


def get_orchestrator(request: Request) -> ToolCallingOrchestrator:
    """Dependency injection provider for conversational ToolCallingOrchestrator."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        service = get_service(request)
        orchestrator = ToolCallingOrchestrator(service=service)
        request.app.state.orchestrator = orchestrator
    return orchestrator


@app.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus exposition metrics endpoint",
)
def prometheus_metrics():
    """Expose Prometheus operational metrics in standard exposition format."""
    return get_metrics_response()


@app.get(
    "/",
    include_in_schema=False,
    summary="Root redirect to Swagger UI documentation",
)
def root_redirect():
    """Redirect root path to interactive OpenAPI documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System Status"],
    summary="Check system health, dataset readiness, and model status",
)
def health_check(
    request: Request,
    service: DecisionSupportService = Depends(get_service),
) -> HealthCheckResponse:
    """Return system readiness, dataset loading state, and model artifact status."""
    base_health = service.get_health_status()
    model_container = getattr(request.app.state, "model_container", None)
    if model_container is None:
        try:
            model_container = get_model(request)
        except Exception:
            model_container = None

    return HealthCheckResponse(
        status=base_health.status,
        dataset_loaded=base_health.dataset_loaded,
        total_units=base_health.total_units,
        total_villages=base_health.total_villages,
        total_census_towns=base_health.total_census_towns,
        districts=base_health.districts,
        archetypes_count=base_health.archetypes_count,
        version=base_health.version,
        model_loaded=(model_container is not None),
        model_artifact_name="village-preparedness-classifier" if model_container else None,
        model_artifact_version=model_container.artifact_version if model_container else None,
        model_feature_schema_status=f"VALID_{len(model_container.features)}_FEATURES" if model_container else None,
        model_provenance=model_container.metadata.get("model_nature") if model_container else None,
    )


@app.get(
    "/api/v1/units/search",
    response_model=List[UnitSummary],
    tags=["Geographic Units"],
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

    valid_districts = {"dakshina kannada", "udupi"}
    if district and district.strip().lower() not in valid_districts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid district '{district}'. Allowed values: 'Dakshina Kannada' or 'Udupi'.",
        )

    return service.search_units(query=clean_q, district_filter=district, limit=limit)


@app.get(
    "/api/v1/units/{identifier}",
    response_model=UnitDetailResponse,
    tags=["Geographic Units"],
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


@app.post(
    "/api/v1/prioritization/scenario",
    response_model=PrioritizationScenarioResponse,
    tags=["Prioritization Scenarios"],
    summary="Compute priority scores and deterministic rankings under custom weights",
)
def compute_prioritization_scenario(
    request: PrioritizationScenarioRequest,
    service: DecisionSupportService = Depends(get_service),
) -> PrioritizationScenarioResponse:
    """Dynamically recompute multidimensional prioritization scores under custom weights.

    Sliders or raw weight inputs are automatically normalized to sum to 1.0.
    """
    valid_districts = {"dakshina kannada", "udupi"}
    if request.district_filter and request.district_filter.strip().lower() not in valid_districts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid district filter '{request.district_filter}'. Allowed: 'Dakshina Kannada', 'Udupi'.",
        )

    return service.compute_prioritization_scenario(request)


@app.post(
    "/api/v1/prioritization/compare",
    response_model=ScenarioComparisonResponse,
    tags=["Prioritization Scenarios"],
    summary="Compare rankings between two weighting scenarios and analyze rank divergence",
)
def compare_prioritization_scenarios(
    request: ScenarioComparisonRequest,
    service: DecisionSupportService = Depends(get_service),
) -> ScenarioComparisonResponse:
    """Evaluate Spearman rank correlation and top diverging units between two policy weighting schemes."""
    return service.compare_scenarios(request)


@app.get(
    "/api/v1/archetypes",
    response_model=ArchetypeCatalogResponse,
    tags=["Vulnerability Archetypes"],
    summary="Retrieve catalog of all four discovered vulnerability archetypes and intervention guidance",
)
def get_archetype_catalog(
    service: DecisionSupportService = Depends(get_service),
) -> ArchetypeCatalogResponse:
    """Return descriptive profiles, district distributions, and intervention checklists for all archetypes."""
    return service.get_archetype_catalog()


# =============================================================================
# Milestone 3B: Integrated Decision-Support Endpoints
# =============================================================================


@app.get(
    "/api/v1/decision-support/summary",
    response_model=RegionalDecisionSummaryResponse,
    tags=["Decision Support Engine"],
    summary="Retrieve regional summary of priority bands, trigger counts, and model concordance",
)
def get_regional_decision_summary(
    service: DecisionSupportService = Depends(get_service),
) -> RegionalDecisionSummaryResponse:
    """Return regional aggregation across all 1,879 records in the study area."""
    return service.get_regional_decision_summary()


@app.get(
    "/api/v1/decision-support/units/{identifier}",
    response_model=DecisionSupportEvaluation,
    tags=["Decision Support Engine"],
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


@app.get(
    "/api/v1/decision-support/units",
    response_model=DecisionEvaluationListResponse,
    tags=["Decision Support Engine"],
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
# Phase 2.4: Registered Model Inference & Diagnostics Endpoints
# =============================================================================


@app.get(
    "/api/v1/model/health",
    response_model=ModelHealthResponse,
    tags=["Model Inference"],
    summary="Check registered model health, active artifact version, and schema integrity",
)
def get_model_health(
    model: RegisteredModelContainer = Depends(get_model),
) -> ModelHealthResponse:
    """Return active artifact version, canonical feature schema status, and model provenance."""
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


@app.post(
    "/api/v1/model/predict",
    response_model=ModelPredictionResponse,
    tags=["Model Inference"],
    summary="Generate calibrated village preparedness tier prediction using W&B registered model",
)
def predict_model_tier(
    request: ModelPredictionRequest,
    service: DecisionSupportService = Depends(get_service),
    model: RegisteredModelContainer = Depends(get_model),
) -> ModelPredictionResponse:
    """Generate calibrated village preparedness tier prediction and boundary case analysis.

    Accepts either an area identifier (PC11 ID, SHRID, or village name) or a feature dictionary.
    Deterministic AHP baseline maintains absolute operational precedence for emergency allocations.
    """
    area_id = None
    village_name = None
    district_name = None
    ahp_tier = None
    ahp_score = None
    ahp_agreement = None
    ahp_precedence_note = None

    # Scenario 1: Direct feature dictionary provided
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

    # Scenario 2: Only area_id provided (lookup in unified dataset)
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

    # Perform inference using registered model container
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

    # Determine agreement status and precedence if AHP tier is available
    if ahp_tier is not None:
        if pred_tier.lower() == ahp_tier.lower():
            ahp_agreement = "FULL_AGREEMENT"
        else:
            ahp_agreement = "BORDERLINE_DISAGREEMENT"

        # Explicit safeguard: Never allow ML model to downgrade an AHP Low classification
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

    # Record Prometheus metrics for successful inference execution
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
# Phase 4.2: Conversational AI Decision Support Assistant Endpoint
# =============================================================================


@app.post(
    "/api/v1/assistant/chat",
    response_model=ChatResponse,
    tags=["AI Decision Support Assistant"],
    summary="Conversational decision support assistant powered by Gemini with deterministic tool grounding",
)
async def chat_with_assistant(
    request: ChatRequest,
    orchestrator: ToolCallingOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Execute a conversational interaction turn with the Gemini tool-calling assistant.

    Processes the user query against registered deterministic tools (community lookup,
    AHP assessment, district comparisons, infrastructure analysis, vulnerability archetypes,
    and rule-based advisories). If the external LLM is unconfigured, unreachable, or rate-limited,
    transparently falls back to direct deterministic decision-matrix outputs.
    """
    return await orchestrator.process_chat(request)

