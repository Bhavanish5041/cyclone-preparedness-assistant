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
import sys
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.schemas import (
    ArchetypeCatalogResponse,
    HealthCheckResponse,
    PrioritizationScenarioRequest,
    PrioritizationScenarioResponse,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
    UnitDetailResponse,
    UnitSummary,
)
from src.assistant.decision_support import DecisionSupportService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for safe, single-time service initialization."""
    logger.info("Initializing DecisionSupportService during application startup...")
    app.state.service = DecisionSupportService()
    logger.info("DecisionSupportService successfully initialized and cached in app.state.")
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


def get_service(request: Request) -> DecisionSupportService:
    """Dependency injection provider for DecisionSupportService."""
    if not hasattr(request.app.state, "service") or request.app.state.service is None:
        logger.info("Creating on-demand DecisionSupportService instance...")
        request.app.state.service = DecisionSupportService()
    return request.app.state.service


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
    summary="Check system health, dataset readiness, and record counts",
)
def health_check(
    service: DecisionSupportService = Depends(get_service),
) -> HealthCheckResponse:
    """Return system readiness, dataset loading state, and administrative unit counts."""
    return service.get_health_status()


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
