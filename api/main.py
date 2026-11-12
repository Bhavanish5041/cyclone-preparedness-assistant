"""FastAPI Application for Cyclone Preparedness Intelligence Platform.

Provides RESTful endpoints for:
- System health and data readiness
- Command-center regional dashboard metrics
- Granular administrative unit search, filtering, and dossier inspection
- Geospatial mapping records and coordinates
- Cross-district comparative intelligence
- Analytical distributions, AHP rankings, and K-Means archetypes
- Logistic Regression surrogate diagnostics and metrics
- Dynamic MCDA policy scenario simulations
- Conversational CycloneLM assistant with deterministic tool grounding
"""

from contextlib import asynccontextmanager
import logging
import os
import sys
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api.dependencies import (
    get_model,
    get_model_singleton,
    get_orchestrator,
    get_orchestrator_singleton,
    get_service,
    get_service_singleton,
)
from api.middleware.request_id import RequestIDMiddleware
from api.routers import (
    analytics_router,
    assistant_router,
    communities_router,
    dashboard_router,
    districts_router,
    health_router,
    legacy_v1_router,
    maps_router,
    models_router,
    simulation_router,
)
from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.orchestrator import ToolCallingOrchestrator
from src.models.load_registered_model import load_registered_model
from src.monitoring.prometheus_metrics import (
    get_metrics_response,
    get_normalized_endpoint,
    record_http_request,
    set_model_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cyclone_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for safe service startup and teardown."""
    logger.info("Initializing Cyclone Preparedness backend services...")
    service = get_service_singleton()
    app.state.service = service
    logger.info(f"Loaded DecisionSupportService with {len(service.ranked_df)} ranked administrative units.")

    orchestrator = get_orchestrator_singleton()
    app.state.orchestrator = orchestrator
    logger.info("Initialized ToolCallingOrchestrator with 7 deterministic tools.")

    # Load verified model artifact
    artifact_address = os.environ.get("WANDB_MODEL_ARTIFACT", "village-preparedness-classifier:v1.0.0")
    try:
        model = load_registered_model(artifact_address=artifact_address)
        app.state.model_container = model
        set_model_status(
            loaded=True,
            artifact_name="village-preparedness-classifier",
            artifact_version=model.artifact_version,
        )
        logger.info(f"Loaded registered surrogate model ({model.artifact_version}).")
    except Exception as e:
        app.state.model_container = None
        set_model_status(loaded=False)
        raise RuntimeError(
            f"FastAPI startup aborted: failed to load required W&B model artifact '{artifact_address}': {e}"
        ) from e

    yield
    logger.info("Cyclone Preparedness backend shut down successfully.")


# Configurable public OpenAPI documentation exposure
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "true").lower() in ("true", "1", "yes")

app = FastAPI(
    title="Cyclone Preparedness Decision Intelligence API",
    description=(
        "Production RESTful API for evidence-based spatial decision support and cyclone preparedness intelligence "
        "across coastal Karnataka (Dakshina Kannada, Udupi, and Uttara Kannada). Grounded in deterministic AHP criteria, "
        "Census 2011 baselines, MoHFW healthcare registries, and 4-tier operational priority allocations."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# CORS configuration
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost",
        "http://localhost:80",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8501",
        "http://127.0.0.1",
        "http://127.0.0.1:80",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8501",
        "https://cyclone-preparedness-assistant.onrender.com",
    ]

# Regex pattern allowing dynamic Cloudflare Quick Tunnel domains (*.trycloudflare.com)
CLOUDFLARE_TUNNEL_ORIGIN_REGEX = r"^https?://.*\.trycloudflare\.com$"

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_origin_regex=CLOUDFLARE_TUNNEL_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add performance timing header to every response."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Centralized exception handler to ensure consistent JSON error responses."""
    logger.exception(f"Unhandled error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the request.",
            "path": request.url.path,
        },
    )


# Root redirect to OpenAPI documentation
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# Prometheus operational metrics exposition endpoint
@app.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus exposition metrics endpoint",
)
def prometheus_metrics():
    """Expose Prometheus operational metrics in standard exposition format."""
    return get_metrics_response()


# Direct health and readiness endpoints for container healthchecks
@app.get(
    "/health",
    tags=["Health & Status"],
    summary="Direct system health, dataset readiness, and model status",
)
def root_health(request: Request):
    service = get_service(request)
    model = get_model(request)
    base_health = service.get_health_status()
    return {
        "status": "healthy",
        "dataset_loaded": len(service.ranked_df) > 0,
        "total_units": len(service.ranked_df),
        "total_villages": base_health.total_villages,
        "total_census_towns": base_health.total_census_towns,
        "districts": sorted(list(service.ranked_df["district_name"].dropna().unique())),
        "archetypes_count": base_health.archetypes_count,
        "version": "2.0.0",
        "model_loaded": (model is not None),
        "model_artifact_name": "village-preparedness-classifier" if model else None,
        "model_artifact_version": model.artifact_version if model else None,
        "model_feature_schema_status": f"VALID_{len(model.features)}_FEATURES" if model else None,
        "model_provenance": model.metadata.get("model_nature") if model else None,
    }


@app.get("/ready", include_in_schema=False)
def root_ready(request: Request):
    service = get_service(request)
    model = get_model(request)
    ready = len(service.ranked_df) > 0 and model is not None
    return {
        "ready": ready,
        "dataset_loaded": len(service.ranked_df) > 0,
        "model_loaded": model is not None,
        "version": "2.0.0",
    }


# Mount API routers under /api
api_prefix = "/api"
app.include_router(health_router, prefix=api_prefix)
app.include_router(dashboard_router, prefix=api_prefix)
app.include_router(districts_router, prefix=api_prefix)
app.include_router(communities_router, prefix=api_prefix)
app.include_router(maps_router, prefix=api_prefix)
app.include_router(analytics_router, prefix=api_prefix)
app.include_router(models_router, prefix=api_prefix)
app.include_router(simulation_router, prefix=api_prefix)
app.include_router(assistant_router, prefix=api_prefix)

# Mount backward-compatibility legacy v1 router
app.include_router(legacy_v1_router)
