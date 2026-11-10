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
from api.routers import (
    analytics_router,
    assistant_router,
    communities_router,
    dashboard_router,
    districts_router,
    health_router,
    maps_router,
    simulation_router,
)
from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.orchestrator import ToolCallingOrchestrator
from src.models.load_registered_model import load_registered_model

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

    model = get_model_singleton()
    app.state.model_container = model
    if model:
        logger.info(f"Loaded registered surrogate model ({model.artifact_version}).")
    else:
        logger.warning("Surrogate ML model not loaded; proceeding with deterministic fallback.")

    yield
    logger.info("Cyclone Preparedness backend shut down successfully.")


app = FastAPI(
    title="Cyclone Preparedness Decision Intelligence API",
    description=(
        "Production RESTful API for evidence-based spatial decision support and cyclone preparedness intelligence "
        "across coastal Karnataka (Dakshina Kannada, Udupi, and Uttara Kannada). Grounded in deterministic AHP criteria, "
        "Census 2011 baselines, MoHFW healthcare registries, and 4-tier operational priority allocations."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS configuration
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://cyclone-preparedness-assistant.onrender.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
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


# Direct health endpoint for container healthchecks
@app.get("/health", include_in_schema=False)
def root_health(request: Request):
    service = get_service(request)
    return {
        "status": "healthy",
        "total_units": len(service.ranked_df),
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
app.include_router(simulation_router, prefix=api_prefix)
app.include_router(assistant_router, prefix=api_prefix)
