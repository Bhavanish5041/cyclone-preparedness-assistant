"""Prometheus metrics and operational observability for Cyclone Preparedness Assistant.

Provides production-grade Prometheus instrumentation for:
- HTTP request counts, status codes, and latency distributions.
- Model inference execution rates, latencies, and error classifications.
- Empirical preparedness-tier prediction distribution tracking (High, Medium, Low).
- Boundary case detection tracking (is_boundary_case=true/false).
- AHP deterministic baseline vs ML empirical surrogate agreement/discordance tracking.
- Model availability gauges and active model artifact metadata.

Cardinality & Security Safeguards:
- Strictly bounded labels: endpoints are parameterized route templates (e.g. /api/v1/units/{identifier}),
  never dynamic IDs, village names, SHRID, or raw input payloads.
- Status codes are bounded string integers.
- Tiers are bounded to ['High', 'Medium', 'Low', 'unknown'].
- Agreement status is bounded to ['FULL_AGREEMENT', 'BORDERLINE_DISAGREEMENT', 'NONE'].
- No sensitive, user-identifying, or spatial coordinate information is exposed.
"""

import logging
import time
from typing import Optional, Set

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
)

logger = logging.getLogger("prometheus_metrics")

# Cardinality boundary sets
ALLOWED_METHODS: Set[str] = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
ALLOWED_TIERS: Set[str] = {"High", "Medium", "Low", "unknown"}
ALLOWED_ERROR_TYPES: Set[str] = {
    "validation_error",
    "not_found",
    "runtime_error",
    "model_unavailable",
    "other",
}
ALLOWED_AGREEMENT_STATUSES: Set[str] = {
    "FULL_AGREEMENT",
    "BORDERLINE_DISAGREEMENT",
    "NONE",
}

# Static standard route paths for zero-overhead resolution
KNOWN_STATIC_PATHS: Set[str] = {
    "/",
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/model/health",
    "/api/v1/model/predict",
    "/api/v1/units/search",
    "/api/v1/prioritization/evaluate",
    "/api/v1/prioritization/scenarios",
    "/api/v1/prioritization/compare",
    "/api/v1/archetypes/catalog",
    "/api/v1/regional/summary",
}

# ------------------------------------------------------------------------------
# Metric Definitions (Registered in default REGISTRY)
# ------------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "cyclone_http_requests_total",
    "Total count of HTTP requests processed by method, endpoint, and status code",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "cyclone_http_request_duration_seconds",
    "HTTP request latency in seconds by method and endpoint",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

MODEL_INFERENCE_REQUESTS_TOTAL = Counter(
    "cyclone_model_inference_requests_total",
    "Total count of model inference requests by execution status",
    ["status"],
)

MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "cyclone_model_inference_duration_seconds",
    "Model inference latency in seconds by active artifact version",
    ["model_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

MODEL_INFERENCE_ERRORS_TOTAL = Counter(
    "cyclone_model_inference_errors_total",
    "Total count of model inference errors by category",
    ["error_type"],
)

MODEL_PREDICTIONS_TIER_TOTAL = Counter(
    "cyclone_model_predictions_tier_total",
    "Distribution of predicted preparedness capacity tiers",
    ["tier"],
)

MODEL_BOUNDARY_CASES_TOTAL = Counter(
    "cyclone_model_boundary_cases_total",
    "Count of detected borderline/boundary cases by predicted tier",
    ["tier"],
)

MODEL_AHP_DISAGREEMENTS_TOTAL = Counter(
    "cyclone_model_ahp_disagreements_total",
    "Comparison of deterministic AHP baseline tier vs ML empirical surrogate tier",
    ["agreement_status", "ahp_tier", "predicted_tier"],
)

MODEL_LOADED = Gauge(
    "cyclone_model_loaded",
    "Indicates if the ML model is currently loaded in memory (1.0 = loaded, 0.0 = not loaded)",
)

MODEL_INFO = Gauge(
    "cyclone_model_info",
    "Active registered model artifact metadata and version",
    ["artifact_name", "artifact_version"],
)

# Initialize default gauge states
MODEL_LOADED.set(0.0)


# ------------------------------------------------------------------------------
# Label Sanitization and Cardinality Protection
# ------------------------------------------------------------------------------

def get_normalized_endpoint(request: Request) -> str:
    """Extract bounded route template path from FastAPI request.

    Guarantees that high-cardinality values (e.g. Census IDs, village names, SHRID)
    are NEVER used as Prometheus labels.
    """
    path = request.url.path

    # Check known static paths first
    if path in KNOWN_STATIC_PATHS:
        return path

    # Check route template matched by Starlette router
    route = request.scope.get("route")
    if route and hasattr(route, "path"):
        return route.path

    # Parameterized fallback checks
    if path.startswith("/api/v1/units/") and path != "/api/v1/units/search":
        return "/api/v1/units/{identifier}"

    return "unmatched"


def sanitize_method(method: str) -> str:
    """Ensure HTTP method is within bounded standard set."""
    m = method.upper()
    return m if m in ALLOWED_METHODS else "OTHER"


def sanitize_tier(tier: Optional[str]) -> str:
    """Ensure preparedness tier is within bounded standard set."""
    if not tier:
        return "unknown"
    t = str(tier).capitalize()
    return t if t in ALLOWED_TIERS else "unknown"


def sanitize_error_type(error_type: str) -> str:
    """Ensure error classification is within bounded standard set."""
    e = str(error_type).lower()
    return e if e in ALLOWED_ERROR_TYPES else "other"


def sanitize_agreement_status(status: Optional[str]) -> str:
    """Ensure agreement status is within bounded set."""
    if not status:
        return "NONE"
    s = str(status).upper()
    return s if s in ALLOWED_AGREEMENT_STATUSES else "NONE"


# ------------------------------------------------------------------------------
# Operational Recording Functions
# ------------------------------------------------------------------------------

def record_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float = 0.0,
    duration: Optional[float] = None,
) -> None:
    """Record HTTP request completion metric and latency."""
    safe_method = sanitize_method(method)
    safe_status = str(status_code)
    dur = duration if duration is not None else duration_seconds
    HTTP_REQUESTS_TOTAL.labels(
        method=safe_method,
        endpoint=endpoint,
        status_code=safe_status,
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=safe_method,
        endpoint=endpoint,
    ).observe(dur)


def record_inference_success(
    duration_seconds: float,
    predicted_tier: str,
    is_boundary: bool,
    model_version: str = "v0",
    ahp_tier: Optional[str] = None,
    agreement_status: Optional[str] = None,
) -> None:
    """Record successful model inference execution, distribution, and agreement."""
    safe_pred = sanitize_tier(predicted_tier)
    safe_version = str(model_version).strip() or "v0"

    # Inference count & duration
    MODEL_INFERENCE_REQUESTS_TOTAL.labels(status="success").inc()
    MODEL_INFERENCE_DURATION_SECONDS.labels(model_version=safe_version).observe(duration_seconds)

    # Prediction tier distribution
    MODEL_PREDICTIONS_TIER_TOTAL.labels(tier=safe_pred).inc()

    # Boundary case detection
    if is_boundary:
        MODEL_BOUNDARY_CASES_TOTAL.labels(tier=safe_pred).inc()

    # AHP comparison agreement
    if ahp_tier is not None:
        safe_ahp = sanitize_tier(ahp_tier)
        safe_agreement = sanitize_agreement_status(agreement_status)
        MODEL_AHP_DISAGREEMENTS_TOTAL.labels(
            agreement_status=safe_agreement,
            ahp_tier=safe_ahp,
            predicted_tier=safe_pred,
        ).inc()


def record_inference_error(error_type: str) -> None:
    """Record model inference failure classified by bounded error type."""
    safe_err = sanitize_error_type(error_type)
    MODEL_INFERENCE_REQUESTS_TOTAL.labels(status="error").inc()
    MODEL_INFERENCE_ERRORS_TOTAL.labels(error_type=safe_err).inc()


def set_model_status(
    loaded: bool,
    artifact_name: Optional[str] = None,
    artifact_version: Optional[str] = None,
) -> None:
    """Update model availability gauge and active artifact version metadata."""
    if loaded:
        MODEL_LOADED.set(1.0)
        if artifact_name and artifact_version:
            MODEL_INFO.labels(
                artifact_name=str(artifact_name),
                artifact_version=str(artifact_version),
            ).set(1.0)
    else:
        MODEL_LOADED.set(0.0)


def get_metrics_response() -> Response:
    """Expose registered Prometheus metrics in standard exposition format."""
    data = generate_latest(REGISTRY)
    return Response(
        content=data,
        media_type=CONTENT_TYPE_LATEST,
    )
