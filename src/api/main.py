"""Deprecated API module for Cyclone Preparedness Decision Support.

.. deprecated:: 2.0.0
   The canonical production FastAPI application has been migrated to `api.main:app`.
   This module is maintained exclusively for backward compatibility with legacy imports
   and will be removed in a future release. All routes, metrics, and compatibility
   handlers are unified in `api.main`.
"""

import warnings
from api.dependencies import (
    get_model,
    get_model_singleton,
    get_orchestrator,
    get_orchestrator_singleton,
    get_service,
    get_service_singleton,
)
from api.main import app, lifespan
from src.models.load_registered_model import load_registered_model

warnings.warn(
    "src.api.main is deprecated and will be removed in a future release. "
    "Please import from api.main directly.",
    DeprecationWarning,
    stacklevel=2,
)

MODEL_PREDICTION_DISCLAIMER = (
    "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate "
    "approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated "
    "physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. "
    "Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
)

__all__ = [
    "app",
    "lifespan",
    "get_service",
    "get_service_singleton",
    "get_model",
    "get_model_singleton",
    "get_orchestrator",
    "get_orchestrator_singleton",
    "load_registered_model",
    "MODEL_PREDICTION_DISCLAIMER",
]
