"""Health check and platform status endpoint."""

import logging
import os
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Response, status
import httpx

from api.dependencies import get_model, get_service
from src.assistant.decision_support import DecisionSupportService
from src.models.load_registered_model import RegisteredModelContainer

logger = logging.getLogger("api_health")

router = APIRouter(prefix="/health", tags=["Health & Status"])


@router.get("", summary="System health, dataset readiness, and model status")
def health_check(
    service: DecisionSupportService = Depends(get_service),
    model: RegisteredModelContainer = Depends(get_model),
) -> Dict[str, Any]:
    """Return health status, dataset load state, record counts, and ML model status."""
    return {
        "status": "healthy",
        "dataset_loaded": True,
        "total_units": len(service.ranked_df),
        "total_assigned_units": len(service.df),
        "districts": ["Dakshina Kannada", "Udupi", "Uttara Kannada"],
        "version": "2.0.0",
        "model_loaded": (model is not None),
        "model_artifact_version": model.artifact_version if model else None,
        "feature_count": len(model.features) if model else 26,
    }


@router.get("/ready", summary="System readiness probe for container orchestrators")
async def readiness_probe(
    response: Response,
    service: DecisionSupportService = Depends(get_service),
    model: RegisteredModelContainer = Depends(get_model),
) -> Dict[str, Any]:
    """Verify backend readiness across data, registered surrogate model, and local LLM."""
    dataset_ready = len(service.ranked_df) > 0
    model_ready = model is not None

    # Check Ollama daemon connectivity
    ollama_ready = False
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{ollama_base_url}/api/tags")
            ollama_ready = (resp.status_code == 200)
    except Exception:
        ollama_ready = False

    is_ready = dataset_ready and model_ready
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    response.status_code = status_code

    return {
        "ready": is_ready,
        "checks": {
            "dataset_loaded": dataset_ready,
            "total_ranked_units": len(service.ranked_df),
            "surrogate_model_loaded": model_ready,
            "surrogate_model_version": getattr(model, "artifact_version", None),
            "ollama_daemon_accessible": ollama_ready,
        },
        "version": "2.0.0",
    }
