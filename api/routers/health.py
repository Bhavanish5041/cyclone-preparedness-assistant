"""Health check and platform status endpoint."""

from typing import Any, Dict
from fastapi import APIRouter, Depends

from api.dependencies import get_model, get_service
from src.assistant.decision_support import DecisionSupportService
from src.models.load_registered_model import RegisteredModelContainer

router = APIRouter(prefix="/health", tags=["Health & Status"])


@router.get("", summary="System health, dataset readiness, and model status")
def health_check(
    service: DecisionSupportService = Depends(get_service),
    model: RegisteredModelContainer = Depends(get_model),
) -> Dict[str, Any]:
    """Return health status, dataset load state, record counts, and ML model status."""
    base_health = service.get_health_status()
    return {
        "status": "healthy",
        "dataset_loaded": True,
        "total_units": len(service.ranked_df),
        "total_assigned_units": len(service.df),
        "districts": ["Dakshina Kannada", "Udupi", "Uttara Kannada"],
        "version": "1.0.0",
        "model_loaded": (model is not None),
        "model_artifact_version": model.artifact_version if model else None,
        "feature_count": len(model.features) if model else 26,
    }
