"""Pydantic schemas for model inference and local model registry endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelPredictionRequest(BaseModel):
    """Input payload for surrogate model tier inference."""

    area_id: Optional[str] = Field(
        None,
        description="Optional Census PC11 ID, SHRID, or village name to extract registered feature vector.",
        examples=["Sasihithlu", "603256", "Tallur"],
    )
    features: Optional[Dict[str, float]] = Field(
        None,
        description="Optional raw 26-feature dictionary. If omitted, features are loaded from registered area_id.",
    )


class ModelPredictionResponse(BaseModel):
    """Calibrated village preparedness tier prediction and boundary analysis."""

    model_name: str
    model_version: str
    area_id: Optional[str] = None
    village_name: Optional[str] = None
    district_name: Optional[str] = None
    predicted_tier: str
    probabilities: Dict[str, float]
    confidence_score: float
    is_boundary_case: bool
    boundary_reason: Optional[str] = None
    deterministic_ahp_tier: Optional[str] = None
    deterministic_ahp_score: Optional[float] = None
    ahp_agreement: Optional[bool] = None
    operational_precedence_note: str


class ModelRegistryItem(BaseModel):
    """Model registry item summary."""

    model_identifier: str
    model_family: str
    active_version: str
    status: str
    total_versions: int
    purpose: str


class ModelRegistryResponse(BaseModel):
    """Catalog of registered models and active versions."""

    registry_version: str
    models: List[ModelRegistryItem]
    artifact_verification_status: str
