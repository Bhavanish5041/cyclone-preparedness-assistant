"""Model inference and local model registry endpoints."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_model, get_service
from api.schemas.models import (
    ModelPredictionRequest,
    ModelPredictionResponse,
    ModelRegistryItem,
    ModelRegistryResponse,
)
from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools.community import resolve_community_row
from src.features.inference_features import engineer_inference_features
from src.models.load_registered_model import RegisteredModelContainer
from src.models.registry.local_registry import LocalModelRegistry

logger = logging.getLogger("api_models")

router = APIRouter(prefix="/models", tags=["Models & Inference"])

OPERATIONAL_PRECEDENCE_NOTE = (
    "OPERATIONAL SAFETY MANDATE: Deterministic Analytical Hierarchy Process (AHP) composite scoring "
    "and 4-tier operational priority bands maintain absolute operational precedence for emergency resource allocation. "
    "Logistic Regression surrogate predictions provide calibrated class probabilities for secondary boundary-case detection, "
    "NOT physical cyclone-risk or hazard prediction."
)


@router.post(
    "/predict",
    response_model=ModelPredictionResponse,
    summary="Generate village preparedness tier prediction and boundary case analysis",
)
def predict_model_tier(
    request: ModelPredictionRequest,
    service: DecisionSupportService = Depends(get_service),
    model: Optional[RegisteredModelContainer] = Depends(get_model),
) -> ModelPredictionResponse:
    """Predict village preparedness tier (High, Medium, Low) and analyze boundary ambiguity.

    Accepts either:
    1. `area_id`: Census PC11 ID, SHRID, or village name.
    2. `features`: 26-feature dictionary.

    Enforces deterministic AHP precedence: AHP scores cannot be downgraded by surrogate ML inference.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Surrogate ML model container is not loaded. Deterministic AHP assessments remain fully active.",
        )

    village_name = None
    district_name = None
    area_id = None
    ahp_tier = None
    ahp_score = None
    ahp_agreement = None

    # Scenario A: Direct feature dictionary provided
    if request.features is not None:
        try:
            X_aligned = engineer_inference_features(request.features)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature validation failed: {err}",
            )

        if request.area_id:
            try:
                row = resolve_community_row(service.ranked_df, request.area_id)
                village_name = str(row["village_name"])
                district_name = str(row["district_name"])
                area_id = str(row["pc11_id"])
                ahp_tier = str(row["preparedness_tier"])
                ahp_score = round(float(row["preparedness_score"]), 2)
            except Exception:
                pass

    # Scenario B: Area ID provided
    elif request.area_id:
        try:
            row = resolve_community_row(service.ranked_df, request.area_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Community identifier '{request.area_id}' not found: {e}",
            )

        village_name = str(row["village_name"])
        district_name = str(row["district_name"])
        area_id = str(row["pc11_id"])
        ahp_tier = str(row["preparedness_tier"])
        ahp_score = round(float(row["preparedness_score"]), 2)

        # Engineer the canonical 26 features from row dictionary
        try:
            X_aligned = engineer_inference_features(row.to_dict())
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature engineering failed for record: {err}",
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request must provide either 'area_id' (village name or Census PC11 ID) or a 'features' dictionary.",
        )

    # Execute inference
    try:
        preds = model.predict(X_aligned)
        predicted_tier = str(preds[0])
        prob_dict_raw = model.predict_proba(X_aligned)
    except Exception as exc:
        logger.error(f"Inference execution failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        )

    # Reconstruct formatted probabilities
    probs = {
        "High": round(float(prob_dict_raw["prob_high"][0]), 4),
        "Medium": round(float(prob_dict_raw["prob_medium"][0]), 4),
        "Low": round(float(prob_dict_raw["prob_low"][0]), 4),
    }

    # Boundary case detection
    sorted_probs = sorted(probs.values(), reverse=True)
    top_p = sorted_probs[0]
    second_p = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = top_p - second_p

    is_boundary = bool(margin < 0.10 or top_p < 0.60)
    boundary_reason = None
    if is_boundary:
        boundary_reason = (
            f"Boundary case detected: top probability margin ({margin * 100:.1f}%) < 10% "
            f"or top confidence ({top_p * 100:.1f}%) < 60%. Manual administrative review recommended."
        )

    if ahp_tier:
        ahp_agreement = bool(predicted_tier.lower() == ahp_tier.lower())

    op_note = OPERATIONAL_PRECEDENCE_NOTE
    if ahp_tier and ahp_tier.lower() == "low" and predicted_tier.lower() != "low":
        op_note = (
            "AHP PRECEDENCE MANDATE: Authoritative AHP capacity tier is 'Low'. "
            "Per operational safety rules, ML model prediction cannot downgrade emergency priority. "
            + OPERATIONAL_PRECEDENCE_NOTE
        )

    return ModelPredictionResponse(
        model_name="village-preparedness-classifier",
        model_version=getattr(model, "artifact_version", "v1.0.0"),
        area_id=area_id,
        village_name=village_name,
        district_name=district_name,
        predicted_tier=predicted_tier,
        probabilities=probs,
        confidence_score=round(float(top_p), 4),
        is_boundary_case=is_boundary,
        boundary_reason=boundary_reason,
        deterministic_ahp_tier=ahp_tier,
        deterministic_ahp_score=ahp_score,
        ahp_agreement=ahp_agreement,
        operational_precedence_note=op_note,
    )


@router.get(
    "/registry",
    response_model=ModelRegistryResponse,
    summary="List all models registered in the local Git-compatible registry",
)
def list_registered_models() -> ModelRegistryResponse:
    """Return catalog of registered models, active versions, and verification status."""
    try:
        registry = LocalModelRegistry()
        models = registry.list_models()
        items = [
            ModelRegistryItem(
                model_identifier=m["model_identifier"],
                model_family=m["model_family"],
                active_version=m["active_version"],
                status=m["status"],
                total_versions=m["total_versions"],
                purpose=m["purpose"],
            )
            for m in models
        ]
        val_results = registry.validate_artifacts()
        all_valid = all(r["status"] == "VALID" for r in val_results)

        return ModelRegistryResponse(
            registry_version="1.0.0",
            models=items,
            artifact_verification_status="ALL_VERIFIED" if all_valid else "VERIFICATION_FAILED",
        )
    except Exception as e:
        logger.error(f"Error accessing model registry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load model registry: {e}",
        )


@router.get(
    "/registry/{model_id}",
    summary="Retrieve detailed metadata and schema for a registered model",
)
def get_registered_model_details(
    model_id: str,
    version: Optional[str] = Query(None, description="Optional version (defaults to active production)"),
) -> Dict[str, Any]:
    """Return complete manifest metadata, input/output schemas, metrics, and history for a model."""
    try:
        registry = LocalModelRegistry()
        return registry.get_model(model_id=model_id, version=version)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
