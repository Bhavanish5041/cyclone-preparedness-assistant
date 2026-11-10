"""Shared dependency injection providers for FastAPI routers."""

import logging
import os
from typing import Any, Optional
from fastapi import Request

from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.orchestrator import ToolCallingOrchestrator
from src.models.load_registered_model import RegisteredModelContainer, load_registered_model

logger = logging.getLogger("api_dependencies")

_service_instance: Optional[DecisionSupportService] = None
_model_container: Optional[RegisteredModelContainer] = None
_orchestrator_instance: Optional[ToolCallingOrchestrator] = None


def get_service_singleton() -> DecisionSupportService:
    """Return or initialize DecisionSupportService singleton."""
    global _service_instance
    if _service_instance is None:
        logger.info("Initializing DecisionSupportService singleton...")
        _service_instance = DecisionSupportService()
    return _service_instance


def get_service(request: Request) -> DecisionSupportService:
    """FastAPI dependency for DecisionSupportService."""
    if hasattr(request.app.state, "service") and request.app.state.service:
        return request.app.state.service
    return get_service_singleton()


def get_model_singleton() -> Any:
    """Return or initialize RegisteredModelContainer singleton."""
    global _model_container
    if _model_container is None:
        artifact_address = os.environ.get("WANDB_MODEL_ARTIFACT", "village-preparedness-classifier:v0")
        try:
            _model_container = load_registered_model(artifact_address=artifact_address)
        except Exception as e:
            logger.warning(f"Could not load W&B registered model ({e}). ML inference may be unavailable.")
            _model_container = None
    return _model_container


def get_model(request: Request) -> Any:
    """FastAPI dependency for registered model container."""
    if hasattr(request.app.state, "model_container") and request.app.state.model_container:
        return request.app.state.model_container
    return get_model_singleton()


def get_orchestrator_singleton() -> ToolCallingOrchestrator:
    """Return or initialize ToolCallingOrchestrator singleton."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        service = get_service_singleton()
        _orchestrator_instance = ToolCallingOrchestrator(service=service)
    return _orchestrator_instance


def get_orchestrator(request: Request) -> ToolCallingOrchestrator:
    """FastAPI dependency for ToolCallingOrchestrator."""
    if hasattr(request.app.state, "orchestrator") and request.app.state.orchestrator:
        return request.app.state.orchestrator
    return get_orchestrator_singleton()
