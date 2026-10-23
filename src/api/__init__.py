"""FastAPI REST API package for Cyclone Preparedness Decision Support."""

from src.api.schemas import (
    ArchetypeCatalogResponse,
    ArchetypeSummary,
    HealthCheckResponse,
    PrioritizationScenarioRequest,
    PrioritizationScenarioResponse,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
    UnitDetailResponse,
    UnitSummary,
    WeightConfiguration,
)

__all__ = [
    "HealthCheckResponse",
    "WeightConfiguration",
    "UnitSummary",
    "UnitDetailResponse",
    "PrioritizationScenarioRequest",
    "PrioritizationScenarioResponse",
    "ScenarioComparisonRequest",
    "ScenarioComparisonResponse",
    "ArchetypeSummary",
    "ArchetypeCatalogResponse",
]
