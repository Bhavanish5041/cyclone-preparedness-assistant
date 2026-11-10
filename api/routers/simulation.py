"""Policy simulation and sensitivity comparison endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from src.api.schemas import (
    PrioritizationScenarioRequest,
    PrioritizationScenarioResponse,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
)
from src.assistant.decision_support import DecisionSupportService

router = APIRouter(prefix="/simulation", tags=["Policy Simulation"])


@router.post("/prioritize", response_model=PrioritizationScenarioResponse, summary="Compute MCDA rankings under custom weights")
def compute_simulation(
    request: PrioritizationScenarioRequest,
    service: DecisionSupportService = Depends(get_service),
) -> PrioritizationScenarioResponse:
    """Dynamically recompute multi-criteria decision analysis rankings based on policy weights."""
    return service.compute_prioritization_scenario(request)


@router.post("/compare", response_model=ScenarioComparisonResponse, summary="Compare baseline and counterfactual policy scenarios")
def compare_simulation(
    request: ScenarioComparisonRequest,
    service: DecisionSupportService = Depends(get_service),
) -> ScenarioComparisonResponse:
    """Compare rank shifts, Spearman rank correlation, and top diverging communities between two policy schemes."""
    return service.compare_scenarios(request)
