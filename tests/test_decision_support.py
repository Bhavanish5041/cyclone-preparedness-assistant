"""Unit tests for decision support service, weight validation, search, and scenario comparison."""

import pytest

from src.api.schemas import (
    PrioritizationScenarioRequest,
    ScenarioComparisonRequest,
    WeightConfiguration,
)
from src.assistant.decision_support import DecisionSupportService


@pytest.fixture(scope="module")
def service() -> DecisionSupportService:
    """Instantiate the DecisionSupportService singleton fixture."""
    return DecisionSupportService()


def test_health_status(service: DecisionSupportService):
    """Verify system health status and total unit counts."""
    health = service.get_health_status()
    assert health.status == "healthy"
    assert health.dataset_loaded is True
    assert health.total_units == 622
    assert health.total_villages == 563
    assert health.total_census_towns == 59
    assert health.archetypes_count == 4
    assert set(health.districts) == {"Dakshina Kannada", "Udupi"}


def test_weight_configuration_auto_normalize():
    """Verify weight auto-normalization from raw slider inputs (e.g. percentages)."""
    # Slider values summing to 100
    cfg = WeightConfiguration(healthcare=40.0, demographic=20.0, infrastructure=20.0, coastal=20.0)
    norm = cfg.get_normalized_weights()
    assert norm["healthcare"] == 0.4
    assert norm["demographic"] == 0.2
    assert norm["infrastructure"] == 0.2
    assert norm["coastal"] == 0.2
    assert pytest.approx(sum(norm.values()), abs=1e-5) == 1.0


def test_weight_configuration_validation_errors():
    """Verify validation errors on negative or zero-sum weights."""
    # Negative weight
    with pytest.raises(ValueError):
        WeightConfiguration(healthcare=-0.1, demographic=0.4, infrastructure=0.4, coastal=0.3)

    # All zero weights
    with pytest.raises(ValueError, match="strictly positive"):
        WeightConfiguration(healthcare=0.0, demographic=0.0, infrastructure=0.0, coastal=0.0)

    # auto_normalize=False with non-1.0 sum
    with pytest.raises(ValueError, match="must sum to 1.0"):
        WeightConfiguration(healthcare=0.5, demographic=0.5, infrastructure=0.5, coastal=0.5, auto_normalize=False)


def test_search_units(service: DecisionSupportService):
    """Verify search functionality by name, PC11 ID, and SHRID."""
    # Search by village name
    res_name = service.search_units("Tallur")
    assert len(res_name) >= 1
    assert any(u.unit_name == "Tallur" for u in res_name)

    # Search by PC11 ID
    res_id = service.search_units("608720")
    assert len(res_id) == 1
    assert res_id[0].unit_name == "Edmoge"

    # Search by SHRID prefix
    res_shrid = service.search_units("11-29-569-05522", limit=5)
    assert len(res_shrid) <= 5
    assert all("11-29-569-05522" in u.shrid2 for u in res_shrid)

    # Non-existent query
    assert service.search_units("NonExistentPlaceXYZ123") == []
    assert service.search_units("   ") == []


def test_get_unit_detail(service: DecisionSupportService):
    """Verify detailed unit inspection and required fields."""
    detail = service.get_unit_detail("608720")  # Edmoge
    assert detail.unit_name == "Edmoge"
    assert detail.district_name == "Udupi"
    assert "healthcare_deficit" in detail.dimension_scores
    assert "coastal_exposure" in detail.dimension_scores
    assert detail.archetype["archetype_id"] in [0, 1, 2, 3]
    assert detail.raw_indicators["total_population"] > 0
    assert "census_vintage" in detail.provenance_and_limitations

    # Invalid identifier raises KeyError
    with pytest.raises(KeyError, match="not found"):
        service.get_unit_detail("INVALID_ID_999999")


def test_compute_prioritization_scenario_determinism(service: DecisionSupportService):
    """Verify deterministic ranking and score recomputation under custom weights."""
    req = PrioritizationScenarioRequest(
        scenario_name="Hazard Priority",
        weights=WeightConfiguration(healthcare=0.1, demographic=0.1, infrastructure=0.2, coastal=0.6),
        top_n=10,
    )

    resp1 = service.compute_prioritization_scenario(req)
    resp2 = service.compute_prioritization_scenario(req)

    assert resp1.total_units_evaluated == 622
    assert len(resp1.top_units) == 10
    # Determinism: top units and scores must be identical
    for u1, u2 in zip(resp1.top_units, resp2.top_units):
        assert u1.shrid2 == u2.shrid2
        assert u1.priority_score == u2.priority_score
        assert u1.priority_rank == u2.priority_rank

    # Coastal-dominant weighting must rank coastal units near top
    assert resp1.top_units[0].unit_name in ["Kilpady", "Shimanthur"]


def test_compare_scenarios(service: DecisionSupportService):
    """Verify scenario comparison, rank divergence, and rank shift calculation."""
    comp_req = ScenarioComparisonRequest(
        scenario_a_name="Equal",
        scenario_a_weights=WeightConfiguration(healthcare=0.25, demographic=0.25, infrastructure=0.25, coastal=0.25),
        scenario_b_name="No Coastal",
        scenario_b_weights=WeightConfiguration(healthcare=0.4, demographic=0.3, infrastructure=0.3, coastal=0.0),
        top_divergent_count=10,
    )

    comp_resp = service.compare_scenarios(comp_req)
    assert -1.0 <= comp_resp.spearman_rank_correlation <= 1.0
    assert -1.0 <= comp_resp.kendall_tau_correlation <= 1.0
    assert len(comp_resp.top_divergent_units) == 10
    assert comp_resp.max_rank_shift > 300

    # Under No-Coastal, foothill villages must surge in priority
    movers = [u.unit_name for u in comp_resp.top_divergent_units if "Surged" in u.direction]
    assert any(name in movers for name in ["Bilinele", "Subramanya", "Shiribagilu", "Kalmakaru"])


def test_get_archetype_catalog(service: DecisionSupportService):
    """Verify archetype catalog returns all 4 clusters with valid statistics."""
    catalog = service.get_archetype_catalog()
    assert catalog.total_archetypes == 4
    total_count = sum(a.unit_count for a in catalog.archetypes)
    assert total_count == 622

    for arch in catalog.archetypes:
        assert 0 <= arch.archetype_id <= 3
        assert len(arch.short_label) > 0
        assert len(arch.detailed_description) > 0
        assert len(arch.primary_intervention_need) > 0
        assert 0.0 <= arch.percentage_of_total <= 100.0
        assert arch.count_dakshina_kannada + arch.count_udupi == arch.unit_count
