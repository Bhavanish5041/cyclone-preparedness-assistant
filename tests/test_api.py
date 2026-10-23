"""Unit and integration tests for FastAPI Cyclone Preparedness Decision Support API.

Uses fastapi.testclient.TestClient to verify:
- Documentation routes (/docs, /openapi.json)
- Health and inventory status (/health)
- Unit search with query and district filtering (/api/v1/units/search)
- Unit detail inspection and 404 error handling (/api/v1/units/{identifier})
- Dynamic prioritization scenario recomputation and weight validation (/api/v1/prioritization/scenario)
- Multi-scenario rank comparison and divergence analysis (/api/v1/prioritization/compare)
- Vulnerability archetype catalog retrieval (/api/v1/archetypes)
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient instance managing app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_docs_and_openapi(client: TestClient):
    """Verify OpenAPI documentation and schema routes exist and return 200."""
    # Test /docs
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    assert "swagger" in docs_resp.text.lower() or "html" in docs_resp.text.lower()

    # Test /openapi.json
    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    assert schema["info"]["title"] == "Cyclone Preparedness Decision Support API"
    assert schema["info"]["version"] == "1.0.0"

    # Verify all 6 endpoints are registered in paths
    paths = schema["paths"]
    assert "/health" in paths
    assert "/api/v1/units/search" in paths
    assert "/api/v1/units/{identifier}" in paths
    assert "/api/v1/prioritization/scenario" in paths
    assert "/api/v1/prioritization/compare" in paths
    assert "/api/v1/archetypes" in paths


def test_root_redirect(client: TestClient):
    """Verify root / redirects to Swagger UI /docs."""
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in [302, 307]
    assert resp.headers["location"] == "/docs"


def test_health_check(client: TestClient):
    """Verify /health endpoint returns dataset inventory and status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["dataset_loaded"] is True
    assert data["total_units"] == 622
    assert data["total_villages"] == 563
    assert data["total_census_towns"] == 59
    assert data["archetypes_count"] == 4
    assert set(data["districts"]) == {"Dakshina Kannada", "Udupi"}


def test_search_units_valid(client: TestClient):
    """Verify unit search by name and PC11 ID."""
    # Search by name
    resp_name = client.get("/api/v1/units/search?query=Tallur")
    assert resp_name.status_code == 200
    results = resp_name.json()
    assert len(results) >= 1
    assert any(u["unit_name"] == "Tallur" for u in results)
    first = results[0]
    assert "priority_score" in first
    assert "vulnerability_archetype_label" in first

    # Search by PC11 ID
    resp_id = client.get("/api/v1/units/search?query=608720")
    assert resp_id.status_code == 200
    results_id = resp_id.json()
    assert len(results_id) == 1
    assert results_id[0]["unit_name"] == "Edmoge"


def test_search_units_with_filter_and_limit(client: TestClient):
    """Verify search filtering by district and result limit."""
    resp = client.get("/api/v1/units/search?query=a&district=Udupi&limit=5")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) <= 5
    assert all(u["district_name"] == "Udupi" for u in results)


def test_search_units_invalid_parameters(client: TestClient):
    """Verify validation errors on whitespace query or invalid district."""
    # Whitespace query -> 422
    resp_blank = client.get("/api/v1/units/search?query=   ")
    assert resp_blank.status_code == 422

    # Invalid district name -> 422
    resp_dist = client.get("/api/v1/units/search?query=Tallur&district=Bangalore")
    assert resp_dist.status_code == 422
    assert "Invalid district" in resp_dist.json()["detail"]


def test_get_unit_detail_valid(client: TestClient):
    """Verify detailed unit inspection by PC11 ID and name."""
    # By PC11 ID (Edmoge)
    resp = client.get("/api/v1/units/608720")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["unit_name"] == "Edmoge"
    assert detail["district_name"] == "Udupi"
    assert "dimension_scores" in detail
    assert "healthcare_deficit" in detail["dimension_scores"]
    assert "raw_indicators" in detail
    assert detail["raw_indicators"]["total_population"] > 0
    assert detail["archetype"]["archetype_id"] in [0, 1, 2, 3]
    assert "provenance_and_limitations" in detail
    assert detail["provenance_and_limitations"]["census_vintage"] == 2011

    # By exact name (case-insensitive)
    resp_name = client.get("/api/v1/units/tallur")
    assert resp_name.status_code == 200
    assert resp_name.json()["unit_name"] == "Tallur"


def test_get_unit_detail_unknown_404(client: TestClient):
    """Verify 404 status code for non-existent unit identifier."""
    resp = client.get("/api/v1/units/999999999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_prioritization_scenario_valid(client: TestClient):
    """Verify custom prioritization scenario calculation and ranking."""
    payload = {
        "scenario_name": "Test Hazard Priority",
        "weights": {
            "healthcare": 10.0,
            "demographic": 10.0,
            "infrastructure": 20.0,
            "coastal": 60.0,
            "auto_normalize": True,
            "include_population_scale": False,
        },
        "top_n": 5,
        "district_filter": "Dakshina Kannada",
    }
    resp = client.post("/api/v1/prioritization/scenario", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_name"] == "Test Hazard Priority"
    assert data["weights_applied"]["coastal"] == 0.6
    assert data["total_units_evaluated"] == 622
    assert len(data["top_units"]) == 5
    assert all(u["district_name"] == "Dakshina Kannada" for u in data["top_units"])
    # Ranks must be 1..5 in order
    assert data["top_units"][0]["priority_rank"] == 1
    assert data["top_units"][0]["priority_score"] >= data["top_units"][1]["priority_score"]


def test_prioritization_scenario_invalid_weights(client: TestClient):
    """Verify validation errors for negative weights or all-zero weights."""
    # Negative weight
    bad_payload = {
        "weights": {"healthcare": -5.0, "demographic": 10.0, "infrastructure": 20.0, "coastal": 60.0}
    }
    resp = client.post("/api/v1/prioritization/scenario", json=bad_payload)
    assert resp.status_code == 422

    # All zeros
    zero_payload = {
        "weights": {"healthcare": 0.0, "demographic": 0.0, "infrastructure": 0.0, "coastal": 0.0}
    }
    resp_zero = client.post("/api/v1/prioritization/scenario", json=zero_payload)
    assert resp_zero.status_code == 422


def test_prioritization_compare_scenarios(client: TestClient):
    """Verify scenario comparison endpoint returns correlations and rank shifts."""
    payload = {
        "scenario_a_name": "Equal Weights",
        "scenario_a_weights": {"healthcare": 0.25, "demographic": 0.25, "infrastructure": 0.25, "coastal": 0.25},
        "scenario_b_name": "No Coastal (Inland Focus)",
        "scenario_b_weights": {"healthcare": 0.40, "demographic": 0.30, "infrastructure": 0.30, "coastal": 0.0},
        "top_divergent_count": 5,
    }
    resp = client.post("/api/v1/prioritization/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_a_name"] == "Equal Weights"
    assert data["scenario_b_name"] == "No Coastal (Inland Focus)"
    assert -1.0 <= data["spearman_rank_correlation"] <= 1.0
    assert data["max_rank_shift"] > 300
    assert len(data["top_divergent_units"]) == 5
    assert all("rank_shift" in item for item in data["top_divergent_units"])


def test_archetype_catalog(client: TestClient):
    """Verify archetype catalog endpoint returns all 4 clusters and descriptions."""
    resp = client.get("/api/v1/archetypes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_archetypes"] == 4
    assert len(data["archetypes"]) == 4

    total_units = sum(a["unit_count"] for a in data["archetypes"])
    assert total_units == 622

    for arch in data["archetypes"]:
        assert 0 <= arch["archetype_id"] <= 3
        assert len(arch["short_label"]) > 0
        assert len(arch["detailed_description"]) > 0
        assert len(arch["primary_intervention_need"]) > 0
        assert "dimension_means" in arch
        assert "healthcare_deficit" in arch["dimension_means"]
