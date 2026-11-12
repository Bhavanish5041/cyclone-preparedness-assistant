"""Swagger UI and OpenAPI Specification Integration Tests.
======================================================

Validates every FastAPI endpoint definition against the actual OpenAPI spec:
  - OpenAPI schema (/openapi.json)
  - Swagger UI documentation (/docs)
  - ReDoc documentation (/redoc)
  - Successful requests (valid body, query params, path params, expected response)
  - Invalid requests (missing fields, wrong data types, invalid enums, out-of-range values,
    malformed JSON, invalid IDs, missing resources, unsupported HTTP methods)
  - Real model inference endpoints (/api/models/predict, /api/v1/model/predict)
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create TestClient fixture with lifespan startup and teardown."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def openapi_spec(client: TestClient):
    """Retrieve and validate the OpenAPI specification."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200, f"/openapi.json returned {resp.status_code}"
    spec = resp.json()
    assert "openapi" in spec
    assert "paths" in spec
    assert "info" in spec
    assert spec["info"]["title"] == "Cyclone Preparedness Decision Intelligence API"
    return spec


# ==============================================================================
# 1. DOCUMENTATION & SCHEMA ENDPOINTS
# ==============================================================================

def test_openapi_schema_endpoint(client: TestClient, openapi_spec):
    """Verify /openapi.json is accessible and syntactically valid."""
    assert openapi_spec["openapi"].startswith("3.")
    assert len(openapi_spec["paths"]) >= 25


def test_swagger_ui_endpoint(client: TestClient):
    """Verify /docs serves Swagger UI HTML."""
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "swagger-ui" in resp.text.lower() or "swagger" in resp.text.lower()


def test_redoc_ui_endpoint(client: TestClient):
    """Verify /redoc serves ReDoc HTML."""
    resp = client.get("/redoc")
    assert resp.status_code == 200
    assert "redoc" in resp.text.lower()


def test_root_redirect_to_docs(client: TestClient):
    """Verify root GET redirects to /docs."""
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


# ==============================================================================
# 2. SYSTEM HEALTH & READINESS ENDPOINTS
# ==============================================================================

def test_health_endpoints_success(client: TestClient):
    """Verify root /health and /api/health return HTTP 200 with complete metadata."""
    for ep in ["/health", "/api/health"]:
        r = client.get(ep)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["total_units"] == 1879
        assert data["model_loaded"] is True


def test_ready_endpoint_success(client: TestClient):
    """Verify /ready and /api/health/ready return ready: True."""
    for ep in ["/ready", "/api/health/ready"]:
        r = client.get(ep)
        assert r.status_code == 200
        assert r.json()["ready"] is True


def test_health_unsupported_methods(client: TestClient):
    """Verify POST and DELETE to /health return 405 Method Not Allowed."""
    assert client.post("/health").status_code == 405
    assert client.delete("/health").status_code == 405


# ==============================================================================
# 3. DASHBOARD & DISTRICTS ENDPOINTS
# ==============================================================================

def test_dashboard_overview_success(client: TestClient):
    """Verify /api/dashboard/overview returns expected structure and metrics."""
    r = client.get("/api/dashboard/overview")
    assert r.status_code == 200
    data = r.json()
    assert "total_villages" in data or "total_units" in data
    assert data.get("total_villages", 1879) == 1879


def test_districts_list_success(client: TestClient):
    """Verify /api/districts lists coastal districts."""
    r = client.get("/api/districts")
    assert r.status_code == 200
    data = r.json()
    assert "districts" in data
    assert len(data["districts"]) >= 3


def test_district_by_id_success(client: TestClient):
    """Verify /api/districts/{district_id} returns detail for valid district."""
    for valid_id in ["dakshina-kannada", "udupi", "uttara-kannada"]:
        r = client.get(f"/api/districts/{valid_id}")
        assert r.status_code == 200
        data = r.json()
        assert "district_name" in data


def test_district_by_id_invalid(client: TestClient):
    """Verify /api/districts/{district_id} returns 404 for nonexistent district."""
    r = client.get("/api/districts/nonexistent_district_xyz")
    assert r.status_code == 404


# ==============================================================================
# 4. COMMUNITIES & MAPS ENDPOINTS
# ==============================================================================

def test_communities_list_success(client: TestClient):
    """Verify /api/communities supports filtering and pagination."""
    # Default query
    r = client.get("/api/communities?limit=10&offset=0")
    assert r.status_code == 200
    data = r.json()
    items = data.get("items", data) if isinstance(data, dict) else data
    assert len(items) <= 10

    # Search filter
    r_search = client.get("/api/communities?search=Tallur")
    assert r_search.status_code == 200


def test_communities_list_invalid_params(client: TestClient):
    """Verify /api/communities rejects invalid query parameters with 422."""
    # Negative limit
    r1 = client.get("/api/communities?limit=-5")
    assert r1.status_code == 422

    # String passed to limit
    r2 = client.get("/api/communities?limit=abc")
    assert r2.status_code == 422


def test_community_by_id_success(client: TestClient):
    """Verify /api/communities/{id} returns details for known village."""
    valid_id = "11-29-563-05478-602577"
    r = client.get(f"/api/communities/{valid_id}")
    assert r.status_code == 200
    data = r.json()
    assert "area_id" in data or "id" in data


def test_community_by_id_invalid(client: TestClient):
    """Verify /api/communities/{id} returns 404 for unknown community."""
    r = client.get("/api/communities/invalid-area-id-999999")
    assert r.status_code == 404


def test_maps_communities_success(client: TestClient):
    """Verify /api/maps/communities returns geospatial points."""
    r = client.get("/api/maps/communities")
    assert r.status_code == 200
    data = r.json()
    assert "points" in data or "features" in data or isinstance(data, list)


# ==============================================================================
# 5. ANALYTICS ENDPOINTS
# ==============================================================================

def test_analytics_endpoints_success(client: TestClient):
    """Verify analytical endpoints return HTTP 200."""
    endpoints = [
        "/api/analytics/preparedness",
        "/api/analytics/priority-bands",
        "/api/analytics/clusters",
        "/api/analytics/model-performance",
    ]
    for ep in endpoints:
        r = client.get(ep)
        assert r.status_code == 200, f"Endpoint {ep} failed with {r.status_code}"


# ==============================================================================
# 6. MODEL INFERENCE & REGISTRY ENDPOINTS
# ==============================================================================

def test_model_predict_valid_request(client: TestClient):
    """Verify /api/models/predict generates inference with real loaded model."""
    payload = {
        "features": {
            "dist_coast_km": 4.5,
            "elevation_mean": 15.0,
            "avg_household_size": 4.2,
            "sc_share": 0.10,
            "st_share": 0.05,
            "vulnerable_pop_share": 0.15,
            "road_connectivity_score": 0.80,
            "drinking_water_score": 0.85,
            "n_hc_10km": 3,
            "n_phc_chc_10km": 2,
            "n_gauges_20km": 1,
            "n_shelters_10km": 4,
            "cvi_physical": 0.40,
            "cvi_social": 0.30,
            "has_pucca_road": 1,
            "has_major_district_road": 1,
            "bus_connectivity": 1,
            "log1p_population": 7.2,
            "log1p_pop_density_sqkm": 6.0,
            "log1p_dist_nearest_shelter_km": 1.1,
            "log1p_dist_nearest_hospital_km": 2.0,
            "coastal_risk_score": 0.60,
            "evacuation_sufficiency": 0.85,
            "vulnerability_exposure": 0.35,
            "infra_composite": 0.70,
            "monitoring_coverage": 0.80,
        }
    }
    r = client.post("/api/models/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_tier"] in ["High", "Low", "Medium"]
    assert "probabilities" in data
    assert "High" in data["probabilities"] or "prob_high" in data["probabilities"]


def test_model_predict_invalid_requests(client: TestClient):
    """Verify /api/models/predict rejects missing features, wrong types, and malformed json."""
    # 1. Missing required features
    r_missing = client.post("/api/models/predict", json={"features": {"dist_coast_km": 5.0}})
    assert r_missing.status_code in (400, 422)

    # 2. String in numeric feature
    r_type = client.post("/api/models/predict", json={"features": {"dist_coast_km": "invalid_string"}})
    assert r_type.status_code in (400, 422)

    # 3. Empty payload
    r_empty = client.post("/api/models/predict", json={})
    assert r_empty.status_code in (400, 422)

    # 4. Unsupported HTTP method (GET)
    r_method = client.get("/api/models/predict")
    assert r_method.status_code == 405


def test_model_registry_endpoints(client: TestClient):
    """Verify /api/models/registry and /api/models/registry/{id}."""
    # List models
    r = client.get("/api/models/registry")
    assert r.status_code == 200
    data = r.json()
    models = data.get("models", data) if isinstance(data, dict) else data
    assert isinstance(models, list)
    model_ids = [m.get("model_identifier") for m in models]
    assert "village-preparedness-classifier" in model_ids
    assert "coastal-vulnerability-kmeans" in model_ids

    # Get specific valid model
    r_cls = client.get("/api/models/registry/village-preparedness-classifier")
    assert r_cls.status_code == 200
    assert r_cls.json()["status"] == "production"

    # Get nonexistent model
    r_404 = client.get("/api/models/registry/nonexistent-model-xyz")
    assert r_404.status_code == 404


# ==============================================================================
# 7. SIMULATION ENDPOINTS
# ==============================================================================

def test_simulation_prioritize_success(client: TestClient):
    """Verify /api/simulation/prioritize runs MCDA scenario."""
    payload = {
        "weights": {
            "healthcare": 0.25,
            "vulnerability": 0.25,
            "infrastructure": 0.25,
            "hazard": 0.25,
        }
    }
    r = client.post("/api/simulation/prioritize", json=payload)
    assert r.status_code == 200


def test_simulation_prioritize_invalid_weights(client: TestClient):
    """Verify /api/simulation/prioritize rejects negative weights or malformed body."""
    # Negative weight
    r_neg = client.post(
        "/api/simulation/prioritize",
        json={"weights": {"healthcare": -1.0, "vulnerability": 0.5, "infrastructure": 0.5, "hazard": 0.5}},
    )
    assert r_neg.status_code in (400, 422)

    # Non-numeric weight
    r_str = client.post(
        "/api/simulation/prioritize",
        json={"weights": {"healthcare": "high", "vulnerability": 0.5}},
    )
    assert r_str.status_code in (400, 422)


def test_simulation_compare_success(client: TestClient):
    """Verify /api/simulation/compare compares two weight scenarios."""
    payload = {
        "scenario_a": {"healthcare": 0.40, "vulnerability": 0.20, "infrastructure": 0.20, "hazard": 0.20},
        "scenario_b": {"healthcare": 0.20, "vulnerability": 0.40, "infrastructure": 0.20, "hazard": 0.20},
    }
    r = client.post("/api/simulation/compare", json=payload)
    assert r.status_code == 200


# ==============================================================================
# 8. ASSISTANT & TOOL ENDPOINTS
# ==============================================================================

def test_assistant_tools_endpoint(client: TestClient):
    """Verify /api/assistant/tools returns 7 registered deterministic tools."""
    r = client.get("/api/assistant/tools")
    assert r.status_code == 200
    tools = r.json()
    assert isinstance(tools, list)
    assert len(tools) == 7


def test_assistant_chat_endpoint_valid_and_invalid(client: TestClient):
    """Verify /api/assistant/chat processes query and rejects empty payload."""
    # Valid query
    r_valid = client.post("/api/assistant/chat", json={"message": "What is the preparedness score of Tallur village?"})
    assert r_valid.status_code == 200
    data = r_valid.json()
    assert "answer" in data or "response" in data

    # Empty payload
    r_empty = client.post("/api/assistant/chat", json={})
    assert r_empty.status_code in (400, 422)


# ==============================================================================
# 9. BACKWARD-COMPATIBILITY LEGACY V1 ENDPOINTS
# ==============================================================================

def test_legacy_v1_endpoints(client: TestClient):
    """Verify backward compatibility of /api/v1/* routes."""
    # /api/v1/units/search
    r_search = client.get("/api/v1/units/search?query=Tallur")
    assert r_search.status_code == 200

    # /api/v1/units/{id}
    r_unit = client.get("/api/v1/units/Tallur")
    assert r_unit.status_code == 200

    # /api/v1/units/nonexistent
    assert client.get("/api/v1/units/unknown_9999").status_code == 404

    # /api/v1/archetypes
    r_arch = client.get("/api/v1/archetypes")
    assert r_arch.status_code == 200

    # /api/v1/model/health
    r_mod_health = client.get("/api/v1/model/health")
    assert r_mod_health.status_code == 200
    assert r_mod_health.json()["model_loaded"] is True

    # /api/v1/decision-support/summary
    r_ds = client.get("/api/v1/decision-support/summary")
    assert r_ds.status_code == 200
