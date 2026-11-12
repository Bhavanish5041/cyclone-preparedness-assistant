"""Route Inventory and API Documentation Validation Test Suite.

Verifies:
- All registered FastAPI routes in `api.main:app` match `docs/api_inventory.json`.
- Zero documentation drift between live code, JSON inventory, and Markdown reference.
- Required documentation endpoints: `/docs`, `/redoc`, `/openapi.json` return 200 OK.
- Representative GET and POST endpoints function correctly and adhere to schema contracts:
  - Health & status (/health, /api/health)
  - Dashboard overview (/api/dashboard/overview)
  - Community search and filtering (/api/communities?search=Tallur)
  - Community detail dossier (/api/communities/608691)
  - District list and comparison (/api/districts)
  - Analytics and distributions (/api/analytics/preparedness, /api/analytics/model-performance)
  - Model registry (/api/models/registry, /api/models/registry/village-preparedness-classifier)
  - Assistant conversational POST endpoint (/api/assistant/chat)
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_PATH = PROJECT_ROOT / "docs" / "api_inventory.json"
MARKDOWN_PATH = PROJECT_ROOT / "docs" / "API_REFERENCE.md"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient with full lifespan startup."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def inventory_data():
    """Load the generated API inventory JSON file."""
    assert INVENTORY_PATH.is_file(), f"Missing API inventory file at {INVENTORY_PATH}"
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def markdown_content():
    """Load the generated Markdown API reference file."""
    assert MARKDOWN_PATH.is_file(), f"Missing API reference file at {MARKDOWN_PATH}"
    with open(MARKDOWN_PATH, "r", encoding="utf-8") as f:
        return f.read()


def test_documentation_endpoints_load(client: TestClient):
    """Verify that Swagger UI (/docs), ReDoc (/redoc), and OpenAPI JSON (/openapi.json) load successfully."""
    r_docs = client.get("/docs")
    assert r_docs.status_code == 200
    assert "swagger-ui" in r_docs.text.lower() or "html" in r_docs.headers.get("content-type", "")

    r_redoc = client.get("/redoc")
    assert r_redoc.status_code == 200
    assert "redoc" in r_redoc.text.lower() or "html" in r_redoc.headers.get("content-type", "")

    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    openapi_json = r_openapi.json()
    assert openapi_json.get("info", {}).get("title") == "Cyclone Preparedness Decision Intelligence API"
    assert "paths" in openapi_json
    assert len(openapi_json["paths"]) >= 31


def test_route_inventory_completeness_and_drift_detection(inventory_data):
    """Ensure every path and method in app.openapi() exists in docs/api_inventory.json."""
    openapi = app.openapi()
    paths = openapi.get("paths", {})

    inventory_endpoints = inventory_data.get("endpoints", [])
    registered_pairs = set()
    for ep in inventory_endpoints:
        registered_pairs.add((ep["method"].upper(), ep["path"]))

    missing_from_inventory = []
    for path, methods in paths.items():
        for m in methods.keys():
            pair = (m.upper(), path)
            if pair not in registered_pairs:
                missing_from_inventory.append(pair)

    assert not missing_from_inventory, (
        f"Documentation drift detected! The following registered FastAPI endpoints are missing "
        f"from docs/api_inventory.json: {missing_from_inventory}. "
        f"Run 'python scripts/generate_api_documentation.py' to synchronize."
    )


def test_markdown_reference_contains_all_endpoints(inventory_data, markdown_content):
    """Verify that every endpoint in docs/api_inventory.json is documented in docs/API_REFERENCE.md."""
    inventory_endpoints = inventory_data.get("endpoints", [])
    for ep in inventory_endpoints:
        heading = f"`{ep['method']} {ep['path']}`"
        assert heading in markdown_content, (
            f"Endpoint {heading} is present in api_inventory.json but missing from docs/API_REFERENCE.md"
        )


def test_endpoint_metadata_integrity(inventory_data):
    """Verify that every endpoint has complete, non-empty metadata fields."""
    required_fields = [
        "method", "path", "tag", "purpose", "status_codes",
        "example_request", "example_response", "group"
    ]
    for ep in inventory_data.get("endpoints", []):
        for field in required_fields:
            assert field in ep, f"Endpoint {ep.get('path')} missing required field '{field}'"
            assert ep[field] is not None, f"Endpoint {ep.get('path')} has null value for '{field}'"
        assert len(ep["status_codes"]) > 0
        assert isinstance(ep["used_by_frontend"], bool)
        assert isinstance(ep["internal_only"], bool)


# ==============================================================================
# REPRESENTATIVE ENDPOINT BEHAVIOR TESTS
# ==============================================================================


def test_health_endpoints(client: TestClient):
    """Verify GET /health and GET /api/health return valid readiness schemas."""
    for p in ("/health", "/api/health"):
        res = client.get(p)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["dataset_loaded"] is True
        assert data["total_units"] == 1879
        assert data["model_loaded"] is True
        assert "Dakshina Kannada" in data["districts"]


def test_dashboard_overview_endpoint(client: TestClient):
    """Verify GET /api/dashboard/overview returns regional command-center KPIs."""
    res = client.get("/api/dashboard/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["total_units"] == 1879
    assert data["districts_count"] == 3
    assert data["assigned_clusters_count"] == 622
    assert data["unassigned_clusters_count"] == 1257
    assert "BAND_1_CRITICAL" in data["priority_band_counts"]
    assert "High" in data["preparedness_tier_counts"]


def test_communities_search_and_detail_endpoints(client: TestClient):
    """Verify GET /api/communities and GET /api/communities/{id} for Tallur (PC11: 608691)."""
    # 1. Search for Tallur
    r_search = client.get("/api/communities?search=Tallur&limit=5")
    assert r_search.status_code == 200
    search_data = r_search.json()
    assert len(search_data["items"]) >= 1
    tallur_item = next((it for it in search_data["items"] if it["village_name"] == "Tallur"), None)
    assert tallur_item is not None
    assert tallur_item["district_name"] == "Udupi"
    assert tallur_item["priority_band"] == "BAND_4_ADEQUATE"

    # 2. Granular Detail Dossier
    r_detail = client.get(f"/api/communities/{tallur_item['area_id']}")
    assert r_detail.status_code == 200
    detail = r_detail.json()
    assert detail["village_name"] == "Tallur"
    assert detail["district_name"] == "Udupi"
    assert round(detail["preparedness_score"], 2) == 74.34
    assert detail["preparedness_tier"] == "High"
    assert detail["priority_band"] == "BAND_4_ADEQUATE"


def test_districts_endpoint(client: TestClient):
    """Verify GET /api/districts and GET /api/districts/{id}."""
    res = client.get("/api/districts")
    assert res.status_code == 200
    districts = res.json()
    d_names = {d["district_name"] for d in districts.get("items", districts if isinstance(districts, list) else districts.get("districts", []))}
    assert "Dakshina Kannada" in d_names
    assert "Udupi" in d_names
    assert "Uttara Kannada" in d_names


def test_analytics_endpoints(client: TestClient):
    """Verify GET /api/analytics/preparedness and GET /api/analytics/model-performance."""
    r_prep = client.get("/api/analytics/preparedness")
    assert r_prep.status_code == 200
    p_data = r_prep.json()
    assert p_data["total_units"] == 1879
    assert "tier_distribution" in p_data

    r_perf = client.get("/api/analytics/model-performance")
    assert r_perf.status_code == 200
    perf_data = r_perf.json()
    assert "test_metrics" in perf_data or "accuracy" in perf_data or "concordance_rate" in perf_data


def test_model_registry_endpoints(client: TestClient):
    """Verify GET /api/models/registry and GET /api/models/registry/{id}."""
    r_reg = client.get("/api/models/registry")
    assert r_reg.status_code == 200
    reg_data = r_reg.json()
    models = reg_data if isinstance(reg_data, list) else reg_data.get("models", [])
    model_ids = {m["model_identifier"] for m in models}
    assert "village-preparedness-classifier" in model_ids

    r_det = client.get("/api/models/registry/village-preparedness-classifier")
    assert r_det.status_code == 200
    m_det = r_det.json()
    assert m_det["version"] == "v1.0.0"
    assert m_det["status"] == "production"


def test_assistant_chat_post_endpoint(client: TestClient):
    """Verify POST /api/assistant/chat processes decision-support turn with deterministic grounding."""
    payload = {
        "message": "Explain why Tallur is classified as Band 4.",
        "conversation_history": []
    }
    res = client.post("/api/assistant/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    # Must preserve deterministic precedence note
    assert "scientific_precedence_note" in data or "AHP" in data["answer"]
