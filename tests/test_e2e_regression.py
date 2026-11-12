"""End-to-End Regression Test Suite (Stage 9 CI Gate).

Verifies the full end-to-end production workflow across canonical FastAPI endpoints:
1. Dashboard overview (/api/dashboard/overview)
2. Community map (/api/maps/communities)
3. Community search (/api/communities?search=Tallur)
4. Community detail (/api/communities/{id}) with Tallur PC11 608691 regression case:
   - Authoritative AHP score: 74.34
   - Preparedness tier: High
   - Priority band: BAND_4_ADEQUATE (Band 4)
   - Strict preservation of the distinction between AHP score, tier, and priority band
5. District comparison (/api/districts, /api/districts/{id})
6. Analytics (/api/analytics/preparedness, /api/analytics/priority-bands, /api/analytics/clusters)
7. Model performance (/api/analytics/model-performance)
8. Assistant conversational intelligence (/api/assistant/chat, /api/assistant/tools)
9. Interactive documentation (/docs, /redoc, /openapi.json)
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

TALLUR_AREA_ID = "11-29-569-05522-608691"
TALLUR_PC11_ID = "608691"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create test client with lifespan context initialized."""
    with TestClient(app) as c:
        yield c


# ==============================================================================
# 1. DASHBOARD OVERVIEW
# ==============================================================================


def test_e2e_dashboard_overview(client: TestClient):
    """Verify regional command-center executive summary."""
    res = client.get("/api/dashboard/overview")
    assert res.status_code == 200
    data = res.json()

    assert data["total_units"] == 1879
    assert data["districts_count"] == 3
    assert data["assigned_clusters_count"] == 622
    assert data["unassigned_clusters_count"] == 1257

    # Verify 4 priority bands exist
    bands = data["priority_band_counts"]
    for b in ["BAND_1_CRITICAL", "BAND_2_SPECIALIZED", "BAND_3_MODERATE", "BAND_4_ADEQUATE"]:
        assert b in bands
        assert bands[b] > 0

    assert sum(bands.values()) == 1879


# ==============================================================================
# 2. COMMUNITY MAP
# ==============================================================================


def test_e2e_community_map(client: TestClient):
    """Verify geospatial mapping points and coordinate bounds."""
    res = client.get("/api/maps/communities")
    assert res.status_code == 200
    data = res.json()

    assert "points" in data or "items" in data
    points = data.get("points", data.get("items", []))
    assert len(points) == 1879

    first = points[0]
    assert "lat" in first and "lon" in first
    assert "band" in first or "priority_band" in first
    assert "score" in first or "preparedness_score" in first


# ==============================================================================
# 3. COMMUNITY SEARCH & 4. DETAIL DOSSIER (TALLUR REGRESSION CASE)
# ==============================================================================


def test_e2e_community_search_tallur(client: TestClient):
    """Verify search returns Tallur village accurately."""
    res = client.get("/api/communities?search=Tallur&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1

    tallur = next((x for x in data["items"] if x["village_name"] == "Tallur"), None)
    assert tallur is not None
    assert tallur["district_name"] == "Udupi"
    assert tallur["priority_band"] == "BAND_4_ADEQUATE"


def test_e2e_tallur_authoritative_regression_case(client: TestClient):
    """Verify Tallur PC11 608691 maintains authoritative AHP score 74.34, High tier, Band 4."""
    res = client.get(f"/api/communities/{TALLUR_AREA_ID}")
    assert res.status_code == 200
    data = res.json()

    # Core Identifiers
    assert data["village_name"] == "Tallur"
    assert data["district_name"] == "Udupi"
    assert data["pc11_id"] == TALLUR_PC11_ID

    # Authoritative Metrics & Strict Metric Distinction
    ahp_score = round(float(data["preparedness_score"]), 2)
    prep_tier = data["preparedness_tier"]
    priority_band = data["priority_band"]

    assert ahp_score == 74.34, f"Authoritative AHP score must be exactly 74.34, got {ahp_score}"
    assert prep_tier == "High", f"Preparedness tier must be High, got {prep_tier}"
    assert priority_band == "BAND_4_ADEQUATE", f"Operational priority band must be BAND_4_ADEQUATE, got {priority_band}"

    # Verify domain indicator breakdowns exist
    for domain in ["medical", "shelters", "infrastructure", "early_warning", "socioeconomic"]:
        key = f"score_{domain}"
        assert key in data, f"Missing domain breakdown: {key}"
        assert data[key] is not None

    # Verify historical hazard context (Tauktae 2021)
    assert "tauktae_affected" in data


# ==============================================================================
# 5. DISTRICT COMPARISON
# ==============================================================================


def test_e2e_districts_intelligence(client: TestClient):
    """Verify district listing and granular district profile for Udupi."""
    res = client.get("/api/districts")
    assert res.status_code == 200
    data = res.json()

    d_list = data["items"] if "items" in data else (data if isinstance(data, list) else data.get("districts", []))
    d_names = {d["district_name"] for d in d_list}
    assert d_names == {"Dakshina Kannada", "Udupi", "Uttara Kannada"}

    # Granular district dossier
    res_udupi = client.get("/api/districts/udupi")
    assert res_udupi.status_code == 200
    u_data = res_udupi.json()
    assert u_data["district_name"] == "Udupi"
    assert u_data["total_units"] == 253


# ==============================================================================
# 6. ANALYTICS (PREPAREDNESS, BANDS, CLUSTERS)
# ==============================================================================


def test_e2e_analytics_distributions(client: TestClient):
    """Verify preparedness distributions, priority bands, and K-Means clusters."""
    # 1. Preparedness
    r_prep = client.get("/api/analytics/preparedness")
    assert r_prep.status_code == 200
    p_data = r_prep.json()
    assert p_data["total_units"] == 1879
    assert "tier_distribution" in p_data

    # 2. Priority bands
    r_bands = client.get("/api/analytics/priority-bands")
    assert r_bands.status_code == 200
    b_data = r_bands.json()
    assert "band_distribution" in b_data
    assert "BAND_1_CRITICAL" in b_data["band_distribution"]

    # 3. K-Means clusters & unassigned UK units
    r_clust = client.get("/api/analytics/clusters")
    assert r_clust.status_code == 200
    c_data = r_clust.json()
    assert c_data["assigned_count"] == 622
    assert c_data["unassigned_count"] == 1257
    assert len(c_data["clusters"]) == 4


# ==============================================================================
# 7. MODEL PERFORMANCE
# ==============================================================================


def test_e2e_model_performance(client: TestClient):
    """Verify surrogate model performance metrics and confusion matrix."""
    res = client.get("/api/analytics/model-performance")
    assert res.status_code == 200
    data = res.json()

    assert data["model_name"] == "village-preparedness-classifier"
    assert "test_metrics" in data
    assert data["test_metrics"]["accuracy"] >= 0.90
    assert "test_confusion_matrix" in data
    assert data["concordance_rate_overall"] >= 85.0


# ==============================================================================
# 8. ASSISTANT CONVERSATIONAL DECISION SUPPORT
# ==============================================================================


def test_e2e_assistant_tools_and_chat(client: TestClient):
    """Verify 7 registered tools and conversational query execution."""
    # 1. Registered tools catalog
    r_tools = client.get("/api/assistant/tools")
    assert r_tools.status_code == 200
    tools = r_tools.json()
    assert len(tools) == 7
    tool_names = {t["name"] for t in tools}
    assert "lookup_community" in tool_names
    assert "explain_preparedness_assessment" in tool_names
    assert "compare_districts" in tool_names

    # 2. Chat execution for Tallur
    chat_payload = {
        "message": "Explain why Tallur is classified as Band 4.",
        "conversation_history": [],
    }
    r_chat = client.post("/api/assistant/chat", json=chat_payload)
    assert r_chat.status_code == 200
    chat_data = r_chat.json()
    assert "answer" in chat_data
    assert len(chat_data["answer"]) > 0
    # Must preserve deterministic scientific precedence note
    assert "scientific_precedence_note" in chat_data
    assert "AHP" in chat_data["scientific_precedence_note"]


# ==============================================================================
# 9. API DOCUMENTATION
# ==============================================================================


def test_e2e_api_documentation_endpoints(client: TestClient):
    """Verify Swagger UI (/docs), ReDoc (/redoc), and OpenAPI specification."""
    # 1. Swagger UI
    r_docs = client.get("/docs")
    assert r_docs.status_code == 200

    # 2. ReDoc
    r_redoc = client.get("/redoc")
    assert r_redoc.status_code == 200

    # 3. OpenAPI JSON
    r_spec = client.get("/openapi.json")
    assert r_spec.status_code == 200
    spec = r_spec.json()
    assert spec["info"]["title"] == "Cyclone Preparedness Decision Intelligence API"
    assert len(spec["paths"]) >= 31
