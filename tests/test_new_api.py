"""Unit and integration tests for the new FastAPI backend (api/main.py)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create TestClient fixture with lifespan events triggered."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_redirect(client: TestClient):
    """Verify root path redirects to docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_health_endpoints(client: TestClient):
    """Verify root health and /api/health."""
    r_root = client.get("/health")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "healthy"
    assert r_root.json()["total_units"] == 1879

    r_api = client.get("/api/health")
    assert r_api.status_code == 200
    data = r_api.json()
    assert data["status"] == "healthy"
    assert data["total_units"] == 1879
    assert data["total_assigned_units"] == 622
    assert len(data["districts"]) == 3


def test_dashboard_overview_exact_baselines(client: TestClient):
    """Verify dashboard overview matches authoritative scientific baseline numbers."""
    response = client.get("/api/dashboard/overview")
    assert response.status_code == 200
    data = response.json()

    assert data["total_units"] == 1879
    assert data["districts_count"] == 3
    assert data["assigned_clusters_count"] == 622
    assert data["unassigned_clusters_count"] == 1257

    # Exact authoritative priority band counts
    bands = data["priority_band_counts"]
    assert bands["BAND_1_CRITICAL"] == 155
    assert bands["BAND_2_SPECIALIZED"] == 829
    assert bands["BAND_3_MODERATE"] == 299
    assert bands["BAND_4_ADEQUATE"] == 596
    assert sum(bands.values()) == 1879

    # Exact authoritative preparedness tier counts
    tiers = data["preparedness_tier_counts"]
    assert tiers["Low"] == 155
    assert tiers["Medium"] == 1128
    assert tiers["High"] == 596
    assert sum(tiers.values()) == 1879

    # 3 districts
    assert len(data["districts"]) == 3
    d_names = [d["district_name"] for d in data["districts"]]
    assert "Dakshina Kannada" in d_names
    assert "Udupi" in d_names
    assert "Uttara Kannada" in d_names


def test_districts_list_and_detail(client: TestClient):
    """Verify districts listing and single district inspection."""
    r_list = client.get("/api/districts")
    assert r_list.status_code == 200
    data = r_list.json()
    assert data["total_districts"] == 3

    # Check Udupi
    r_udupi = client.get("/api/districts/udupi")
    assert r_udupi.status_code == 200
    u_data = r_udupi.json()
    assert u_data["district_name"] == "Udupi"
    assert u_data["total_units"] == 253
    assert u_data["band_counts"]["BAND_4_ADEQUATE"] == 190
    assert u_data["band_counts"]["BAND_2_SPECIALIZED"] == 20
    assert u_data["subdistricts"] is not None

    # Check Dakshina Kannada
    r_dk = client.get("/api/districts/dakshina-kannada")
    assert r_dk.status_code == 200
    dk_data = r_dk.json()
    assert dk_data["total_units"] == 369

    # Check Uttara Kannada
    r_uk = client.get("/api/districts/uttara-kannada")
    assert r_uk.status_code == 200
    uk_data = r_uk.json()
    assert uk_data["total_units"] == 1257

    # Non-existent district
    r_none = client.get("/api/districts/non-existent-district")
    assert r_none.status_code == 404


def test_communities_list_and_filtering(client: TestClient):
    """Verify community search, filters, pagination, and sorting."""
    # Default list
    r_list = client.get("/api/communities?limit=10&page=1")
    assert r_list.status_code == 200
    data = r_list.json()
    assert data["total"] == 1879
    assert data["page"] == 1
    assert data["limit"] == 10
    assert len(data["items"]) == 10

    # Search filter
    r_search = client.get("/api/communities?search=Tallur")
    assert r_search.status_code == 200
    s_data = r_search.json()
    assert s_data["total"] >= 1
    names = [c["village_name"] for c in s_data["items"]]
    assert any("tallur" in n.lower() for n in names)

    # District filter
    r_dk = client.get("/api/communities?district=Dakshina+Kannada")
    assert r_dk.status_code == 200
    assert r_dk.json()["total"] == 369

    # Priority band filter
    r_b1 = client.get("/api/communities?priority_band=BAND_1_CRITICAL")
    assert r_b1.status_code == 200
    assert r_b1.json()["total"] == 155

    # Tier filter
    r_low = client.get("/api/communities?tier=Low")
    assert r_low.status_code == 200
    assert r_low.json()["total"] == 155


def test_community_detail_dossier(client: TestClient):
    """Verify granular community inspection dossier for Tallur."""
    response = client.get("/api/communities/Tallur")
    assert response.status_code == 200
    data = response.json()

    assert data["village_name"] == "Tallur"
    assert data["district_name"] == "Udupi"
    assert data["preparedness_tier"] == "High"
    assert data["preparedness_score"] > 60.0
    assert data["priority_band"] == "BAND_4_ADEQUATE"
    assert data["lat"] > 0
    assert data["lon"] > 0
    assert len(data["actionable_advisories"]) > 0
    assert "census_vintage" in data["provenance_and_limitations"]

    # Not found case
    r_none = client.get("/api/communities/definitely_not_a_real_village_123")
    assert r_none.status_code == 404


def test_maps_communities_endpoint(client: TestClient):
    """Verify geospatial mapping points endpoint."""
    response = client.get("/api/maps/communities")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1879
    assert len(data["points"]) == 1879

    # Test filtering by district on map
    r_udupi = client.get("/api/maps/communities?district=Udupi")
    assert r_udupi.status_code == 200
    assert r_udupi.json()["total"] == 253


def test_analytics_endpoints(client: TestClient):
    """Verify preparedness, priority bands, clusters, and model performance analytics."""
    # Preparedness
    r_prep = client.get("/api/analytics/preparedness")
    assert r_prep.status_code == 200
    assert r_prep.json()["total_units"] == 1879
    assert r_prep.json()["tier_distribution"]["High"] == 596

    # Priority Bands
    r_bands = client.get("/api/analytics/priority-bands")
    assert r_bands.status_code == 200
    assert r_bands.json()["band_distribution"]["BAND_1_CRITICAL"] == 155
    assert r_bands.json()["band_distribution"]["BAND_2_SPECIALIZED"] == 829

    # Clusters
    r_clusters = client.get("/api/analytics/clusters")
    assert r_clusters.status_code == 200
    c_data = r_clusters.json()
    assert c_data["total_units"] == 1879
    assert c_data["assigned_count"] == 622
    assert c_data["unassigned_count"] == 1257
    assert len(c_data["clusters"]) == 4

    # Model Performance
    r_model = client.get("/api/analytics/model-performance")
    assert r_model.status_code == 200
    m_data = r_model.json()
    assert m_data["model_name"] == "village-preparedness-classifier"
    assert m_data["features_count"] == 26
    assert m_data["test_metrics"]["accuracy"] > 0.90
    assert len(m_data["test_confusion_matrix"]["matrix"]) == 3


def test_assistant_tools_catalog(client: TestClient):
    """Verify assistant tool catalog exposes all 7 deterministic tools."""
    response = client.get("/api/assistant/tools")
    assert response.status_code == 200
    tools = response.json()
    assert len(tools) == 7
    names = [t["name"] for t in tools]
    assert "lookup_community" in names
    assert "explain_preparedness_assessment" in names
    assert "compare_districts" in names
    assert "filter_and_rank_communities" in names


def test_simulation_prioritize(client: TestClient):
    """Verify dynamic policy scenario re-ranking."""
    payload = {
        "scenario_name": "Test Coastal Hazard",
        "weights": {
            "healthcare": 0.1,
            "demographic": 0.1,
            "infrastructure": 0.1,
            "coastal": 0.7,
            "include_population_scale": False,
        },
        "top_n": 5,
    }
    response = client.post("/api/simulation/prioritize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["top_units"]) == 5
    assert data["weights_applied"]["coastal"] == 0.7
