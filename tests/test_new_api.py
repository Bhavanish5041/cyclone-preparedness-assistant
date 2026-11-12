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


# =============================================================================
# Phase 3: Comprehensive Regression Tests for Community Explorer & CORS
# =============================================================================


def test_phase3_01_community_list_returns_200(client: TestClient):
    """1. Community list endpoint returns HTTP 200."""
    response = client.get("/api/communities")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] == 1879


def test_phase3_02_community_records_expected_schema(client: TestClient):
    """2. Community records are returned with the expected frontend schema."""
    response = client.get("/api/communities?limit=5")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 5

    required_keys = {
        "area_id",
        "pc11_id",
        "shrid2",
        "village_name",
        "district_name",
        "unit_type",
        "lat",
        "lon",
        "preparedness_score",
        "preparedness_tier",
        "priority_band",
        "priority_rank",
        "band_rank",
    }
    for item in items:
        for key in required_keys:
            assert key in item, f"Missing required key '{key}' in community record"
        assert item["unit_type"] in ("VILLAGE", "TOWN")
        assert isinstance(item["lat"], float) and -90 <= item["lat"] <= 90
        assert isinstance(item["lon"], float) and -180 <= item["lon"] <= 180
        assert item["preparedness_tier"] in ("High", "Medium", "Low")
        assert item["priority_band"].startswith("BAND_")


def test_phase3_03_search_by_community_name(client: TestClient):
    """3. Search by community name works case-insensitively."""
    response = client.get("/api/communities?search=tallur")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    found_names = [c["village_name"].lower() for c in data["items"]]
    assert any("tallur" in name for name in found_names)


def test_phase3_04_search_by_pc11_id(client: TestClient):
    """4. Search by Census PC11 ID works correctly."""
    response = client.get("/api/communities?search=608720")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    item = data["items"][0]
    assert item["pc11_id"] == "608720"
    assert item["village_name"] == "Edmoge"


def test_phase3_05_search_by_shrid(client: TestClient):
    """5. Search by SHRID works correctly."""
    # Lookup first unit's SHRID
    ref = client.get("/api/communities?limit=1").json()["items"][0]
    shrid = ref["shrid2"]
    assert shrid is not None

    response = client.get(f"/api/communities?search={shrid}")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any(c["shrid2"] == shrid for c in data["items"])


def test_phase3_06_district_filtering(client: TestClient):
    """6. District filtering returns exact counts for each district."""
    # Dakshina Kannada: 369 units
    r_dk = client.get("/api/communities?district=Dakshina+Kannada")
    assert r_dk.status_code == 200
    assert r_dk.json()["total"] == 369
    assert all(c["district_name"] == "Dakshina Kannada" for c in r_dk.json()["items"])

    # Udupi: 253 units
    r_ud = client.get("/api/communities?district=Udupi")
    assert r_ud.status_code == 200
    assert r_ud.json()["total"] == 253
    assert all(c["district_name"] == "Udupi" for c in r_ud.json()["items"])

    # Uttara Kannada: 1,257 units
    r_uk = client.get("/api/communities?district=Uttara+Kannada")
    assert r_uk.status_code == 200
    assert r_uk.json()["total"] == 1257
    assert all(c["district_name"] == "Uttara Kannada" for c in r_uk.json()["items"])

    # Sum must equal total
    assert 369 + 253 + 1257 == 1879


def test_phase3_07_priority_band_filtering(client: TestClient):
    """7. Priority Band filtering returns exact baseline allocations."""
    expected_band_counts = {
        "BAND_1_CRITICAL": 155,
        "BAND_2_SPECIALIZED": 829,
        "BAND_3_MODERATE": 299,
        "BAND_4_ADEQUATE": 596,
    }
    for band, expected_count in expected_band_counts.items():
        resp = client.get(f"/api/communities?priority_band={band}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == expected_count, f"Mismatch for {band}: got {data['total']}, expected {expected_count}"
        assert all(c["priority_band"] == band for c in data["items"])


def test_phase3_08_ahp_tier_filtering(client: TestClient):
    """8. AHP Tier filtering returns exact counts."""
    expected_tier_counts = {
        "High": 596,
        "Medium": 1128,
        "Low": 155,
    }
    for tier, expected_count in expected_tier_counts.items():
        resp = client.get(f"/api/communities?tier={tier}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == expected_count, f"Mismatch for {tier}: got {data['total']}, expected {expected_count}"
        assert all(c["preparedness_tier"] == tier for c in data["items"])


def test_phase3_09_combined_filtering(client: TestClient):
    """9. Multiple filters work in combination."""
    # Udupi + Band 4 Adequate
    resp = client.get("/api/communities?district=Udupi&priority_band=BAND_4_ADEQUATE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 190
    for item in data["items"]:
        assert item["district_name"] == "Udupi"
        assert item["priority_band"] == "BAND_4_ADEQUATE"

    # Dakshina Kannada + Low Tier + Band 1 Critical
    resp_crit = client.get("/api/communities?district=Dakshina+Kannada&tier=Low&priority_band=BAND_1_CRITICAL")
    assert resp_crit.status_code == 200
    data_crit = resp_crit.json()
    assert data_crit["total"] == 2
    for item in data_crit["items"]:
        assert item["district_name"] == "Dakshina Kannada"
        assert item["preparedness_tier"] == "Low"
        assert item["priority_band"] == "BAND_1_CRITICAL"


def test_phase3_10_pagination_and_total_count(client: TestClient):
    """10. Pagination operates correctly and total reflects full filtered count."""
    resp_p1 = client.get("/api/communities?page=1&limit=25")
    assert resp_p1.status_code == 200
    d1 = resp_p1.json()
    assert d1["total"] == 1879
    assert d1["page"] == 1
    assert d1["limit"] == 25
    assert d1["total_pages"] == (1879 + 25 - 1) // 25
    assert len(d1["items"]) == 25

    resp_p2 = client.get("/api/communities?page=2&limit=25")
    assert resp_p2.status_code == 200
    d2 = resp_p2.json()
    assert d2["page"] == 2
    assert len(d2["items"]) == 25

    # Check non-overlapping items
    p1_ids = {c["area_id"] for c in d1["items"]}
    p2_ids = {c["area_id"] for c in d2["items"]}
    assert len(p1_ids.intersection(p2_ids)) == 0


def test_phase3_11_empty_search_results(client: TestClient):
    """11. Search yielding zero matches returns 200 with empty list, not an error."""
    resp = client.get("/api/communities?search=NonExistentFictionalTown999")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["total_pages"] == 1


def test_phase3_12_invalid_query_parameters(client: TestClient):
    """12. Invalid query parameters return HTTP 422 validation errors."""
    # page < 1
    r_bad_page = client.get("/api/communities?page=0")
    assert r_bad_page.status_code == 422

    # limit > 100
    r_bad_limit = client.get("/api/communities?limit=500")
    assert r_bad_limit.status_code == 422

    # invalid sort_order
    r_bad_order = client.get("/api/communities?sort_order=sideways")
    assert r_bad_order.status_code == 422


def test_phase3_13_dataset_availability_and_coverage(client: TestClient):
    """13. Full dataset of 1,879 records is queryable without gaps."""
    r_all = client.get("/api/communities?limit=100&page=1")
    assert r_all.status_code == 200
    data = r_all.json()
    assert data["total"] == 1879
    for item in data["items"]:
        assert item["area_id"]
        assert item["pc11_id"]
        assert item["village_name"]
        assert item["lat"] != 0.0
        assert item["lon"] != 0.0


def test_cors_preflight_and_origins(client: TestClient):
    """14. CORS headers allow localhost, 127.0.0.1, and Cloudflare quick tunnels."""
    # Test Cloudflare Quick Tunnel Origin regex matching
    cf_origin = "https://indianapolis-lamp-picking-feeding.trycloudflare.com"
    r_opt = client.options(
        "/api/assistant/chat",
        headers={
            "Origin": cf_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r_opt.status_code == 200
    assert r_opt.headers.get("access-control-allow-origin") == cf_origin

    # Test another dynamic Cloudflare tunnel origin
    dynamic_cf = "https://random-words-subdomain-42.trycloudflare.com"
    r_dyn = client.options(
        "/api/assistant/chat",
        headers={
            "Origin": dynamic_cf,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r_dyn.status_code == 200
    assert r_dyn.headers.get("access-control-allow-origin") == dynamic_cf

    # Test localhost origin
    r_local = client.options(
        "/api/assistant/chat",
        headers={
            "Origin": "http://localhost",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r_local.status_code == 200
    assert r_local.headers.get("access-control-allow-origin") == "http://localhost"

