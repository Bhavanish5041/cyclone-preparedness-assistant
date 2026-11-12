"""Unit and integration tests for DecisionSupportApiClient (Milestone 4B.1).

Verifies:
1. Direct-mode regional summary.
2. Direct-mode individual unit lookup (PC11, name, SHRID).
3. Direct-mode filtering and pagination (by district, band, limits).
4. REST-mode successful responses via FastAPI test transport.
5. REST-mode connection failures (verifying no silent direct fallback).
6. REST-mode HTTP error handling (500, 422, malformed JSON).
7. Invalid unit identifiers (404, empty, whitespace).
8. Missing/null field preservation (Uttara Kannada cluster nulls).
9. Full parity and consistency between direct and REST modes.
10. Correct authoritative priority-band aggregation across all 1,879 units.
11. Local client caching and force-refresh behaviors.
12. Environment variable overrides and context manager lifecycle.
"""

from unittest.mock import MagicMock
import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.api.schemas import (
    DecisionEvaluationListResponse,
    DecisionSupportEvaluation,
    RegionalDecisionSummaryResponse,
)
from src.assistant.api_client import (
    AdapterMode,
    ApiClientConnectionError,
    ApiClientError,
    ApiClientNotFoundError,
    ApiClientResponseError,
    ApiClientValidationError,
    DecisionSupportApiClient,
    get_api_client,
)
from src.assistant.decision_support import DecisionSupportService


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    """Provide a shared, pre-loaded DecisionSupportService instance."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def direct_client(shared_service: DecisionSupportService) -> DecisionSupportApiClient:
    """Provide DecisionSupportApiClient in direct mode with pre-loaded service."""
    return DecisionSupportApiClient(mode="direct", service=shared_service)


@pytest.fixture(scope="module")
def rest_client() -> DecisionSupportApiClient:
    """Provide DecisionSupportApiClient in REST mode using FastAPI TestClient."""
    test_client = TestClient(app)
    return DecisionSupportApiClient(mode="rest", http_client=test_client)


# ==============================================================================
# TEST 1: DIRECT-MODE REGIONAL SUMMARY
# ==============================================================================


def test_direct_mode_regional_summary(direct_client: DecisionSupportApiClient):
    """Verify direct-mode retrieval of regional summary across all 1,879 records."""
    summary = direct_client.get_regional_summary()
    assert isinstance(summary, RegionalDecisionSummaryResponse)
    assert summary.total_units == 1879

    # Verify authoritative four-band counts
    assert summary.band_counts["BAND_1_CRITICAL"] == 155
    assert summary.band_counts["BAND_2_SPECIALIZED"] == 829
    assert summary.band_counts["BAND_3_MODERATE"] == 299
    assert summary.band_counts["BAND_4_ADEQUATE"] == 596
    assert sum(summary.band_counts.values()) == 1879

    # Verify all 3 districts are represented
    breakdown = summary.district_band_breakdown
    assert set(breakdown.keys()) == {"Dakshina Kannada", "Udupi", "Uttara Kannada"}
    assert sum(breakdown["Dakshina Kannada"].values()) == 369
    assert sum(breakdown["Udupi"].values()) == 253
    assert sum(breakdown["Uttara Kannada"].values()) == 1257

    # Model agreement counts
    assert summary.model_agreement_counts["FULL_AGREEMENT"] == 1792
    assert summary.model_agreement_counts["BORDERLINE_DISAGREEMENT"] == 87

    # Provenance accuracies
    assert summary.provenance_agreement is not None
    assert summary.provenance_agreement["TRAIN_IN_SAMPLE"]["agreement_rate"] == 0.9589
    assert summary.provenance_agreement["VAL_OUT_OF_SAMPLE"]["agreement_rate"] == 0.9397
    assert summary.provenance_agreement["TEST_OUT_OF_SAMPLE"]["agreement_rate"] == 0.9433


# ==============================================================================
# TEST 2: DIRECT-MODE INDIVIDUAL UNIT LOOKUP
# ==============================================================================


def test_direct_mode_unit_lookup(direct_client: DecisionSupportApiClient):
    """Verify direct-mode lookup by PC11 code, village name, and SHRID2."""
    # 1. Lookup by PC11 code (Edmoge, Udupi)
    unit_pc11 = direct_client.get_unit("608720")
    assert isinstance(unit_pc11, DecisionSupportEvaluation)
    assert unit_pc11.village_name == "Edmoge"
    assert unit_pc11.district_name == "Udupi"
    assert unit_pc11.provisional_priority_band == "BAND_3_MODERATE"
    assert unit_pc11.preparedness_score == 49.84
    assert unit_pc11.preparedness_tier == "Medium"
    assert len(unit_pc11.domain_scores) == 5
    assert "score_shelters" in unit_pc11.domain_scores
    assert "score_medical" in unit_pc11.domain_scores
    assert "score_infrastructure" in unit_pc11.domain_scores
    assert len(unit_pc11.recommended_advisories) > 0

    # 2. Lookup by village name (Tallur, Udupi)
    unit_name = direct_client.get_unit("Tallur")
    assert isinstance(unit_name, DecisionSupportEvaluation)
    assert unit_name.pc11_id == "608691"
    assert unit_name.district_name == "Udupi"
    assert unit_name.provisional_priority_band == "BAND_4_ADEQUATE"
    assert unit_name.band_rank >= 1

    # 3. Lookup by SHRID2 (Edmoge)
    shrid = unit_pc11.shrid2
    unit_shrid = direct_client.get_unit(shrid)
    assert unit_shrid.pc11_id == "608720"
    assert unit_shrid.overall_priority_rank == unit_pc11.overall_priority_rank


# ==============================================================================
# TEST 3: DIRECT-MODE FILTERING AND PAGINATION
# ==============================================================================


def test_direct_mode_filtering_and_pagination(direct_client: DecisionSupportApiClient):
    """Verify direct-mode filtering by district and priority band with pagination."""
    # 1. District filter: Dakshina Kannada
    res_dk = direct_client.get_units(district="Dakshina Kannada", limit=50)
    assert isinstance(res_dk, DecisionEvaluationListResponse)
    assert res_dk.total_evaluated == 1879
    assert res_dk.total_matching == 369
    assert len(res_dk.units) == 50
    assert all(u.district_name == "Dakshina Kannada" for u in res_dk.units)

    # 2. District filter: Udupi
    res_udupi = direct_client.get_units(filters={"district": "Udupi"}, pagination={"limit": 20})
    assert res_udupi.total_matching == 253
    assert len(res_udupi.units) == 20
    assert all(u.district_name == "Udupi" for u in res_udupi.units)

    # 3. Priority band filter: BAND_1_CRITICAL
    res_b1 = direct_client.get_units(priority_band="BAND_1_CRITICAL", limit=200)
    assert res_b1.total_matching == 155
    assert len(res_b1.units) == 155
    assert all(u.provisional_priority_band == "BAND_1_CRITICAL" for u in res_b1.units)

    # 4. Priority band filter: BAND_2_SPECIALIZED
    res_b2 = direct_client.get_units(priority_band="BAND_2_SPECIALIZED", limit=10)
    assert res_b2.total_matching == 829
    assert len(res_b2.units) == 10

    # 5. Combined filter: Uttara Kannada + BAND_1_CRITICAL
    res_uk_b1 = direct_client.get_units(district="Uttara Kannada", priority_band="BAND_1_CRITICAL", limit=200)
    assert res_uk_b1.total_matching == 153
    assert len(res_uk_b1.units) == 153

    # 6. Pagination offset ordering check
    page1 = direct_client.get_units(limit=5, offset=0)
    page2 = direct_client.get_units(limit=5, offset=5)
    assert len(page1.units) == 5
    assert len(page2.units) == 5
    ids_p1 = [u.shrid2 for u in page1.units]
    ids_p2 = [u.shrid2 for u in page2.units]
    assert set(ids_p1).isdisjoint(set(ids_p2))
    assert page1.units[-1].overall_priority_rank < page2.units[0].overall_priority_rank


# ==============================================================================
# TEST 4: REST-MODE SUCCESSFUL RESPONSES
# ==============================================================================


def test_rest_mode_successful_responses(rest_client: DecisionSupportApiClient):
    """Verify REST-mode queries using FastAPI test transport."""
    # 1. Summary
    summary = rest_client.get_regional_summary()
    assert isinstance(summary, RegionalDecisionSummaryResponse)
    assert summary.total_units == 1879
    assert summary.band_counts["BAND_1_CRITICAL"] == 155

    # 2. Individual unit
    unit = rest_client.get_unit("608720")
    assert isinstance(unit, DecisionSupportEvaluation)
    assert unit.village_name == "Edmoge"
    assert unit.provisional_priority_band == "BAND_3_MODERATE"

    # 3. Filtered collection
    units_resp = rest_client.get_units(district="Uttara Kannada", limit=15)
    assert isinstance(units_resp, DecisionEvaluationListResponse)
    assert units_resp.total_matching == 1257
    assert len(units_resp.units) == 15

    # 4. Priority distribution
    dist = rest_client.get_priority_distribution()
    assert dist["BAND_1_CRITICAL"] == 155
    assert dist["BAND_2_SPECIALIZED"] == 829


# ==============================================================================
# TEST 5: REST-MODE CONNECTION FAILURES (NO SILENT FALLBACK)
# ==============================================================================


def test_rest_mode_connection_failures():
    """Verify that network failures raise ApiClientConnectionError without silent fallback."""
    # Mock client that raises ConnectError
    mock_http = MagicMock()
    mock_http.get.side_effect = httpx.ConnectError("Connection refused to port 8000")

    client = DecisionSupportApiClient(mode="rest", http_client=mock_http)
    assert client.mode == AdapterMode.REST

    # 1. get_regional_summary must raise ApiClientConnectionError
    with pytest.raises(ApiClientConnectionError) as exc_info:
        client.get_regional_summary()
    assert "Failed to connect" in str(exc_info.value) or "Connection refused" in str(exc_info.value)

    # 2. get_unit must raise ApiClientConnectionError
    with pytest.raises(ApiClientConnectionError):
        client.get_unit("608720")

    # 3. get_units must raise ApiClientConnectionError
    with pytest.raises(ApiClientConnectionError):
        client.get_units(limit=10)

    # 4. Verify mock was called and direct service was NOT invoked as a silent fallback
    assert mock_http.get.called
    assert client._service is None


# ==============================================================================
# TEST 6: REST-MODE HTTP ERROR HANDLING
# ==============================================================================


def test_rest_mode_http_errors():
    """Verify REST-mode handling of HTTP 500, 422, and malformed JSON."""
    mock_http = MagicMock()
    client = DecisionSupportApiClient(mode="rest", http_client=mock_http)

    # 1. HTTP 500 Internal Server Error
    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    mock_resp_500.text = "Internal Server Error in Decision Engine"
    mock_http.get.return_value = mock_resp_500

    with pytest.raises(ApiClientResponseError) as exc_info:
        client.get_regional_summary()
    assert "500" in str(exc_info.value)

    # 2. HTTP 422 Unprocessable Entity
    mock_resp_422 = MagicMock()
    mock_resp_422.status_code = 422
    mock_resp_422.text = "Invalid district 'InvalidDistrict'"
    mock_http.get.return_value = mock_resp_422

    with pytest.raises(ApiClientValidationError):
        client.get_unit("608720")

    # 3. Malformed JSON (status 200 but invalid schema)
    mock_resp_bad_json = MagicMock()
    mock_resp_bad_json.status_code = 200
    mock_resp_bad_json.json.return_value = {"broken_field": 123}
    mock_http.get.return_value = mock_resp_bad_json

    with pytest.raises(ApiClientResponseError):
        client.get_regional_summary()


# ==============================================================================
# TEST 7: INVALID UNIT IDENTIFIERS & PARAMETERS
# ==============================================================================


def test_invalid_unit_identifiers(direct_client: DecisionSupportApiClient, rest_client: DecisionSupportApiClient):
    """Verify error behavior on invalid, empty, or non-existent identifiers."""
    # 1. Non-existent unit in direct mode
    with pytest.raises(ApiClientNotFoundError) as exc_direct:
        direct_client.get_unit("NonExistentVillage1234567")
    assert isinstance(exc_direct.value, KeyError)

    # 2. Non-existent unit in REST mode
    with pytest.raises(ApiClientNotFoundError) as exc_rest:
        rest_client.get_unit("NonExistentVillage1234567")
    assert isinstance(exc_rest.value, KeyError)

    # 3. Empty identifier string
    with pytest.raises(ApiClientValidationError) as exc_empty:
        direct_client.get_unit("")
    assert isinstance(exc_empty.value, ValueError)

    with pytest.raises(ApiClientValidationError):
        direct_client.get_unit("   ")

    # 4. Invalid district filter validation
    with pytest.raises(ApiClientValidationError):
        direct_client.get_units(district="Atlantis")

    with pytest.raises(ApiClientValidationError):
        rest_client.get_units(district="Atlantis")

    # 5. Invalid priority band validation
    with pytest.raises(ApiClientValidationError):
        direct_client.get_units(priority_band="BAND_99_UNKNOWN")

    # 6. Invalid pagination limit
    with pytest.raises(ApiClientValidationError):
        direct_client.get_units(limit=0)

    with pytest.raises(ApiClientValidationError):
        direct_client.get_units(limit=2000)

    with pytest.raises(ApiClientValidationError):
        direct_client.get_units(offset=-1)


# ==============================================================================
# TEST 8: MISSING AND NULL FIELD PRESERVATION
# ==============================================================================


def test_missing_and_null_fields_preservation(direct_client: DecisionSupportApiClient):
    """Verify null values in Uttara Kannada records are preserved and not replaced by defaults."""
    unit_uk = direct_client.get_unit("Bappalgundi")
    assert unit_uk.district_name == "Uttara Kannada"

    # Uttara Kannada units have NO K-Means cluster assignment
    assert unit_uk.vulnerability_archetype_id is None
    assert unit_uk.vulnerability_archetype_label is None
    assert unit_uk.cluster_assignment_status == "UNASSIGNED_OUT_OF_STUDY_AREA"

    # Verify probability distribution structure
    assert isinstance(unit_uk.class_probabilities, dict)
    assert set(unit_uk.class_probabilities.keys()) == {"prob_low", "prob_medium", "prob_high"}
    assert all(isinstance(v, float) for v in unit_uk.class_probabilities.values())

    # Verify provenance and caveats are intact
    assert unit_uk.prediction_provenance in (
        "TRAIN_IN_SAMPLE",
        "VAL_OUT_OF_SAMPLE",
        "TEST_OUT_OF_SAMPLE",
    )
    assert len(unit_uk.advisory_disclaimer) > 0


# ==============================================================================
# TEST 9: CONSISTENCY BETWEEN DIRECT AND REST MODES
# ==============================================================================


def test_consistency_between_direct_and_rest_modes(
    direct_client: DecisionSupportApiClient,
    rest_client: DecisionSupportApiClient,
):
    """Verify that direct and REST modes return 100% identical responses."""
    # 1. Compare regional summaries
    summary_direct = direct_client.get_regional_summary(force_refresh=True)
    summary_rest = rest_client.get_regional_summary(force_refresh=True)

    assert summary_direct.total_units == summary_rest.total_units
    assert summary_direct.band_counts == summary_rest.band_counts
    assert summary_direct.district_band_breakdown == summary_rest.district_band_breakdown
    assert summary_direct.model_agreement_counts == summary_rest.model_agreement_counts
    assert summary_direct.uttara_kannada_trigger_breakdown == summary_rest.uttara_kannada_trigger_breakdown

    # 2. Compare individual unit (Edmoge)
    unit_direct = direct_client.get_unit("608720", force_refresh=True)
    unit_rest = rest_client.get_unit("608720", force_refresh=True)
    assert unit_direct.model_dump() == unit_rest.model_dump()

    # 3. Compare individual unit (Bappalgundi in UK)
    uk_direct = direct_client.get_unit("Bappalgundi", force_refresh=True)
    uk_rest = rest_client.get_unit("Bappalgundi", force_refresh=True)
    assert uk_direct.model_dump() == uk_rest.model_dump()

    # 4. Compare priority distributions
    assert direct_client.get_priority_distribution() == rest_client.get_priority_distribution()


# ==============================================================================
# TEST 10: CORRECT PRIORITY-BAND AGGREGATION ACROSS ALL 1,879 UNITS
# ==============================================================================


def test_priority_band_aggregation_integrity(direct_client: DecisionSupportApiClient):
    """Verify exact priority-band totals and district cross-tabulation across 1,879 records."""
    dist = direct_client.get_priority_distribution()

    # Verify authoritative totals
    assert dist["BAND_1_CRITICAL"] == 155
    assert dist["BAND_2_SPECIALIZED"] == 829
    assert dist["BAND_3_MODERATE"] == 299
    assert dist["BAND_4_ADEQUATE"] == 596
    assert sum(dist.values()) == 1879

    # Verify district allocations
    summary = direct_client.get_regional_summary()
    breakdown = summary.district_band_breakdown

    # Dakshina Kannada (369 units)
    assert breakdown["Dakshina Kannada"]["BAND_1_CRITICAL"] == 2
    assert breakdown["Dakshina Kannada"]["BAND_2_SPECIALIZED"] == 123
    assert breakdown["Dakshina Kannada"]["BAND_3_MODERATE"] == 45
    assert breakdown["Dakshina Kannada"]["BAND_4_ADEQUATE"] == 199
    assert sum(breakdown["Dakshina Kannada"].values()) == 369

    # Udupi (253 units)
    assert breakdown["Udupi"]["BAND_1_CRITICAL"] == 0
    assert breakdown["Udupi"]["BAND_2_SPECIALIZED"] == 20
    assert breakdown["Udupi"]["BAND_3_MODERATE"] == 43
    assert breakdown["Udupi"]["BAND_4_ADEQUATE"] == 190
    assert sum(breakdown["Udupi"].values()) == 253

    # Uttara Kannada (1,257 units)
    assert breakdown["Uttara Kannada"]["BAND_1_CRITICAL"] == 153
    assert breakdown["Uttara Kannada"]["BAND_2_SPECIALIZED"] == 686
    assert breakdown["Uttara Kannada"]["BAND_3_MODERATE"] == 211
    assert breakdown["Uttara Kannada"]["BAND_4_ADEQUATE"] == 207
    assert sum(breakdown["Uttara Kannada"].values()) == 1257

    # Overall reconciliation
    total_dk = sum(breakdown["Dakshina Kannada"].values())
    total_udupi = sum(breakdown["Udupi"].values())
    total_uk = sum(breakdown["Uttara Kannada"].values())
    assert total_dk + total_udupi + total_uk == 1879


# ==============================================================================
# TEST 11: CLIENT CACHING & UNIT INDEX
# ==============================================================================


def test_client_caching_and_unit_index(direct_client: DecisionSupportApiClient):
    """Verify local client caching and unit index extraction."""
    # 1. Summary caching
    summary_1 = direct_client.get_regional_summary()
    summary_2 = direct_client.get_regional_summary()
    assert summary_1 is summary_2  # Same object reference from cache

    # Refresh
    summary_3 = direct_client.get_regional_summary(force_refresh=True)
    assert summary_3 is not None
    assert summary_3.total_units == 1879

    # 2. Unit caching
    unit_1 = direct_client.get_unit("608720")
    unit_2 = direct_client.get_unit("608720")
    assert unit_1 is unit_2  # Cached object reference

    # 3. Unit index dataframe
    index_df = direct_client.get_unit_index()
    assert len(index_df) == 1879
    expected_cols = {
        "shrid2",
        "pc11_id",
        "village_name",
        "district_name",
        "provisional_priority_band",
        "overall_priority_rank",
        "band_rank",
        "preparedness_score",
        "preparedness_tier",
        "lat",
        "lon",
    }
    assert expected_cols.issubset(set(index_df.columns))


# ==============================================================================
# TEST 12: INITIALIZATION & CONTEXT MANAGER
# ==============================================================================


def test_adapter_initialization_and_lifecycle(monkeypatch):
    """Verify adapter mode parsing, env var overrides, and context management."""
    # 1. Invalid mode raises ApiClientValidationError
    with pytest.raises(ApiClientValidationError):
        DecisionSupportApiClient(mode="unsupported_mode")

    # 2. Env var mode override
    monkeypatch.setenv("CYCLONE_ADAPTER_MODE", "rest")
    monkeypatch.setenv("CYCLONE_API_BASE_URL", "http://api.internal:8000")
    client_env = DecisionSupportApiClient()
    assert client_env.mode == AdapterMode.REST
    assert client_env.base_url == "http://api.internal:8000"

    # 3. Context manager
    with DecisionSupportApiClient(mode="direct") as client:
        assert client.mode == AdapterMode.DIRECT
        assert client.get_regional_summary().total_units == 1879

    # 4. Factory helper get_api_client
    c1 = get_api_client(mode="direct", force_new=True)
    c2 = get_api_client()
    assert c1 is c2
