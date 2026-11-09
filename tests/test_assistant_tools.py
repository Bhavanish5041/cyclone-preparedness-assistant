"""Tests for the Assistant Deterministic Tool Suite and Tool Registry (Phase 4.1).

Verifies:
- All seven tools execute deterministically with verified scientific correctness.
- Argument validation, schema checking, and limits enforcement.
- Community resolution with exact keys and multi-unit ambiguity handling.
- K-Means archetype assignments (Dakshina Kannada & Udupi) and unassigned fallbacks (Uttara Kannada).
- Bounded payload sizes and 100% JSON-serializable outputs.
- Prevention of secret leakage and uncontrolled dataset dumps.
- Multi-provider schema export parity (OpenAI, Gemini, Anthropic).
"""

import json
import pytest

from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools import (
    AmbiguousCommunityError,
    AnalyzeInfrastructureAndExposureTool,
    BaseTool,
    CompareDistrictsTool,
    ExplainPreparednessAssessmentTool,
    ExplainVulnerabilityArchetypeTool,
    FilterAndRankCommunitiesTool,
    GetInterventionRecommendationsTool,
    LookupCommunityTool,
    ToolRegistry,
    ToolResult,
    get_default_registry,
    resolve_community_row,
)


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    """Module-scoped DecisionSupportService instance to share loaded datasets across tests."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def registry(shared_service: DecisionSupportService) -> ToolRegistry:
    """Module-scoped configured ToolRegistry."""
    return get_default_registry(service=shared_service)


# ==============================================================================
# 1. TOOL REGISTRY DISCOVERY & SCHEMA TESTS
# ==============================================================================


def test_registry_discovery_and_registration(registry: ToolRegistry):
    """Verify registry discovers and indexes all seven tools."""
    expected_tools = [
        "analyze_infrastructure_and_exposure",
        "compare_districts",
        "explain_preparedness_assessment",
        "explain_vulnerability_archetype",
        "filter_and_rank_communities",
        "get_intervention_recommendations",
        "lookup_community",
    ]
    registered = registry.list_tools()
    assert registered == expected_tools
    assert len(registered) == 7

    for tool_name in expected_tools:
        assert registry.has_tool(tool_name)
        tool = registry.get_tool(tool_name)
        assert tool is not None
        assert isinstance(tool, BaseTool)
        assert tool.name == tool_name
        assert len(tool.description) > 20


def test_registry_duplicate_registration_rejected(registry: ToolRegistry):
    """Verify registering a duplicate tool name raises ValueError."""
    dummy_tool = LookupCommunityTool()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(dummy_tool)


def test_registry_schema_exports(registry: ToolRegistry):
    """Verify multi-provider tool schema generation."""
    # 1. Standard Schemas
    schemas = registry.get_all_schemas()
    assert len(schemas) == 7
    for s in schemas:
        assert "name" in s
        assert "description" in s
        assert "parameters" in s
        assert s["parameters"].get("type") == "object"

    # 2. OpenAI Tool Definitions
    openai_tools = registry.export_openai_tool_definitions()
    assert len(openai_tools) == 7
    for ot in openai_tools:
        assert ot["type"] == "function"
        fn = ot["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn

    # 3. Gemini Declarations
    gemini_tools = registry.export_gemini_tool_declarations()
    assert len(gemini_tools) == 7
    for gt in gemini_tools:
        assert "name" in gt
        assert "description" in gt
        assert "parameters" in gt

    # 4. Anthropic Tool Definitions
    anthropic_tools = registry.export_anthropic_tool_definitions()
    assert len(anthropic_tools) == 7
    for at in anthropic_tools:
        assert "name" in at
        assert "description" in at
        assert "input_schema" in at


def test_registry_unknown_tool_execution(registry: ToolRegistry):
    """Verify executing an unknown tool returns structured UNKNOWN_TOOL error envelope."""
    result = registry.execute("non_existent_tool", {"foo": "bar"})
    assert isinstance(result, ToolResult)
    assert not result.success
    assert result.error_type == "UNKNOWN_TOOL"
    assert "Unknown tool 'non_existent_tool'" in result.error
    assert "available_tools" in result.metadata


# ==============================================================================
# 2. TOOL 1: lookup_community TESTS
# ==============================================================================


def test_lookup_community_by_name(registry: ToolRegistry):
    """Test lookup_community resolves unique name 'Tallur'."""
    res = registry.execute("lookup_community", {"query": "Tallur"})
    assert res.success
    data = res.data
    assert data["query"] == "Tallur"
    assert data["total_matches"] >= 1
    assert data["returned_count"] >= 1

    first = data["matches"][0]
    assert first["village_name"] == "Tallur"
    assert first["pc11_id"] == "608691"
    assert first["district_name"] == "Udupi"
    assert first["priority_band"] == "BAND_4_ADEQUATE"
    assert first["preparedness_score"] > 60.0


def test_lookup_community_by_pc11_id(registry: ToolRegistry):
    """Test lookup_community resolves by Census PC11 ID."""
    res = registry.execute("lookup_community", {"query": "608691"})
    assert res.success
    assert res.data["total_matches"] == 1
    assert res.data["matches"][0]["village_name"] == "Tallur"


def test_lookup_community_with_district_filter(registry: ToolRegistry):
    """Test lookup_community restricts results by district."""
    # Search for common substring 'halli' in Dakshina Kannada only
    res = registry.execute("lookup_community", {"query": "alli", "district": "Dakshina Kannada", "limit": 5})
    assert res.success
    assert res.data["returned_count"] <= 5
    for item in res.data["matches"]:
        assert item["district_name"] == "Dakshina Kannada"


def test_lookup_community_validation(registry: ToolRegistry):
    """Test lookup_community validation errors on invalid inputs."""
    # Empty query
    res = registry.execute("lookup_community", {"query": "   "})
    assert not res.success
    assert res.error_type == "VALIDATION_ERROR"

    # Invalid district
    res = registry.execute("lookup_community", {"query": "Tallur", "district": "Mysore"})
    assert not res.success
    assert res.error_type == "VALIDATION_ERROR"

    # Out of range limit (> 15)
    res = registry.execute("lookup_community", {"query": "Tallur", "limit": 25})
    assert not res.success
    assert res.error_type == "VALIDATION_ERROR"


# ==============================================================================
# 3. TOOL 2: explain_preparedness_assessment TESTS
# ==============================================================================


def test_explain_preparedness_tallur(registry: ToolRegistry):
    """Verify Tallur preparedness assessment matches authoritative baseline."""
    res = registry.execute("explain_preparedness_assessment", {"identifier": "Tallur"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Tallur"
    assert data["pc11_id"] == "608691"
    assert data["district_name"] == "Udupi"

    # AHP baseline
    ahp = data["deterministic_ahp"]
    assert ahp["preparedness_tier"] == "High"
    assert ahp["preparedness_score"] > 60.0
    assert "score_shelters" in ahp["domain_scores"]
    assert "score_medical" in ahp["domain_scores"]

    # Operational priority: High tier maps to Band 4 Adequate
    prioritization = data["operational_prioritization"]
    assert prioritization["priority_band"] == "BAND_4_ADEQUATE"
    assert prioritization["overall_priority_rank"] > 0
    assert "band_4_adequate" in prioritization["priority_explanation"].lower()

    # Precedence mandate and disclaimer
    assert "AHP PRECEDENCE MANDATE" in data["scientific_precedence_mandate"]
    assert "ADVISORY PLANNING NOTICE" in data["disclaimer"]


def test_explain_preparedness_edmoge(registry: ToolRegistry):
    """Verify Edmoge (remote foothill assigned unit) evaluation."""
    res = registry.execute("explain_preparedness_assessment", {"identifier": "Edmoge"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Edmoge"
    assert data["vulnerability_archetype"]["cluster_assignment_status"] == "ASSIGNED"
    assert data["vulnerability_archetype"]["archetype_id"] == 1
    assert "Foothill" in data["vulnerability_archetype"]["archetype_label"]


def test_explain_preparedness_bappalgundi_uttara_kannada(registry: ToolRegistry):
    """Verify Bappalgundi (Uttara Kannada unassigned unit) evaluation."""
    res = registry.execute("explain_preparedness_assessment", {"identifier": "Bappalgundi"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Bappalgundi"
    assert data["district_name"] == "Uttara Kannada"
    assert data["vulnerability_archetype"]["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"
    assert data["vulnerability_archetype"]["archetype_id"] is None
    assert "Unassigned" in data["operational_prioritization"]["priority_explanation"]


def test_explain_preparedness_ambiguous_community(registry: ToolRegistry):
    """Verify querying duplicate name 'Balkur' triggers AMBIGUOUS_COMMUNITY error."""
    res = registry.execute("explain_preparedness_assessment", {"identifier": "Balkur"})
    assert not res.success
    assert res.error_type == "AMBIGUOUS_COMMUNITY"
    assert "matches 2 administrative units" in res.error
    assert len(res.data["candidates"]) == 2

    # Disambiguate using exact PC11 ID for Balkur (Kundapura, Udupi: 608740)
    res_pc11 = registry.execute("explain_preparedness_assessment", {"identifier": "608740"})
    assert res_pc11.success
    assert res_pc11.data["community_name"] == "Balkur"
    assert res_pc11.data["district_name"] == "Udupi"


def test_explain_preparedness_not_found(registry: ToolRegistry):
    """Verify non-existent community returns COMMUNITY_NOT_FOUND error."""
    res = registry.execute("explain_preparedness_assessment", {"identifier": "NonExistentVillageXYZ"})
    assert not res.success
    assert res.error_type == "COMMUNITY_NOT_FOUND"
    assert "not found in coastal Karnataka study area" in res.error


# ==============================================================================
# 4. TOOL 3: compare_districts TESTS
# ==============================================================================


def test_compare_districts_dakshina_kannada_vs_udupi(registry: ToolRegistry):
    """Verify comparative metrics between Dakshina Kannada and Udupi."""
    res = registry.execute(
        "compare_districts",
        {"district_a": "Dakshina Kannada", "district_b": "Udupi"},
    )
    assert res.success
    data = res.data

    assert data["districts_compared"] == ["Dakshina Kannada", "Udupi"]
    comp = data["comparison"]

    assert comp["Dakshina Kannada"]["total_units"] == 369
    assert comp["Udupi"]["total_units"] == 253

    assert comp["Dakshina Kannada"]["clustering_coverage"].startswith("Assigned")
    assert comp["Udupi"]["clustering_coverage"].startswith("Assigned")

    assert len(data["key_observations"]) >= 1
    assert "methodological_note" in data


def test_compare_districts_with_uttara_kannada(registry: ToolRegistry):
    """Verify comparative metrics with Uttara Kannada note unassigned status."""
    res = registry.execute(
        "compare_districts",
        {"district_a": "Uttara Kannada", "district_b": "Dakshina Kannada"},
    )
    assert res.success
    data = res.data
    comp = data["comparison"]

    assert comp["Uttara Kannada"]["total_units"] == 1257
    assert "Unassigned" in comp["Uttara Kannada"]["clustering_coverage"]


def test_compare_districts_validation(registry: ToolRegistry):
    """Verify validation errors for compare_districts."""
    # Identical districts
    res = registry.execute(
        "compare_districts",
        {"district_a": "Udupi", "district_b": "Udupi"},
    )
    assert not res.success
    assert res.error_type == "VALIDATION_ERROR"
    assert "must be different districts" in res.error

    # Invalid district name
    res = registry.execute(
        "compare_districts",
        {"district_a": "Dakshina Kannada", "district_b": "Shimoga"},
    )
    assert not res.success
    assert res.error_type == "VALIDATION_ERROR"


# ==============================================================================
# 5. TOOL 4: filter_and_rank_communities TESTS
# ==============================================================================


def test_filter_and_rank_band_1_critical(registry: ToolRegistry):
    """Verify filtering for Band 1 Critical units across all districts."""
    res = registry.execute(
        "filter_and_rank_communities",
        {"priority_band": "BAND_1_CRITICAL", "limit": 10},
    )
    assert res.success
    data = res.data

    assert data["total_matching"] == 155  # Authoritative Band 1 total
    assert data["returned_count"] == 10
    assert len(data["units"]) == 10

    for u in data["units"]:
        assert u["priority_band"] == "BAND_1_CRITICAL"
        assert u["overall_priority_rank"] <= 155


def test_filter_and_rank_with_district_and_sorting(registry: ToolRegistry):
    """Verify filtering by district and sorting by coastal proximity."""
    res = registry.execute(
        "filter_and_rank_communities",
        {
            "district": "Udupi",
            "priority_band": "BAND_2_SPECIALIZED",
            "sort_by": "dist_coast_km",
            "ascending": True,
            "limit": 5,
        },
    )
    assert res.success
    data = res.data

    assert data["returned_count"] == 5
    for u in data["units"]:
        assert u["district_name"] == "Udupi"
        assert u["priority_band"] == "BAND_2_SPECIALIZED"


def test_filter_and_rank_strict_limit_enforcement(registry: ToolRegistry):
    """Verify filter_and_rank_communities rejects limit > 20."""
    res = registry.execute("filter_and_rank_communities", {"limit": 25})
    assert not res.success
    assert res.error_type == "VALIDATION_ERROR"
    assert "limit" in res.error.lower()


# ==============================================================================
# 6. TOOL 5: analyze_infrastructure_and_exposure TESTS
# ==============================================================================


def test_analyze_infrastructure_tallur(registry: ToolRegistry):
    """Verify physical indicators and operational disclaimers for Tallur."""
    res = registry.execute("analyze_infrastructure_and_exposure", {"identifier": "Tallur"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Tallur"
    assert data["pc11_id"] == "608691"

    # Exposure indicators
    exp = data["physical_exposure"]
    assert exp["dist_coast_km"] is not None
    assert exp["dist_coast_km"] < 5.0  # Tallur is near coast (~1.68 km)

    # Healthcare indicators
    hc = data["healthcare_access"]
    assert hc["dist_nearest_hospital_km"] is not None
    assert hc["dist_nearest_phc_km"] is not None

    # Provenance caveats
    lim = data["operational_limitations"]
    assert lim["census_vintage"] == 2011
    assert "operational_caveat" in lim
    assert "NOT confirmed active doctor staffing" in lim["operational_caveat"]


def test_analyze_infrastructure_ambiguous_and_missing(registry: ToolRegistry):
    """Verify error handling on ambiguous and missing communities."""
    res_amb = registry.execute("analyze_infrastructure_and_exposure", {"identifier": "Balkur"})
    assert not res_amb.success
    assert res_amb.error_type == "AMBIGUOUS_COMMUNITY"

    res_missing = registry.execute("analyze_infrastructure_and_exposure", {"identifier": "UnknownTown"})
    assert not res_missing.success
    assert res_missing.error_type == "COMMUNITY_NOT_FOUND"


# ==============================================================================
# 7. TOOL 6: explain_vulnerability_archetype TESTS
# ==============================================================================


def test_explain_archetype_by_id(registry: ToolRegistry):
    """Verify inspecting archetype details by cluster ID (0..3)."""
    for cid in range(4):
        res = registry.execute("explain_vulnerability_archetype", {"archetype_id": cid})
        assert res.success
        data = res.data
        assert data["archetype_id"] == cid
        assert len(data["short_label"]) > 5
        assert len(data["detailed_description"]) > 20
        assert "dimension_means" in data
        assert "spatial_boundary_note" in data


def test_explain_archetype_invalid_id(registry: ToolRegistry):
    """Verify invalid cluster ID (> 3) is rejected."""
    res = registry.execute("explain_vulnerability_archetype", {"archetype_id": 4})
    assert not res.success
    assert res.error_type == "VALIDATION_ERROR"


def test_explain_archetype_by_assigned_community(registry: ToolRegistry):
    """Verify inspecting archetype via assigned community (Edmoge)."""
    res = registry.execute("explain_vulnerability_archetype", {"community_identifier": "Edmoge"})
    assert res.success
    data = res.data
    assert data["community_name"] == "Edmoge"
    assert data["cluster_assignment_status"] == "ASSIGNED"
    assert data["archetype_id"] == 1
    assert "community_dimension_scores" in data


def test_explain_archetype_by_unassigned_community(registry: ToolRegistry):
    """Verify inspecting archetype via unassigned community in Uttara Kannada (Bappalgundi)."""
    res = registry.execute("explain_vulnerability_archetype", {"community_identifier": "Bappalgundi"})
    assert res.success
    data = res.data
    assert data["community_name"] == "Bappalgundi"
    assert data["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"
    assert data["archetype_id"] is None
    assert "fallback_trigger_evaluation" in data
    assert "coastal_proximity_trigger (<5km)" in data["fallback_trigger_evaluation"]


def test_explain_archetype_catalog_overview(registry: ToolRegistry):
    """Verify omitting arguments returns catalog overview of all 4 archetypes."""
    res = registry.execute("explain_vulnerability_archetype", {})
    assert res.success
    data = res.data
    assert data["total_archetypes"] == 4
    assert len(data["archetypes"]) == 4


# ==============================================================================
# 8. TOOL 7: get_intervention_recommendations TESTS
# ==============================================================================


def test_get_advisories_tallur(registry: ToolRegistry):
    """Verify rule-based advisories for Tallur (coastal proximity triggers)."""
    res = registry.execute("get_intervention_recommendations", {"identifier": "Tallur"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Tallur"
    assert data["advisories_count"] >= 1
    assert len(data["advisories"]) == data["advisories_count"]

    # Verify advisory structure
    for adv in data["advisories"]:
        assert "directive_id" in adv
        assert "advisory_focus" in adv
        assert "operational_recommendation" in adv
        assert "lead_coordinating_agency" in adv
        assert "supporting_evidence" in adv


def test_get_advisories_edmoge(registry: ToolRegistry):
    """Verify rule-based advisories for Edmoge (foothill/hospital isolation)."""
    res = registry.execute("get_intervention_recommendations", {"identifier": "Edmoge"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Edmoge"
    assert data["cluster_assignment_status"] == "ASSIGNED"
    directive_ids = [a["directive_id"] for a in data["advisories"]]
    assert any("HOSPITAL" in d or "SHELTER" in d for d in directive_ids)


def test_get_advisories_bappalgundi(registry: ToolRegistry):
    """Verify UK fallback advisories for Bappalgundi."""
    res = registry.execute("get_intervention_recommendations", {"identifier": "Bappalgundi"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Bappalgundi"
    assert data["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"
    for adv in data["advisories"]:
        assert adv["directive_id"].startswith("ADV_UK_")


# ==============================================================================
# 9. JSON SERIALIZATION & BOUNDED PAYLOAD TESTS
# ==============================================================================


def test_all_tools_json_serializable_and_bounded_size(registry: ToolRegistry):
    """Verify every tool returns 100% JSON-serializable outputs bounded below 10 KB."""
    test_invocations = [
        ("lookup_community", {"query": "Tallur"}),
        ("explain_preparedness_assessment", {"identifier": "Tallur"}),
        ("explain_preparedness_assessment", {"identifier": "Edmoge"}),
        ("explain_preparedness_assessment", {"identifier": "Bappalgundi"}),
        ("compare_districts", {"district_a": "Dakshina Kannada", "district_b": "Udupi"}),
        ("filter_and_rank_communities", {"priority_band": "BAND_1_CRITICAL", "limit": 10}),
        ("analyze_infrastructure_and_exposure", {"identifier": "Tallur"}),
        ("explain_vulnerability_archetype", {"archetype_id": 0}),
        ("explain_vulnerability_archetype", {"community_identifier": "Edmoge"}),
        ("explain_vulnerability_archetype", {"community_identifier": "Bappalgundi"}),
        ("explain_vulnerability_archetype", {}),
        ("get_intervention_recommendations", {"identifier": "Tallur"}),
        ("get_intervention_recommendations", {"identifier": "Bappalgundi"}),
    ]

    for tool_name, args in test_invocations:
        res = registry.execute(tool_name, args)
        assert res.success, f"Tool '{tool_name}' failed with {args}: {res.error}"

        # 1. JSON serialization test
        json_str = res.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["success"] is True

        # 2. Bounded payload size (< 10 KB)
        payload_bytes = len(json_str.encode("utf-8"))
        assert payload_bytes < 10240, f"Tool '{tool_name}' produced oversized payload ({payload_bytes} bytes)"


def test_no_internal_secrets_or_filesystem_leakage(registry: ToolRegistry):
    """Verify tool error responses do not leak absolute workspace paths or internal keys."""
    res = registry.execute("explain_preparedness_assessment", {"identifier": "UnknownVillage"})
    json_str = res.to_json()

    prohibited_strings = [
        "/home/bhavanish",
        "WANDB_API_KEY",
        "Traceback (most recent call last)",
        "SELECT *",
    ]
    for prohibited in prohibited_strings:
        assert prohibited not in json_str, f"Found leaked internal string '{prohibited}' in response envelope."
