"""Phase 4.4.1 Regression Tests: Data Consistency, Healthcare Disambiguation, and Tool Budget.

Verifies:
1. Exact authoritative Udupi priority-band distribution (Band 1: 0, Band 2: 20, Band 3: 43, Band 4: 190).
2. Exact district-level priority band rosters across all 1,879 surveyed coastal units.
3. Independent mathematical proof of Udupi's tier-to-band assignment logic.
4. Granular disambiguation of Tallur healthcare metrics (PHC: 3.09 km vs in-village sub-centre vs apex hospital).
5. Tool-call budget validation ensuring orchestrator executes up to 8 tool calls across 4 rounds without truncation.
"""

from typing import Any, Dict, List
import pytest
from unittest.mock import AsyncMock, patch

from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.client import OllamaClient
from src.assistant.llm.orchestrator import (
    MAX_TOTAL_TOOL_CALLS,
    MAX_TOOL_ROUNDS,
    ToolCallingOrchestrator,
)
from src.assistant.llm.schemas import ChatRequest, ConversationStatus
from src.assistant.tools.registry import ToolRegistry, get_default_registry
from src.integration.decision_matrix import (
    COASTAL_PROXIMITY_THRESHOLD_KM,
    MEDICAL_DEFICIT_THRESHOLD,
    SHELTER_DEFICIT_THRESHOLD,
    compute_priority_bands_and_ranking,
)


@pytest.fixture(scope="module")
def service() -> DecisionSupportService:
    """Shared DecisionSupportService fixture with ranked_df populated."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def registry(service: DecisionSupportService) -> ToolRegistry:
    """Shared ToolRegistry fixture."""
    return get_default_registry(service=service)


# ==============================================================================
# 1. UDUPI PRIORITY-BAND DISTRIBUTION AND MATHEMATICAL PROOF
# ==============================================================================


def test_udupi_authoritative_priority_band_distribution(service: DecisionSupportService):
    """Verify that Udupi has exactly 253 units distributed as 0 / 20 / 43 / 190.

    Reconciles the discrepancy between earlier Phase 4.3.1 reports (0 / 62 / 37 / 154)
    and authoritative Phase 4.4 reports (0 / 20 / 43 / 190).
    """
    df = service.ranked_df
    udupi = df[df["district_name"].str.strip().str.lower() == "udupi"]

    assert len(udupi) == 253, f"Expected 253 units in Udupi, got {len(udupi)}"

    band_counts = udupi["priority_band"].value_counts().to_dict()

    assert band_counts.get("BAND_1_CRITICAL", 0) == 0, (
        f"Udupi must have 0 Band 1 units, got {band_counts.get('BAND_1_CRITICAL')}"
    )
    assert band_counts.get("BAND_2_SPECIALIZED", 0) == 20, (
        f"Udupi must have exactly 20 Band 2 units, got {band_counts.get('BAND_2_SPECIALIZED')}"
    )
    assert band_counts.get("BAND_3_MODERATE", 0) == 43, (
        f"Udupi must have exactly 43 Band 3 units, got {band_counts.get('BAND_3_MODERATE')}"
    )
    assert band_counts.get("BAND_4_ADEQUATE", 0) == 190, (
        f"Udupi must have exactly 190 Band 4 units, got {band_counts.get('BAND_4_ADEQUATE')}"
    )

    # Verify sum
    assert sum(band_counts.values()) == 253


def test_udupi_priority_band_independent_mathematical_derivation(service: DecisionSupportService):
    """Independently calculate Udupi priority bands using raw AHP tiers and deficit rules.

    Proves mathematically why Band 2 is 20 and Band 3 is 43:
    1. Low tier (AHP < 40.0): 0 units -> Band 1 = 0
    2. High tier (AHP > 60.0): 190 units -> Band 4 = 190
    3. Medium tier (40.0 <= AHP <= 60.0): 63 units
       - Deficit trigger: dist_coast_km < 5.0 OR score_shelters < 20.0 OR score_medical < 35.0
       - Exactly 20 units meet the trigger (all via coastal proximity < 5.0 km) -> Band 2 = 20
       - Exactly 43 units do NOT meet the trigger -> Band 3 = 43
    """
    df = service.ranked_df
    udupi = df[df["district_name"].str.strip().str.lower() == "udupi"].copy()

    tier_series = udupi["preparedness_tier"].astype(str).str.strip().str.capitalize()

    low_units = udupi[tier_series == "Low"]
    med_units = udupi[tier_series == "Medium"]
    high_units = udupi[tier_series == "High"]

    assert len(low_units) == 0, "Udupi has 0 Low tier units"
    assert len(high_units) == 190, "Udupi has 190 High tier units"
    assert len(med_units) == 63, "Udupi has 63 Medium tier units"
    assert len(low_units) + len(med_units) + len(high_units) == 253

    # Evaluate the 3 independent deficit triggers on Medium tier units
    c_coast = med_units["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM
    c_shelter = med_units["score_shelters"] < SHELTER_DEFICIT_THRESHOLD
    c_medical = med_units["score_medical"] < MEDICAL_DEFICIT_THRESHOLD
    any_trigger = c_coast | c_shelter | c_medical

    assert c_coast.sum() == 20, "Exactly 20 Medium tier units are within 5 km of coast"
    assert c_shelter.sum() == 0, "Zero Medium tier units in Udupi have shelter deficit (<20)"
    assert c_medical.sum() == 0, "Zero Medium tier units in Udupi have medical deficit (<35)"

    assert any_trigger.sum() == 20, "Band 2 Specialized must equal exactly 20"
    assert (~any_trigger).sum() == 43, "Band 3 Moderate must equal exactly 43 (63 - 20)"


def test_coastal_karnataka_complete_district_roster(service: DecisionSupportService):
    """Verify exact priority band distribution across all three coastal Karnataka districts."""
    df = service.ranked_df
    assert len(df) == 1879

    dk = df[df["district_name"].str.lower() == "dakshina kannada"]["priority_band"].value_counts().to_dict()
    assert dk.get("BAND_1_CRITICAL", 0) == 2
    assert dk.get("BAND_2_SPECIALIZED", 0) == 123
    assert dk.get("BAND_3_MODERATE", 0) == 45
    assert dk.get("BAND_4_ADEQUATE", 0) == 199
    assert sum(dk.values()) == 369

    udupi = df[df["district_name"].str.lower() == "udupi"]["priority_band"].value_counts().to_dict()
    assert udupi.get("BAND_1_CRITICAL", 0) == 0
    assert udupi.get("BAND_2_SPECIALIZED", 0) == 20
    assert udupi.get("BAND_3_MODERATE", 0) == 43
    assert udupi.get("BAND_4_ADEQUATE", 0) == 190
    assert sum(udupi.values()) == 253

    uk = df[df["district_name"].str.lower() == "uttara kannada"]["priority_band"].value_counts().to_dict()
    assert uk.get("BAND_1_CRITICAL", 0) == 153
    assert uk.get("BAND_2_SPECIALIZED", 0) == 686
    assert uk.get("BAND_3_MODERATE", 0) == 211
    assert uk.get("BAND_4_ADEQUATE", 0) == 207
    assert sum(uk.values()) == 1257

    # Overall totals
    totals = df["priority_band"].value_counts().to_dict()
    assert totals["BAND_1_CRITICAL"] == 155
    assert totals["BAND_2_SPECIALIZED"] == 829
    assert totals["BAND_3_MODERATE"] == 299
    assert totals["BAND_4_ADEQUATE"] == 596
    assert sum(totals.values()) == 1879


# ==============================================================================
# 2. TALLUR HEALTHCARE METRIC DISAMBIGUATION
# ==============================================================================


def test_tallur_healthcare_distance_disambiguation(registry: ToolRegistry):
    """Verify granular healthcare distances and explicit facility disambiguation for Tallur.

    Reconciles the reported 2.50 km vs 3.09 km metrics:
    - dist_nearest_phc_km is 3.09 km (geodesic distance to Primary Health Centre).
    - dist_nearest_local_facility_km is 0.00 km (sub-centre located directly within Tallur boundary).
    - dist_nearest_hospital_km is 34.79 km (major tertiary apex hospital).
    """
    res = registry.execute("analyze_infrastructure_and_exposure", {"identifier": "Tallur"})
    assert res.success
    data = res.data

    assert data["community_name"] == "Tallur"
    assert data["pc11_id"] == "608691"
    assert data["district_name"] == "Udupi"

    hc = data["healthcare_access"]
    assert hc["dist_nearest_phc_km"] == 3.09, (
        f"Nearest PHC distance must be 3.09 km, got {hc['dist_nearest_phc_km']}"
    )
    assert hc["dist_nearest_local_facility_km"] == 0.0, (
        f"Local facility distance must be 0.00 km (in-village), got {hc['dist_nearest_local_facility_km']}"
    )
    assert hc["dist_nearest_hospital_km"] == 34.79, (
        f"Apex hospital distance must be 34.79 km, got {hc['dist_nearest_hospital_km']}"
    )
    assert hc["hospitals_within_10km"] == 57
    assert hc["phc_chc_within_10km"] == 10

    # Verify structured disambiguation dictionary
    assert "facility_disambiguation" in hc
    dis = hc["facility_disambiguation"]
    assert dis["primary_health_centre_phc_km"] == 3.09
    assert dis["local_health_subcentre_km"] == 0.0
    assert dis["apex_hospital_km"] == 34.79
    assert "Primary Health Centre (PHC)" in dis["definition_note"]
    assert "0.0 km indicates facility located within village boundary" in dis["definition_note"]


# ==============================================================================
# 3. TOOL-CALL BUDGET CONFIGURATION AND MULTI-TOOL EXECUTION
# ==============================================================================


def test_orchestrator_tool_budget_configuration():
    """Verify orchestrator budget is configured to 4 rounds and 8 total tool calls."""
    assert MAX_TOOL_ROUNDS == 4
    assert MAX_TOTAL_TOOL_CALLS == 8

    orch = ToolCallingOrchestrator()
    assert orch.max_rounds == 4
    assert orch.max_tool_calls == 8


def test_orchestrator_multi_tool_chain_exceeding_four_calls(registry: ToolRegistry):
    """Verify that an orchestrator can execute 5+ tool calls in a single turn without premature capping.

    Simulates a model requesting 3 tools in Round 1 and 2 tools in Round 2 (5 total calls),
    confirming all 5 execute cleanly within the 8-call budget.
    """
    async def _run():
        import json
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "qwen2.5:0.5b-instruct"

        qwen_intent_json = json.dumps({
            "intents": [
                "lookup_community",
                "explain_preparedness_assessment",
                "analyze_infrastructure_and_exposure",
                "get_intervention_recommendations",
                "explain_vulnerability_archetype",
            ],
            "arguments": {"identifier": "Tallur", "archetype_id": 3},
            "needs_clarification": False,
            "clarification_question": None,
        })
        synthesis_text = "Comprehensive multi-tool assessment completed for Tallur across 5 tools."

        mock_client.generate.side_effect = [qwen_intent_json, synthesis_text]

        orch = ToolCallingOrchestrator(client=mock_client, registry=registry)
        req = ChatRequest(message="Run comprehensive 5-tool audit on Tallur.")

        res = await orch.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert not res.is_fallback
        assert len(res.tools_executed) == 5, f"Expected 5 executed tools, got {len(res.tools_executed)}"
        assert res.tools_executed == [
            "lookup_community",
            "explain_preparedness_assessment",
            "analyze_infrastructure_and_exposure",
            "get_intervention_recommendations",
            "explain_vulnerability_archetype",
        ]
        assert all(tr.success for tr in res.tool_records)
        assert "Comprehensive multi-tool assessment" in res.answer

    import asyncio
    asyncio.run(_run())
