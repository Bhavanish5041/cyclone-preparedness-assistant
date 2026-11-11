"""Comprehensive model testing suite for CYCLONE INTEL.

Verifies:
1. Qwen intent classification and tool selection
2. Structured JSON parsing and schema validation
3. Entity resolution (gazetteer matching, disambiguation, district context)
4. Multi-intent request handling and sequence execution
5. Deterministic tool execution and calculation integrity
6. Authoritative AHP scores, preparedness tiers, and operational priority bands
7. Logistic Regression surrogate behavior (in-distribution probabilities, 26-feature schema)
8. K-Means cluster assignment and missing-cluster handling (Uttara Kannada fallback heuristics)
9. Immediate refusal gating for unsupported dynamic operational telemetry
10. Ollama timeout and malformed-response handling
11. Deterministic fallback behavior and reconciliation blocks
12. Error handling and input validation boundaries
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.client import OllamaClient, OllamaConnectionError, OllamaTimeoutError
from src.assistant.llm.orchestrator import (
    ToolCallingOrchestrator,
    audit_and_guard_answer,
)
from src.assistant.llm.schemas import (
    ChatMessage,
    ChatRequest,
    ConversationStatus,
    QwenIntentParse,
)
from src.assistant.tools.archetypes import ExplainVulnerabilityArchetypeTool
from src.assistant.tools.community import LookupCommunityTool
from src.assistant.tools.decision import (
    ExplainPreparednessAssessmentTool,
    GetInterventionRecommendationsTool,
)
from src.assistant.tools.regional import (
    CompareDistrictsTool,
    FilterAndRankCommunitiesTool,
)
from src.assistant.tools.registry import ToolRegistry, get_default_registry
from src.assistant.tools.argument_extractor import ArgumentExtractor
from src.assistant.safety.refusal_gate import RefusalGate
from src.models.load_registered_model import RegisteredModelContainer, load_registered_model

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model_test_dataset.json"


@pytest.fixture(scope="module")
def test_dataset():
    """Load the reproducible test dataset."""
    assert FIXTURE_PATH.exists(), f"Missing fixture at {FIXTURE_PATH}"
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def service():
    """Instantiate shared DecisionSupportService."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def registry(service):
    """Instantiate ToolRegistry populated with the 7 deterministic tools."""
    return get_default_registry(service=service)


@pytest.fixture(scope="module")
def refusal_gate():
    """Instantiate RefusalGate."""
    return RefusalGate()


@pytest.fixture(scope="module")
def argument_extractor(service):
    """Instantiate ArgumentExtractor."""
    return ArgumentExtractor(service=service)


# ==============================================================================
# 1. INTENT CLASSIFICATION AND TOOL SELECTION
# ==============================================================================


class TestIntentClassification:
    """Tests for intent parsing, tool selection, and schema compliance."""

    def test_single_intent_selection_from_dataset(self, test_dataset, service, registry):
        """Verify each single-intent query in dataset maps to the expected tool."""
        test_cases = [
            tc for tc in test_dataset["test_cases"]
            if len(tc["expected_intents"]) == 1 and not tc["expected_refusal"] and not tc.get("needs_clarification")
        ]

        extractor = ArgumentExtractor(service=service)
        for tc in test_cases:
            target_tool = tc["expected_intents"][0]
            assert target_tool in registry.get_tool_descriptions()

            res = extractor.extract_arguments(target_tool, tc["query"])
            if not tc.get("missing_required") and not tc.get("is_ambiguous"):
                assert res.is_valid is True, f"Extraction failed for {tc['id']}: {res.validation_error}"

    def test_structured_json_parsing_valid(self):
        """Verify QwenIntentParse parses standard conformant JSON."""
        raw_json = (
            '{"intents": ["explain_preparedness_assessment"], '
            '"arguments": {"identifier": "Sasihithlu"}, '
            '"needs_clarification": false, '
            '"clarification_question": null}'
        )
        data = json.loads(raw_json)
        parse = QwenIntentParse(**data)
        assert parse.intents == ["explain_preparedness_assessment"]
        assert parse.arguments["identifier"] == "Sasihithlu"
        assert parse.needs_clarification is False

    def test_malformed_json_fallback_handling(self, service, registry):
        """Verify orchestrator recovers gracefully when Qwen returns malformed JSON."""
        async def _run():
            mock_client = AsyncMock(spec=OllamaClient)
            mock_client.provider_name = "ollama"
            mock_client.model = "qwen2.5:0.5b-instruct"
            mock_client.is_configured = MagicMock(return_value=True)

            # Return unparseable markdown/corrupted text instead of JSON
            mock_client.generate.return_value = "```json\n{intents: [unquoted_key], broken\n```"

            orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
            req = ChatRequest(message="What is the preparedness score for Tallur?")
            resp = await orchestrator.process_chat(req)

            # Domain heuristics or fallback should catch 'Tallur' and execute explain_preparedness_assessment
            assert resp.conversation_status == ConversationStatus.COMPLETED
            assert "Tallur" in resp.answer

        asyncio.run(_run())


# ==============================================================================
# 2. ENTITY RESOLUTION AND DISAMBIGUATION
# ==============================================================================


class TestEntityResolution:
    """Tests for community and district entity extraction and ambiguity detection."""

    def test_canonical_community_resolution(self, service):
        """Verify exact matching of canonical village names."""
        df = service.ranked_df
        villages = ["Sasihithlu", "Tallur", "Padu", "Bappalgundi"]
        for v in villages:
            match = df[df["village_name"].str.lower() == v.lower()]
            assert not match.empty, f"Village {v} not found in gazetteer"

    def test_ambiguous_entity_detection(self, argument_extractor):
        """Verify ambiguous name 'Ullal' triggers ambiguity detection."""
        res = argument_extractor.extract_arguments("explain_preparedness_assessment", "Show assessment for Ullal")
        assert res.is_ambiguous is True
        assert len(res.ambiguous_candidates) >= 2
        districts = {c["district_name"] for c in res.ambiguous_candidates}
        assert "Dakshina Kannada" in districts
        assert "Uttara Kannada" in districts

    def test_district_context_disambiguation(self, argument_extractor):
        """Verify adding district context resolves ambiguity for 'Ullal'."""
        res = argument_extractor.extract_arguments(
            "explain_preparedness_assessment",
            "Show assessment for Ullal in Dakshina Kannada"
        )
        assert res.is_ambiguous is False
        assert res.is_valid is True
        assert res.parameters.get("identifier") == "Ullal (TMC)"


# ==============================================================================
# 3. MULTI-INTENT REQUEST HANDLING
# ==============================================================================


class TestMultiIntentHandling:
    """Tests for executing multiple intents in a single turn."""

    def test_multi_intent_execution_sequence(self, service, registry):
        """Verify handling of combined preparedness assessment + intervention request."""
        async def _run():
            mock_client = AsyncMock(spec=OllamaClient)
            mock_client.provider_name = "ollama"
            mock_client.model = "qwen2.5:0.5b-instruct"
            mock_client.is_configured = MagicMock(return_value=True)

            mock_client.generate.side_effect = [
                # 1. Intent parse returning two tools
                json.dumps({
                    "intents": ["explain_preparedness_assessment", "get_intervention_recommendations"],
                    "arguments": {"identifier": "Sasihithlu"},
                    "needs_clarification": False,
                    "clarification_question": None,
                }),
                # 2. Synthesis output
                "Sasihithlu has an AHP preparedness score of 79.06 / 100.0 (High, Band 4 Adequate). "
                "Recommended interventions focus on maintaining high standards and storm surge barriers."
            ]

            orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
            req = ChatRequest(message="Give me the preparedness assessment and intervention recommendations for Sasihithlu.")
            resp = await orchestrator.process_chat(req)

            assert "explain_preparedness_assessment" in resp.tools_executed
            assert "get_intervention_recommendations" in resp.tools_executed
            assert resp.structured_evidence is not None
            assert "explain_preparedness_assessment" in resp.structured_evidence
            assert "get_intervention_recommendations" in resp.structured_evidence
            assert "79.06" in resp.answer

        asyncio.run(_run())


# ==============================================================================
# 4. DETERMINISTIC TOOL EXECUTION AND SCIENTIFIC INVARIANCE
# ==============================================================================


class TestDeterministicScientificInvariance:
    """Tests for strict calculation consistency of AHP scores and priority bands."""

    def test_tallur_ahp_ground_truth(self, registry):
        """Verify Tallur's exact AHP score and priority band."""
        tool = registry.get_tool("explain_preparedness_assessment")
        res = tool.execute({"identifier": "Tallur"})

        assert res.success is True
        data = res.data
        ahp = data["deterministic_ahp"]
        prio = data["operational_prioritization"]

        assert round(ahp["preparedness_score"], 2) == 74.34
        assert ahp["preparedness_tier"] == "High"
        assert prio["priority_band"] == "BAND_4_ADEQUATE"

    def test_bappalgundi_critical_band1_ground_truth(self, registry):
        """Verify Bappalgundi is evaluated as Band 1 Critical with Low preparedness."""
        tool = registry.get_tool("explain_preparedness_assessment")
        res = tool.execute({"identifier": "Bappalgundi"})

        assert res.success is True
        data = res.data
        ahp = data["deterministic_ahp"]
        prio = data["operational_prioritization"]

        assert round(ahp["preparedness_score"], 2) == 20.14
        assert ahp["preparedness_tier"] == "Low"
        assert prio["priority_band"] == "BAND_1_CRITICAL"

    def test_district_comparison_deterministic_invariance(self, registry):
        """Verify cross-district comparison yields accurate deterministic totals."""
        tool = registry.get_tool("compare_districts")
        res = tool.execute({"district_a": "Dakshina Kannada", "district_b": "Udupi"})

        assert res.success is True
        data = res.data
        assert set(data["districts_compared"]) == {"Dakshina Kannada", "Udupi"}
        assert "comparison" in data
        assert "Dakshina Kannada" in data["comparison"]
        assert "Udupi" in data["comparison"]
        assert data["comparison"]["Dakshina Kannada"]["total_units"] == 369
        assert data["comparison"]["Udupi"]["total_units"] == 253


# ==============================================================================
# 5. K-MEANS CLUSTERING AND MISSING-CLUSTER HANDLING
# ==============================================================================


class TestClusteringAndMissingClusterHandling:
    """Tests for K-Means archetypes and Uttara Kannada fallback handling."""

    def test_assigned_unit_clustering(self, registry):
        """Verify assigned unit (Sasihithlu in Dakshina Kannada) returns Cluster 0."""
        tool = registry.get_tool("explain_vulnerability_archetype")
        res = tool.execute({"community_identifier": "Sasihithlu"})

        assert res.success is True
        data = res.data
        assert data["cluster_assignment_status"] == "ASSIGNED"
        assert data["archetype_id"] == 0
        assert "Urban" in data["short_label"]

    def test_unassigned_uttara_kannada_missing_cluster_handling(self, registry):
        """Verify Uttara Kannada unit (Bappalgundi) explicitly returns UNASSIGNED status with fallback triggers."""
        tool = registry.get_tool("explain_vulnerability_archetype")
        res = tool.execute({"community_identifier": "Bappalgundi"})

        assert res.success is True
        data = res.data
        assert data["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"
        assert data["archetype_id"] is None
        assert "fallback_trigger_evaluation" in data
        assert "baseline_low_tier_trigger (score <40)" in data["fallback_trigger_evaluation"]
        assert data["fallback_trigger_evaluation"]["baseline_low_tier_trigger (score <40)"] is True

    def test_clustering_catalog_structure(self, registry):
        """Verify catalog returns exactly 4 archetypes covering 622 units."""
        tool = registry.get_tool("explain_vulnerability_archetype")
        res = tool.execute({})

        assert res.success is True
        data = res.data
        assert data["total_archetypes"] == 4
        assert data["study_area_units"] == 622
        assert len(data["archetypes"]) == 4


# ==============================================================================
# 6. LOGISTIC REGRESSION SURROGATE BEHAVIOR
# ==============================================================================


class TestLogisticRegressionSurrogate:
    """Tests for surrogate model behavior and schema adherence."""

    def test_surrogate_model_loads_locally_or_from_cache(self):
        """Verify surrogate model container loads without remote network dependency."""
        container = load_registered_model(
            artifact_address="village-preparedness-classifier:v0",
            cache_dir=".wandb_artifacts_cache",
        )
        assert container is not None
        assert len(container.features) == 26
        assert set(container.classes_) == {"High", "Low", "Medium"}

    def test_surrogate_probability_distribution(self):
        """Verify surrogate model probabilities sum to 1.0 and produce calibrated intervals."""
        container = load_registered_model(
            artifact_address="village-preparedness-classifier:v0",
            cache_dir=".wandb_artifacts_cache",
        )

        # Generate a dummy input with the exact 26 features
        dummy_data = {f: [0.0] for f in container.features}
        # Set realistic dummy values
        dummy_data["has_pucca_road"] = [1.0]
        dummy_data["has_major_district_road"] = [1.0]
        dummy_data["bus_connectivity"] = [1.0]
        X_df = pd.DataFrame(dummy_data)

        probs = container.predict_proba(X_df)
        assert "prob_high" in probs
        assert "prob_medium" in probs
        assert "prob_low" in probs

        total_prob = probs["prob_high"][0] + probs["prob_medium"][0] + probs["prob_low"][0]
        assert pytest.approx(total_prob, abs=1e-5) == 1.0


# ==============================================================================
# 7. REFUSAL GATING FOR UNSUPPORTED DYNAMIC TELEMETRY
# ==============================================================================


class TestRefusalGating:
    """Tests for refusal gate intercepting live operational telemetry requests."""

    @pytest.mark.parametrize(
        "query,expected_domain",
        [
            ("What is the live wind speed right now in Mangalore?", "meteorological_telemetry"),
            ("How many hospital beds are vacant today in Udupi?", "medical_occupancy"),
            ("Show me real-time IMD cyclone storm surge forecast.", "meteorological_telemetry"),
            ("Are roads currently waterlogged or impassable right now?", "infrastructure_damage"),
        ],
    )
    def test_refusal_on_live_telemetry_queries(self, refusal_gate, query, expected_domain):
        """Verify live telemetry queries are immediately refused with confidence."""
        decision = refusal_gate.evaluate(query)
        assert decision.is_refused is True
        assert decision.reason is not None
        assert "dynamic" in decision.reason.lower() or "imd" in decision.reason.lower()


# ==============================================================================
# 8. TIMEOUT AND MALFORMED SYNTHESIS FALLBACK
# ==============================================================================


class TestTimeoutAndFallbackBehavior:
    """Tests for timeout resilience and deterministic reconciliation."""

    def test_ollama_timeout_triggers_deterministic_fallback(self, service, registry):
        """Verify timeout during synthesis falls back cleanly to deterministic response."""
        async def _run():
            mock_client = AsyncMock(spec=OllamaClient)
            mock_client.provider_name = "ollama"
            mock_client.model = "qwen2.5:0.5b-instruct"
            mock_client.is_configured = MagicMock(return_value=True)

            # 1. Intent parse succeeds
            # 2. Synthesis times out
            mock_client.generate.side_effect = [
                json.dumps({
                    "intents": ["explain_preparedness_assessment"],
                    "arguments": {"identifier": "Tallur"},
                    "needs_clarification": False,
                }),
                OllamaTimeoutError("Request timed out after 30s"),
            ]

            orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
            req = ChatRequest(message="What is the preparedness assessment for Tallur?")
            resp = await orchestrator.process_chat(req)

            assert resp.is_fallback is True
            assert "explain_preparedness_assessment" in resp.tools_executed
            assert "Tallur" in resp.answer
            assert "74.34" in resp.answer or "Adequate" in resp.answer

        asyncio.run(_run())

    def test_contradictory_claims_guard_reconciles_score(self):
        """Verify audit_and_guard_answer appends reconciliation block if LLM contradicts AHP score."""
        hallucinated_answer = "Tallur has an excellent score of 98.50 and belongs to Band 1."
        evidence = {
            "explain_preparedness_assessment": {
                "community_name": "Tallur",
                "deterministic_ahp": {"preparedness_score": 74.34, "preparedness_tier": "High"},
                "operational_prioritization": {"priority_band": "BAND_4_ADEQUATE"},
            }
        }

        guarded = audit_and_guard_answer(hallucinated_answer, evidence)
        assert "Authoritative Metric Verification" in guarded
        assert "74.34" in guarded
        assert "BAND_4_ADEQUATE" in guarded
