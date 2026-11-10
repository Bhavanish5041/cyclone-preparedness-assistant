"""Tests for Qwen-based intent understanding, multi-intent execution, and deterministic safeguards."""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.client import OllamaClient, OllamaTimeoutError
from src.assistant.llm.orchestrator import ToolCallingOrchestrator, audit_and_guard_answer
from src.assistant.llm.schemas import ChatRequest, ConversationStatus
from src.assistant.tools.registry import get_default_registry


@pytest.fixture(scope="module")
def service():
    return DecisionSupportService()


@pytest.fixture(scope="module")
def registry(service):
    return get_default_registry(service=service)


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=OllamaClient)
    client.model = "qwen2.5:0.5b-instruct"
    client.provider_name = "ollama"
    return client


# ------------------------------------------------------------------------------
# 1. Simple community lookup
# ------------------------------------------------------------------------------
def test_simple_community_lookup(service, registry, mock_client):
    """Verify lookup_community intent resolution and execution."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["lookup_community"], "arguments": {"query": "Belapu"}, "needs_clarification": false}',
            "Belapu is located in Kaup taluk of Udupi district.",
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="Where is Belapu located?")
        res = await orchestrator.process_chat(req)

        assert "lookup_community" in res.tools_executed
        assert res.conversation_status == ConversationStatus.COMPLETED
        assert not res.is_fallback
        assert "Belapu" in res.answer

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 2. Preparedness assessment
# ------------------------------------------------------------------------------
def test_preparedness_assessment(service, registry, mock_client):
    """Verify explain_preparedness_assessment retrieves exact AHP metrics."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["explain_preparedness_assessment"], "arguments": {"identifier": "Tallur"}, "needs_clarification": false}',
            "Tallur has an authoritative AHP score of 74.34 and is classified under Band 4 Adequate.",
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="What is the preparedness assessment for Tallur?")
        res = await orchestrator.process_chat(req)

        assert "explain_preparedness_assessment" in res.tools_executed
        assert "74.34" in res.answer
        assert "Band 4 Adequate" in res.answer

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 3. District comparison
# ------------------------------------------------------------------------------
def test_district_comparison(service, registry, mock_client):
    """Verify compare_districts executes and returns comparative metrics."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["compare_districts"], "arguments": {"district_a": "Dakshina Kannada", "district_b": "Udupi"}, "needs_clarification": false}',
            "Dakshina Kannada has 369 units with mean preparedness 61.06, while Udupi has 253 units with mean preparedness 65.04.",
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="Compare Dakshina Kannada and Udupi.")
        res = await orchestrator.process_chat(req)

        assert "compare_districts" in res.tools_executed
        assert "compare_districts" in res.structured_evidence
        comp = res.structured_evidence["compare_districts"]["comparison"]
        assert comp["Dakshina Kannada"]["total_units"] == 369
        assert comp["Udupi"]["total_units"] == 253

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 4. Infrastructure vulnerability analysis
# ------------------------------------------------------------------------------
def test_infrastructure_vulnerability_analysis(service, registry, mock_client):
    """Verify analyze_infrastructure_and_exposure executes for village."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["analyze_infrastructure_and_exposure"], "arguments": {"identifier": "Edmoge"}, "needs_clarification": false}',
            "Edmoge infrastructure analysis shows specific pucca housing and hospital distance metrics.",
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="Analyze infrastructure vulnerabilities for Edmoge.")
        res = await orchestrator.process_chat(req)

        assert "analyze_infrastructure_and_exposure" in res.tools_executed
        assert "analyze_infrastructure_and_exposure" in res.structured_evidence

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 5. Intervention recommendations
# ------------------------------------------------------------------------------
def test_intervention_recommendations(service, registry, mock_client):
    """Verify get_intervention_recommendations executes."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["get_intervention_recommendations"], "arguments": {"identifier": "Edmoge"}, "needs_clarification": false}',
            "Recommended interventions for Edmoge prioritize shelter access and all-weather road connectivity.",
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="What policy interventions are recommended for Edmoge?")
        res = await orchestrator.process_chat(req)

        assert "get_intervention_recommendations" in res.tools_executed
        assert "advisories" in res.structured_evidence["get_intervention_recommendations"]

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 6. Multi-intent failed query reproduction
# ------------------------------------------------------------------------------
def test_reproduction_of_failed_query(service, registry, mock_client):
    """Specifically reproduce and pass the multi-clause query that previously failed with 0.32 confidence."""
    async def _run():
        failed_query = (
            "Compare the cyclone preparedness of Dakshina Kannada and Udupi. "
            "Which district has greater infrastructure vulnerabilities, what are the major "
            "differences in their preparedness, and what interventions should local authorities prioritise?"
        )

        mock_client.generate.side_effect = [
            (
                '{"intents": ["compare_districts", "explain_preparedness_assessment"], '
                '"arguments": {"district_a": "Dakshina Kannada", "district_b": "Udupi"}, '
                '"needs_clarification": false}'
            ),
            (
                "Comparative Assessment: Dakshina Kannada (369 units, mean preparedness score 61.06) "
                "has 2 Band 1 Critical units and greater hospital distance (37.86 km) than Udupi "
                "(253 units, mean preparedness score 65.04, 0 Band 1 Critical units, 25.4 km hospital distance)."
            ),
        ]

        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message=failed_query)
        res = await orchestrator.process_chat(req)

        # Must NOT fail with low-confidence abstention
        assert "Low classification confidence" not in res.answer
        assert "abstains from automatic routing" not in res.answer
        assert "compare_districts" in res.tools_executed
        assert res.conversation_status == ConversationStatus.COMPLETED
        assert not res.is_fallback
        assert "Dakshina Kannada" in res.answer
        assert "Udupi" in res.answer

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 7. Paraphrased and conversational queries
# ------------------------------------------------------------------------------
def test_paraphrased_conversational_queries(service, registry, mock_client):
    """Verify conversational phrasing routes correctly without softmax degradation."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["compare_districts"], "arguments": {"district_a": "Dakshina Kannada", "district_b": "Udupi"}, "needs_clarification": false}',
            "Dakshina Kannada and Udupi comparison briefed.",
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="Could you provide a detailed side by side breakdown of Udupi vs Dakshina Kannada?")
        res = await orchestrator.process_chat(req)

        assert "compare_districts" in res.tools_executed
        assert res.conversation_status == ConversationStatus.COMPLETED

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 8. Ambiguous community names
# ------------------------------------------------------------------------------
def test_ambiguous_community_names(service, registry, mock_client):
    """Verify ambiguous community name triggers deterministic disambiguation."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["explain_preparedness_assessment"], "arguments": {"identifier": "Ullal"}, "needs_clarification": false}',
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="What is the preparedness of Ullal?")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert len(res.tools_executed) == 0  # Does not blindly execute ambiguous target
        assert "Disambiguation" in res.answer or "Ambiguous" in res.answer
        assert "603256" in res.answer or "803182" in res.answer

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 9. Missing arguments
# ------------------------------------------------------------------------------
def test_missing_arguments(service, registry, mock_client):
    """Verify missing required parameter triggers deterministic clarification prompt."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["compare_districts"], "arguments": {"district_a": "Dakshina Kannada"}, "needs_clarification": false}',
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="Compare Dakshina Kannada.")
        res = await orchestrator.process_chat(req)

        assert len(res.tools_executed) == 0
        assert "Comparison requires two coastal districts" in res.answer

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 10. Unsupported live-data requests
# ------------------------------------------------------------------------------
def test_unsupported_live_data_requests(service, registry, mock_client):
    """Verify live dynamic request is rejected by refusal gate without calling LLM."""
    async def _run():
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="What is the current wind speed right now in Mangalore?")
        res = await orchestrator.process_chat(req)

        assert res.model_name == "deterministic-gate"
        assert len(res.tools_executed) == 0
        mock_client.generate.assert_not_called()  # LLM never invoked for live dynamic requests
        assert "live dynamic operational telemetry" in res.answer.lower() or "imd" in res.answer.lower()

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 11. Malformed Qwen JSON
# ------------------------------------------------------------------------------
def test_malformed_qwen_json(service, registry, mock_client):
    """Verify corrupted JSON from LLM gracefully falls back to deterministic rule parser."""
    async def _run():
        mock_client.generate.side_effect = [
            "Here is what I think: { corrupted json",
            "Tallur has an authoritative AHP score of 74.34.",
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="Explain preparedness for Tallur.")
        res = await orchestrator.process_chat(req)

        assert "explain_preparedness_assessment" in res.tools_executed
        assert res.conversation_status == ConversationStatus.COMPLETED

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 12. Ollama timeout and deterministic fallback
# ------------------------------------------------------------------------------
def test_ollama_timeout_and_deterministic_fallback(service, registry, mock_client):
    """Verify CPU timeout during synthesis activates deterministic template formatting."""
    async def _run():
        mock_client.generate.side_effect = [
            '{"intents": ["compare_districts"], "arguments": {"district_a": "Dakshina Kannada", "district_b": "Udupi"}, "needs_clarification": false}',
            OllamaTimeoutError("CPU inference timed out after 30s"),
        ]
        orchestrator = ToolCallingOrchestrator(client=mock_client, registry=registry, service=service)
        req = ChatRequest(message="Compare Dakshina Kannada and Udupi.")
        res = await orchestrator.process_chat(req)

        assert res.is_fallback is True
        assert "timed out" in res.fallback_reason
        assert "Dakshina Kannada" in res.answer
        assert "Udupi" in res.answer
        assert "369" in res.answer
        assert "253" in res.answer

    asyncio.run(_run())


# ------------------------------------------------------------------------------
# 13. Incorrect or contradictory Qwen-generated claims
# ------------------------------------------------------------------------------
def test_contradictory_qwen_claims_guarded():
    """Verify post-generation safety audit appends authoritative reconciliation block."""
    evidence = {
        "explain_preparedness_assessment": {
            "community_name": "Tallur",
            "deterministic_ahp": {"preparedness_score": 74.34},
            "operational_prioritization": {"priority_band": "BAND_4_ADEQUATE"},
        }
    }
    # Hallucinated answer claiming wrong band and score
    hallucinated = "Tallur is classified as Band 1 Critical with an AHP score of 12.5."
    guarded = audit_and_guard_answer(hallucinated, evidence)

    assert "Authoritative Metric Verification (Deterministic Ground Truth):" in guarded
    assert "74.34 / 100.0" in guarded
    assert "BAND_4_ADEQUATE" in guarded
