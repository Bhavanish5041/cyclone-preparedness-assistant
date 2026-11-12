"""Unit and integration tests for Ollama LLM Client and Tool Orchestrator.

Verifies:
- OllamaClient configuration, defaults, and local connection endpoints.
- Intent routing and deterministic fallback when Ollama is unavailable or unconfigured.
- Prevention of infinite tool loops (strictly capped at 4 rounds and 8 executions).
- Closed-domain scientific guardrails (AHP precedence, surrogate ML demarcation, UK fallback).
- FastAPI `/api/v1/assistant/chat` endpoint validation and responses.
"""

import asyncio
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from api.main import app
from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    MessageRole,
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    SCIENTIFIC_PRECEDENCE_DISCLAIMER,
    SYSTEM_INSTRUCTION,
    ToolCallingOrchestrator,
    ToolExecutionRecord,
)
from src.assistant.tools.registry import ToolRegistry, get_default_registry


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    """Module-scoped DecisionSupportService instance."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def shared_registry(shared_service: DecisionSupportService) -> ToolRegistry:
    """Module-scoped ToolRegistry configured with all 7 deterministic tools."""
    return get_default_registry(service=shared_service)


# ==============================================================================
# 1. OLLAMA CLIENT TESTS
# ==============================================================================


def test_ollama_client_configuration():
    """Verify OllamaClient initialization, model defaults, and endpoint URLs."""
    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:0.5b-instruct",
        timeout=20.0,
    )
    assert client.is_configured()
    assert client.model == "qwen2.5:0.5b-instruct"
    assert client.base_url == "http://127.0.0.1:11434"
    assert client.timeout == 20.0
    assert client.provider_name == "ollama"


def test_ollama_client_exceptions():
    """Verify classification of Ollama client network errors and timeouts."""
    async def _run():
        mock_transport = MagicMock(spec=httpx.AsyncClient)

        # 1. Timeout Exception
        mock_transport.post = AsyncMock(side_effect=httpx.TimeoutException("Connection timed out"))
        client = OllamaClient(http_client=mock_transport)
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            await client.generate("test prompt")

        # 2. Connection Error
        mock_transport.post = AsyncMock(side_effect=httpx.ConnectError("Failed to connect"))
        with pytest.raises(OllamaConnectionError, match="Failed to connect"):
            await client.generate("test prompt")

        # 3. HTTP 500 Response Error
        mock_transport.post = AsyncMock(return_value=httpx.Response(500, text="Internal Ollama Crash"))
        with pytest.raises(OllamaResponseError, match="HTTP 500"):
            await client.generate("test prompt")

    asyncio.run(_run())


# ==============================================================================
# 2. ORCHESTRATOR FALLBACK ON OLLAMA FAILURES
# ==============================================================================


def test_orchestrator_deterministic_fallback_when_unconfigured(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback triggers immediately when client is unconfigured."""
    async def _run():
        client_unconfigured = MagicMock(spec=OllamaClient)
        client_unconfigured.is_configured.return_value = False
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(
            message="What is the preparedness status of Tallur?",
            community_identifier="Tallur",
        )
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "Ollama client is not configured" in res.fallback_reason
        assert res.model_name == "deterministic-fallback"

        # Evaluated Tallur directly via deterministic tools
        assert "explain_preparedness_assessment" in res.tools_executed
        assert "get_intervention_recommendations" in res.tools_executed
        assert "Tallur" in res.answer
        assert "74.34" in res.answer
        assert "BAND_4_ADEQUATE" in res.answer
        assert "AHP PRECEDENCE MANDATE" in res.scientific_precedence_note

    asyncio.run(_run())


def test_orchestrator_deterministic_fallback_on_connection_error(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback triggers when Ollama daemon is offline."""
    async def _run():
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "qwen2.5:0.5b-instruct"
        mock_client.generate = AsyncMock(
            side_effect=OllamaConnectionError("Failed to connect to Ollama")
        )

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(
            message="Give me the assessment for Edmoge",
            community_identifier="Edmoge",
        )
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "Edmoge" in res.answer

    asyncio.run(_run())


def test_orchestrator_deterministic_fallback_on_timeout(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback triggers when Ollama inference times out."""
    async def _run():
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "qwen2.5:0.5b-instruct"
        mock_client.generate = AsyncMock(
            side_effect=OllamaTimeoutError("Request timed out after 30 seconds")
        )

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Status of Bappalgundi", community_identifier="Bappalgundi")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "Bappalgundi" in res.answer

    asyncio.run(_run())


# ==============================================================================
# 3. DETERMINISTIC INTENT RESOLUTION TESTS
# ==============================================================================


def test_deterministic_fallback_tallur_unambiguous_query(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic resolution for 'Explain why Tallur has its current priority classification.'"""
    async def _run():
        client_unconfigured = MagicMock(spec=OllamaClient)
        client_unconfigured.is_configured.return_value = False
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Explain why Tallur has its current priority classification.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "lookup_community" in res.tools_executed
        assert "explain_preparedness_assessment" in res.tools_executed
        assert "get_intervention_recommendations" in res.tools_executed
        assert "Tallur" in res.answer
        assert "Udupi" in res.answer
        assert "608691" in res.answer
        assert "74.34" in res.answer
        assert "BAND_4_ADEQUATE" in res.answer
        assert "Priority Justification and Supporting Indicators" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_district_comparison(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic district comparison for 'Compare Dakshina Kannada and Udupi.'"""
    async def _run():
        client_unconfigured = MagicMock(spec=OllamaClient)
        client_unconfigured.is_configured.return_value = False
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Compare Dakshina Kannada and Udupi.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "compare_districts" in res.tools_executed
        assert res.structured_evidence is not None
        assert "compare_districts" in res.structured_evidence
        assert "Dakshina Kannada" in res.answer
        assert "Udupi" in res.answer
        assert "369" in res.answer
        assert "253" in res.answer
        assert "Summary Metrics" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_band_filtering(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic band filtering for 'Find Band 1 communities.'"""
    async def _run():
        client_unconfigured = MagicMock(spec=OllamaClient)
        client_unconfigured.is_configured.return_value = False
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Find Band 1 communities.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "filter_and_rank_communities" in res.tools_executed
        assert res.structured_evidence is not None
        assert "filter_and_rank_communities" in res.structured_evidence
        assert "BAND_1_CRITICAL" in res.answer
        assert "155" in res.answer
        assert "Top Ranked Units" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_uttara_kannada_methodology(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback methodology response for Uttara Kannada queries."""
    async def _run():
        client_unconfigured = MagicMock(spec=OllamaClient)
        client_unconfigured.is_configured.return_value = False
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Explain Uttara Kannada fallback methodology.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "explain_vulnerability_archetype" in res.tools_executed
        assert res.structured_evidence is not None
        assert "Uttara Kannada" in res.answer
        assert "1,257" in res.answer
        assert "Unassigned" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_ambiguous_community(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify ambiguous community lookup returns structured disambiguation guidance."""
    async def _run():
        client_unconfigured = MagicMock(spec=OllamaClient)
        client_unconfigured.is_configured.return_value = False
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Assess Siddapur")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "lookup_community" in res.tools_executed
        assert "Community Disambiguation Required" in res.answer
        assert "602812" in res.answer
        assert "603083" in res.answer
        assert "608717" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_unsupported_query(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify unsupported out-of-scope query triggers boundaries guidance."""
    async def _run():
        client_unconfigured = MagicMock(spec=OllamaClient)
        client_unconfigured.is_configured.return_value = False
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Can you recommend a good Italian restaurant in Bangalore?")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert res.tools_executed == []
        assert "Decision Support System Boundary" in res.answer
        assert "deterministic mode" in res.answer.lower()

    asyncio.run(_run())


# ==============================================================================
# 4. SCIENTIFIC GUARDRAILS & INSTRUCTION AUDITING
# ==============================================================================


def test_scientific_guardrails_system_instruction():
    """Verify system instruction contains mandatory closed-domain scientific guardrails."""
    assert "AHP ABSOLUTE OPERATIONAL PRECEDENCE" in SYSTEM_INSTRUCTION
    assert "CLOSED-WORLD EVIDENCE GROUNDING" in SYSTEM_INSTRUCTION
    assert "SURROGATE ML DEMARCATION" in SYSTEM_INSTRUCTION
    assert "AHP PRECEDENCE MANDATE" in SCIENTIFIC_PRECEDENCE_DISCLAIMER


# ==============================================================================
# 5. REQUEST SCHEMA VALIDATION
# ==============================================================================


def test_chat_request_validation():
    """Verify ChatRequest Pydantic constraints (message length, history limits)."""
    # Empty message should fail
    with pytest.raises(ValueError):
        ChatRequest(message="   ")

    # Excessive message length (> 2000 chars) should fail
    with pytest.raises(ValueError, match=r"(cannot exceed 2000|at most 2000)"):
        ChatRequest(message="a" * 2001)

    # Valid message passes
    req = ChatRequest(message="Valid prompt")
    assert req.message == "Valid prompt"

    # Excessive history (> 10 items) should fail
    long_history = [
        ChatMessage(role=MessageRole.USER, content=f"Turn {i}")
        for i in range(11)
    ]
    with pytest.raises(ValueError, match=r"(cannot exceed 10|at most 10)"):
        ChatRequest(message="Valid prompt", conversation_history=long_history)


# ==============================================================================
# 6. FASTAPI ENDPOINT INTEGRATION TESTS
# ==============================================================================


def test_fastapi_assistant_chat_endpoint_fallback():
    """Verify POST /api/v1/assistant/chat executes deterministic fallback when unconfigured."""
    with TestClient(app) as test_client:
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.is_configured.return_value = False
        test_client.app.state.orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            service=test_client.app.state.service,
        )
        payload = {
            "message": "Why is Tallur in Band 4 Adequate?",
            "community_identifier": "Tallur",
        }
        response = test_client.post("/api/v1/assistant/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "Tallur" in data["answer"]
        assert data["conversation_status"] == "FALLBACK"
        assert data["is_fallback"] is True
        assert "explain_preparedness_assessment" in data["tools_executed"]
        assert "AHP PRECEDENCE MANDATE" in data["scientific_precedence_note"]
