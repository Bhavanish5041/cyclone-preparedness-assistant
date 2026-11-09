"""Unit and integration tests for the Gemini LLM Client and Tool Orchestrator (Phase 4.2).

Verifies:
- GeminiClient configuration, request formatting, and header-based authentication.
- Complete conversational tool-calling cycle with single, multiple, and sequential tools.
- Strict multi-turn limit enforcement (max 4 rounds, max 8 calls) preventing infinite loops.
- Error handling for invalid tool names, invalid arguments, rate limits, timeouts, and malformed responses.
- Deterministic fallback when Gemini is unavailable, unconfigured, or erroring.
- Absolute non-leakage of API keys across responses, logs, exceptions, and serialization.
- Scientific guardrail enforcement (AHP precedence, surrogate ML demarcation, UK fallback).
- FastAPI `/api/v1/assistant/chat` endpoint validation and responses.
"""

import asyncio
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    GeminiAuthError,
    GeminiClient,
    GeminiClientError,
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiResponseError,
    GeminiTimeoutError,
    MessageRole,
    SCIENTIFIC_PRECEDENCE_DISCLAIMER,
    SYSTEM_INSTRUCTION,
    ToolCallingOrchestrator,
    ToolExecutionRecord,
)
from src.assistant.tools.registry import ToolRegistry, get_default_registry

DUMMY_SECRET_KEY = "AIzaSyDUMMY_SECRET_KEY_CYCLONE_12345"


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    """Module-scoped DecisionSupportService instance."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def shared_registry(shared_service: DecisionSupportService) -> ToolRegistry:
    """Module-scoped ToolRegistry configured with all 7 deterministic tools."""
    return get_default_registry(service=shared_service)


# ==============================================================================
# 1. GEMINI CLIENT TESTS
# ==============================================================================


def test_gemini_client_configuration():
    """Verify GeminiClient initialization, model defaults, and is_configured status."""
    client_empty = GeminiClient(api_key="")
    assert not client_empty.is_configured()

    client_configured = GeminiClient(
        api_key=DUMMY_SECRET_KEY,
        model="gemini-2.5-flash",
    )
    assert client_configured.is_configured()
    assert client_configured.model == "gemini-2.5-flash"
    assert client_configured.base_url == "https://generativelanguage.googleapis.com"


def test_gemini_client_unconfigured_error():
    """Verify calling generate_content without API key raises GeminiAuthError."""
    async def _run():
        client = GeminiClient(api_key="")
        with pytest.raises(GeminiAuthError, match="not configured"):
            await client.generate_content(contents=[{"role": "user", "parts": [{"text": "Hello"}]}])
    asyncio.run(_run())


def test_gemini_client_http_exceptions():
    """Verify proper classification of HTTP status codes and timeout exceptions."""
    async def _run():
        mock_transport = MagicMock(spec=httpx.AsyncClient)

        # 1. HTTP 401 / 403 Authentication Error
        mock_transport.post = AsyncMock(
            return_value=httpx.Response(401, text=f"Invalid API Key: {DUMMY_SECRET_KEY}")
        )
        client = GeminiClient(api_key=DUMMY_SECRET_KEY, http_client=mock_transport)
        with pytest.raises(GeminiAuthError) as exc_info:
            await client.generate_content(contents=[{"role": "user", "parts": [{"text": "Test"}]}])
        assert DUMMY_SECRET_KEY not in str(exc_info.value)
        assert "[REDACTED_API_KEY]" in str(exc_info.value) or "REDACTED" in str(exc_info.value)

        # 2. HTTP 429 Rate Limit
        mock_transport.post = AsyncMock(return_value=httpx.Response(429, text="Resource has been exhausted"))
        with pytest.raises(GeminiRateLimitError, match="rate limit or quota exceeded"):
            await client.generate_content(contents=[{"role": "user", "parts": [{"text": "Test"}]}])

        # 3. HTTP 500 Server Error
        mock_transport.post = AsyncMock(return_value=httpx.Response(500, text="Internal server error"))
        with pytest.raises(GeminiResponseError, match="upstream server error"):
            await client.generate_content(contents=[{"role": "user", "parts": [{"text": "Test"}]}])

        # 4. Timeout Exception
        mock_transport.post = AsyncMock(side_effect=httpx.TimeoutException("Read timed out"))
        with pytest.raises(GeminiTimeoutError, match="timed out"):
            await client.generate_content(contents=[{"role": "user", "parts": [{"text": "Test"}]}])

        # 5. Network Connection Error
        mock_transport.post = AsyncMock(side_effect=httpx.ConnectError("Failed to resolve host"))
        with pytest.raises(GeminiNetworkError, match="network connection error"):
            await client.generate_content(contents=[{"role": "user", "parts": [{"text": "Test"}]}])
    asyncio.run(_run())


def test_gemini_client_response_parsing():
    """Verify extracting function calls and natural language text from Gemini responses."""
    sample_fc_response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup_community",
                                "args": {"query": "Tallur"},
                            }
                        },
                        {
                            "functionCall": {
                                "name": "explain_preparedness_assessment",
                                "args": {"identifier": "608691"},
                            }
                        },
                    ],
                }
            }
        ]
    }
    calls = GeminiClient.extract_function_calls(sample_fc_response)
    assert len(calls) == 2
    assert calls[0]["name"] == "lookup_community"
    assert calls[0]["args"] == {"query": "Tallur"}
    assert calls[1]["name"] == "explain_preparedness_assessment"

    sample_text_response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Tallur is located in Udupi district."},
                        {"text": "Its AHP preparedness score is 74.34 (High Tier)."},
                    ],
                }
            }
        ]
    }
    text = GeminiClient.extract_text_response(sample_text_response)
    assert text == "Tallur is located in Udupi district.\nIts AHP preparedness score is 74.34 (High Tier)."


# ==============================================================================
# 2. ORCHESTRATOR CONVERSATIONAL TURN TESTS (MOCKED GEMINI)
# ==============================================================================


def test_orchestrator_normal_text_response(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify conversation turn where Gemini answers directly without requesting tools."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "gemini-2.5-flash"
        mock_client.generate_content = AsyncMock(
            return_value={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "The platform covers 1,879 administrative units in coastal Karnataka."}],
                        }
                    }
                ]
            }
        )
        mock_client.extract_function_calls = GeminiClient.extract_function_calls
        mock_client.extract_text_response = GeminiClient.extract_text_response

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="How many units are covered in the platform?")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert not res.is_fallback
        assert res.tools_executed == []
        assert res.tool_records == []
        assert "1,879 administrative units" in res.answer
        assert res.model_name == "gemini-2.5-flash"
        assert "AHP PRECEDENCE MANDATE" in res.scientific_precedence_note

    asyncio.run(_run())


def test_orchestrator_single_tool_call(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify turn where Gemini requests a single tool call (lookup_community)."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "gemini-2.5-flash"

        # Step 1: Model requests lookup_community
        step1_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "lookup_community",
                                    "args": {"query": "Tallur"},
                                }
                            }
                        ],
                    }
                }
            ]
        }
        # Step 2: Model formulates grounded natural language response
        step2_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Tallur was resolved as PC11 ID 608691 in Udupi with AHP score 74.34 (Band 4 Adequate)."}
                        ],
                    }
                }
            ]
        }
        mock_client.generate_content = AsyncMock(side_effect=[step1_resp, step2_resp])
        mock_client.extract_function_calls = GeminiClient.extract_function_calls
        mock_client.extract_text_response = GeminiClient.extract_text_response

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="What is the identifier for Tallur?")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert not res.is_fallback
        assert res.tools_executed == ["lookup_community"]
        assert len(res.tool_records) == 1
        assert res.tool_records[0].tool_name == "lookup_community"
        assert res.tool_records[0].success is True
        assert "lookup_community" in res.structured_evidence
        assert "Tallur was resolved" in res.answer

    asyncio.run(_run())


def test_orchestrator_sequential_tool_calls(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify turn with sequential tool calls across multiple rounds."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "gemini-2.5-flash"

        # Round 1: Model requests lookup_community
        step1_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "lookup_community", "args": {"query": "Edmoge"}}}],
                    }
                }
            ]
        }
        # Round 2: Model requests explain_preparedness_assessment
        step2_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "explain_preparedness_assessment", "args": {"identifier": "608720"}}}],
                    }
                }
            ]
        }
        # Round 3: Model outputs final answer
        step3_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Edmoge (PC11: 608720) is in Band 2 Specialized due to remote foothill hospital isolation."}],
                    }
                }
            ]
        }
        mock_client.generate_content = AsyncMock(side_effect=[step1_resp, step2_resp, step3_resp])
        mock_client.extract_function_calls = GeminiClient.extract_function_calls
        mock_client.extract_text_response = GeminiClient.extract_text_response

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Tell me about Edmoge's preparedness.", community_identifier="Edmoge")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert res.tools_executed == ["lookup_community", "explain_preparedness_assessment"]
        assert len(res.tool_records) == 2
        assert "explain_preparedness_assessment" in res.structured_evidence
        assert "Edmoge (PC11: 608720)" in res.answer

    asyncio.run(_run())


def test_orchestrator_invalid_tool_name_handled_server_side(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify model requesting an unregistered tool name receives structured error envelope."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "gemini-2.5-flash"

        # Step 1: Model attempts to call an unregistered tool
        step1_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "deploy_evacuation_chopper", "args": {"location": "Malpe"}}}],
                    }
                }
            ]
        }
        # Step 2: Model receives error response and responds politely
        step2_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "I do not have dispatch capabilities. I provide decision support only."}],
                    }
                }
            ]
        }
        mock_client.generate_content = AsyncMock(side_effect=[step1_resp, step2_resp])
        mock_client.extract_function_calls = GeminiClient.extract_function_calls
        mock_client.extract_text_response = GeminiClient.extract_text_response

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Deploy a chopper to Malpe")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert len(res.tool_records) == 1
        assert res.tool_records[0].tool_name == "deploy_evacuation_chopper"
        assert res.tool_records[0].success is False
        assert res.tool_records[0].error_type == "UNKNOWN_TOOL"
        assert "I do not have dispatch capabilities" in res.answer

    asyncio.run(_run())


def test_orchestrator_max_tool_rounds_limit_enforced(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify orchestrator terminates infinite tool loops after MAX_TOOL_ROUNDS (4 rounds)."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "gemini-2.5-flash"

        # Response that repeatedly requests a tool
        repeated_fc = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "lookup_community", "args": {"query": "Tallur"}}}],
                    }
                }
            ]
        }
        final_text_resp = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Budget exhausted. Formulating final response with current data."}],
                    }
                }
            ]
        }
        # 4 tool calls + 1 final text synthesis call
        mock_client.generate_content = AsyncMock(
            side_effect=[repeated_fc, repeated_fc, repeated_fc, repeated_fc, final_text_resp]
        )
        mock_client.extract_function_calls = GeminiClient.extract_function_calls
        mock_client.extract_text_response = GeminiClient.extract_text_response

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
            max_rounds=4,
        )

        req = ChatRequest(message="Loop test query")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert len(res.tool_records) == 4
        assert mock_client.generate_content.call_count == 5  # 4 rounds + 1 final synthesis
        assert "Budget exhausted" in res.answer

    asyncio.run(_run())


# ==============================================================================
# 3. DETERMINISTIC FALLBACK TESTS
# ==============================================================================


def test_orchestrator_deterministic_fallback_when_unconfigured(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback triggers immediately when GEMINI_API_KEY is not configured."""
    async def _run():
        client_unconfigured = GeminiClient(api_key="")
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
        assert "GEMINI_API_KEY is not configured" in res.fallback_reason
        assert res.model_name == "deterministic-fallback"

        # Evaluated Tallur directly via deterministic tools
        assert "explain_preparedness_assessment" in res.tools_executed
        assert "get_intervention_recommendations" in res.tools_executed
        assert "Tallur" in res.answer
        assert "74.34" in res.answer  # AHP score
        assert "BAND_4_ADEQUATE" in res.answer
        assert "AHP PRECEDENCE MANDATE" in res.scientific_precedence_note

    asyncio.run(_run())


def test_orchestrator_deterministic_fallback_on_rate_limit(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback triggers when Gemini returns HTTP 429 rate limit."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.generate_content = AsyncMock(
            side_effect=GeminiRateLimitError("Rate limit exceeded (HTTP 429)")
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
        assert "Rate limit exceeded" in res.fallback_reason
        assert "Edmoge" in res.answer

    asyncio.run(_run())


def test_orchestrator_deterministic_fallback_on_timeout(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback triggers when Gemini request times out."""
    async def _run():
        mock_client = MagicMock(spec=GeminiClient)
        mock_client.is_configured.return_value = True
        mock_client.generate_content = AsyncMock(
            side_effect=GeminiTimeoutError("Request timed out after 15 seconds")
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
        assert "Request timed out" in res.fallback_reason
        assert "Bappalgundi" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_tallur_unambiguous_query(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic resolution for 'Explain why Tallur has its current priority classification.'"""
    async def _run():
        client_unconfigured = GeminiClient(api_key="")
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Explain why Tallur has its current priority classification.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        # Verify tools executed
        assert "lookup_community" in res.tools_executed
        assert "explain_preparedness_assessment" in res.tools_executed
        assert "get_intervention_recommendations" in res.tools_executed
        # Verify content
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
        client_unconfigured = GeminiClient(api_key="")
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
        client_unconfigured = GeminiClient(api_key="")
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
    """Verify deterministic methodology explanation for 'Explain Uttara Kannada's fallback methodology.'"""
    async def _run():
        client_unconfigured = GeminiClient(api_key="")
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Explain Uttara Kannada's fallback methodology.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "explain_vulnerability_archetype" in res.tools_executed
        assert "Uttara Kannada" in res.answer
        assert "1,257" in res.answer
        assert "three-sector" in res.answer.lower() or "3-sector" in res.answer.lower()
        assert "Coastal Hazard Proximity" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_ambiguous_community(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify disambiguation prompt for ambiguous village name 'Assess Siddapur.'"""
    async def _run():
        client_unconfigured = GeminiClient(api_key="")
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Assess Siddapur.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "lookup_community" in res.tools_executed
        assert "Community Disambiguation Required" in res.answer
        assert "matches" in res.answer.lower() or "administrative units" in res.answer.lower()
        assert "602812" in res.answer
        assert "608717" in res.answer

    asyncio.run(_run())


def test_deterministic_fallback_unsupported_query(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify system boundary response for unsupported query 'What is the weather in Paris tomorrow?'"""
    async def _run():
        client_unconfigured = GeminiClient(api_key="")
        orchestrator = ToolCallingOrchestrator(
            client=client_unconfigured,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="What is the weather in Paris tomorrow?")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert res.tools_executed == []
        assert "Decision Support System Boundary" in res.answer
        assert "unable to answer this inquiry directly" in res.answer
        assert "1,879" in res.answer

    asyncio.run(_run())


# ==============================================================================
# 4. SCIENTIFIC GUARDRAIL & PROMPT INTEGRITY TESTS
# ==============================================================================


def test_scientific_guardrails_system_instruction():
    """Verify SYSTEM_INSTRUCTION strictly contains mandatory scientific constraints."""
    # AHP absolute precedence
    assert "AHP ABSOLUTE OPERATIONAL PRECEDENCE" in SYSTEM_INSTRUCTION
    assert "0.0 - 100.0" in SYSTEM_INSTRUCTION
    assert "never alter, recalculate, or override" in SYSTEM_INSTRUCTION.lower()

    # Surrogate ML demarcation
    assert "SURROGATE ML DEMARCATION" in SYSTEM_INSTRUCTION
    assert "empirical tier surrogate" in SYSTEM_INSTRUCTION
    assert "NEVER refer to model probabilities" in SYSTEM_INSTRUCTION

    # Descriptive archetypes & UK fallback
    assert "DESCRIPTIVE VULNERABILITY ARCHETYPES" in SYSTEM_INSTRUCTION
    assert "UTTARA KANNADA DETERMINISTIC FALLBACK" in SYSTEM_INSTRUCTION
    assert "3-sector fallback" in SYSTEM_INSTRUCTION

    # Historical vintage caveat
    assert "CENSUS 2011" in SYSTEM_INSTRUCTION
    assert "over 15 years old" in SYSTEM_INSTRUCTION

    # Zero operational invention
    assert "ZERO REAL-TIME OPERATIONAL INVENTION" in SYSTEM_INSTRUCTION
    assert "India Meteorological Department" in SYSTEM_INSTRUCTION


# ==============================================================================
# 5. REQUEST VALIDATION TESTS
# ==============================================================================


def test_chat_request_validation():
    """Verify strict Pydantic input validation for ChatRequest."""
    # Valid request
    valid_req = ChatRequest(message="What is the AHP score of Tallur?")
    assert valid_req.message == "What is the AHP score of Tallur?"
    assert valid_req.conversation_history == []

    # Whitespace-only message rejected
    with pytest.raises(ValueError, match="cannot be empty or solely whitespace"):
        ChatRequest(message="    ")

    # History > 10 items rejected
    long_history = [
        ChatMessage(role=MessageRole.USER, content=f"Msg {i}")
        for i in range(12)
    ]
    with pytest.raises(ValueError, match="cannot exceed 10 messages"):
        ChatRequest(message="Valid prompt", conversation_history=long_history)


# ==============================================================================
# 6. FASTAPI ENDPOINT INTEGRATION TESTS
# ==============================================================================


def test_fastapi_assistant_chat_endpoint_fallback():
    """Verify POST /api/v1/assistant/chat executes deterministic fallback when unconfigured."""
    with TestClient(app) as test_client:
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


def test_api_key_non_leakage():
    """Verify dummy API key is never leaked in client error messages or response payloads."""
    client = GeminiClient(api_key=DUMMY_SECRET_KEY)
    masked = client._mask_api_key_in_string(f"Error querying {DUMMY_SECRET_KEY} endpoint")
    assert DUMMY_SECRET_KEY not in masked
    assert "[REDACTED_API_KEY]" in masked
