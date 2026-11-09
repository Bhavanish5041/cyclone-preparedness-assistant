"""Unit and integration tests for the OpenRouter LLM Client and Tool Orchestrator.

Verifies:
- OpenRouterClient configuration, defaults, and header-based authentication.
- Complete conversational tool-calling cycle with single, multiple, and sequential tools.
- Strict multi-turn limit enforcement (max 4 rounds, max 8 calls) preventing infinite loops.
- Error handling for invalid tool names, invalid arguments, rate limits, timeouts, and malformed responses.
- Deterministic fallback when OpenRouter is unconfigured, unreachable, or rate-limited.
- Absolute non-leakage of API keys across responses, logs, exceptions, and serialization.
- Scientific guardrail enforcement (AHP precedence, surrogate ML demarcation, UK fallback).
- Telemetry record validation and preservation across execution turns.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    MessageRole,
    OpenRouterAuthError,
    OpenRouterClient,
    OpenRouterClientError,
    OpenRouterNetworkError,
    OpenRouterRateLimitError,
    OpenRouterResponseError,
    OpenRouterTimeoutError,
    SCIENTIFIC_PRECEDENCE_DISCLAIMER,
    SYSTEM_INSTRUCTION,
    ToolCallingOrchestrator,
    ToolExecutionRecord,
    get_default_llm_client,
)
from src.assistant.tools.registry import ToolRegistry, get_default_registry
from src.assistant.views.chat_assistant import _is_valid_telemetry_record

DUMMY_OPENROUTER_KEY = "sk-or-v1-dummy-secret-key-cyclone-platform-12345"


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    """Module-scoped DecisionSupportService instance."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def shared_registry(shared_service: DecisionSupportService) -> ToolRegistry:
    """Module-scoped ToolRegistry configured with all deterministic tools."""
    return get_default_registry(service=shared_service)


# ==============================================================================
# 1. OPENROUTER CLIENT UNIT TESTS
# ==============================================================================


def test_openrouter_client_configuration():
    """Verify OpenRouterClient initialization, defaults, and is_configured status."""
    client_empty = OpenRouterClient(api_key="")
    assert not client_empty.is_configured()
    assert client_empty.provider_name == "openrouter"

    client_configured = OpenRouterClient(
        api_key=DUMMY_OPENROUTER_KEY,
        model="nvidia/nemotron-3.5-lightning:free",
    )
    assert client_configured.is_configured()
    assert client_configured.provider_name == "openrouter"
    assert client_configured.model == "nvidia/nemotron-3.5-lightning:free"
    assert client_configured.base_url == "https://openrouter.ai/api/v1"
    assert client_configured.timeout == 30.0


def test_openrouter_client_unconfigured_error():
    """Verify OpenRouterClient raises OpenRouterAuthError if create_chat_completion is called without key."""
    client = OpenRouterClient(api_key="")

    async def _run():
        with pytest.raises(OpenRouterAuthError) as exc_info:
            await client.create_chat_completion(messages=[{"role": "user", "content": "Hello"}])
        assert "OPENROUTER_API_KEY environment variable" in str(exc_info.value)

    asyncio.run(_run())


def test_openrouter_client_successful_response():
    """Verify OpenRouterClient sends correct headers, payload, and parses JSON response."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    expected_response = {
        "id": "gen-123",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Tallur is classified in Band 4 Adequate.",
                }
            }
        ],
    }
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = expected_response
    mock_http.post = AsyncMock(return_value=mock_resp)

    client = OpenRouterClient(
        api_key=DUMMY_OPENROUTER_KEY,
        http_client=mock_http,
    )

    async def _run():
        result = await client.create_chat_completion(
            messages=[{"role": "user", "content": "Explain Tallur"}],
            temperature=0.0,
        )
        assert result == expected_response
        assert mock_http.post.called

        _, kwargs = mock_http.post.call_args
        assert kwargs["headers"]["Authorization"] == f"Bearer {DUMMY_OPENROUTER_KEY}"
        assert kwargs["headers"]["HTTP-Referer"] == "https://cyclone-preparedness-assistant.onrender.com"
        assert kwargs["headers"]["X-Title"] == "Cyclone Preparedness Assistant"
        assert kwargs["json"]["model"] == "nvidia/nemotron-3.5-lightning:free"
        assert kwargs["json"]["temperature"] == 0.0

    asyncio.run(_run())


def test_openrouter_client_extract_tool_calls():
    """Verify extraction of tool calls from OpenRouter completion payload."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "lookup_community",
                                "arguments": json.dumps({"query": "Tallur", "limit": 5}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    calls = OpenRouterClient.extract_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0]["id"] == "call_abc123"
    assert calls[0]["name"] == "lookup_community"
    assert calls[0]["args"] == {"query": "Tallur", "limit": 5}


def test_openrouter_client_extract_tool_calls_dict_arguments():
    """Verify extraction handles pre-parsed dictionary arguments gracefully."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_xyz789",
                            "type": "function",
                            "function": {
                                "name": "compare_districts",
                                "arguments": {"district_a": "Dakshina Kannada", "district_b": "Udupi"},
                            },
                        }
                    ],
                }
            }
        ]
    }
    calls = OpenRouterClient.extract_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0]["args"] == {"district_a": "Dakshina Kannada", "district_b": "Udupi"}


def test_openrouter_client_extract_text_response():
    """Verify extraction of natural language text from OpenRouter completion payload."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Tallur has an authoritative AHP preparedness score of 74.34.",
                }
            }
        ]
    }
    text = OpenRouterClient.extract_text_response(payload)
    assert text == "Tallur has an authoritative AHP preparedness score of 74.34."

    empty_payload = {"choices": []}
    assert OpenRouterClient.extract_text_response(empty_payload) is None


# ==============================================================================
# 2. OPENROUTER CLIENT ERROR CLASSIFICATION & NON-LEAKAGE
# ==============================================================================


@pytest.mark.parametrize(
    "status_code, expected_exc",
    [
        (401, OpenRouterAuthError),
        (403, OpenRouterAuthError),
        (429, OpenRouterRateLimitError),
        (500, OpenRouterResponseError),
        (502, OpenRouterResponseError),
        (503, OpenRouterResponseError),
    ],
)
def test_openrouter_client_error_classification(status_code: int, expected_exc: type):
    """Verify HTTP status codes map to specific OpenRouter exception types."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.text = f"HTTP {status_code} Error from provider containing {DUMMY_OPENROUTER_KEY}"
    mock_http.post = AsyncMock(return_value=mock_resp)

    client = OpenRouterClient(
        api_key=DUMMY_OPENROUTER_KEY,
        http_client=mock_http,
    )

    async def _run():
        with pytest.raises(expected_exc) as exc_info:
            await client.create_chat_completion(messages=[{"role": "user", "content": "Hi"}])

        err_msg = str(exc_info.value)
        # Ensure secret API key is scrubbed
        assert DUMMY_OPENROUTER_KEY not in err_msg
        if status_code in (401, 403):
            assert "[REDACTED_API_KEY]" in err_msg

    asyncio.run(_run())


def test_openrouter_client_timeout_handling():
    """Verify request timeout raises OpenRouterTimeoutError without leaking key."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(side_effect=httpx.TimeoutException(f"Timeout on key={DUMMY_OPENROUTER_KEY}"))

    client = OpenRouterClient(api_key=DUMMY_OPENROUTER_KEY, http_client=mock_http)

    async def _run():
        with pytest.raises(OpenRouterTimeoutError) as exc_info:
            await client.create_chat_completion(messages=[{"role": "user", "content": "Hi"}])
        assert DUMMY_OPENROUTER_KEY not in str(exc_info.value)

    asyncio.run(_run())


def test_openrouter_client_network_error_handling():
    """Verify connection error raises OpenRouterNetworkError without leaking key."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(side_effect=httpx.ConnectError(f"Connection failed for {DUMMY_OPENROUTER_KEY}"))

    client = OpenRouterClient(api_key=DUMMY_OPENROUTER_KEY, http_client=mock_http)

    async def _run():
        with pytest.raises(OpenRouterNetworkError) as exc_info:
            await client.create_chat_completion(messages=[{"role": "user", "content": "Hi"}])
        assert DUMMY_OPENROUTER_KEY not in str(exc_info.value)

    asyncio.run(_run())


# ==============================================================================
# 3. CONVERSATIONAL TOOL ORCHESTRATION WITH OPENROUTER
# ==============================================================================


def test_orchestrator_openrouter_single_tool_round(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify complete single-tool turn: User -> Tool Call -> Tool Result -> Final Answer."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.model = "nvidia/nemotron-3.5-lightning:free"

        # Round 1: Model requests lookup_community for Tallur
        step1_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_community",
                                    "arguments": json.dumps({"query": "Tallur"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Round 2: Model receives tool result and provides final answer
        step2_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tallur (PC11: 608720) was successfully located in Kundapura taluk, Udupi district.",
                    }
                }
            ]
        }

        mock_client.create_chat_completion = AsyncMock(side_effect=[step1_resp, step2_resp])

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Where is Tallur located?")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert res.is_fallback is False
        assert res.model_name == "nvidia/nemotron-3.5-lightning:free"
        assert "Kundapura taluk" in res.answer
        assert res.tools_executed == ["lookup_community"]
        assert len(res.tool_records) == 1

        rec = res.tool_records[0]
        assert rec.tool_name == "lookup_community"
        assert rec.arguments == {"query": "Tallur"}
        assert rec.success is True
        assert rec.execution_time_ms > 0
        assert _is_valid_telemetry_record(rec)

        # Verify second call passed tool response message with matching tool_call_id
        assert mock_client.create_chat_completion.call_count == 2
        second_call_args = mock_client.create_chat_completion.call_args_list[1]
        sent_messages = second_call_args[1]["messages"]
        tool_msg = [m for m in sent_messages if m.get("role") == "tool"][0]
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["name"] == "lookup_community"
        parsed_body = json.loads(tool_msg["content"])
        assert parsed_body["total_matches"] == 1
        assert parsed_body["matches"][0]["village_name"] == "Tallur"

    asyncio.run(_run())


def test_orchestrator_openrouter_sequential_tool_calls(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify sequential tool cycle: lookup -> assessment -> final explanation."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.model = "nvidia/nemotron-3.5-lightning:free"

        # Turn 1: Call lookup_community
        resp1 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_seq_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_community",
                                    "arguments": json.dumps({"query": "Tallur"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Turn 2: Call explain_preparedness_assessment
        resp2 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_seq_2",
                                "type": "function",
                                "function": {
                                    "name": "explain_preparedness_assessment",
                                    "arguments": json.dumps({"identifier": "608720"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        # Turn 3: Formulate final grounded answer
        resp3 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tallur has an authoritative AHP score of 74.34 and is classified in Band 4 Adequate.",
                    }
                }
            ]
        }

        mock_client.create_chat_completion = AsyncMock(side_effect=[resp1, resp2, resp3])

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Explain why Tallur is in Band 4.")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert res.tools_executed == ["lookup_community", "explain_preparedness_assessment"]
        assert len(res.tool_records) == 2
        assert all(r.success for r in res.tool_records)
        assert "74.34" in res.answer
        assert "Band 4 Adequate" in res.answer
        assert "explain_preparedness_assessment" in res.structured_evidence

    asyncio.run(_run())


def test_orchestrator_openrouter_invalid_tool_name(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify graceful handling when OpenRouter attempts to invoke an unregistered tool."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.model = "nvidia/nemotron-3.5-lightning:free"

        resp1 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "deploy_evacuation_drones",
                                    "arguments": json.dumps({"count": 5}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        resp2 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I do not have dispatch authority or operational drone capabilities.",
                    }
                }
            ]
        }

        mock_client.create_chat_completion = AsyncMock(side_effect=[resp1, resp2])

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Deploy 5 drones to Tallur")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert len(res.tool_records) == 1
        assert res.tool_records[0].tool_name == "deploy_evacuation_drones"
        assert res.tool_records[0].success is False
        assert res.tool_records[0].error_type == "UNKNOWN_TOOL"
        assert "I do not have dispatch authority" in res.answer

    asyncio.run(_run())


def test_orchestrator_openrouter_invalid_arguments(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify tool parameter validation errors are reported and captured in telemetry."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.model = "nvidia/nemotron-3.5-lightning:free"

        # compare_districts requires both district_a and district_b
        resp1 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_bad_args",
                                "type": "function",
                                "function": {
                                    "name": "compare_districts",
                                    "arguments": json.dumps({"district_a": "Dakshina Kannada"}),  # missing district_b
                                },
                            }
                        ],
                    }
                }
            ]
        }

        resp2 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Please specify two districts to compare.",
                    }
                }
            ]
        }

        mock_client.create_chat_completion = AsyncMock(side_effect=[resp1, resp2])

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Compare Dakshina Kannada")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.COMPLETED
        assert len(res.tool_records) == 1
        assert res.tool_records[0].success is False
        assert res.tool_records[0].error_type == "VALIDATION_ERROR"
        assert "Please specify two districts" in res.answer

    asyncio.run(_run())


def test_orchestrator_openrouter_max_tool_rounds_limit(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify orchestrator terminates infinite tool loops after MAX_TOOL_ROUNDS (4 rounds)."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.model = "nvidia/nemotron-3.5-lightning:free"

        repeated_tc = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_loop",
                                "type": "function",
                                "function": {
                                    "name": "lookup_community",
                                    "arguments": json.dumps({"query": "Tallur"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        final_synthesis = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tool budget reached. Synthesizing final briefing.",
                    }
                }
            ]
        }

        # 4 loop turns + 1 final synthesis turn
        mock_client.create_chat_completion = AsyncMock(
            side_effect=[repeated_tc, repeated_tc, repeated_tc, repeated_tc, final_synthesis]
        )

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
        assert mock_client.create_chat_completion.call_count == 5
        assert "Tool budget reached" in res.answer

    asyncio.run(_run())


# ==============================================================================
# 4. DETERMINISTIC FALLBACK ON OPENROUTER FAILURE
# ==============================================================================


def test_orchestrator_openrouter_fallback_when_unconfigured(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify deterministic fallback triggers immediately when OPENROUTER_API_KEY is missing."""
    async def _run():
        client_unconfigured = OpenRouterClient(api_key="")
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
        assert "OPENROUTER_API_KEY is not configured" in res.fallback_reason
        assert res.model_name == "deterministic-fallback"

        # Grounded Tallur metrics evaluated directly
        assert "explain_preparedness_assessment" in res.tools_executed
        assert "Tallur" in res.answer
        assert "74.34" in res.answer
        assert "BAND_4_ADEQUATE" in res.answer
        assert "AHP PRECEDENCE MANDATE" in res.scientific_precedence_note

    asyncio.run(_run())


def test_orchestrator_openrouter_fallback_on_429_rate_limit(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify rate limit error (HTTP 429) triggers deterministic fallback cleanly."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.create_chat_completion = AsyncMock(
            side_effect=OpenRouterRateLimitError("Rate limit exceeded (HTTP 429).")
        )

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(
            message="Assess Tallur",
            community_identifier="Tallur",
        )
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "OpenRouter API failure" in res.fallback_reason
        assert "74.34" in res.answer
        assert "BAND_4_ADEQUATE" in res.answer

    asyncio.run(_run())


def test_orchestrator_openrouter_fallback_on_network_timeout(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify request timeout triggers deterministic fallback cleanly."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.create_chat_completion = AsyncMock(
            side_effect=OpenRouterTimeoutError("OpenRouter request timed out after 30.0 seconds.")
        )

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Compare Dakshina Kannada and Udupi")
        res = await orchestrator.process_chat(req)

        assert res.conversation_status == ConversationStatus.FALLBACK
        assert res.is_fallback is True
        assert "compare_districts" in res.tools_executed
        assert "Dakshina Kannada" in res.answer
        assert "Udupi" in res.answer

    asyncio.run(_run())


# ==============================================================================
# 5. PROVIDER RESOLUTION & SCIENTIFIC SAFEGUARDS
# ==============================================================================


def test_get_default_llm_client_resolution():
    """Verify client factory resolution precedence."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "dummy-gemini"}, clear=False):
        c1 = get_default_llm_client()
        assert c1.provider_name == "gemini"

    with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "dummy-or"}, clear=False):
        c2 = get_default_llm_client()
        assert c2.provider_name == "openrouter"

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "dummy-or", "GEMINI_API_KEY": "dummy-gemini"}, clear=True):
        # OpenRouter takes precedence over Gemini when LLM_PROVIDER is not set
        c3 = get_default_llm_client()
        assert c3.provider_name == "openrouter"

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GEMINI_API_KEY": "dummy-gemini"}, clear=False):
        c4 = get_default_llm_client()
        assert c4.provider_name == "gemini"


def test_scientific_safeguard_precedence_in_response(
    shared_service: DecisionSupportService,
    shared_registry: ToolRegistry,
):
    """Verify scientific precedence disclaimer is present in every ChatResponse."""
    async def _run():
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.is_configured.return_value = True
        mock_client.provider_name = "openrouter"
        mock_client.model = "nvidia/nemotron-3.5-lightning:free"
        mock_client.create_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Operational brief.",
                        }
                    }
                ]
            }
        )

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        req = ChatRequest(message="Quick check")
        res = await orchestrator.process_chat(req)

        assert res.scientific_precedence_note == SCIENTIFIC_PRECEDENCE_DISCLAIMER
        assert "AHP PRECEDENCE MANDATE" in res.scientific_precedence_note
        assert "empirical tier surrogate" in res.scientific_precedence_note

    asyncio.run(_run())


def test_audit_and_guard_answer_reconciles_discrepancy():
    """Verify audit_and_guard_answer detects when LLM omits or contradicts score/band and reconciles it."""
    from src.assistant.llm.orchestrator import audit_and_guard_answer

    evidence = {
        "explain_preparedness_assessment": {
            "community_name": "Tallur",
            "deterministic_ahp": {"preparedness_score": 74.34},
            "operational_prioritization": {"priority_band": "BAND_4_ADEQUATE"},
        }
    }

    # Case 1: Model text correctly cites 74.34 and Band 4 Adequate -> leaves text unmodified
    good_text = "Tallur has an authoritative AHP score of 74.34 and is in Band 4 Adequate."
    assert audit_and_guard_answer(good_text, evidence) == good_text

    # Case 2: Model text incorrectly claims Band 1 and omits 74.34 -> appends ground-truth reconciliation
    bad_text = "Tallur is in critical condition and categorized in Band 1."
    reconciled = audit_and_guard_answer(bad_text, evidence)
    assert "Authoritative Metric Verification (Deterministic Ground Truth)" in reconciled
    assert "74.34 / 100.0" in reconciled
    assert "BAND_4_ADEQUATE" in reconciled
    assert "Tallur" in reconciled

    # Case 3: Empty evidence -> leaves text intact
    assert audit_and_guard_answer(bad_text, None) == bad_text
