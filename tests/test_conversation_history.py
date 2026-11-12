"""Unit and regression tests for safe conversation history management (Phase 4.3 hotfix).

Verifies:
- ChatMessage Pydantic schema strictly enforces 5,000-character ceiling without validation crash.
- Intelligent compaction of large prose messages at sentence boundaries.
- Preservation of valid JSON structure when compacting large tool dumps.
- Extraction and sanitization of history from Streamlit session state.
- Graceful handling of malformed, empty, or orphaned historical entries.
- Preservation of the current user question in full.
- Multi-turn conversational tool execution across local Ollama (Qwen) with compacted history.
- Zero Pydantic ValidationError crashes during multi-turn interactions.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    MessageRole,
    OllamaClient,
    SCIENTIFIC_PRECEDENCE_DISCLAIMER,
    ToolCallingOrchestrator,
    compact_message_content,
    prepare_conversation_history,
)
from src.assistant.tools.registry import get_default_registry
from src.assistant.views.chat_assistant import execute_assistant_turn


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    return DecisionSupportService()


@pytest.fixture(scope="module")
def shared_registry(shared_service: DecisionSupportService):
    return get_default_registry(service=shared_service)


# ==============================================================================
# 1. CHATMESSAAGE SCHEMA & BEFORE-VALIDATOR TESTS
# ==============================================================================


def test_chat_message_schema_bounds_large_content():
    """Verify ChatMessage before-validator bounds strings > 5,000 characters without raising ValidationError."""
    large_text = "Operational memo content. " * 300  # ~7,800 characters
    assert len(large_text) > 5000

    # Direct construction must NOT raise pydantic_core.ValidationError
    msg = ChatMessage(role=MessageRole.ASSISTANT, content=large_text)
    assert len(msg.content) <= 5000
    assert "[... prior response condensed to preserve conversational context ...]" in msg.content or "[... content truncated" in msg.content


def test_chat_message_schema_normal_content_unchanged():
    """Verify normal short messages are not truncated or modified."""
    text = "Explain why Tallur is in Band 4 Adequate."
    msg = ChatMessage(role=MessageRole.USER, content=text)
    assert msg.content == text


def test_chat_message_schema_empty_content_rejected():
    """Verify empty or purely whitespace content raises ValueError."""
    with pytest.raises(ValueError):
        ChatMessage(role=MessageRole.USER, content="")

    with pytest.raises(ValueError):
        ChatMessage(role=MessageRole.USER, content="   \n\t   ")


# ==============================================================================
# 2. MESSAGE COMPACTION & SYNTACTIC INTEGRITY TESTS
# ==============================================================================


def test_compact_message_content_prose_sentence_boundary():
    """Verify prose messages are truncated cleanly at sentence boundaries below 4,000 chars."""
    sentences = [f"Paragraph {i}: Detailed analysis of coastal infrastructure." for i in range(120)]
    long_prose = "\n\n".join(sentences)  # ~7,200 chars
    assert len(long_prose) > 5000

    compacted = compact_message_content(long_prose, max_chars=4000)
    assert len(compacted) <= 4000
    assert "[... prior response condensed to preserve conversational context ...]" in compacted
    assert compacted.startswith("Paragraph 0:")


def test_compact_message_content_json_validity():
    """Verify large raw JSON data is compacted into valid JSON syntax without blind truncation."""
    data = {
        "taluk": "Kundapura",
        "communities": [
            {"id": f"608{i:03d}", "name": f"Village_{i}", "scores": {"ahp": 70.0 + (i % 20)}}
            for i in range(150)
        ],
    }
    raw_json = json.dumps(data)
    assert len(raw_json) > 8000

    compacted = compact_message_content(raw_json, max_chars=4000)
    assert len(compacted) <= 4000
    # Must be parseable as valid JSON
    parsed = json.loads(compacted)
    assert parsed["taluk"] == "Kundapura"
    assert len(parsed["communities"]) <= 6  # 5 items + 1 omission indicator


def test_compact_message_content_markdown_code_fence_closure():
    """Verify unclosed markdown code fences resulting from truncation are cleanly closed."""
    content = "Here is the data:\n```json\n" + ("{'village': 'Tallur'},\n" * 400)
    assert len(content) > 5000

    compacted = compact_message_content(content, max_chars=4000)
    assert len(compacted) <= 4000
    # Code fences must be balanced (even count of triple backticks)
    assert compacted.count("```") % 2 == 0


# ==============================================================================
# 3. CONVERSATION HISTORY EXTRACTION & SANITIZATION
# ==============================================================================


def test_prepare_conversation_history_large_assistant_message():
    """Verify prepare_conversation_history converts large assistant turn into bounded ChatMessage."""
    raw_history = [
        {"role": "user", "content": "Explain Tallur."},
        {"role": "assistant", "content": "Comprehensive briefing memo. " * 350},  # ~10,500 chars
    ]

    history = prepare_conversation_history(raw_history, max_turns=10, max_content_chars=4000)
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[0].content == "Explain Tallur."
    assert history[1].role == MessageRole.ASSISTANT
    assert len(history[1].content) <= 4000
    assert history[1].content.endswith("[... prior response condensed to preserve conversational context ...]")


def test_prepare_conversation_history_multi_turn_accumulation():
    """Verify history bounding retains at most max_turns (default 10) most recent messages."""
    raw_history = []
    for turn in range(25):
        raw_history.append({"role": "user", "content": f"User question {turn}"})
        raw_history.append({"role": "assistant", "content": f"Assistant response {turn} with metrics."})

    history = prepare_conversation_history(raw_history, max_turns=10, max_content_chars=4000)
    assert len(history) == 10
    # Must be the most recent turns
    assert history[-1].content == "Assistant response 24 with metrics."
    assert history[-2].content == "User question 24"


def test_prepare_conversation_history_malformed_entries():
    """Verify malformed, empty, or unsupported historical records are skipped gracefully."""
    raw_history = [
        None,
        "not a dictionary",
        {"role": "unknown_role", "content": "Invalid role"},
        {"role": "unsupported_metadata", "content": "Orphaned metadata message"},
        {"role": "user", "content": ""},  # empty content
        {"role": "user", "content": "   "},  # whitespace only
        {"role": "user"},  # missing content key
        {"role": "user", "content": "Valid user prompt."},
        {"role": "assistant", "content": "Valid assistant response."},
    ]

    history = prepare_conversation_history(raw_history, max_turns=10)
    assert len(history) == 2
    assert history[0].content == "Valid user prompt."
    assert history[1].content == "Valid assistant response."


def test_normal_short_conversation_history_preserved():
    """Verify standard short conversational turns are preserved identically."""
    raw_history = [
        {"role": "user", "content": "What is Tallur's AHP score?"},
        {"role": "assistant", "content": "Tallur has an authoritative AHP score of 74.34 (Band 4 Adequate)."},
    ]

    history = prepare_conversation_history(raw_history, max_turns=10)
    assert len(history) == 2
    assert history[0].content == raw_history[0]["content"]
    assert history[1].content == raw_history[1]["content"]


# ==============================================================================
# 4. MULTI-TURN ORCHESTRATION WITH LOCAL QWEN (OLLAMA)
# ==============================================================================


def test_ollama_orchestrator_with_large_compacted_history(
    shared_service: DecisionSupportService,
    shared_registry,
):
    """Verify Ollama (Qwen) orchestrator turn cycle executes cleanly with large historical turns."""
    async def _run():
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.is_configured.return_value = True
        mock_client.model = "qwen2.5:0.5b-instruct"

        qwen_intent_json = json.dumps({
            "intents": ["lookup_community", "explain_preparedness_assessment"],
            "arguments": {"identifier": "Tallur"},
            "needs_clarification": False,
            "clarification_question": None,
        })
        synthesis_text = "Follow-up analysis formulated successfully."
        mock_client.generate = AsyncMock(side_effect=[qwen_intent_json, synthesis_text])

        orchestrator = ToolCallingOrchestrator(
            client=mock_client,
            registry=shared_registry,
            service=shared_service,
        )

        # 4 historical turns, including an 8,000-character previous memo
        raw_history = [
            {"role": "user", "content": "Initial query about Tallur."},
            {"role": "assistant", "content": "Comprehensive memo. " * 300},  # ~6,000 chars
            {"role": "user", "content": "What about shelters?"},
            {"role": "assistant", "content": "Shelter analysis details. " * 300},  # ~7,500 chars
        ]

        prepared_history = prepare_conversation_history(raw_history)
        req = ChatRequest(
            message="Now compare with Kundapura taluk average.",
            conversation_history=prepared_history,
        )

        res = await orchestrator.process_chat(req)
        assert res.conversation_status == ConversationStatus.COMPLETED
        assert res.is_fallback is False
        assert "Follow-up analysis" in res.answer

    asyncio.run(_run())


# ==============================================================================
# 5. STREAMLIT INTERFACE INTEGRATION & ERROR PROTECTION
# ==============================================================================


def test_execute_assistant_turn_with_huge_previous_response():
    """Verify Streamlit execute_assistant_turn does not crash when session state contains >5000 char message."""
    mock_api_client = MagicMock()
    mock_response = ChatResponse(
        answer="Turn 2 assistant answer.",
        tools_executed=[],
        tool_records=[],
        is_fallback=True,
        fallback_reason="DEMO",
        conversation_status=ConversationStatus.COMPLETED,
        model_name="deterministic-fallback",
        total_duration_ms=10.0,
        scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
    )
    mock_api_client.chat.return_value = mock_response

    session_state = {
        "assistant_messages": [
            {"role": "user", "content": "Turn 1 question"},
            # Previous assistant message with 7,500 characters (the exact crash condition)
            {"role": "assistant", "content": "Large briefing text. " * 350},
        ],
        "pending_prompt": None,
        "is_processing": False,
    }

    with patch("src.assistant.views.chat_assistant.st.session_state", session_state), \
         patch("src.assistant.views.chat_assistant.st.spinner"):
        # Must execute cleanly without raising pydantic_core.ValidationError
        execute_assistant_turn(
            client=mock_api_client,
            user_prompt="Turn 2 follow-up question",
        )

        # Assert Turn 2 was appended successfully
        messages = session_state["assistant_messages"]
        assert len(messages) == 4  # Turn 1 (user+asst) + Turn 2 (user+asst)
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Turn 2 assistant answer."

        # Verify the ChatRequest passed to client.chat had valid, bounded ChatMessage objects
        chat_arg = mock_api_client.chat.call_args[0][0]
        assert isinstance(chat_arg, ChatRequest)
        for h_msg in chat_arg.conversation_history:
            assert isinstance(h_msg, ChatMessage)
            assert len(h_msg.content) <= 5000


def test_execute_assistant_turn_rejects_excessive_user_prompt():
    """Verify user prompt exceeding 2,000 characters displays warning and does not crash."""
    mock_api_client = MagicMock()
    session_state = {
        "assistant_messages": [],
        "pending_prompt": None,
        "is_processing": False,
    }

    huge_user_prompt = "Query " * 500  # 3,000 chars

    with patch("src.assistant.views.chat_assistant.st.session_state", session_state), \
         patch("src.assistant.views.chat_assistant.st.warning") as mock_warning:
        execute_assistant_turn(client=mock_api_client, user_prompt=huge_user_prompt)

        assert mock_warning.called
        assert "exceeds maximum allowed length of 2,000 characters" in mock_warning.call_args[0][0]
        # Api client should NOT have been called with an invalid payload
        assert not mock_api_client.chat.called
        # No message appended to history
        assert len(session_state["assistant_messages"]) == 0


# ==============================================================================
# 6. STRUCTURED TOOL-RESULT INTEGRITY & LIFECYCLE TESTS
# ==============================================================================


def test_chat_message_role_tool_with_json_result():
    """Verify ChatMessage supports role='tool', retains tool_call_id/name, and compacts JSON safely."""
    huge_json_dict = {
        "status": "success",
        "district": "Udupi",
        "communities": [
            {
                "village_name": f"Village_{i}",
                "priority_band": "BAND_1_CRITICAL",
                "preparedness_score": 35.5 + (i * 0.5),
                "notes": "Extensive vulnerability narrative with deep infrastructure assessment. " * 5,
            }
            for i in range(50)
        ],
    }
    raw_json = json.dumps(huge_json_dict)
    assert len(raw_json) > 15000  # Well above 5000 chars

    msg = ChatMessage(
        role=MessageRole.TOOL,
        content=raw_json,
        name="filter_and_rank_communities",
        tool_call_id="call_999",
    )

    assert msg.role == MessageRole.TOOL
    assert msg.name == "filter_and_rank_communities"
    assert msg.tool_call_id == "call_999"
    assert len(msg.content) <= 5000

    # MUST parse successfully as valid JSON without JSONDecodeError
    parsed = json.loads(msg.content)
    assert isinstance(parsed, dict)
    assert "status" in parsed
    assert "communities" in parsed
    assert len(parsed["communities"]) > 0


def test_compact_json_extreme_payload_never_produces_invalid_syntax():
    """Verify extreme JSON structures (huge arrays, massive strings, deeply nested) always remain valid JSON."""
    extreme_payload = {
        f"long_key_{i}": "Deep textual analysis of cyclone vulnerability parameters. " * 50
        for i in range(25)
    }
    raw_json = json.dumps(extreme_payload)
    assert len(raw_json) > 50000

    compacted = compact_message_content(raw_json, max_chars=4000)
    assert len(compacted) <= 4000

    # MUST parse as valid JSON
    parsed = json.loads(compacted)
    assert isinstance(parsed, dict)


def test_http_client_lifecycle_across_asyncio_loops():
    """Verify OllamaClient safely handles execution across multiple independent asyncio event loops."""
    client = OllamaClient(base_url="http://127.0.0.1:11434")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "Loop test response."
    }

    # Simulate Turn 1 in loop 1
    async def turn1():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await client.generate(prompt="Turn 1")
            assert res == "Loop test response."
        await client.close()

    asyncio.run(turn1())
    assert client._http_client is None

    # Simulate Turn 2 in loop 2 without error
    async def turn2():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await client.generate(prompt="Turn 2")
            assert res == "Loop test response."
        await client.close()

    asyncio.run(turn2())
    assert client._http_client is None

    # Simulate Turn 3 in loop 3 without error
    async def turn3():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await client.generate(prompt="Turn 3")
            assert res == "Loop test response."
        await client.close()

    asyncio.run(turn3())
    assert client._http_client is None


def test_http_client_reused_within_same_loop():
    """Verify HTTP client instance is reused across multiple calls within the same event loop."""
    async def _run():
        client = OllamaClient(base_url="http://127.0.0.1:11434")
        c1 = client._get_http_client()
        c2 = client._get_http_client()
        assert c1 is c2  # Exact same instance reused within loop
        await client.close()
        assert client._http_client is None

    asyncio.run(_run())


def test_direct_mode_chat_cleans_up_client_in_finally():
    """Verify DecisionSupportApiClient.chat cleans up orchestrator client in finally block."""
    from src.assistant.api_client import DecisionSupportApiClient

    api_client = DecisionSupportApiClient(mode="direct")
    req = ChatRequest(message="Ping test query")

    # In mock orchestrator, verify client.close is called
    mock_orchestrator = MagicMock()
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    mock_orchestrator.client = mock_client
    mock_orchestrator.process_chat = AsyncMock(
        return_value=ChatResponse(
            answer="Direct test response",
            tools_executed=[],
            tool_records=[],
            is_fallback=True,
            fallback_reason="TEST",
            conversation_status=ConversationStatus.COMPLETED,
            model_name="test-model",
            total_duration_ms=5.0,
            scientific_precedence_note=SCIENTIFIC_PRECEDENCE_DISCLAIMER,
        )
    )

    api_client._orchestrator = mock_orchestrator
    resp = api_client.chat(req)

    assert resp.answer == "Direct test response"
    assert mock_client.close.called
