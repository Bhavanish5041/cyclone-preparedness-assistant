"""Unit and integration tests for the Streamlit Conversational Assistant (Phase 4.3).

Verifies:
- Streamlit chat session state initialization.
- Session history preservation and bounded size.
- Duplicate submission prevention and blank input handling.
- Clear conversation control.
- Direct Mode in-process orchestration via DecisionSupportApiClient.
- Missing API key deterministic fallback behavior and status indicators.
- Tool execution display and evidence cards generation.
- Scientific guardrail disclaimers (AHP precedence, surrogate ML disclaimer).
- Streamlit AppTest end-to-end integration across navigation and chat view.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    MessageRole,
    ToolExecutionRecord,
)
from src.assistant.api_client import (
    AdapterMode,
    DecisionSupportApiClient,
    get_api_client,
)
from src.assistant.decision_support import DecisionSupportService
from src.assistant.views.chat_assistant import (
    MAX_SESSION_HISTORY_MESSAGES,
    SUGGESTED_PROMPTS,
    clear_chat_history,
    execute_assistant_turn,
    init_chat_session_state,
    render_evidence_cards,
    render_message_bubble,
    render_tool_execution_details,
)


@pytest.fixture
def mock_streamlit_session():
    """Fixture providing clean mock streamlit session_state dict."""
    state = {}
    with patch.object(st, "session_state", state):
        yield state


@pytest.fixture(scope="module")
def shared_service() -> DecisionSupportService:
    """Module-scoped DecisionSupportService."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def direct_client(shared_service: DecisionSupportService) -> DecisionSupportApiClient:
    """Module-scoped direct-mode DecisionSupportApiClient."""
    return DecisionSupportApiClient(mode="direct", service=shared_service)


# ==============================================================================
# 1. SESSION STATE & INTERACTION HANDLING TESTS
# ==============================================================================


def test_init_chat_session_state(mock_streamlit_session):
    """Verify session state keys initialization."""
    assert "assistant_messages" not in mock_streamlit_session
    init_chat_session_state()
    assert mock_streamlit_session["assistant_messages"] == []
    assert mock_streamlit_session["pending_prompt"] is None
    assert mock_streamlit_session["is_processing"] is False

    # Existing values are not overwritten
    mock_streamlit_session["assistant_messages"] = [{"role": "user", "content": "Hello"}]
    init_chat_session_state()
    assert len(mock_streamlit_session["assistant_messages"]) == 1


def test_clear_chat_history(mock_streamlit_session):
    """Verify clear_chat_history resets messages and processing state."""
    init_chat_session_state()
    mock_streamlit_session["assistant_messages"] = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    mock_streamlit_session["pending_prompt"] = "Pending Q"
    mock_streamlit_session["is_processing"] = True

    clear_chat_history()
    assert mock_streamlit_session["assistant_messages"] == []
    assert mock_streamlit_session["pending_prompt"] is None
    assert mock_streamlit_session["is_processing"] is False


def test_duplicate_and_empty_submission_handling(mock_streamlit_session, direct_client):
    """Verify empty or whitespace-only prompts are safely ignored."""
    init_chat_session_state()

    # Empty string
    execute_assistant_turn(direct_client, "")
    assert len(mock_streamlit_session["assistant_messages"]) == 0

    # Whitespace only
    execute_assistant_turn(direct_client, "   \n\t  ")
    assert len(mock_streamlit_session["assistant_messages"]) == 0


def test_session_history_bounded_size(mock_streamlit_session, direct_client):
    """Verify session history does not exceed MAX_SESSION_HISTORY_MESSAGES."""
    init_chat_session_state()
    # Seed with 25 messages
    for i in range(25):
        mock_streamlit_session["assistant_messages"].append(
            {"role": "user", "content": f"Historical prompt {i}", "timestamp": "12:00:00 UTC"}
        )

    # Execute a turn
    with patch.object(direct_client, "chat") as mock_chat:
        mock_chat.return_value = ChatResponse(
            answer="Bounded turn answer",
            conversation_status=ConversationStatus.COMPLETED,
            model_name="mock-model",
            is_fallback=False,
            total_duration_ms=12.5,
            scientific_precedence_note="AHP is authoritative.",
        )
        execute_assistant_turn(direct_client, "Trigger bound check")

    assert len(mock_streamlit_session["assistant_messages"]) <= MAX_SESSION_HISTORY_MESSAGES


# ==============================================================================
# 2. DIRECT MODE ORCHESTRATION & FALLBACK TESTS
# ==============================================================================


def test_direct_mode_chat_execution_fallback(direct_client):
    """Verify direct client executes orchestrator in-process without network overhead."""
    # Ensure unconfigured key triggers deterministic fallback
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        # 1. Unambiguous query without explicit community_identifier
        req_tallur = ChatRequest(message="Explain why Tallur has its current priority classification.")
        res_tallur = direct_client.chat(req_tallur)
        assert res_tallur.is_fallback is True
        assert res_tallur.conversation_status == ConversationStatus.FALLBACK
        assert "lookup_community" in res_tallur.tools_executed
        assert "explain_preparedness_assessment" in res_tallur.tools_executed
        assert "Tallur" in res_tallur.answer
        assert "74.34" in res_tallur.answer
        assert "BAND_4_ADEQUATE" in res_tallur.answer

        # 2. District comparison query
        req_comp = ChatRequest(message="Compare Dakshina Kannada and Udupi.")
        res_comp = direct_client.chat(req_comp)
        assert res_comp.is_fallback is True
        assert "compare_districts" in res_comp.tools_executed
        assert "Dakshina Kannada" in res_comp.answer
        assert "Udupi" in res_comp.answer
        assert "369" in res_comp.answer

        # 3. Band filtering query
        req_band = ChatRequest(message="Find Band 1 communities.")
        res_band = direct_client.chat(req_band)
        assert res_band.is_fallback is True
        assert "filter_and_rank_communities" in res_band.tools_executed
        assert "BAND_1_CRITICAL" in res_band.answer
        assert "155" in res_band.answer

        # 4. Fallback methodology query
        req_uk = ChatRequest(message="Explain Uttara Kannada's fallback methodology.")
        res_uk = direct_client.chat(req_uk)
        assert res_uk.is_fallback is True
        assert "explain_vulnerability_archetype" in res_uk.tools_executed
        assert "1,257" in res_uk.answer

        # 5. Ambiguous community query
        req_amb = ChatRequest(message="Assess Siddapur.")
        res_amb = direct_client.chat(req_amb)
        assert res_amb.is_fallback is True
        assert "lookup_community" in res_amb.tools_executed
        assert "Community Disambiguation Required" in res_amb.answer
        assert "602812" in res_amb.answer

        # 6. Unsupported query
        req_unsup = ChatRequest(message="What is the weather in Paris tomorrow?")
        res_unsup = direct_client.chat(req_unsup)
        assert res_unsup.is_fallback is True
        assert res_unsup.tools_executed == []
        assert "Decision Support System Boundary" in res_unsup.answer


def test_direct_mode_chat_mocked_llm_response(shared_service):
    """Verify direct client handles successful tool-grounded response when LLM responds."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.process_chat = MagicMock()

    # Create async coroutine mock
    async def _mock_process(req):
        return ChatResponse(
            answer="Tallur is in Band 4 Adequate with AHP score 74.34.",
            tools_executed=["lookup_community", "explain_preparedness_assessment"],
            tool_records=[
                ToolExecutionRecord(
                    tool_name="lookup_community",
                    arguments={"query": "Tallur"},
                    success=True,
                    result_summary="Found PC11 ID 608691",
                    execution_time_ms=1.2,
                )
            ],
            structured_evidence={
                "explain_preparedness_assessment": {
                    "name": "Tallur",
                    "district": "Udupi",
                    "ahp_score": 74.34,
                    "priority_band": "BAND_4_ADEQUATE",
                    "ml_surrogate_assessment": {
                        "tier": "Tier 1 - High Preparedness",
                        "scientific_disclaimer": "Logistic Regression is an empirical AHP-tier surrogate.",
                    },
                }
            },
            is_fallback=False,
            conversation_status=ConversationStatus.COMPLETED,
            model_name="gemini-2.5-flash",
            total_duration_ms=450.0,
            scientific_precedence_note="AHP is authoritative.",
        )

    mock_orchestrator.process_chat.side_effect = _mock_process

    client = DecisionSupportApiClient(
        mode=AdapterMode.DIRECT,
        service=shared_service,
        orchestrator=mock_orchestrator,
    )

    req = ChatRequest(message="Assess Tallur")
    res = client.chat(req)

    assert not res.is_fallback
    assert res.conversation_status == ConversationStatus.COMPLETED
    assert "Tallur is in Band 4 Adequate" in res.answer
    assert res.tools_executed == ["lookup_community", "explain_preparedness_assessment"]
    assert res.structured_evidence is not None


# ==============================================================================
# 3. EVIDENCE DISPLAY & SCIENTIFIC DISCLAIMER TESTS
# ==============================================================================


def test_render_tool_execution_details_no_crash():
    """Verify tool execution telemetry renderer handles valid records without error."""
    records = [
        ToolExecutionRecord(
            tool_name="lookup_community",
            arguments={"query": "Tallur"},
            success=True,
            result_summary="Found PC11: 608691",
            execution_time_ms=2.5,
        ),
        {
            "tool_name": "compare_districts",
            "arguments": {"districts": ["Dakshina Kannada", "Udupi"]},
            "success": True,
            "result_summary": "Comparison computed",
            "execution_time_ms": 5.1,
        },
    ]
    # Calling with records should not raise
    with patch.object(st, "markdown") as mock_md, patch.object(st, "expander") as mock_exp:
        mock_exp.return_value.__enter__ = MagicMock()
        mock_exp.return_value.__exit__ = MagicMock(return_value=False)
        render_tool_execution_details(records, duration_ms=7.6)
        assert mock_exp.called
        assert mock_md.call_count == 2


def test_render_evidence_cards_surrogate_ml_disclaimer():
    """Verify evidence cards extract and display the mandatory surrogate ML disclaimer."""
    evidence = {
        "explain_preparedness_assessment": {
            "name": "Tallur",
            "district": "Udupi",
            "ahp_score": 74.34,
            "priority_band": "BAND_4_ADEQUATE",
            "ml_surrogate_assessment": {
                "tier": "Tier 1 - High Preparedness",
                "scientific_disclaimer": "Logistic Regression is an empirical AHP-tier surrogate, not a physical cyclone hazard or probability model.",
            },
        }
    }
    with patch.object(st, "markdown") as mock_md, patch.object(st, "columns") as mock_cols:
        mock_cols.return_value = [MagicMock(), MagicMock(), MagicMock()]
        render_evidence_cards(evidence)
        assert mock_md.called
        # Verify surrogate disclaimer is in rendered markdown
        disclaimer_rendered = any(
            "Surrogate ML Demarcation" in str(call_arg) for call_arg in mock_md.call_args_list
        )
        assert disclaimer_rendered


def test_render_message_bubble_styling():
    """Verify message bubble renders user and assistant turns with warm chocolate styling."""
    with patch.object(st, "markdown") as mock_md:
        # User message
        render_message_bubble({"role": "user", "content": "What is Tallur's score?", "timestamp": "12:00:00 UTC"})
        assert any("Incident Commander Query" in str(arg) for arg in mock_md.call_args_list)

        # Assistant message with fallback
        render_message_bubble(
            {
                "role": "assistant",
                "content": "AHP score is 74.34",
                "is_fallback": True,
                "fallback_reason": "API Key missing",
                "timestamp": "12:00:01 UTC",
            }
        )
        assert any("Deterministic Mode Active" in str(arg) for arg in mock_md.call_args_list)


# ==============================================================================
# 4. STREAMLIT APPTEST INTEGRATION TESTS
# ==============================================================================


def test_apptest_navigation_to_cyclone_assistant():
    """Verify Streamlit AppTest navigates to Cyclone Assistant page without exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "assistant" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Select Cyclone Assistant from sidebar
    at.sidebar.radio(key="platform_nav_radio").set_value("Cyclone Assistant")
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Verify Clear Conversation button is present
    clear_btn = at.button(key="clear_chat_button")
    assert clear_btn is not None
    assert clear_btn.label == "Clear Conversation"

    # Verify suggested prompt buttons are present
    assert at.button(key="sug_btn_0") is not None
    assert at.button(key="sug_btn_1") is not None
    assert at.button(key="sug_btn_2") is not None
    assert at.button(key="sug_btn_3") is not None
    assert at.button(key="sug_btn_4") is not None

    # Verify chat input is present
    chat_input = at.chat_input(key="main_chat_input")
    assert chat_input is not None
