"""Conversational AI Decision Support Assistant Interface for Streamlit.

Warm chocolate editorial styling, strict scientific guardrails, and deterministic tool transparency.
Supports:
- Direct Mode in-process orchestration (no external HTTP required).
- Conversational message history stored in Streamlit session state.
- Suggested question prompts for quick-fire operational inquiries.
- Tool execution transparency with timing, arguments, and status.
- Authoritative evidence cards (AHP score, priority band, ML surrogate disclaimer).
- Deterministic fallback status indication when GEMINI_API_KEY is not configured or unavailable.
- Strictly zero emojis, zero em dashes, and zero invented operational facts.
"""

from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional

import streamlit as st

from src.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    ToolExecutionRecord,
    prepare_conversation_history,
)
from src.assistant.api_client import DecisionSupportApiClient
from src.assistant.theme import (
    BAND_COLORS,
    THEME_COLORS,
    render_disclaimer,
    render_header,
    render_kpi_card,
)

# Suggested standard operational prompts
SUGGESTED_PROMPTS = [
    "Explain why Tallur has its current priority classification.",
    "Compare Dakshina Kannada and Udupi.",
    "Find Band 1 communities.",
    "Explain Uttara Kannada's fallback methodology.",
    "Assess a community's healthcare accessibility.",
]

MAX_SESSION_HISTORY_MESSAGES = 20
MAX_REQUEST_HISTORY_TURNS = 10


def init_chat_session_state() -> None:
    """Initialize session state keys for conversational assistant."""
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []
    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False


def clear_chat_history() -> None:
    """Reset conversational message history in session state."""
    st.session_state["assistant_messages"] = []
    st.session_state["pending_prompt"] = None
    st.session_state["is_processing"] = False


logger = logging.getLogger(__name__)


def _is_valid_telemetry_record(rec: Any) -> bool:
    """Validate whether an item is a supported ToolExecutionRecord or dictionary record."""
    if rec is None or isinstance(rec, (bool, int, float, str, bytes, list, set, tuple)):
        return False
    if isinstance(rec, ToolExecutionRecord):
        return True
    if isinstance(rec, dict):
        return True
    # Support duck-typed records only if they have the mandatory tool_name attribute
    return hasattr(rec, "tool_name")


def _extract_telemetry_field(rec: Any, field_name: str, default: Any = None) -> Any:
    """Safely extract a telemetry attribute or dict key without invoking non-existent methods."""
    if isinstance(rec, dict):
        return rec.get(field_name, default)
    val = getattr(rec, field_name, None)
    return val if val is not None else default


def render_tool_execution_details(tool_records: List[Any], duration_ms: Optional[float] = None) -> None:
    """Render collapsible tool execution telemetry panel."""
    if not tool_records:
        return

    valid_records = [r for r in tool_records if _is_valid_telemetry_record(r)]
    if not valid_records:
        return

    count = len(valid_records)
    dur_str = f" in {duration_ms:.1f} ms" if duration_ms is not None else ""
    summary_label = f"Tool Execution Telemetry: {count} tool(s) invoked{dur_str}"

    with st.expander(summary_label, expanded=False):
        for idx, rec in enumerate(valid_records, 1):
            name = _extract_telemetry_field(rec, "tool_name", "unknown_tool")
            success = _extract_telemetry_field(rec, "success", True)
            if success is None:
                success = True
            args = _extract_telemetry_field(rec, "arguments", {})
            summary = _extract_telemetry_field(rec, "result_summary", "")
            rec_dur = _extract_telemetry_field(rec, "execution_time_ms", 0.0)
            if rec_dur is None:
                rec_dur = 0.0
            error_msg = _extract_telemetry_field(rec, "error", None)

            # Fallback for summary if missing
            if not summary:
                if error_msg:
                    summary = f"Error: {error_msg}"
                else:
                    summary = f"Status: {'ok' if success else 'failed'}"

            status_badge = (
                "<span style='color:#7A8B63; font-weight:600;'>SUCCESS</span>"
                if success
                else "<span style='color:#C95D4A; font-weight:600;'>FAILED</span>"
            )

            dur_display = f"{rec_dur:.1f} ms" if isinstance(rec_dur, (int, float)) else str(rec_dur)

            st.markdown(
                f"""
                <div style="background:#2B211B; border:1px solid #514033; border-radius:3px; padding:0.6rem 0.8rem; margin-bottom:0.5rem; font-size:0.82rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span style="color:#D6B28C; font-weight:600;">{idx}. {name}</span>
                        <span>{status_badge} <span style="color:#918170;">({dur_display})</span></span>
                    </div>
                    <div style="color:#BDAE9C; margin-bottom:0.2rem;"><b>Arguments:</b> <code>{args}</code></div>
                    <div style="color:#918170; font-style:italic;">{summary}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_evidence_cards(structured_evidence: Optional[Dict[str, Any]]) -> None:
    """Render authoritative evidence cards extracted from tool executions."""
    if not structured_evidence:
        return

    # Check for explain_preparedness_assessment tool evidence
    assessment = structured_evidence.get("explain_preparedness_assessment")
    if assessment and isinstance(assessment, dict):
        unit_name = assessment.get("community_name") or assessment.get("name", "Unknown Unit")
        district = assessment.get("district_name") or assessment.get("district", "Unknown District")
        ahp_obj = assessment.get("deterministic_ahp", {})
        ahp_score = (
            ahp_obj.get("preparedness_score")
            if isinstance(ahp_obj, dict) and "preparedness_score" in ahp_obj
            else assessment.get("ahp_score")
        )
        prio_obj = assessment.get("operational_prioritization", {})
        band = (
            prio_obj.get("priority_band")
            if isinstance(prio_obj, dict) and "priority_band" in prio_obj
            else assessment.get("priority_band", "BAND_3_MODERATE")
        )

        band_meta = BAND_COLORS.get(band, {})
        band_name = band_meta.get("name", band)
        band_color = band_meta.get("hex", "#D6B28C")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                render_kpi_card(
                    "Community Assessment",
                    unit_name,
                    caption=f"District: {district}",
                    color="#F3E9DC",
                    badge_label="Target Unit",
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_kpi_card(
                    "Authoritative AHP Score",
                    f"{ahp_score:.2f}" if isinstance(ahp_score, (int, float)) else "N/A",
                    caption="Immutable Multi-Criteria Index (0-100)",
                    color="#C18B62",
                    badge_label="Authoritative",
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                render_kpi_card(
                    "Priority Classification",
                    band_meta.get("short_label", band),
                    caption=band_name,
                    color=band_color,
                    badge_label="Intervention Band",
                ),
                unsafe_allow_html=True,
            )

        # Scientific disclaimer on ML surrogate
        ml_data = assessment.get("surrogate_ml") or assessment.get("ml_surrogate_assessment")
        if ml_data and isinstance(ml_data, dict):
            ml_tier = ml_data.get("predicted_preparedness_tier") or ml_data.get("tier", "N/A")
            ml_disc = ml_data.get(
                "scientific_disclaimer",
                "Logistic Regression is an empirical AHP-tier surrogate, not a physical cyclone hazard or probability model.",
            )
            st.markdown(
                f"""
                <div style="background:#2B211B; border:1px solid #514033; border-left:3px solid #8D9A78; border-radius:3px; padding:0.5rem 0.8rem; margin-top:0.4rem; font-size:0.80rem; color:#BDAE9C;">
                    <b style="color:#D6B28C;">Surrogate ML Demarcation:</b> Model Tier: <code>{ml_tier}</code>. {ml_disc}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_message_bubble(msg: Dict[str, Any]) -> None:
    """Render a single conversational turn with warm chocolate styling."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    is_fallback = msg.get("is_fallback", False)
    fallback_reason = msg.get("fallback_reason")
    tool_records = msg.get("tool_records", [])
    structured_evidence = msg.get("structured_evidence")
    precedence_note = msg.get("scientific_precedence_note")
    duration_ms = msg.get("total_duration_ms")
    timestamp = msg.get("timestamp", "")

    if role == "user":
        st.markdown(
            f"""
            <div style="background-color: #342820; border: 1px solid #514033; border-radius: 4px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                    <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #918170; font-weight: 600;">Incident Commander Query</span>
                    <span style="font-size: 0.72rem; color: #918170;">{timestamp}</span>
                </div>
                <div style="color: #F3E9DC; font-size: 0.95rem; line-height: 1.55;">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Assistant Response Container
        st.markdown(
            f"""
            <div style="background-color: #2B211B; border: 1px solid #514033; border-left: 3px solid #C18B62; border-radius: 4px; padding: 1.0rem 1.25rem; margin-bottom: 0.8rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #C18B62; font-weight: 600;">Preparedness Intelligence Assistant</span>
                    <span style="font-size: 0.72rem; color: #918170;">{timestamp}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Deterministic fallback banner if applicable
        if is_fallback:
            reason_text = f": {fallback_reason}" if fallback_reason else ""
            st.markdown(
                f"""
                <div style="background-color: #342820; border: 1px solid #D48B47; border-radius: 3px; padding: 0.45rem 0.75rem; margin-bottom: 0.75rem; font-size: 0.78rem; color: #BDAE9C;">
                    <b style="color: #D48B47;">Deterministic Mode Active</b>{reason_text}. Evaluated directly via local decision support matrix.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Main assistant answer text (supports markdown formatting)
        st.markdown(content)

        # Structured evidence cards (if present)
        if structured_evidence:
            try:
                render_evidence_cards(structured_evidence)
            except Exception as ev_err:
                logger.warning("Failed to render structured evidence: %s", ev_err)

        # Tool execution details (if tools were executed)
        if tool_records:
            try:
                render_tool_execution_details(tool_records, duration_ms)
            except Exception as tel_err:
                logger.warning("Failed to render tool execution telemetry: %s", tel_err)

        # Scientific precedence disclaimer
        if precedence_note:
            st.markdown(
                f"""
                <div style="border-top: 1px solid #403127; margin-top: 0.75rem; padding-top: 0.45rem; font-size: 0.73rem; color: #918170; line-height: 1.4;">
                    <b>Scientific Governance:</b> {precedence_note}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


def execute_assistant_turn(
    client: DecisionSupportApiClient,
    user_prompt: str,
    selected_community: Optional[str] = None,
) -> None:
    """Execute a conversational interaction turn and record results in session state."""
    clean_prompt = user_prompt.strip()
    if not clean_prompt:
        return

    if len(clean_prompt) > 2000:
        st.warning(
            f"Operational prompt exceeds maximum allowed length of 2,000 characters "
            f"(current: {len(clean_prompt):,} characters). Please shorten your inquiry to avoid context truncation."
        )
        return

    now_iso = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # 1. Append user prompt to session state
    st.session_state["assistant_messages"].append(
        {
            "role": "user",
            "content": clean_prompt,
            "timestamp": now_iso,
        }
    )

    # 2. Build safe, bounded conversation history for request payload
    prior_messages = prepare_conversation_history(
        st.session_state["assistant_messages"][-MAX_REQUEST_HISTORY_TURNS - 1 : -1],
        max_turns=MAX_REQUEST_HISTORY_TURNS,
        max_content_chars=4000,
    )

    try:
        request_payload = ChatRequest(
            message=clean_prompt,
            conversation_history=prior_messages,
            community_identifier=selected_community,
        )
    except Exception as req_err:
        logger.error("Failed to construct ChatRequest payload: %s", req_err)
        err_turn = {
            "role": "assistant",
            "content": f"Decision support assistant could not prepare request payload: {str(req_err)}",
            "is_fallback": True,
            "fallback_reason": type(req_err).__name__,
            "tools_executed": [],
            "tool_records": [],
            "structured_evidence": None,
            "scientific_precedence_note": "AHP deterministic metrics remain authoritative.",
            "model_name": "error-handler",
            "total_duration_ms": 0.0,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        }
        st.session_state["assistant_messages"].append(err_turn)
        return

    # 3. Invoke client adapter (Direct Mode in-process or REST mode)
    with st.spinner("Analyzing preparedness metrics with tool grounding..."):
        try:
            response: ChatResponse = client.chat(request_payload)

            # Record assistant turn
            assistant_turn = {
                "role": "assistant",
                "content": response.answer,
                "is_fallback": response.is_fallback,
                "fallback_reason": response.fallback_reason,
                "tools_executed": response.tools_executed,
                "tool_records": response.tool_records,
                "structured_evidence": response.structured_evidence,
                "scientific_precedence_note": response.scientific_precedence_note,
                "model_name": response.model_name,
                "total_duration_ms": response.total_duration_ms,
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            }
            st.session_state["assistant_messages"].append(assistant_turn)

        except Exception as err:
            err_turn = {
                "role": "assistant",
                "content": f"Decision support assistant encountered an execution error: {str(err)}",
                "is_fallback": True,
                "fallback_reason": type(err).__name__,
                "tools_executed": [],
                "tool_records": [],
                "structured_evidence": None,
                "scientific_precedence_note": "AHP deterministic metrics remain authoritative.",
                "model_name": "error-handler",
                "total_duration_ms": 0.0,
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            }
            st.session_state["assistant_messages"].append(err_turn)

    # Bounded session history to prevent browser memory bloat
    if len(st.session_state["assistant_messages"]) > MAX_SESSION_HISTORY_MESSAGES:
        st.session_state["assistant_messages"] = st.session_state["assistant_messages"][-MAX_SESSION_HISTORY_MESSAGES:]


def render_assistant_page(client: DecisionSupportApiClient) -> None:
    """Render the Conversational AI Decision Support Assistant page."""
    init_chat_session_state()

    # Header masthead
    st.markdown(
        render_header(
            title="AI Decision Support Assistant",
            subtitle="Conversational operational decision support grounded in deterministic AHP metrics, Census baselines, and multi-sector advisories.",
            badge="Direct Mode" if client.mode.value == "direct" else "REST Mode",
        ),
        unsafe_allow_html=True,
    )

    # System status telemetry banner
    openrouter_key_set = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY", "").strip())
    explicit_provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if explicit_provider == "gemini":
        active_provider = "Gemini"
        active_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        is_active_configured = gemini_key_set
        missing_key_name = "GEMINI_API_KEY"
    elif explicit_provider == "openrouter":
        active_provider = "OpenRouter"
        active_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        is_active_configured = openrouter_key_set
        missing_key_name = "OPENROUTER_API_KEY"
    elif openrouter_key_set:
        active_provider = "OpenRouter"
        active_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        is_active_configured = True
        missing_key_name = ""
    elif gemini_key_set:
        active_provider = "Gemini"
        active_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        is_active_configured = True
        missing_key_name = ""
    else:
        active_provider = "OpenRouter"
        active_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        is_active_configured = False
        missing_key_name = "OPENROUTER_API_KEY"

    status_col1, status_col2, status_col3 = st.columns([2, 1, 1])
    with status_col1:
        if is_active_configured:
            st.markdown(
                f"""
                <div style="background:#2B211B; border:1px solid #514033; border-radius:3px; padding:0.45rem 0.75rem; font-size:0.78rem; color:#BDAE9C;">
                    <b>LLM Engine:</b> {active_provider} configured (<code>{active_model}</code>). Tool calling active.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background:#2B211B; border:1px solid #D48B47; border-radius:3px; padding:0.45rem 0.75rem; font-size:0.78rem; color:#BDAE9C;">
                    <b style="color: #D48B47;">Deterministic Mode:</b> {missing_key_name} unconfigured. Operating via local decision-support matrix.
                </div>
                """,
                unsafe_allow_html=True,
            )
    with status_col2:
        st.markdown(
            f"""
            <div style="background:#2B211B; border:1px solid #514033; border-radius:3px; padding:0.45rem 0.75rem; font-size:0.78rem; color:#BDAE9C; text-align:center;">
                <b>Adapter:</b> <code>{client.mode.value.upper()}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_col3:
        if st.button("Clear Conversation", use_container_width=True, key="clear_chat_button"):
            clear_chat_history()
            st.rerun()

    # Suggested Prompts Bar
    st.markdown(
        """
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#918170; margin-top:0.8rem; margin-bottom:0.4rem; font-weight:600;">
            Suggested Operational Inquiries
        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt_cols = st.columns(len(SUGGESTED_PROMPTS))
    selected_suggested_prompt: Optional[str] = None
    for idx, prompt_text in enumerate(SUGGESTED_PROMPTS):
        with prompt_cols[idx]:
            # Clean label for button
            btn_label = prompt_text.split(".")[0]
            if len(btn_label) > 28:
                btn_label = btn_label[:25] + "..."
            if st.button(btn_label, key=f"sug_btn_{idx}", help=prompt_text, use_container_width=True):
                selected_suggested_prompt = prompt_text

    st.markdown("---")

    # Display conversational message history
    if not st.session_state["assistant_messages"]:
        st.markdown(
            """
            <div style="background:#2B211B; border:1px solid #514033; border-radius:4px; padding:1.5rem; text-align:center; margin:1.5rem 0;">
                <div style="font-family:'DM Serif Display', Georgia, serif; font-size:1.35rem; color:#F3E9DC; margin-bottom:0.4rem;">
                    Preparedness Intelligence Assistant Ready
                </div>
                <div style="color:#918170; font-size:0.88rem; max-width:42rem; margin:0 auto; line-height:1.6;">
                    Ask questions regarding community preparedness classifications, comparative district readiness,
                    infrastructure deficits, or vulnerability archetypes across coastal Karnataka.
                    All responses are strictly grounded in deterministic multi-criteria evaluations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state["assistant_messages"]:
            render_message_bubble(msg)

    # Process prompt from suggested button if clicked
    prompt_to_process = selected_suggested_prompt

    # Chat input widget
    user_input = st.chat_input(
        placeholder="Enter operational inquiry (e.g., assess Tallur, compare districts, list Band 1 units)...",
        key="main_chat_input",
    )

    if user_input:
        prompt_to_process = user_input

    # Execute turn if prompt available and not already processing
    if prompt_to_process and not st.session_state["is_processing"]:
        st.session_state["is_processing"] = True
        try:
            execute_assistant_turn(client, prompt_to_process)
        finally:
            st.session_state["is_processing"] = False
        st.rerun()
