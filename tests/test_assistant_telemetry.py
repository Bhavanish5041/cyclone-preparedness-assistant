"""Targeted regression tests for Cyclone Assistant tool telemetry rendering and AHP sub-score presentation.

Phase 4.2 / UI Telemetry Hardening:
- Verifies real ToolExecutionRecord instance rendering without AttributeError.
- Verifies record containing explicit result_summary.
- Verifies records with optional fields missing, None, or defaulted.
- Verifies multiple sequential tool execution records with correct indexing and counts.
- Verifies empty telemetry list ([]) and None gracefully handled.
- Verifies dictionary-format telemetry records compatibility.
- Verifies malformed telemetry entries do not crash or suppress the assistant response in render_message_bubble.
- Verifies AHP domain sub-scores presentation using canonical score_* keys.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from src.assistant.llm.prompts import format_community_assessment_response
from src.assistant.llm.schemas import ToolExecutionRecord
from src.assistant.views.chat_assistant import (
    _extract_telemetry_field,
    _is_valid_telemetry_record,
    render_message_bubble,
    render_tool_execution_details,
)


class FakeExpanderContext:
    """Non-suppressing context manager mock for st.expander."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Strictly re-raise any exception to expose rendering bugs
        return False


# ==============================================================================
# 1. TELEMETRY EXTRACTION HELPER UNIT TESTS
# ==============================================================================


def test_extract_telemetry_field_from_pydantic_object():
    """Verify safe attribute extraction from Pydantic ToolExecutionRecord."""
    rec = ToolExecutionRecord(
        tool_name="lookup_community",
        arguments={"query": "Tallur"},
        success=True,
        execution_time_ms=12.4,
        result_summary="Found unit 608691",
    )
    assert _extract_telemetry_field(rec, "tool_name") == "lookup_community"
    assert _extract_telemetry_field(rec, "success") is True
    assert _extract_telemetry_field(rec, "execution_time_ms") == 12.4
    assert _extract_telemetry_field(rec, "result_summary") == "Found unit 608691"
    # Non-existent attribute must return default, never raise AttributeError
    assert _extract_telemetry_field(rec, "non_existent_key", "default_val") == "default_val"


def test_extract_telemetry_field_from_dict():
    """Verify safe key extraction from dictionary records."""
    rec = {
        "tool_name": "compare_districts",
        "success": False,
        "arguments": {"districts": ["Dakshina Kannada", "Udupi"]},
        "execution_time_ms": 5.0,
        "error": "Timeout",
    }
    assert _extract_telemetry_field(rec, "tool_name") == "compare_districts"
    assert _extract_telemetry_field(rec, "success") is False
    assert _extract_telemetry_field(rec, "result_summary") is None
    assert _extract_telemetry_field(rec, "result_summary", "default_summary") == "default_summary"
    assert _extract_telemetry_field(rec, "error") == "Timeout"


def test_is_valid_telemetry_record():
    """Verify strict telemetry record validation rejects non-record types."""
    # Accepted types
    valid_record = ToolExecutionRecord(
        tool_name="lookup_community",
        arguments={},
        success=True,
        execution_time_ms=1.0,
    )
    assert _is_valid_telemetry_record(valid_record) is True
    assert _is_valid_telemetry_record({"tool_name": "lookup_community"}) is True

    class DuckRecord:
        tool_name = "custom_tool"

    assert _is_valid_telemetry_record(DuckRecord()) is True

    # Rejected types
    assert _is_valid_telemetry_record(None) is False
    assert _is_valid_telemetry_record(12345) is False
    assert _is_valid_telemetry_record(45.67) is False
    assert _is_valid_telemetry_record("corrupted_string") is False
    assert _is_valid_telemetry_record(True) is False
    assert _is_valid_telemetry_record([1, 2, 3]) is False
    assert _is_valid_telemetry_record((1, 2)) is False
    assert _is_valid_telemetry_record({1, 2}) is False
    assert _is_valid_telemetry_record(object()) is False


# ==============================================================================
# 2. RENDER TOOL EXECUTION DETAILS TESTS
# ==============================================================================


def test_render_tool_execution_details_real_pydantic_record():
    """Verify render_tool_execution_details successfully renders real ToolExecutionRecord."""
    rec = ToolExecutionRecord(
        tool_name="lookup_community",
        arguments={"query": "Tallur"},
        success=True,
        execution_time_ms=8.5,
        result_summary="Identified administrative unit PC11 608691 (Tallur)",
    )

    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ) as mock_exp:
        render_tool_execution_details([rec], duration_ms=8.5)

        assert mock_exp.called
        assert mock_md.called
        # Verify rendered html content contains tool name, duration, and summary
        rendered_html = mock_md.call_args[0][0]
        assert "lookup_community" in rendered_html
        assert "SUCCESS" in rendered_html
        assert "8.5 ms" in rendered_html
        assert "Identified administrative unit PC11 608691" in rendered_html


def test_render_tool_execution_details_with_result_summary():
    """Verify result summary is explicitly rendered in italic telemetry description."""
    rec = ToolExecutionRecord(
        tool_name="explain_preparedness_assessment",
        arguments={"identifier": "Tallur"},
        success=True,
        execution_time_ms=15.2,
        result_summary="Tallur: AHP 74.34 (High), Band 4 Adequate",
    )

    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ):
        render_tool_execution_details([rec], duration_ms=15.2)

        rendered_html = mock_md.call_args[0][0]
        assert "Tallur: AHP 74.34 (High), Band 4 Adequate" in rendered_html


def test_render_tool_execution_details_optional_fields_missing_or_none():
    """Verify records with missing or None optional fields default cleanly without crashing."""
    rec = ToolExecutionRecord(
        tool_name="filter_and_rank_communities",
        arguments={},
        success=True,
        execution_time_ms=0.0,
        result_summary=None,
        error=None,
        error_type=None,
    )

    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ):
        render_tool_execution_details([rec], duration_ms=None)

        rendered_html = mock_md.call_args[0][0]
        assert "filter_and_rank_communities" in rendered_html
        assert "SUCCESS" in rendered_html
        assert "Status: ok" in rendered_html


def test_render_tool_execution_details_failed_record_with_error():
    """Verify failed tool executions display FAILED badge and error summary."""
    rec = ToolExecutionRecord(
        tool_name="lookup_community",
        arguments={"query": "NonexistentVillageXYZ"},
        success=False,
        execution_time_ms=3.1,
        result_summary=None,
        error="Community 'NonexistentVillageXYZ' not found in database",
        error_type="NOT_FOUND",
    )

    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ):
        render_tool_execution_details([rec], duration_ms=3.1)

        rendered_html = mock_md.call_args[0][0]
        assert "lookup_community" in rendered_html
        assert "FAILED" in rendered_html
        assert "Error: Community &#39;NonexistentVillageXYZ&#39; not found" in rendered_html or "Error: Community 'NonexistentVillageXYZ' not found" in rendered_html


def test_render_tool_execution_details_multiple_records():
    """Verify multiple tool execution records render sequentially with 1., 2., 3. indices."""
    records = [
        ToolExecutionRecord(
            tool_name="lookup_community",
            arguments={"query": "Tallur"},
            success=True,
            execution_time_ms=4.1,
            result_summary="Found unit 608691",
        ),
        ToolExecutionRecord(
            tool_name="explain_preparedness_assessment",
            arguments={"identifier": "608691"},
            success=True,
            execution_time_ms=9.2,
            result_summary="AHP 74.34 (High)",
        ),
        ToolExecutionRecord(
            tool_name="get_intervention_recommendations",
            arguments={"identifier": "608691"},
            success=True,
            execution_time_ms=2.3,
            result_summary="3 field advisories generated",
        ),
    ]

    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ) as mock_exp:
        render_tool_execution_details(records, duration_ms=15.6)

        assert mock_exp.called
        # Verify expander label mentions 3 tools
        expander_label = mock_exp.call_args[0][0]
        assert "3 tool(s) invoked" in expander_label
        assert "15.6 ms" in expander_label

        # Verify 3 separate markdown cards were rendered
        assert mock_md.call_count == 3
        calls = [c[0][0] for c in mock_md.call_args_list]
        assert "1. lookup_community" in calls[0]
        assert "2. explain_preparedness_assessment" in calls[1]
        assert "3. get_intervention_recommendations" in calls[2]


def test_render_tool_execution_details_empty_and_none_list():
    """Verify empty or None telemetry list returns immediately without opening an expander."""
    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ) as mock_exp:
        render_tool_execution_details([])
        render_tool_execution_details(None)
        render_tool_execution_details([None, None])

        assert not mock_exp.called
        assert not mock_md.called


def test_render_tool_execution_details_dict_records():
    """Verify dictionary records are supported seamlessly."""
    records = [
        {
            "tool_name": "compare_districts",
            "arguments": {"district_a": "Dakshina Kannada", "district_b": "Udupi"},
            "success": True,
            "execution_time_ms": 11.0,
            "result_summary": "Comparison matrix generated across 622 units",
        }
    ]

    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ):
        render_tool_execution_details(records, duration_ms=11.0)

        assert mock_md.called
        rendered_html = mock_md.call_args[0][0]
        assert "compare_districts" in rendered_html
        assert "SUCCESS" in rendered_html
        assert "Comparison matrix generated" in rendered_html


# ==============================================================================
# 3. MESSAGE BUBBLE RESILIENCE & ERROR ISOLATION TESTS
# ==============================================================================


def test_render_message_bubble_malformed_telemetry_does_not_crash_assistant_response():
    """Verify malformed telemetry records cannot hide or crash the main assistant response."""
    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": "Tallur is in Band 4 Adequate Baseline with an authoritative AHP score of 74.34 / 100.0.",
        "is_fallback": True,
        "fallback_reason": "GEMINI_API_KEY is not configured",
        "tools_executed": ["lookup_community"],
        "tool_records": [
            None,
            12345,
            "corrupted_record_string",
            {"malformed": "dictionary_without_standard_keys"},
        ],
        "structured_evidence": None,
        "scientific_precedence_note": "Deterministic AHP scores maintain absolute operational precedence.",
        "total_duration_ms": 42.0,
        "timestamp": "12:00:00 UTC",
    }

    with patch.object(st, "markdown") as mock_md, patch.object(
        st, "expander", return_value=FakeExpanderContext()
    ):
        # Must execute cleanly without raising any exception
        render_message_bubble(msg)

        # Confirm the main answer was rendered
        all_md_content = " ".join(str(c[0][0]) for c in mock_md.call_args_list)
        assert "Tallur is in Band 4 Adequate Baseline" in all_md_content
        assert "74.34 / 100.0" in all_md_content
        assert "Deterministic AHP scores maintain absolute operational precedence" in all_md_content


# ==============================================================================
# 4. AHP DOMAIN SUB-SCORES PRESENTATION TESTS
# ==============================================================================


def test_ahp_domain_sub_scores_presentation_with_canonical_keys():
    """Verify format_community_assessment_response renders canonical domain sub-scores."""
    community_eval = {
        "community_name": "Tallur",
        "district_name": "Udupi",
        "pc11_id": "608691",
        "shrid2": "11-29-569-05522-608691",
        "subdistrict_name": "Kundapura",
        "ahp_score": 74.34,
        "preparedness_tier": "High",
        "deterministic_ahp": {
            "preparedness_score": 74.34,
            "preparedness_tier": "High",
            "domain_scores": {
                "score_medical": 71.94,
                "score_shelters": 68.59,
                "score_infrastructure": 68.0,
                "score_early_warning": 82.11,
                "score_socioeconomic": 88.58,
            },
        },
        "operational_prioritization": {
            "priority_band": "BAND_4_ADEQUATE",
            "overall_priority_rank": 1833,
            "band_rank": 550,
            "priority_explanation": "Unit assigned to BAND_4_ADEQUATE with score 74.34.",
        },
    }

    resp = format_community_assessment_response(
        query="Assess Tallur",
        reason="GEMINI_API_KEY unconfigured",
        community_eval=community_eval,
        advisories=[],
    )

    assert "**AHP Domain Sub-Scores:**" in resp
    assert "Primary Emergency Shelters: 68.6 / 100.0" in resp
    assert "Primary Healthcare Access: 71.9 / 100.0" in resp
    assert "Infrastructure & Physical Connectivity: 68.0 / 100.0" in resp
    assert "Early Warning & Communication: 82.1 / 100.0" in resp
    assert "Socioeconomic Vulnerability Baseline: 88.6 / 100.0" in resp
    # Ensure no 'N/A / 100.0' formatting anomaly occurs
    assert "N/A / 100.0" not in resp


def test_ahp_domain_sub_scores_presentation_with_missing_keys():
    """Verify missing domain sub-scores display clean 'N/A' without appending '/ 100.0'."""
    community_eval = {
        "community_name": "Sample Village",
        "district_name": "Udupi",
        "deterministic_ahp": {
            "preparedness_score": 50.0,
            "preparedness_tier": "Medium",
            "domain_scores": {
                "score_medical": 45.0,
                # All other scores missing
            },
        },
        "operational_prioritization": {
            "priority_band": "BAND_3_MODERATE",
            "overall_priority_rank": 100,
            "band_rank": 20,
        },
    }

    resp = format_community_assessment_response(
        query="Assess Sample Village",
        reason="GEMINI_API_KEY unconfigured",
        community_eval=community_eval,
        advisories=[],
    )

    assert "Primary Healthcare Access: 45.0 / 100.0" in resp
    assert "Primary Emergency Shelters: N/A" in resp
    assert "N/A / 100.0" not in resp
