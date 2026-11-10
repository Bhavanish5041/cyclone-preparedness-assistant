"""Unit tests for CycloneLM structured tool protocol and orchestration."""

import pytest

from src.cyclonelm.tools.protocol import (
    ToolCallRequest,
    ToolExecutionRecord,
    ToolOrchestrator,
)


def test_extract_tool_calls():
    orchestrator = ToolOrchestrator()
    sample_text = (
        'Here is the analysis:\n'
        '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Ullal"}}</tool_call>\n'
        'Awaiting results.'
    )
    calls = orchestrator.extract_tool_calls(sample_text)
    assert len(calls) == 1
    assert calls[0].tool == "lookup_community"
    assert calls[0].parameters == {"query": "Ullal"}


def test_malformed_json_handling():
    orchestrator = ToolOrchestrator()
    sample_text = '<tool_call>{"tool": "lookup_community", broken_json</tool_call>'
    calls = orchestrator.extract_tool_calls(sample_text)
    assert len(calls) == 0  # Safely returns empty without exception


def test_execute_deterministic_lookup():
    orchestrator = ToolOrchestrator()
    call = ToolCallRequest(tool="lookup_community", parameters={"query": "Ullal"})
    rec = orchestrator.execute_tool_call(call, round_idx=1)
    assert rec.success is True
    assert rec.data is not None
    assert rec.data.get("total_matches", 0) > 0


def test_unregistered_tool_blocked():
    orchestrator = ToolOrchestrator()
    call = ToolCallRequest(tool="execute_arbitrary_shell_cmd", parameters={"cmd": "ls"})
    rec = orchestrator.execute_tool_call(call, round_idx=1)
    assert rec.success is False
    assert rec.error_type == "ToolNotFoundError"


def test_orchestration_session_deduplication():
    orchestrator = ToolOrchestrator()
    mock_turns = [
        '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Ullal"}}</tool_call>',
        '<tool_call>{"tool": "lookup_community", "parameters": {"query": "Ullal"}}</tool_call>',
    ]
    res = orchestrator.run_session(prompt="Test prompt", mock_generator_outputs=mock_turns)
    assert len(res.records) == 2
    assert res.records[0].success is True
    assert res.records[1].success is False
    assert res.records[1].error_type == "DuplicateCallError"
