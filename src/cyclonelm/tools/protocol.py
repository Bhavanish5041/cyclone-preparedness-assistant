"""Structured Tool-Use Protocol and Orchestration Layer for CycloneLM.

Reuses the authoritative deterministic tool registry from `src.assistant.tools`.
Guarantees:
- Safe JSON extraction from model outputs without code execution (no eval/exec).
- Tool name validation against the registered deterministic tools.
- Strict Pydantic argument validation.
- Duplicate call detection and loop prevention.
- Max round limits.
- Structured evidence capture and final synthesis formatting.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from src.assistant.tools.base import ToolResult
from src.assistant.tools.registry import ToolRegistry, get_default_registry

logger = logging.getLogger(__name__)

TOOL_CALL_REGEX = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)


@dataclass
class ToolCallRequest:
    """Parsed structured tool call request."""

    tool: str
    parameters: Dict[str, Any]
    raw_text: str = ""


@dataclass
class ToolExecutionRecord:
    """Record of an executed deterministic tool call."""

    round_idx: int
    tool: str
    parameters: Dict[str, Any]
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationResult:
    """Comprehensive result of an orchestrated multi-round tool execution session."""

    initial_prompt: str
    rounds_executed: int
    records: List[ToolExecutionRecord] = field(default_factory=list)
    final_evidence: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_ready: bool = False
    error: Optional[str] = None

    def format_evidence_context(self) -> str:
        """Format captured structured evidence as context for final model synthesis."""
        if not self.records:
            return "No tool evidence collected."

        lines = ["=== VERIFIED DETERMINISTIC EVIDENCE ==="]
        for rec in self.records:
            status_str = "SUCCESS" if rec.success else "FAILED"
            lines.append(f"Tool: {rec.tool} (Status: {status_str})")
            lines.append(f"Parameters: {json.dumps(rec.parameters, ensure_ascii=False)}")
            if rec.success and rec.data:
                lines.append(f"Data: {json.dumps(rec.data, ensure_ascii=False, indent=2)}")
            elif rec.error:
                lines.append(f"Error: {rec.error} ({rec.error_type})")
            lines.append("-" * 40)
        return "\n".join(lines)


class ToolOrchestrator:
    """Orchestration layer coordinating CycloneLM with deterministic tool execution."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        max_rounds: int = 3,
    ) -> None:
        self.registry = registry or get_default_registry()
        self.max_rounds = max_rounds

    def extract_tool_calls(self, text: str) -> List[ToolCallRequest]:
        """Extract structured <tool_call>{...}</tool_call> tags safely without eval/exec."""
        matches = TOOL_CALL_REGEX.findall(text)
        requests: List[ToolCallRequest] = []

        for match in matches:
            cleaned = match.strip()
            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    continue
                tool_name = parsed.get("tool")
                params = parsed.get("parameters", {})
                if isinstance(tool_name, str) and isinstance(params, dict):
                    requests.append(
                        ToolCallRequest(
                            tool=tool_name,
                            parameters=params,
                            raw_text=cleaned,
                        )
                    )
            except json.JSONDecodeError as exc:
                logger.warning(f"Malformed JSON in tool call: {cleaned!r} -> {exc}")

        return requests

    def execute_tool_call(
        self,
        call: ToolCallRequest,
        round_idx: int = 1,
    ) -> ToolExecutionRecord:
        """Validate and execute a single tool request against the deterministic registry."""
        tool_name = call.tool
        params = call.parameters

        # 1. Validate tool existence
        if not self.registry.has_tool(tool_name):
            return ToolExecutionRecord(
                round_idx=round_idx,
                tool=tool_name,
                parameters=params,
                success=False,
                error=f"Unrecognized tool: {tool_name!r}. Available tools: {self.registry.list_tools()}",
                error_type="ToolNotFoundError",
            )

        # 2. Execute through registry (performs Pydantic schema validation internally)
        result: ToolResult = self.registry.execute(tool_name, params)

        return ToolExecutionRecord(
            round_idx=round_idx,
            tool=tool_name,
            parameters=params,
            success=result.success,
            data=result.data,
            error=result.error,
            error_type=result.error_type,
        )

    def run_session(
        self,
        prompt: str,
        mock_generator_outputs: Optional[List[str]] = None,
    ) -> OrchestrationResult:
        """Run tool orchestration session with duplicate detection and round limits.

        Args:
            prompt: User inquiry or instruction.
            mock_generator_outputs: Optional list of simulated model outputs per round.
        """
        result = OrchestrationResult(initial_prompt=prompt, rounds_executed=0)
        seen_calls: Set[Tuple[str, str]] = set()

        generator_turns = list(mock_generator_outputs or [])
        round_idx = 0

        while round_idx < self.max_rounds and generator_turns:
            round_idx += 1
            result.rounds_executed = round_idx
            model_text = generator_turns.pop(0)

            # Extract tool calls
            calls = self.extract_tool_calls(model_text)
            if not calls:
                # No more tool calls; final synthesis reached
                result.synthesis_ready = True
                break

            for call in calls:
                # Deduplication check
                call_key = (call.tool, json.dumps(call.parameters, sort_keys=True))
                if call_key in seen_calls:
                    record = ToolExecutionRecord(
                        round_idx=round_idx,
                        tool=call.tool,
                        parameters=call.parameters,
                        success=False,
                        error="Duplicate tool call detected in same session.",
                        error_type="DuplicateCallError",
                    )
                else:
                    seen_calls.add(call_key)
                    record = self.execute_tool_call(call, round_idx=round_idx)

                result.records.append(record)
                if record.success and record.data:
                    result.final_evidence.append({
                        "tool": record.tool,
                        "data": record.data,
                    })

        result.synthesis_ready = True
        return result
