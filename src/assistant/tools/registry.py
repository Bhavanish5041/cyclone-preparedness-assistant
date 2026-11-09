"""Centralized registry for assistant tools.

Provides:
- Tool registration, discovery, and lookup.
- Safe argument validation and controlled execution.
- Export of tool definitions for OpenAI, Gemini, and Anthropic tool-calling protocols.
- Factory function creating a pre-configured registry with all seven deterministic tools.
"""

from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional

from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools.archetypes import ExplainVulnerabilityArchetypeTool
from src.assistant.tools.base import BaseTool, ToolNotFoundError, ToolResult
from src.assistant.tools.community import AnalyzeInfrastructureAndExposureTool, LookupCommunityTool
from src.assistant.tools.decision import (
    ExplainPreparednessAssessmentTool,
    GetInterventionRecommendationsTool,
)
from src.assistant.tools.regional import CompareDistrictsTool, FilterAndRankCommunitiesTool

logger = logging.getLogger(__name__)


UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "examples",
    "example",
    "$schema",
    "$defs",
    "definitions",
    "additionalProperties",
}


def sanitize_gemini_schema(schema: Any) -> Any:
    """Recursively normalize a JSON Schema dictionary for Google Gemini Function Calling.

    Complies with Google Gemini's OpenAPI 3.0 Schema protobuf subset:
    - Recursively strips unsupported metadata (`examples`, `example`, `$schema`, `$defs`,
      `definitions`, `additionalProperties`) that cause HTTP 400 INVALID_ARGUMENT.
    - Flattens nullable unions (`anyOf` or `oneOf` with `type: "null"`) into concrete types
      with `nullable: True`.
    - Normalizes list-based types (`['string', 'null']`) to scalar type with `nullable: True`.
    - Strictly preserves parameter names, types, descriptions, titles, required fields,
      enums, nested properties, array items, and numeric/string constraints (minimum, maximum,
      minLength, maxLength, pattern).
    """
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [sanitize_gemini_schema(item) for item in schema]
        return schema

    schema_copy = deepcopy(schema)

    # 1. Handle type as list (e.g. ['string', 'null'])
    raw_type = schema_copy.get("type")
    if isinstance(raw_type, list):
        has_null = "null" in raw_type
        non_null_types = [t for t in raw_type if t != "null"]
        if non_null_types:
            schema_copy["type"] = non_null_types[0]
        if has_null:
            schema_copy["nullable"] = True

    # 2. Normalize anyOf / oneOf unions into a single schema with nullable flag
    for union_key in ("anyOf", "oneOf"):
        if union_key in schema_copy and isinstance(schema_copy[union_key], list):
            variants = schema_copy[union_key]
            has_null = any(isinstance(v, dict) and v.get("type") == "null" for v in variants)
            non_null = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
            if non_null:
                primary = deepcopy(non_null[0])
                for pk, pv in primary.items():
                    if pk not in schema_copy:
                        schema_copy[pk] = pv
                if has_null:
                    schema_copy["nullable"] = True
            del schema_copy[union_key]

    # 3. Recursively clean remaining fields
    cleaned: Dict[str, Any] = {}
    for k, v in schema_copy.items():
        if k in UNSUPPORTED_GEMINI_SCHEMA_KEYS:
            continue
        cleaned[k] = sanitize_gemini_schema(v)

    # 4. Clean required array if present
    if "required" in cleaned and isinstance(cleaned["required"], list):
        cleaned["required"] = [str(r) for r in cleaned["required"]]

    return cleaned


class ToolRegistry:
    """Centralized catalog and dispatcher for deterministic assistant tools."""

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._service = service

    @property
    def service(self) -> Optional[DecisionSupportService]:
        """Shared DecisionSupportService instance if injected."""
        return self._service

    def register(self, tool: BaseTool) -> None:
        """Register a new assistant tool.

        Args:
            tool: BaseTool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if not tool.name or not tool.name.strip():
            raise ValueError("Tool name must be a non-empty string.")

        canonical_name = tool.name.strip().lower()
        if canonical_name in self._tools:
            raise ValueError(f"Tool '{canonical_name}' is already registered.")

        self._tools[canonical_name] = tool
        logger.info(f"Registered tool: '{canonical_name}' ({tool.__class__.__name__})")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Look up a tool by name (case-insensitive)."""
        if not name:
            return None
        return self._tools.get(name.strip().lower())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        if not name:
            return False
        return name.strip().lower() in self._tools

    def list_tools(self) -> List[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return a dictionary mapping tool names to their descriptions."""
        return {name: tool.description for name, tool in sorted(self._tools.items())}

    def execute(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute a tool by name with arguments and return a standardized ToolResult.

        Args:
            name: Name of tool to execute.
            arguments: Dictionary of keyword arguments for the tool.

        Returns:
            ToolResult containing execution success, data payload, or error details.
        """
        tool = self.get_tool(name)
        if not tool:
            available = self.list_tools()
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool '{name}'. Available tools: {available}",
                error_type="UNKNOWN_TOOL",
                metadata={"available_tools": available},
            )

        return tool.execute(arguments)

    # ==========================================================================
    # SCHEMA EXPORT ADAPTERS FOR LLM PROVIDERS
    # ==========================================================================

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Return format-agnostic JSON schemas for all registered tools."""
        schemas = []
        for name in self.list_tools():
            tool = self._tools[name]
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_json_schema(),
                }
            )
        return schemas

    def export_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        """Export tool declarations in OpenAI / Azure OpenAI tool-calling format.

        Format:
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup_community",
                    "description": "...",
                    "parameters": {...}
                }
            }
        ]
        """
        definitions = []
        for name in self.list_tools():
            tool = self._tools[name]
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": deepcopy(tool.get_json_schema()),
                    },
                }
            )
        return definitions

    def export_gemini_tool_declarations(self) -> List[Dict[str, Any]]:
        """Export tool declarations in Google Gemini Function Calling format.

        Format:
        [
            {
                "name": "lookup_community",
                "description": "...",
                "parameters": {...}
            }
        ]
        """
        declarations = []
        for name in self.list_tools():
            tool = self._tools[name]
            raw_schema = deepcopy(tool.get_json_schema())
            sanitized_parameters = sanitize_gemini_schema(raw_schema)
            declarations.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": sanitized_parameters,
                }
            )
        return declarations

    def export_anthropic_tool_definitions(self) -> List[Dict[str, Any]]:
        """Export tool declarations in Anthropic Claude tool use format.

        Format:
        [
            {
                "name": "lookup_community",
                "description": "...",
                "input_schema": {...}
            }
        ]
        """
        definitions = []
        for name in self.list_tools():
            tool = self._tools[name]
            definitions.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": deepcopy(tool.get_json_schema()),
                }
            )
        return definitions


def get_default_registry(service: Optional[DecisionSupportService] = None) -> ToolRegistry:
    """Create and populate a ToolRegistry with all 7 default deterministic tools.

    Args:
        service: Optional DecisionSupportService instance. If omitted, tools
                 will use or initialize the singleton service.

    Returns:
        Fully configured ToolRegistry instance.
    """
    shared_service = service or DecisionSupportService()
    registry = ToolRegistry(service=shared_service)

    # 1. Community tools
    registry.register(LookupCommunityTool(service=shared_service))
    registry.register(AnalyzeInfrastructureAndExposureTool(service=shared_service))

    # 2. Decision support tools
    registry.register(ExplainPreparednessAssessmentTool(service=shared_service))
    registry.register(GetInterventionRecommendationsTool(service=shared_service))

    # 3. Regional analysis tools
    registry.register(CompareDistrictsTool(service=shared_service))
    registry.register(FilterAndRankCommunitiesTool(service=shared_service))

    # 4. Vulnerability archetype tools
    registry.register(ExplainVulnerabilityArchetypeTool(service=shared_service))

    return registry
