"""Centralized registry for assistant tools.

Provides:
- Tool registration, discovery, and lookup.
- Safe argument validation and controlled execution.
- Export of tool definitions for OpenAI and Anthropic tool-calling protocols.
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
