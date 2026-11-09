"""Base classes, data contracts, and result models for assistant tools.

Provides:
- Standardized `ToolResult` envelope ensuring 100% JSON-serializable outputs.
- `BaseTool` abstract base class with automated argument validation and timing metadata.
- Standard exception hierarchy for deterministic error reporting without internal leakage.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS HIERARCHY
# ==============================================================================


class ToolError(Exception):
    """Base exception for all tool execution and validation errors."""

    pass


class ToolValidationError(ToolError):
    """Raised when tool arguments fail schema validation or contain invalid values."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool does not exist in the registry."""

    pass


class CommunityNotFoundError(ToolError):
    """Raised when an administrative unit cannot be resolved in the dataset."""

    pass


class AmbiguousCommunityError(ToolError):
    """Raised when a village name matches multiple administrative units."""

    def __init__(self, name: str, candidates: List[Dict[str, Any]]) -> None:
        self.name = name
        self.candidates = candidates
        cand_str = "; ".join(
            f"PC11: {c.get('pc11_id', 'N/A')}, SHRID: {c.get('shrid2', 'N/A')} "
            f"({c.get('village_name')}, Taluk: {c.get('subdistrict_name')}, District: {c.get('district_name')})"
            for c in candidates
        )
        msg = (
            f"Ambiguous village name '{name}' matches {len(candidates)} administrative units. "
            f"Please specify the Census PC11 ID or SHRID to disambiguate: {cand_str}"
        )
        super().__init__(msg)


# ==============================================================================
# STANDARDIZED RESULT ENVELOPE
# ==============================================================================


class ToolResult(BaseModel):
    """Standardized result envelope returned by all assistant tools.

    Guarantees:
    - 100% JSON-serializable output.
    - Explicit boolean `success` indicator.
    - Structured `data` payload on success, or `error` string on failure.
    - Rich `metadata` containing execution duration, timestamp, and query details.
    """

    tool_name: str = Field(..., description="Name of the executed tool.")
    success: bool = Field(..., description="Whether the tool executed successfully.")
    data: Optional[Dict[str, Any]] = Field(None, description="Output payload on successful execution.")
    error: Optional[str] = Field(None, description="Error message on failure.")
    error_type: Optional[str] = Field(None, description="Machine-readable error category.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result envelope to standard dictionary."""
        return self.model_dump()

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize result envelope to valid JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ==============================================================================
# ABSTRACT BASE TOOL
# ==============================================================================


class BaseTool(ABC):
    """Abstract base class for all deterministic assistant tools."""

    name: str
    description: str
    input_schema: Type[BaseModel]

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the deterministic core logic with validated kwargs and return serializable dict.

        Args:
            **kwargs: Validated arguments matching input_schema.

        Returns:
            Dictionary containing result data (must be JSON-serializable).
        """
        pass

    def execute(self, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Validate input arguments and execute the tool, wrapping output in ToolResult.

        Handles:
        - Argument schema validation via Pydantic.
        - Runtime execution timing.
        - Exception capture and structured error envelope formatting.
        - Zero exposure of internal Python traces, filesystem paths, or secrets.

        Args:
            arguments: Raw input dictionary from caller.

        Returns:
            ToolResult envelope.
        """
        start_time = time.perf_counter()
        raw_args = arguments or {}

        # 1. Input Schema Validation
        try:
            validated_args = self.input_schema.model_validate(raw_args)
        except ValidationError as val_err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            error_msgs = []
            for e in val_err.errors():
                loc_str = " -> ".join(str(x) for x in e["loc"]) if e["loc"] else "input"
                error_msgs.append(f"{loc_str}: {e['msg']}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="; ".join(error_msgs),
                error_type="VALIDATION_ERROR",
                metadata={"execution_time_ms": duration_ms},
            )

        # 2. Deterministic Execution
        try:
            data = self.run(**validated_args.model_dump())
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                metadata={
                    "execution_time_ms": duration_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except AmbiguousCommunityError as err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(err),
                error_type="AMBIGUOUS_COMMUNITY",
                data={"name": err.name, "candidates": err.candidates},
                metadata={"execution_time_ms": duration_ms},
            )
        except CommunityNotFoundError as err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(err),
                error_type="COMMUNITY_NOT_FOUND",
                metadata={"execution_time_ms": duration_ms},
            )
        except ToolValidationError as err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(err),
                error_type="VALIDATION_ERROR",
                metadata={"execution_time_ms": duration_ms},
            )
        except Exception as err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception(f"Unhandled error during tool '{self.name}' execution: {err}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Internal tool execution error: {type(err).__name__}: {str(err)}",
                error_type="EXECUTION_ERROR",
                metadata={"execution_time_ms": duration_ms},
            )

    def get_json_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema for tool parameters from input_schema."""
        return self.input_schema.model_json_schema()
