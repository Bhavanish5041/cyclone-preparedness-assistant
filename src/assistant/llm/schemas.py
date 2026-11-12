"""Pydantic request, response, and message schemas for the Cyclone Preparedness Assistant.

Provides:
- Strict validation for user input, conversation history, and community context.
- Structured tool execution recording for auditing and transparency.
- Response envelope with explicit fallback indicators and scientific precedence mandates.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class MessageRole(str, Enum):
    """Permitted conversation roles."""

    USER = "user"
    ASSISTANT = "assistant"
    MODEL = "model"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationStatus(str, Enum):
    """Status of the conversation turn."""

    COMPLETED = "COMPLETED"
    FALLBACK = "FALLBACK"
    ERROR = "ERROR"


class ChatMessage(BaseModel):
    """Single turn in client-supplied conversation history."""

    role: MessageRole = Field(
        ...,
        description="Role of the message sender ('user', 'assistant', 'model', 'system', or 'tool').",
        examples=["user", "assistant", "tool"],
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text content of the message.",
    )
    name: Optional[str] = Field(
        None,
        description="Optional tool function name when role is 'tool'.",
    )
    tool_call_id: Optional[str] = Field(
        None,
        description="Optional tool call identifier when role is 'tool'.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> str:
        """Sanitize content and strictly enforce 5,000-character ceiling to prevent validation crashes.

        Safely compacts structured JSON payloads and text without destroying JSON syntax.
        """
        if v is None:
            raise ValueError("Message content cannot be None.")
        clean = str(v).strip()
        if not clean:
            raise ValueError("Message content cannot be empty or solely whitespace.")
        if len(clean) > 5000:
            from src.assistant.llm.history import compact_message_content
            return compact_message_content(clean, max_chars=5000)
        return clean


class ChatRequest(BaseModel):
    """Request payload for /api/v1/assistant/chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question or incident command prompt for the assistant.",
        examples=[
            "Why is Tallur classified as Band 4 Adequate despite coastal proximity?",
            "Compare preparedness between Dakshina Kannada and Udupi.",
            "What are the recommended interventions for Edmoge?",
        ],
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Recent conversation history for multi-turn context (maximum 10 turns).",
    )
    community_identifier: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional Census PC11 ID, SHRID, or village name for contextual focus.",
        examples=["608691", "Tallur", "Edmoge"],
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional sampling temperature override (default 0.0 for deterministic grounding).",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_history_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "query" in data and ("message" not in data or not data.get("message")):
                data["message"] = data.pop("query")
            if "history" in data and ("conversation_history" not in data or not data.get("conversation_history")):
                data["conversation_history"] = data.pop("history")
            if "context_community_id" in data and ("community_identifier" not in data or not data.get("community_identifier")):
                data["community_identifier"] = data.pop("context_community_id")
        return data

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Message cannot be empty or solely whitespace.")
        return clean

    @field_validator("conversation_history")
    @classmethod
    def validate_history_length(cls, v: Optional[List[ChatMessage]]) -> List[ChatMessage]:
        if v is None:
            return []
        if len(v) > 10:
            raise ValueError("Conversation history cannot exceed 10 messages to prevent context exhaustion.")
        return v

    @field_validator("community_identifier")
    @classmethod
    def validate_community_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        return clean if clean else None


class ToolExecutionRecord(BaseModel):
    """Audit record of a single deterministic tool execution."""

    tool_name: str = Field(..., description="Name of the deterministic tool executed.")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool.")
    success: bool = Field(..., description="Whether the tool execution succeeded.")
    execution_time_ms: float = Field(..., description="Tool execution duration in milliseconds.")
    result_summary: Optional[str] = Field(
        None, description="Human-readable summary of the tool execution result."
    )
    error: Optional[str] = Field(None, description="Error message if tool execution failed.")
    error_type: Optional[str] = Field(None, description="Categorized error type if applicable.")


class ChatResponse(BaseModel):
    """Response payload returned by the conversational decision support engine."""

    answer: str = Field(..., description="Grounded natural language assessment or advisory memo.")
    tools_executed: List[str] = Field(
        default_factory=list,
        description="Names of all tools executed during this conversation turn.",
    )
    tool_records: List[ToolExecutionRecord] = Field(
        default_factory=list,
        description="Audit records for all tool executions during this turn.",
    )
    structured_evidence: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured dictionary of factual evidence extracted from executed tools.",
    )
    is_fallback: bool = Field(
        False,
        description="True if deterministic fallback was activated (e.g. Ollama daemon offline or unconfigured).",
    )
    fallback_reason: Optional[str] = Field(
        None,
        description="Reason why deterministic fallback was activated if applicable.",
    )
    conversation_status: ConversationStatus = Field(
        ConversationStatus.COMPLETED,
        description="'COMPLETED', 'FALLBACK', or 'ERROR'.",
    )
    model_name: Optional[str] = Field(
        None,
        description="Exact model identifier used for response generation (e.g., 'qwen2.5:0.5b-instruct').",
    )
    total_duration_ms: float = Field(
        ...,
        description="Total round-trip processing duration in milliseconds.",
    )
    scientific_precedence_note: str = Field(
        ...,
        description="Mandatory scientific governance notice reinforcing AHP precedence and Census baseline limitations.",
    )


class QwenIntentParse(BaseModel):
    """Structured intent understanding and argument extraction schema from Qwen."""

    intents: List[str] = Field(
        default_factory=list,
        description="List of registered deterministic tool names identified from the user query.",
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted parameters and entity arguments for the identified tools.",
    )
    needs_clarification: bool = Field(
        False,
        description="Whether user query is underspecified, ambiguous, or requires clarification.",
    )
    clarification_question: Optional[str] = Field(
        None,
        description="Clarification prompt to return if needs_clarification is True.",
    )

