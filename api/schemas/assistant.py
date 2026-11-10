"""Pydantic schemas for AI Decision Support Assistant endpoint."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Re-export or inherit from src.assistant.llm.schemas
from src.assistant.llm.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    MessageRole,
    ToolExecutionRecord,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConversationStatus",
    "MessageRole",
    "ToolExecutionRecord",
]
