"""LLM integration package for Cyclone Preparedness Intelligence Platform.

Exposes:
- `GeminiClient` and client exceptions.
- `ToolCallingOrchestrator` managing the multi-turn loop.
- Request, response, and message schemas (`ChatRequest`, `ChatResponse`, `ChatMessage`, `ToolExecutionRecord`).
- System prompts and scientific guardrail templates.
"""

from src.assistant.llm.client import (
    GeminiAuthError,
    GeminiClient,
    GeminiClientError,
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiResponseError,
    GeminiTimeoutError,
)
from src.assistant.llm.prompts import (
    SCIENTIFIC_PRECEDENCE_DISCLAIMER,
    SYSTEM_INSTRUCTION,
    format_deterministic_fallback_response,
)
from src.assistant.llm.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    MessageRole,
    ToolExecutionRecord,
)


def __getattr__(name: str):
    """Lazy import for ToolCallingOrchestrator to break circular dependencies."""
    if name == "ToolCallingOrchestrator":
        from src.assistant.llm.orchestrator import ToolCallingOrchestrator
        return ToolCallingOrchestrator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Client & Exceptions
    "GeminiClient",
    "GeminiClientError",
    "GeminiAuthError",
    "GeminiRateLimitError",
    "GeminiTimeoutError",
    "GeminiNetworkError",
    "GeminiResponseError",
    # Orchestrator
    "ToolCallingOrchestrator",
    # Schemas
    "ChatRequest",
    "ChatResponse",
    "ChatMessage",
    "MessageRole",
    "ConversationStatus",
    "ToolExecutionRecord",
    # Prompts & Disclaimers
    "SYSTEM_INSTRUCTION",
    "SCIENTIFIC_PRECEDENCE_DISCLAIMER",
    "format_deterministic_fallback_response",
]
