"""Conversational AI Decision Support Assistant endpoints."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends

from api.dependencies import get_orchestrator, get_service
from src.assistant.decision_support import DecisionSupportService
from src.assistant.llm.orchestrator import ToolCallingOrchestrator
from src.assistant.llm.schemas import ChatRequest, ChatResponse
from src.assistant.tools.registry import get_default_registry

router = APIRouter(prefix="/assistant", tags=["CycloneLM Assistant"])


@router.post("/chat", response_model=ChatResponse, summary="Conversational decision support turn with deterministic tool grounding")
async def chat_with_assistant(
    request: ChatRequest,
    orchestrator: ToolCallingOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Execute conversational turn with LLM and deterministic tool-calling safeguards."""
    return await orchestrator.process_chat(request)


@router.get("/tools", summary="List registered deterministic tools and execution schemas")
def list_assistant_tools(
    service: DecisionSupportService = Depends(get_service),
) -> List[Dict[str, Any]]:
    """Return catalog of all 7 registered deterministic tools and their schemas."""
    registry = get_default_registry(service=service)
    tools = []
    for name, desc in registry.get_tool_descriptions().items():
        tool = registry.get_tool(name)
        tools.append({
            "name": name,
            "description": desc,
            "parameters": tool.get_json_schema() if tool else {},
        })
    return tools
