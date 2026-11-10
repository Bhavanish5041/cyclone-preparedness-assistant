"""Unit tests for Ollama client, IntentRouter pipeline, and local AI orchestration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.assistant.llm.client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaTimeoutError,
    get_default_llm_client,
)
from src.assistant.llm.orchestrator import ToolCallingOrchestrator
from src.assistant.llm.schemas import ChatRequest, ConversationStatus


class TestOllamaClient:
    def test_client_init_and_properties(self):
        client = OllamaClient(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5:0.5b-instruct",
            timeout=25.0,
        )
        assert client.provider_name == "ollama"
        assert client.model == "qwen2.5:0.5b-instruct"
        assert client.base_url == "http://127.0.0.1:11434"
        assert client.timeout == 25.0
        assert client.is_configured() is True

    def test_factory_returns_ollama_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        client = get_default_llm_client()
        assert isinstance(client, OllamaClient)
        assert client.provider_name == "ollama"

    def test_ollama_chat_success(self):
        async def _run():
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {"content": "Test synthesis response."}
            }

            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response

            client = OllamaClient(http_client=mock_http)
            result = await client.chat([{"role": "user", "content": "Hello"}])
            assert result == "Test synthesis response."
            mock_http.post.assert_called_once()
        asyncio.run(_run())

    def test_ollama_generate_success(self):
        async def _run():
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Generated briefing memo."
            }

            mock_http = AsyncMock()
            mock_http.post.return_value = mock_response

            client = OllamaClient(http_client=mock_http)
            result = await client.generate(prompt="Explain Ullal", system="System rule")
            assert result == "Generated briefing memo."
        asyncio.run(_run())

    def test_ollama_timeout_error(self):
        async def _run():
            import httpx

            mock_http = AsyncMock()
            mock_http.post.side_effect = httpx.TimeoutException("Timeout")

            client = OllamaClient(http_client=mock_http)
            with pytest.raises(OllamaTimeoutError):
                await client.generate("test prompt")
        asyncio.run(_run())


class TestOrchestratorOllamaPipeline:
    def test_pipeline_refusal_for_live_telemetry(self):
        async def _run():
            orchestrator = ToolCallingOrchestrator()
            req = ChatRequest(message="What is the live wind speed right now in Mangalore?")
            resp = await orchestrator.process_chat(req)

            assert resp.conversation_status == ConversationStatus.COMPLETED
            assert resp.tools_executed == []
            assert "dynamic" in resp.answer.lower() or "imd" in resp.answer.lower()
            assert resp.structured_evidence is None
        asyncio.run(_run())

    def test_pipeline_out_of_scope(self):
        async def _run():
            orchestrator = ToolCallingOrchestrator()
            req = ChatRequest(message="Write a python binary search tree function.")
            resp = await orchestrator.process_chat(req)

            assert resp.conversation_status == ConversationStatus.COMPLETED
            assert resp.tools_executed == []
            assert "scope" in resp.answer.lower()
        asyncio.run(_run())

    def test_pipeline_routed_tool_execution(self):
        async def _run():
            mock_client = AsyncMock()
            mock_client.provider_name = "ollama"
            mock_client.model = "qwen2.5:0.5b-instruct"
            mock_client.is_configured = MagicMock(return_value=True)
            mock_client.generate.return_value = (
                "Tallur has an authoritative AHP score of 74.34, categorized in Band 4 Adequate."
            )

            orchestrator = ToolCallingOrchestrator(client=mock_client)
            req = ChatRequest(message="What is the preparedness assessment for Tallur?")
            resp = await orchestrator.process_chat(req)

            assert "explain_preparedness_assessment" in resp.tools_executed
            assert resp.structured_evidence is not None
            assert "explain_preparedness_assessment" in resp.structured_evidence
            assert "74.34" in resp.answer
            assert "Band 4" in resp.answer or "Adequate" in resp.answer
        asyncio.run(_run())

    def test_pipeline_fallback_on_ollama_failure(self):
        async def _run():
            mock_client = AsyncMock()
            mock_client.provider_name = "ollama"
            mock_client.model = "qwen2.5:0.5b-instruct"
            mock_client.is_configured = MagicMock(return_value=True)
            mock_client.generate.side_effect = OllamaConnectionError("Ollama daemon offline")

            orchestrator = ToolCallingOrchestrator(client=mock_client)
            req = ChatRequest(message="What is the preparedness assessment for Tallur?")
            resp = await orchestrator.process_chat(req)

            assert resp.is_fallback is True
            assert "explain_preparedness_assessment" in resp.tools_executed
            assert resp.structured_evidence is not None
            assert "Tallur" in resp.answer
        asyncio.run(_run())
