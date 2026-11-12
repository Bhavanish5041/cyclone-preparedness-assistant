"""Asynchronous HTTP client for local Ollama REST API (Qwen 2.5).

Provides:
- Lightweight asynchronous calls via `httpx.AsyncClient` without heavy external SDKs.
- Local self-hosted Qwen 2.5 inference (`qwen2.5:0.5b-instruct`) via Ollama.
- Zero external network dependencies, third-party API keys, or cloud costs.
- Single-concurrency inference via asyncio.Semaphore(1) to prevent CPU starvation.
- Robust exception classification (Connection, Timeout, Response, Client).
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx


def _get_config_var(name: str, default: str = "") -> str:
    """Retrieve configuration variable from environment, with fallback to local .env file.

    If the variable is explicitly present in os.environ (including empty string in tests),
    that value takes strict precedence without reading .env.
    """
    if name in os.environ:
        return os.environ[name].strip()
    try:
        from dotenv import dotenv_values
        vals = dotenv_values()
        if name in vals and vals[name] is not None:
            return str(vals[name]).strip()
    except Exception:
        pass
    return default


logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b-instruct"
DEFAULT_OLLAMA_TIMEOUT = 30.0


# ==============================================================================
# EXCEPTIONS - OLLAMA
# ==============================================================================


class OllamaClientError(Exception):
    """Base exception for all Ollama API interaction errors."""

    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when connection to Ollama daemon fails or daemon is offline."""

    pass


class OllamaTimeoutError(OllamaClientError):
    """Raised when an Ollama inference request times out."""

    pass


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns an error status code or malformed JSON payload."""

    pass


# ==============================================================================
# CLIENT IMPLEMENTATION - OLLAMA (LOCAL QWEN)
# ==============================================================================


class OllamaClient:
    """Asynchronous client for local self-hosted Ollama REST API.

    Guarantees:
    - Zero external network dependencies after initial weights are downloaded.
    - Single-concurrency inference via asyncio.Semaphore(1) to prevent CPU starvation.
    - Explicit timeout management (defaults to 30.0s).
    - Support for /api/chat and /api/generate endpoints.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        raw_base_url = (
            base_url if base_url is not None else _get_config_var("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        )
        self._base_url = str(raw_base_url).rstrip("/")

        raw_model = model if model is not None else _get_config_var("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self._model = str(raw_model).strip() or DEFAULT_OLLAMA_MODEL

        raw_timeout = (
            timeout
            if timeout is not None
            else float(_get_config_var("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT)))
        )
        self._timeout = float(raw_timeout)

        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._loop = None
        self._semaphore = asyncio.Semaphore(1)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> float:
        return self._timeout

    def is_configured(self) -> bool:
        """Local Ollama is always considered configured (no API key required)."""
        return True

    def _get_http_client(self) -> httpx.AsyncClient:
        if not self._owns_http_client and self._http_client is not None:
            return self._http_client
        current_loop = asyncio.get_running_loop()
        if self._http_client is None or self._http_client.is_closed or self._loop != current_loop:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
            self._loop = current_loop
            self._owns_http_client = True
        return self._http_client

    async def close(self) -> None:
        if self._owns_http_client and self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def is_available(self) -> bool:
        """Check if Ollama daemon is running and responsive."""
        client = self._get_http_client()
        try:
            resp = await client.get(f"{self._base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        """Send chat messages to Ollama /api/chat with bounded CPU concurrency."""
        async with self._semaphore:
            client = self._get_http_client()
            payload = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            }
            try:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.TimeoutException as exc:
                raise OllamaTimeoutError(f"Ollama request timed out after {self._timeout}s: {exc}") from exc
            except httpx.ConnectError as exc:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {self._base_url}: {exc}") from exc
            except Exception as exc:
                raise OllamaClientError(f"Ollama HTTP communication failed: {exc}") from exc

            if resp.status_code != 200:
                raise OllamaResponseError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

            try:
                data = resp.json()
                msg = data.get("message", {})
                content = msg.get("content", "")
                return str(content).strip()
            except Exception as exc:
                raise OllamaResponseError(f"Failed to parse Ollama response JSON: {exc}") from exc

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        num_predict: int = 350,
        repeat_penalty: float = 1.15,
    ) -> str:
        """Send prompt to Ollama /api/generate with bounded CPU concurrency."""
        async with self._semaphore:
            client = self._get_http_client()
            payload: Dict[str, Any] = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "repeat_penalty": repeat_penalty,
                },
            }
            if system:
                payload["system"] = system

            try:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.TimeoutException as exc:
                raise OllamaTimeoutError(f"Ollama request timed out after {self._timeout}s: {exc}") from exc
            except httpx.ConnectError as exc:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {self._base_url}: {exc}") from exc
            except Exception as exc:
                raise OllamaClientError(f"Ollama HTTP communication failed: {exc}") from exc

            if resp.status_code != 200:
                raise OllamaResponseError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

            try:
                data = resp.json()
                return str(data.get("response", "")).strip()
            except Exception as exc:
                raise OllamaResponseError(f"Failed to parse Ollama response JSON: {exc}") from exc


# ==============================================================================
# CLIENT FACTORY
# ==============================================================================


def get_default_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaClient:
    """Factory creating the production Ollama (Qwen) LLM client."""
    return OllamaClient(model=model, base_url=base_url)
