"""Asynchronous HTTP clients for OpenRouter and Google Gemini REST APIs.

Provides:
- Lightweight asynchronous calls via `httpx.AsyncClient` without heavy external SDKs.
- OpenRouter OpenAI-compatible chat completion client with native function/tool calling.
- Multi-turn function calling payload assembly complying with REST specifications.
- Secure header-based authentication preventing URL query-string leaks.
- Zero logging or exposure of API credentials across logs, errors, and traces.
- Robust exception classification (Auth, RateLimit, Timeout, Network, Response).
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union
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

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 30.0

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_OPENROUTER_TIMEOUT = 30.0

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
# EXCEPTIONS - GEMINI
# ==============================================================================


class GeminiClientError(Exception):
    """Base exception for all Gemini API interaction errors."""

    pass


class GeminiAuthError(GeminiClientError):
    """Raised when authentication fails (HTTP 401/403) or API key is invalid."""

    pass


class GeminiRateLimitError(GeminiClientError):
    """Raised when API quotas or rate limits are exceeded (HTTP 429)."""

    pass


class GeminiTimeoutError(GeminiClientError):
    """Raised when an API request times out."""

    pass


class GeminiNetworkError(GeminiClientError):
    """Raised when network transport or connection fails."""

    pass


class GeminiResponseError(GeminiClientError):
    """Raised when the API returns an unexpected error or malformed payload."""

    pass


# ==============================================================================
# EXCEPTIONS - OPENROUTER
# ==============================================================================


class OpenRouterClientError(Exception):
    """Base exception for all OpenRouter API interaction errors."""

    pass


class OpenRouterAuthError(OpenRouterClientError):
    """Raised when OpenRouter authentication fails (HTTP 401/403) or API key is invalid."""

    pass


class OpenRouterRateLimitError(OpenRouterClientError):
    """Raised when OpenRouter rate limits or upstream provider quotas are hit (HTTP 429)."""

    pass


class OpenRouterTimeoutError(OpenRouterClientError):
    """Raised when an OpenRouter API request times out."""

    pass


class OpenRouterNetworkError(OpenRouterClientError):
    """Raised when network connection to OpenRouter fails."""

    pass


class OpenRouterResponseError(OpenRouterClientError):
    """Raised when OpenRouter returns an unexpected error code or malformed payload."""

    pass


# ==============================================================================
# CLIENT IMPLEMENTATION - GEMINI
# ==============================================================================


class GeminiClient:
    """Asynchronous client for Google Gemini v1beta generateContent REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize client with environment variable fallbacks.

        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            model: Model name. Defaults to GEMINI_MODEL environment variable or 'gemini-2.5-flash'.
            base_url: Base URL for Gemini API. Defaults to GEMINI_API_BASE_URL or default Google endpoint.
            timeout: Request timeout in seconds. Defaults to GEMINI_TIMEOUT_SECONDS or 30.0.
            http_client: Optional custom httpx.AsyncClient (e.g. for testing mocks).
        """
        raw_key = api_key if api_key is not None else _get_config_var("GEMINI_API_KEY", "")
        self._api_key = str(raw_key).strip()

        raw_model = model if model is not None else _get_config_var("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._model = str(raw_model).strip() or DEFAULT_GEMINI_MODEL

        raw_base_url = (
            base_url if base_url is not None else _get_config_var("GEMINI_API_BASE_URL", DEFAULT_GEMINI_BASE_URL)
        )
        self._base_url = str(raw_base_url).rstrip("/")

        raw_timeout = (
            timeout
            if timeout is not None
            else float(_get_config_var("GEMINI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        )
        self._timeout = float(raw_timeout)

        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._loop = None

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "gemini"

    @property
    def model(self) -> str:
        """Active model identifier."""
        return self._model

    @property
    def base_url(self) -> str:
        """Active API base URL."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Configured timeout in seconds."""
        return self._timeout

    def is_configured(self) -> bool:
        """Check whether an API key is provided and non-empty."""
        return bool(self._api_key)

    def _get_http_client(self) -> httpx.AsyncClient:
        """Obtain or lazily create the asynchronous HTTP client."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._owns_http_client:
            if self._http_client is None or self._http_client.is_closed or self._loop != current_loop:
                old_client = self._http_client
                old_loop = self._loop
                if old_client is not None and not old_client.is_closed:
                    if old_loop is not None and not old_loop.is_closed() and old_loop.is_running():
                        try:
                            asyncio.run_coroutine_threadsafe(old_client.aclose(), old_loop)
                        except Exception:
                            pass
                self._http_client = httpx.AsyncClient(timeout=self._timeout)
                self._loop = current_loop
        elif self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)

        return self._http_client

    async def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_http_client and self._http_client is not None:
            if not self._http_client.is_closed:
                try:
                    await self._http_client.aclose()
                except Exception:
                    pass
            self._http_client = None
            self._loop = None

    def _mask_api_key_in_string(self, text: str) -> str:
        """Scrub any occurrence of the active API key from error strings or logs."""
        if self._api_key and len(self._api_key) >= 6:
            return text.replace(self._api_key, "[REDACTED_API_KEY]")
        return text

    async def generate_content(
        self,
        contents: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch a generateContent request to the Gemini REST API.

        Args:
            contents: Multi-turn contents array matching Gemini schema.
            tools: Optional tool definitions (e.g. `[{"functionDeclarations": [...]}]`).
            system_instruction: Optional system instruction text.
            generation_config: Optional generation configuration dictionary.

        Returns:
            Parsed JSON response dictionary from Gemini.

        Raises:
            GeminiAuthError: If authentication fails (401/403) or API key is missing.
            GeminiRateLimitError: If rate limit is hit (429).
            GeminiTimeoutError: If the request times out.
            GeminiNetworkError: If transport/network connection fails.
            GeminiResponseError: If server returns 5xx or invalid payload.
        """
        if not self.is_configured():
            raise GeminiAuthError(
                "Gemini API key is not configured. Set the GEMINI_API_KEY environment variable."
            )

        endpoint_url = f"{self._base_url}/v1beta/models/{self._model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        body: Dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if tools:
            body["tools"] = tools

        config = {"temperature": 0.0, "maxOutputTokens": 2048}
        if generation_config:
            config.update(generation_config)
        body["generationConfig"] = config

        client = self._get_http_client()

        try:
            resp = await client.post(endpoint_url, json=body, headers=headers)
        except (httpx.TimeoutException, TimeoutError) as err:
            clean_err = self._mask_api_key_in_string(str(err))
            logger.warning(f"Gemini API request timed out ({self._timeout}s): {clean_err}")
            raise GeminiTimeoutError(f"Gemini request timed out after {self._timeout} seconds.") from err
        except (httpx.ConnectError, httpx.NetworkError, httpx.TransportError) as err:
            clean_err = self._mask_api_key_in_string(str(err))
            logger.warning(f"Gemini API network connection error: {clean_err}")
            raise GeminiNetworkError(f"Gemini network connection error: {clean_err}") from err
        except Exception as err:
            clean_err = self._mask_api_key_in_string(str(err))
            logger.error(f"Unexpected transport failure calling Gemini API: {clean_err}")
            raise GeminiClientError(f"Unexpected error querying Gemini API: {clean_err}") from err

        # Process HTTP response codes
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as err:
                raise GeminiResponseError(f"Failed to parse Gemini response as JSON: {err}") from err

        clean_resp_text = self._mask_api_key_in_string(resp.text)

        if resp.status_code in (401, 403):
            logger.error(f"Gemini API authentication failed (HTTP {resp.status_code}): {clean_resp_text}")
            raise GeminiAuthError(
                f"Gemini API authentication failed (HTTP {resp.status_code}): {clean_resp_text}"
            )
        elif resp.status_code == 429:
            logger.warning(f"Gemini API rate limit exceeded (HTTP 429): {clean_resp_text}")
            raise GeminiRateLimitError(
                "Gemini API rate limit or quota exceeded (HTTP 429). Please retry shortly."
            )
        elif resp.status_code == 404 and "no longer available to new users" in clean_resp_text:
            # Fallback for newer Google AI Studio API accounts where gemini-2.5-flash is retired
            fallback_model = "gemini-3.5-flash"
            logger.warning(
                f"Configured model '{self._model}' is retired by Google API ({clean_resp_text}). "
                f"Automatically retrying with active model '{fallback_model}'."
            )
            fallback_url = f"{self._base_url}/v1beta/models/{fallback_model}:generateContent"
            retry_resp = await client.post(fallback_url, json=body, headers=headers)
            if retry_resp.status_code == 200:
                self._model = fallback_model
                try:
                    return retry_resp.json()
                except Exception as err:
                    raise GeminiResponseError(f"Failed to parse Gemini response as JSON: {err}") from err
            clean_resp_text = self._mask_api_key_in_string(retry_resp.text)
            resp = retry_resp
            if resp.status_code >= 500:
                raise GeminiResponseError(f"Gemini upstream server error (HTTP {resp.status_code}): {clean_resp_text}")
            else:
                raise GeminiResponseError(f"Gemini API returned HTTP {resp.status_code}: {clean_resp_text}")
        elif resp.status_code >= 500:
            logger.error(f"Gemini API server error (HTTP {resp.status_code}): {clean_resp_text}")
            raise GeminiResponseError(
                f"Gemini upstream server error (HTTP {resp.status_code}): {clean_resp_text}"
            )
        else:
            logger.error(f"Gemini API returned unexpected HTTP {resp.status_code}: {clean_resp_text}")
            raise GeminiResponseError(
                f"Gemini API returned HTTP {resp.status_code}: {clean_resp_text}"
            )

    @staticmethod
    def extract_function_calls(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract function call objects from a Gemini candidate response.

        Returns:
            List of dictionaries, each containing:
            {'name': 'tool_name', 'args': {...}}
        """
        calls: List[Dict[str, Any]] = []
        candidates = response_json.get("candidates", [])
        if not candidates:
            return calls

        content = candidates[0].get("content", {})
        for part in content.get("parts", []):
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append(
                    {
                        "name": str(fc.get("name", "")),
                        "args": fc.get("args") or {},
                    }
                )
        return calls

    @staticmethod
    def extract_text_response(response_json: Dict[str, Any]) -> Optional[str]:
        """Extract natural language text from a Gemini candidate response.

        Returns:
            String text or None if no text parts are present.
        """
        candidates = response_json.get("candidates", [])
        if not candidates:
            return None

        content = candidates[0].get("content", {})
        texts: List[str] = []
        for part in content.get("parts", []):
            if "text" in part:
                texts.append(part["text"])

        return "\n".join(texts).strip() if texts else None


# ==============================================================================
# CLIENT IMPLEMENTATION - OPENROUTER
# ==============================================================================


class OpenRouterClient:
    """Asynchronous client for OpenRouter /chat/completions OpenAI-compatible REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize OpenRouter client with environment variable fallbacks.

        Args:
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY environment variable.
            model: Model identifier. Defaults to OPENROUTER_MODEL or 'nvidia/nemotron-3.5-lightning:free'.
            base_url: Base URL for OpenRouter API. Defaults to OPENROUTER_BASE_URL or 'https://openrouter.ai/api/v1'.
            timeout: Request timeout in seconds. Defaults to OPENROUTER_TIMEOUT or 30.0.
            http_client: Optional custom httpx.AsyncClient (e.g. for testing mocks).
        """
        raw_key = api_key if api_key is not None else _get_config_var("OPENROUTER_API_KEY", "")
        self._api_key = str(raw_key).strip()

        raw_model = model if model is not None else _get_config_var("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        self._model = str(raw_model).strip() or DEFAULT_OPENROUTER_MODEL

        raw_base_url = (
            base_url if base_url is not None else _get_config_var("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
        )
        self._base_url = str(raw_base_url).rstrip("/")

        raw_timeout = (
            timeout
            if timeout is not None
            else float(_get_config_var("OPENROUTER_TIMEOUT", str(DEFAULT_OPENROUTER_TIMEOUT)))
        )
        self._timeout = float(raw_timeout)

        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._loop = None

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "openrouter"

    @property
    def model(self) -> str:
        """Active model identifier."""
        return self._model

    @property
    def base_url(self) -> str:
        """Active API base URL."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Configured timeout in seconds."""
        return self._timeout

    def is_configured(self) -> bool:
        """Check whether an API key is provided and non-empty."""
        return bool(self._api_key)

    def _get_http_client(self) -> httpx.AsyncClient:
        """Obtain or lazily create the asynchronous HTTP client."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._owns_http_client:
            if self._http_client is None or self._http_client.is_closed or self._loop != current_loop:
                old_client = self._http_client
                old_loop = self._loop
                if old_client is not None and not old_client.is_closed:
                    if old_loop is not None and not old_loop.is_closed() and old_loop.is_running():
                        try:
                            asyncio.run_coroutine_threadsafe(old_client.aclose(), old_loop)
                        except Exception:
                            pass
                self._http_client = httpx.AsyncClient(timeout=self._timeout)
                self._loop = current_loop
        elif self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)

        return self._http_client

    async def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_http_client and self._http_client is not None:
            if not self._http_client.is_closed:
                try:
                    await self._http_client.aclose()
                except Exception:
                    pass
            self._http_client = None
            self._loop = None

    def _mask_api_key_in_string(self, text: str) -> str:
        """Scrub any occurrence of the active API key from error strings or logs."""
        if self._api_key and len(self._api_key) >= 6:
            return text.replace(self._api_key, "[REDACTED_API_KEY]")
        return text

    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Dispatch a chat completion request to the OpenRouter REST API.

        Args:
            messages: List of OpenAI-compatible message dictionaries.
            tools: Optional tool definitions in standard OpenAI format.
            temperature: Sampling temperature (default 0.0 for deterministic outputs).
            max_tokens: Maximum tokens in generated completion.

        Returns:
            Parsed JSON response dictionary from OpenRouter.

        Raises:
            OpenRouterAuthError: If authentication fails (401/403) or API key is missing.
            OpenRouterRateLimitError: If rate limit or upstream quota is hit (429).
            OpenRouterTimeoutError: If the request times out.
            OpenRouterNetworkError: If transport/network connection fails.
            OpenRouterResponseError: If server returns 5xx or invalid payload.
        """
        if not self.is_configured():
            raise OpenRouterAuthError(
                "OpenRouter API key is not configured. Set the OPENROUTER_API_KEY environment variable."
            )

        endpoint_url = f"{self._base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://cyclone-preparedness-assistant.onrender.com",
            "X-Title": "Cyclone Preparedness Assistant",
        }

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            body["tools"] = tools

        client = self._get_http_client()

        try:
            resp = await client.post(endpoint_url, json=body, headers=headers)
        except (httpx.TimeoutException, TimeoutError) as err:
            clean_err = self._mask_api_key_in_string(str(err))
            logger.warning(f"OpenRouter API request timed out ({self._timeout}s): {clean_err}")
            raise OpenRouterTimeoutError(f"OpenRouter request timed out after {self._timeout} seconds.") from err
        except (httpx.ConnectError, httpx.NetworkError, httpx.TransportError) as err:
            clean_err = self._mask_api_key_in_string(str(err))
            logger.warning(f"OpenRouter API network connection error: {clean_err}")
            raise OpenRouterNetworkError(f"OpenRouter network connection error: {clean_err}") from err
        except Exception as err:
            clean_err = self._mask_api_key_in_string(str(err))
            logger.error(f"Unexpected transport failure calling OpenRouter API: {clean_err}")
            raise OpenRouterClientError(f"Unexpected error querying OpenRouter API: {clean_err}") from err

        # Process HTTP response codes
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as err:
                raise OpenRouterResponseError(f"Failed to parse OpenRouter response as JSON: {err}") from err

        clean_resp_text = self._mask_api_key_in_string(resp.text)

        if resp.status_code in (401, 403):
            logger.error(f"OpenRouter API authentication failed (HTTP {resp.status_code}): {clean_resp_text}")
            raise OpenRouterAuthError(
                f"OpenRouter API authentication failed (HTTP {resp.status_code}): {clean_resp_text}"
            )
        elif resp.status_code == 429:
            logger.warning(f"OpenRouter API rate limit exceeded (HTTP 429): {clean_resp_text}")
            raise OpenRouterRateLimitError(
                "OpenRouter API rate limit or upstream quota exceeded (HTTP 429). Please retry shortly."
            )
        elif resp.status_code >= 500:
            logger.error(f"OpenRouter API server error (HTTP {resp.status_code}): {clean_resp_text}")
            raise OpenRouterResponseError(
                f"OpenRouter upstream server error (HTTP {resp.status_code}): {clean_resp_text}"
            )
        else:
            logger.error(f"OpenRouter API returned unexpected HTTP {resp.status_code}: {clean_resp_text}")
            raise OpenRouterResponseError(
                f"OpenRouter API returned HTTP {resp.status_code}: {clean_resp_text}"
            )

    @staticmethod
    def extract_tool_calls(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool call objects from an OpenRouter chat completion response.

        Returns:
            List of dictionaries, each containing:
            {'id': 'call_xxx', 'name': 'tool_name', 'args': {...}}
        """
        calls: List[Dict[str, Any]] = []
        choices = response_json.get("choices", [])
        if not choices:
            return calls

        message = choices[0].get("message", {})
        raw_tool_calls = message.get("tool_calls", [])
        if not raw_tool_calls:
            return calls

        for tc in raw_tool_calls:
            tc_id = str(tc.get("id") or "")
            func = tc.get("function", {})
            name = str(func.get("name") or "")
            args_raw = func.get("arguments", "{}")

            if isinstance(args_raw, dict):
                args = args_raw
            elif isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw.strip() else {}
                except Exception:
                    args = {}
            else:
                args = {}

            calls.append(
                {
                    "id": tc_id,
                    "name": name,
                    "args": args,
                }
            )
        return calls

    @staticmethod
    def extract_text_response(response_json: Dict[str, Any]) -> Optional[str]:
        """Extract natural language text from an OpenRouter chat completion response.

        Returns:
            String text or None if no content is present.
        """
        choices = response_json.get("choices", [])
        if not choices:
            return None

        message = choices[0].get("message", {})
        content = message.get("content")
        return str(content).strip() if content is not None else None


# ==============================================================================
# CLIENT IMPLEMENTATION - OLLAMA (LOCAL)
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
        # Local Ollama is always considered configured (no API key required)
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
) -> Union[OllamaClient, OpenRouterClient, GeminiClient]:
    """Factory creating the configured LLM client.

    Resolution order:
    1. Explicit provider parameter ('ollama', 'openrouter', or 'gemini').
    2. LLM_PROVIDER environment variable ('ollama', 'openrouter', or 'gemini').
    3. If OPENROUTER_API_KEY is present in environment, OpenRouter is primary.
    4. If GEMINI_API_KEY is present in environment, Gemini is fallback.
    5. Default to OllamaClient (local self-hosted mode).
    """
    raw_provider = provider or _get_config_var("LLM_PROVIDER")
    if raw_provider:
        prov = str(raw_provider).strip().lower()
        if prov == "ollama":
            return OllamaClient(model=model, base_url=base_url)
        elif prov == "gemini":
            return GeminiClient(api_key=api_key, model=model, base_url=base_url)
        elif prov == "openrouter":
            return OpenRouterClient(api_key=api_key, model=model, base_url=base_url)

    if _get_config_var("OPENROUTER_API_KEY", "").strip():
        return OpenRouterClient(api_key=api_key, model=model)

    if _get_config_var("GEMINI_API_KEY", "").strip():
        return GeminiClient(api_key=api_key, model=model)

    return OllamaClient(model=model, base_url=base_url)
