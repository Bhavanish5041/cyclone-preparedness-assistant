"""Asynchronous HTTP client for the Google Gemini Generative Language REST API.

Provides:
- Lightweight asynchronous calls via `httpx.AsyncClient` without heavy external SDKs.
- Multi-turn function calling payload assembly complying with Gemini v1beta REST specifications.
- Secure header-based authentication via `x-goog-api-key` preventing URL query-string leaks.
- Zero logging or exposure of API credentials across logs, errors, and traces.
- Robust exception classification (Auth, RateLimit, Timeout, Network, Response).
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 30.0


# ==============================================================================
# EXCEPTIONS
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
# CLIENT IMPLEMENTATION
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
            timeout: Request timeout in seconds. Defaults to GEMINI_TIMEOUT_SECONDS or 15.0.
            http_client: Optional custom httpx.AsyncClient (e.g. for testing mocks).
        """
        raw_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self._api_key = str(raw_key).strip()

        raw_model = model if model is not None else os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._model = str(raw_model).strip() or DEFAULT_GEMINI_MODEL

        raw_base_url = (
            base_url if base_url is not None else os.environ.get("GEMINI_API_BASE_URL", DEFAULT_GEMINI_BASE_URL)
        )
        self._base_url = str(raw_base_url).rstrip("/")

        raw_timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("GEMINI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        )
        self._timeout = float(raw_timeout)

        self._http_client = http_client
        self._owns_http_client = http_client is None

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
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
        return self._http_client

    async def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

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
