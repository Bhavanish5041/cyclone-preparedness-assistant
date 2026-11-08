"""Unified API client and data adapter for Cyclone Preparedness Decision Support.

Provides a standardized integration layer between the Streamlit frontend and the
decision-support backend supporting two execution modes:

1. Direct in-memory service mode (DEFAULT):
   Directly queries the in-process DecisionSupportService for ultra-low latency (<1ms)
   and standalone single-process execution without network overhead.

2. FastAPI REST mode (OPTIONAL):
   Dispatches HTTP requests to FastAPI REST endpoints (/api/v1/decision-support/...)
   using httpx, enabling distributed deployments and validating external API parity.

Both modes normalize data contracts, returning identical Pydantic response models
and preserving all scientific metadata, surrogate ML probabilities, provenance flags,
and explicit null values.
"""

from enum import Enum
import logging
import os
from typing import Any, Dict, List, Optional, Union
import httpx
import pandas as pd

from src.api.schemas import (
    DecisionEvaluationListResponse,
    DecisionSupportEvaluation,
    RegionalDecisionSummaryResponse,
)
from src.assistant.decision_support import DecisionSupportService

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS & EXCEPTIONS
# ==============================================================================

DEFAULT_ADAPTER_MODE = "direct"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 10.0

VALID_DISTRICTS = {"dakshina kannada", "udupi", "uttara kannada"}
VALID_PRIORITY_BANDS = {
    "BAND_1_CRITICAL",
    "BAND_2_SPECIALIZED",
    "BAND_3_MODERATE",
    "BAND_4_ADEQUATE",
}


class AdapterMode(str, Enum):
    """Execution mode for the decision support API client."""

    DIRECT = "direct"
    REST = "rest"


class ApiClientError(Exception):
    """Base exception for all decision support API client errors."""


class ApiClientConnectionError(ApiClientError):
    """Raised when REST connection fails, times out, or cannot reach backend."""


class ApiClientNotFoundError(ApiClientError, KeyError):
    """Raised when the requested administrative unit is not found."""


class ApiClientValidationError(ApiClientError, ValueError):
    """Raised when query or filter parameters are invalid."""


class ApiClientResponseError(ApiClientError):
    """Raised when backend response is malformed or returns an unexpected status."""


# ==============================================================================
# CLIENT IMPLEMENTATION
# ==============================================================================


class DecisionSupportApiClient:
    """Standardized client interface for cyclone preparedness decision support.

    Transparently abstracts direct in-memory service execution and FastAPI REST calls,
    ensuring identical method signatures, error semantics, and Pydantic response models.
    """

    def __init__(
        self,
        mode: Optional[Union[str, AdapterMode]] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        service: Optional[DecisionSupportService] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        """Initialize client adapter.

        Args:
            mode: 'direct' or 'rest'. Defaults to CYCLONE_ADAPTER_MODE env var or 'direct'.
            base_url: Base URL for REST mode. Defaults to CYCLONE_API_BASE_URL or 'http://localhost:8000'.
            timeout: HTTP request timeout in seconds for REST mode. Defaults to CYCLONE_API_TIMEOUT or 10.0.
            service: Optional pre-initialized DecisionSupportService instance for direct mode.
            http_client: Optional custom HTTP client (e.g. httpx.Client or TestClient) for REST mode.
        """
        # 1. Mode determination
        raw_mode = mode or os.getenv("CYCLONE_ADAPTER_MODE", DEFAULT_ADAPTER_MODE)
        if isinstance(raw_mode, AdapterMode):
            self._mode = raw_mode
        else:
            clean_mode = str(raw_mode).strip().lower()
            if clean_mode not in ("direct", "rest"):
                raise ApiClientValidationError(
                    f"Invalid adapter mode '{raw_mode}'. Allowed modes: 'direct', 'rest'."
                )
            self._mode = AdapterMode(clean_mode)

        # 2. In-memory service configuration
        self._service: Optional[DecisionSupportService] = service

        # 3. REST configuration
        raw_base_url = (
            base_url
            or os.getenv("CYCLONE_API_BASE_URL")
            or os.getenv("API_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self._base_url = raw_base_url.rstrip("/")

        raw_timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("CYCLONE_API_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
        )
        self._timeout = float(raw_timeout)
        self._http_client = http_client
        self._owns_http_client = http_client is None

        # 4. Local client-side caches (independent of Streamlit session state)
        self._summary_cache: Optional[RegionalDecisionSummaryResponse] = None
        self._unit_cache: Dict[str, DecisionSupportEvaluation] = {}
        self._unit_index_cache: Optional[pd.DataFrame] = None

        logger.info(
            f"Initialized DecisionSupportApiClient in '{self._mode.value}' mode "
            f"(base_url='{self._base_url}', timeout={self._timeout}s)."
        )

    @property
    def mode(self) -> AdapterMode:
        """Active execution mode ('direct' or 'rest')."""
        return self._mode

    @property
    def base_url(self) -> str:
        """Configured REST API base URL."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Configured REST timeout in seconds."""
        return self._timeout

    @property
    def service(self) -> DecisionSupportService:
        """Lazy-loaded DecisionSupportService instance for direct mode."""
        if self._service is None:
            self._service = DecisionSupportService()
        return self._service

    def _get_http_client(self) -> Any:
        """Obtain or lazily create an HTTP client instance."""
        if self._http_client is not None:
            return self._http_client
        self._http_client = httpx.Client(base_url=self._base_url, timeout=self._timeout)
        return self._http_client

    # --------------------------------------------------------------------------
    # CORE API CONTRACTS
    # --------------------------------------------------------------------------

    def get_regional_summary(
        self,
        force_refresh: bool = False,
    ) -> RegionalDecisionSummaryResponse:
        """Retrieve regional summary of priority bands, trigger counts, and model concordance.

        Args:
            force_refresh: If True, bypasses local summary cache and re-queries backend.

        Returns:
            RegionalDecisionSummaryResponse Pydantic model.

        Raises:
            ApiClientConnectionError: If REST connection fails or times out.
            ApiClientResponseError: If backend returns non-200 or invalid schema.
        """
        if not force_refresh and self._summary_cache is not None:
            return self._summary_cache

        if self._mode == AdapterMode.DIRECT:
            summary = self.service.get_regional_decision_summary()
        else:
            url = "/api/v1/decision-support/summary"
            client = self._get_http_client()
            try:
                resp = client.get(url)
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as err:
                raise ApiClientConnectionError(
                    f"Failed to connect to decision-support API at {self._base_url}: {err}"
                ) from err
            except Exception as err:
                raise ApiClientConnectionError(
                    f"Network error querying {url}: {err}"
                ) from err

            if resp.status_code != 200:
                raise ApiClientResponseError(
                    f"API returned HTTP {resp.status_code} for {url}: {resp.text}"
                )

            try:
                summary = RegionalDecisionSummaryResponse.model_validate(resp.json())
            except Exception as err:
                raise ApiClientResponseError(
                    f"Failed to parse RegionalDecisionSummaryResponse: {err}"
                ) from err

        self._summary_cache = summary
        return summary

    def get_unit(
        self,
        identifier: str,
        force_refresh: bool = False,
    ) -> DecisionSupportEvaluation:
        """Retrieve full decision-support evaluation and grounded advisories for a unit.

        Args:
            identifier: Census PC11 ID (e.g. '608720'), SHRID (e.g. '11-29-569-05522-608720'),
                        or village name (e.g. 'Edmoge').
            force_refresh: If True, bypasses local unit cache.

        Returns:
            DecisionSupportEvaluation Pydantic model.

        Raises:
            ApiClientValidationError: If identifier is empty or invalid.
            ApiClientNotFoundError: If unit cannot be found.
            ApiClientConnectionError: If REST connection fails or times out.
            ApiClientResponseError: If backend returns unexpected error.
        """
        if not identifier or not str(identifier).strip():
            raise ApiClientValidationError("Unit identifier must be a non-empty string.")

        clean_id = str(identifier).strip()
        cache_key = clean_id.lower()

        if not force_refresh and cache_key in self._unit_cache:
            return self._unit_cache[cache_key]

        if self._mode == AdapterMode.DIRECT:
            try:
                evaluation = self.service.evaluate_decision_unit(clean_id)
            except KeyError as err:
                raise ApiClientNotFoundError(
                    f"Administrative unit '{clean_id}' not found in unified preparedness dataset."
                ) from err
        else:
            url = f"/api/v1/decision-support/units/{clean_id}"
            client = self._get_http_client()
            try:
                resp = client.get(url)
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as err:
                raise ApiClientConnectionError(
                    f"Failed to connect to decision-support API at {self._base_url}: {err}"
                ) from err
            except Exception as err:
                raise ApiClientConnectionError(
                    f"Network error querying {url}: {err}"
                ) from err

            if resp.status_code == 404:
                raise ApiClientNotFoundError(
                    f"Administrative unit '{clean_id}' not found in unified preparedness dataset."
                )
            elif resp.status_code == 422:
                raise ApiClientValidationError(
                    f"Validation error querying unit '{clean_id}': {resp.text}"
                )
            elif resp.status_code != 200:
                raise ApiClientResponseError(
                    f"API returned HTTP {resp.status_code} for {url}: {resp.text}"
                )

            try:
                evaluation = DecisionSupportEvaluation.model_validate(resp.json())
            except Exception as err:
                raise ApiClientResponseError(
                    f"Failed to parse DecisionSupportEvaluation for '{clean_id}': {err}"
                ) from err

        # Cache by query key and standard identifiers
        self._unit_cache[cache_key] = evaluation
        if evaluation.pc11_id:
            self._unit_cache[evaluation.pc11_id.lower()] = evaluation
        if evaluation.shrid2:
            self._unit_cache[evaluation.shrid2.lower()] = evaluation
        if evaluation.village_name:
            self._unit_cache[evaluation.village_name.lower()] = evaluation

        return evaluation

    def get_units(
        self,
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[Dict[str, Any]] = None,
        *,
        district: Optional[str] = None,
        priority_band: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DecisionEvaluationListResponse:
        """Retrieve paginated decision evaluations with optional district and priority band filtering.

        Supports both dictionary parameters (`filters`, `pagination`) and explicit keyword arguments.

        Args:
            filters: Optional dict containing 'district' and/or 'priority_band'.
            pagination: Optional dict containing 'limit' and/or 'offset'.
            district: Optional district filter ('Dakshina Kannada', 'Udupi', 'Uttara Kannada').
            priority_band: Optional priority band ('BAND_1_CRITICAL', 'BAND_2_SPECIALIZED', etc.).
            limit: Maximum number of units to return per page (1 to 1879, default 50).
            offset: Offset index for pagination (default 0).

        Returns:
            DecisionEvaluationListResponse Pydantic model.

        Raises:
            ApiClientValidationError: If filters, limit, or offset are invalid.
            ApiClientConnectionError: If REST connection fails or times out.
            ApiClientResponseError: If backend returns unexpected error.
        """
        # Extract from dictionary structures if passed
        if filters:
            if "district" in filters and district is None:
                district = filters["district"]
            if "priority_band" in filters and priority_band is None:
                priority_band = filters["priority_band"]

        if pagination:
            if "limit" in pagination and limit == 50:
                limit = pagination["limit"]
            if "offset" in pagination and offset == 0:
                offset = pagination["offset"]

        # Validate parameters symmetrically across both modes
        if district and str(district).strip().lower() not in VALID_DISTRICTS:
            raise ApiClientValidationError(
                f"Invalid district '{district}'. Allowed: 'Dakshina Kannada', 'Udupi', 'Uttara Kannada'."
            )

        if priority_band and str(priority_band).strip().upper() not in VALID_PRIORITY_BANDS:
            raise ApiClientValidationError(
                f"Invalid priority_band '{priority_band}'. Allowed: {sorted(VALID_PRIORITY_BANDS)}."
            )

        if limit < 1 or limit > 1879:
            raise ApiClientValidationError(f"Limit must be between 1 and 1879, got {limit}.")

        if offset < 0:
            raise ApiClientValidationError(f"Offset must be >= 0, got {offset}.")

        if self._mode == AdapterMode.DIRECT:
            return self.service.get_decision_units(
                district=district,
                priority_band=priority_band,
                limit=limit,
                offset=offset,
            )
        else:
            url = "/api/v1/decision-support/units"
            params: Dict[str, Any] = {"limit": limit, "offset": offset}
            if district:
                params["district"] = str(district).strip()
            if priority_band:
                params["priority_band"] = str(priority_band).strip().upper()

            client = self._get_http_client()
            try:
                resp = client.get(url, params=params)
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as err:
                raise ApiClientConnectionError(
                    f"Failed to connect to decision-support API at {self._base_url}: {err}"
                ) from err
            except Exception as err:
                raise ApiClientConnectionError(
                    f"Network error querying {url}: {err}"
                ) from err

            if resp.status_code == 422:
                raise ApiClientValidationError(
                    f"Validation error querying decision units: {resp.text}"
                )
            elif resp.status_code != 200:
                raise ApiClientResponseError(
                    f"API returned HTTP {resp.status_code} for {url}: {resp.text}"
                )

            try:
                return DecisionEvaluationListResponse.model_validate(resp.json())
            except Exception as err:
                raise ApiClientResponseError(
                    f"Failed to parse DecisionEvaluationListResponse: {err}"
                ) from err

    def get_priority_distribution(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, int]:
        """Retrieve authoritative priority band counts across all 1,879 records.

        Returns:
            Dict mapping band names to unit counts:
            {'BAND_1_CRITICAL': 155, 'BAND_2_SPECIALIZED': 829, 'BAND_3_MODERATE': 299, 'BAND_4_ADEQUATE': 596}
        """
        summary = self.get_regional_summary(force_refresh=force_refresh)
        return summary.band_counts

    def get_unit_index(self) -> pd.DataFrame:
        """Retrieve a lightweight DataFrame index of all 1,879 units for instant search and map plotting.

        Returns:
            DataFrame with columns:
            ['shrid2', 'pc11_id', 'village_name', 'district_name', 'subdistrict_name',
             'provisional_priority_band', 'overall_priority_rank', 'band_rank',
             'preparedness_score', 'preparedness_tier', 'lat', 'lon']
        """
        if self._unit_index_cache is not None:
            return self._unit_index_cache

        target_columns = [
            "shrid2",
            "pc11_id",
            "village_name",
            "district_name",
            "subdistrict_name",
            "provisional_priority_band",
            "overall_priority_rank",
            "band_rank",
            "preparedness_score",
            "preparedness_tier",
            "vulnerability_archetype_id",
            "vulnerability_archetype_label",
            "lat",
            "lon",
        ]

        if self._mode == AdapterMode.DIRECT:
            ranked_df = self.service.ranked_df
            df = ranked_df.copy()
            # Normalize column name if present as 'priority_band'
            if "provisional_priority_band" not in df.columns and "priority_band" in df.columns:
                df["provisional_priority_band"] = df["priority_band"]

            avail_cols = [c for c in target_columns if c in df.columns]
            self._unit_index_cache = df[avail_cols].copy()
        else:
            # Query all 1,879 units from REST endpoint
            response = self.get_units(limit=1879)
            records = []
            for unit in response.units:
                records.append(
                    {
                        "shrid2": unit.shrid2,
                        "pc11_id": unit.pc11_id,
                        "village_name": unit.village_name,
                        "district_name": unit.district_name,
                        "subdistrict_name": unit.subdistrict_name,
                        "provisional_priority_band": unit.provisional_priority_band,
                        "overall_priority_rank": unit.overall_priority_rank,
                        "band_rank": unit.band_rank,
                        "preparedness_score": unit.preparedness_score,
                        "preparedness_tier": unit.preparedness_tier,
                        "vulnerability_archetype_id": unit.vulnerability_archetype_id,
                        "vulnerability_archetype_label": unit.vulnerability_archetype_label,
                        "lat": unit.lat,
                        "lon": unit.lon,
                    }
                )
            self._unit_index_cache = pd.DataFrame(records)

        return self._unit_index_cache

    def close(self) -> None:
        """Close internal HTTP client if owned by this instance."""
        if self._owns_http_client and self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

    def __enter__(self) -> "DecisionSupportApiClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ==============================================================================
# FACTORY & SINGLETON UTILITIES
# ==============================================================================

_CLIENT_SINGLETON: Optional[DecisionSupportApiClient] = None


def get_api_client(
    mode: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    force_new: bool = False,
) -> DecisionSupportApiClient:
    """Factory function returning a singleton or newly initialized DecisionSupportApiClient.

    Can be safely called within Streamlit components or standalone scripts.

    Args:
        mode: Optional mode override ('direct' or 'rest').
        base_url: Optional base URL override for REST mode.
        timeout: Optional timeout override in seconds for REST mode.
        force_new: If True, creates and caches a new instance.

    Returns:
        Configured DecisionSupportApiClient instance.
    """
    global _CLIENT_SINGLETON
    if force_new or _CLIENT_SINGLETON is None:
        _CLIENT_SINGLETON = DecisionSupportApiClient(
            mode=mode,
            base_url=base_url,
            timeout=timeout,
        )
    return _CLIENT_SINGLETON
