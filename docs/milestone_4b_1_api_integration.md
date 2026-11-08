# Milestone 4B.1: API Adapter and Data Integration Report
**Project:** Cyclone Preparedness Spatial Decision Support System  
**Repository:** `/home/bhavanish/cyclone-preparedness-assistant`  
**Status:** COMPLETE & VERIFIED (Milestone 4B.1)  
**Date:** November 2026  
**Module:** `src/assistant/api_client.py`  
**Test Suite:** `tests/test_api_client.py`  

---

## 1. Executive Summary

Milestone 4B.1 establishes a production-grade, standardized integration adapter (`src/assistant/api_client.py`) connecting the Streamlit frontend to the decision-support backend. The adapter transparently encapsulates two operational modes:

1. **Direct In-Memory Service Mode (Default)**: Direct in-process execution using `DecisionSupportService` via cached singleton. Provides sub-millisecond query responses (<1 ms), eliminates local network overhead, and allows single-command dashboard startup (`streamlit run src/assistant/app.py`).
2. **FastAPI REST Mode (Optional)**: Dispatches HTTP requests to FastAPI endpoints (`/api/v1/decision-support/...`) using `httpx`. Enables decoupled, distributed deployment and validates end-to-end API parity.

Both execution modes return identical Pydantic response models (`RegionalDecisionSummaryResponse`, `DecisionSupportEvaluation`, `DecisionEvaluationListResponse`), normalizing data contracts while preserving all scientific metadata, surrogate ML probabilities, provenance flags, and explicit null values without fabricated defaults.

---

## 2. Investigation of Audit vs Backend Priority-Band Counts Discrepancy

During the Milestone 4A architecture audit, an initial proposed draft KPI table listed provisional counts of `Band 1: 155`, `Band 2: 105`, `Band 3: 997`, `Band 4: 622`.

### Cause of Discrepancy
The audit draft reflected an earlier exploratory baseline before the finalized Medium-tier Band 2 rule was applied. In the approved Milestone 3A/3B specification, Medium-tier units with coastal proximity (<5 km), severe shelter deficits (`score_shelters` < 20), or severe medical deficits (`score_medical` < 35) are promoted into **Band 2 (Specialized Readiness)**.

### Authoritative Backend Reconciliation Across All 1,879 Units
The adapter dynamically retrieves and verifies the authoritative counts directly from the backend without hardcoding:

| Priority Band | Classification | Dakshina Kannada | Udupi | Uttara Kannada | Study Area Total | Percentage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Band 1** | Critical Priority | 2 | 0 | 153 | **155** | 8.25% |
| **Band 2** | Specialized Readiness | 123 | 20 | 686 | **829** | 44.12% |
| **Band 3** | Moderate Priority | 45 | 43 | 211 | **299** | 15.91% |
| **Band 4** | Adequate Baseline | 199 | 190 | 207 | **596** | 31.72% |
| **Total** | **All Units** | **369** | **253** | **1,257** | **1,879** | **100.00%** |

*Note: All 1,879 units are fully accounted for. The sum across all bands ($155 + 829 + 299 + 596 = 1,879$) and across all districts ($369 + 253 + 1,257 = 1,879$) reconciles with zero omissions.*

---

## 3. Adapter Architecture & Execution Modes

```
                              ┌────────────────────────────────────────┐
                              │          Streamlit Frontend            │
                              │  (Dashboard, Inspector, Map, Analytics)│
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │       DecisionSupportApiClient         │
                              │      (src/assistant/api_client.py)     │
                              └───────┬────────────────────────┬───────┘
                                      │                        │
                      [Mode: "direct"]│                        │[Mode: "rest"]
                                      ▼                        ▼
               ┌──────────────────────────────┐        ┌──────────────────────────────┐
               │    DecisionSupportService    │        │         FastAPI App          │
               │ (In-Process Cached Singleton)│        │     (http://localhost:8000)  │
               └──────────────┬───────────────┘        └──────────────┬───────────────┘
                              │                                       │
                              │                                       ▼
                              │                        ┌──────────────────────────────┐
                              │                        │  REST Endpoints (/api/v1/..) │
                              │                        └──────────────┬───────────────┘
                              │                                       │
                              ▼                                       ▼
               ┌──────────────────────────────────────────────────────────────┐
               │           Unified Processed Parquet Storage & Models         │
               │   • unified_village_preparedness_v2.parquet (1,879 units)    │
               │   • coastal_vulnerability_archetypes.parquet (622 units)     │
               │   • logistic_regression_tier_model.pkl & kmeans_model.pkl   │
               └──────────────────────────────────────────────────────────────┘
```

### 3.1 Mode Configuration & Environment Variables

The adapter determines its operational mode through constructor parameters or environment variables:

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `CYCLONE_ADAPTER_MODE` | `str` | `"direct"` | Operational mode: `"direct"` or `"rest"`. |
| `CYCLONE_API_BASE_URL` | `str` | `"http://localhost:8000"` | Base URL for FastAPI REST service. |
| `CYCLONE_API_TIMEOUT` | `float` | `10.0` | HTTP request timeout in seconds. |

### 3.2 Constructor & Factory Function

```python
from src.assistant.api_client import DecisionSupportApiClient, get_api_client

# Direct mode (default)
client = get_api_client(mode="direct")

# REST mode with custom endpoint
client_rest = DecisionSupportApiClient(
    mode="rest",
    base_url="http://api.internal:8000",
    timeout=15.0,
)
```

---

## 4. Method Contracts & Normalized Data Contracts

The adapter implements four core methods plus a lightweight indexing utility, with symmetrical signatures across both modes:

### 4.1 `get_regional_summary(force_refresh: bool = False) -> RegionalDecisionSummaryResponse`
- **Output Model**: `RegionalDecisionSummaryResponse`
- **Key Fields**:
  - `total_units`: `int` (1,879).
  - `band_counts`: `Dict[str, int]` (`BAND_1_CRITICAL`: 155, `BAND_2_SPECIALIZED`: 829, `BAND_3_MODERATE`: 299, `BAND_4_ADEQUATE`: 596).
  - `district_band_breakdown`: `Dict[str, Dict[str, int]]` (Full cross-tabulation across DK, Udupi, and UK).
  - `model_agreement_counts`: `Dict[str, int]` (`FULL_AGREEMENT`: 1,792, `BORDERLINE_DISAGREEMENT`: 87).
  - `provenance_agreement`: Accuracy breakdown: Train: 95.89%, Val: 93.97%, Test: 94.33%.
  - `uttara_kannada_trigger_breakdown`: Sector trigger overlap distribution.
  - `methodological_disclaimer`: Mandatory scientific caveat string.

### 4.2 `get_unit(identifier: str, force_refresh: bool = False) -> DecisionSupportEvaluation`
- **Identifier Support**: Census PC11 ID (e.g. `"608720"`), SHRID2 (e.g. `"11-29-569-05522-608720"`), or Village Name (e.g. `"Edmoge"`, `"Tallur"`, `"Bappalgundi"`).
- **Output Model**: `DecisionSupportEvaluation`
- **Data Preservation Standards**:
  - Null preservation: For Uttara Kannada units, `vulnerability_archetype_id` and `vulnerability_archetype_label` remain strictly `None`.
  - Cluster status: Explicitly tagged as `"UNASSIGNED_OUT_OF_STUDY_AREA"` for Uttara Kannada.
  - ML Probabilities: Preserved as raw class scores (`prob_low`, `prob_medium`, `prob_high`) without misrepresenting them as physical landfall probabilities.
  - Advisories: Iterated list of `InterventionItem` objects detailing directive ID, agency, directive focus, eligibility rationale, and supporting evidence metrics.

### 4.3 `get_units(...) -> DecisionEvaluationListResponse`
- **Signatures Supported**:
  - Symmetrical dictionary parameters: `get_units(filters={"district": "..."}, pagination={"limit": 50, "offset": 0})`.
  - Explicit keyword arguments: `get_units(district="Dakshina Kannada", priority_band="BAND_1_CRITICAL", limit=50, offset=0)`.
- **Validation**: Enforces strict validation across both modes:
  - Allowed districts: `Dakshina Kannada`, `Udupi`, `Uttara Kannada` (case-insensitive).
  - Allowed priority bands: `BAND_1_CRITICAL`, `BAND_2_SPECIALIZED`, `BAND_3_MODERATE`, `BAND_4_ADEQUATE`.
  - Limits: $1 \le \text{limit} \le 1879$; $\text{offset} \ge 0$.

### 4.4 `get_priority_distribution(force_refresh: bool = False) -> Dict[str, int]`
- Returns a clean dictionary of band counts for high-level UI metrics.

### 4.5 `get_unit_index() -> pd.DataFrame`
- Returns a cached lightweight DataFrame (1,879 rows $\times$ 12 columns) containing unit identifiers, names, coordinates, priority bands, overall ranks, and scores for high-speed map marker plotting and dropdown autocomplete indexing.

---

## 5. Error Handling & Failure Semantics

The adapter implements a robust, unified exception hierarchy ensuring consistent error handling regardless of mode:

```
Exception
└── ApiClientError
    ├── ApiClientConnectionError   # Raised on REST network timeouts or connection refusal
    ├── ApiClientNotFoundError    # Raised on 404 / missing unit (inherits from KeyError)
    ├── ApiClientValidationError   # Raised on 422 / invalid filters (inherits from ValueError)
    └── ApiClientResponseError     # Raised on 500 / malformed backend payloads
```

> [!IMPORTANT]
> **Strict Non-Fallback Rule**: If REST mode was explicitly requested and the connection fails (e.g. FastAPI server is not running), the adapter raises `ApiClientConnectionError`. It **never silently falls back** to direct in-memory execution, guaranteeing that integration failures are surfaced immediately to the user or test runner.

---

## 6. Caching Strategy

The adapter implements multi-tiered, client-side caching independent of mutable Streamlit session state:

1. **Regional Summary Cache (`_summary_cache`)**: Caches `RegionalDecisionSummaryResponse` in memory. Bypassed when `force_refresh=True`.
2. **Unit Evaluation Cache (`_unit_cache`)**: Caches individual `DecisionSupportEvaluation` objects indexed by PC11 ID, SHRID2, and village name. Repeated inspections of the same unit execute in 0 ms.
3. **Unit Index Cache (`_unit_index_cache`)**: Caches the 1,879-unit indexing DataFrame, eliminating repetitive slicing and serialization overhead across reruns.

---

## 7. Verification & Test Suite Results

The adapter is covered by 12 comprehensive unit and integration tests in [`tests/test_api_client.py`](file:///home/bhavanish/cyclone-preparedness-assistant/tests/test_api_client.py).

### 7.1 New Adapter Test Results (`tests/test_api_client.py`)

| Test Name | Coverage & Invariants Verified | Status |
| :--- | :--- | :---: |
| `test_direct_mode_regional_summary` | Total units = 1,879; band counts (155, 829, 299, 596); 3 districts; model agreement (1,792/87); Train: 95.89%, Val: 93.97%, Test: 94.33%. | **PASSED** |
| `test_direct_mode_unit_lookup` | Lookup by PC11 ("608720"), Name ("Tallur"), and SHRID2. Domain scores, ranks, advisories. | **PASSED** |
| `test_direct_mode_filtering_and_pagination` | Filtering by district (DK: 369, Udupi: 253, UK: 1257), bands (Band 1: 155, Band 2: 829), pagination disjoint offsets. | **PASSED** |
| `test_rest_mode_successful_responses` | End-to-end REST dispatch via FastAPI test transport; summary, unit, and list models. | **PASSED** |
| `test_rest_mode_connection_failures` | Connection refused simulation; verifies `ApiClientConnectionError` raised without silent fallback. | **PASSED** |
| `test_rest_mode_http_errors` | HTTP 500 (`ApiClientResponseError`), HTTP 422 (`ApiClientValidationError`), malformed JSON schema. | **PASSED** |
| `test_invalid_unit_identifiers` | Non-existent units raise `ApiClientNotFoundError` (KeyError); empty/whitespace strings raise `ApiClientValidationError`. | **PASSED** |
| `test_missing_and_null_fields_preservation` | Uttara Kannada unit ("Bappalgundi"): archetype ID is `None`, status is `UNASSIGNED_OUT_OF_STUDY_AREA`, no fabricated defaults. | **PASSED** |
| `test_consistency_between_direct_and_rest_modes` | 100% field-for-field parity between Direct and REST models for summaries, units, and distributions. | **PASSED** |
| `test_priority_band_aggregation_integrity` | Sum across bands = 1,879; sum across districts = 1,879 (DK: 369, Udupi: 253, UK: 1257). | **PASSED** |
| `test_client_caching_and_unit_index` | Summary and unit cache hit verification; force_refresh bypass; unit index DataFrame 1,879 rows $\times$ 12 cols. | **PASSED** |
| `test_adapter_initialization_and_lifecycle` | Invalid mode validation, `CYCLONE_ADAPTER_MODE` env var parsing, context manager (`__enter__`/`__exit__`). | **PASSED** |

### 7.2 Full Repository Test Suite Status

```bash
.venv/bin/pytest
```
- **Total Tests**: **175 passed** (163 existing baseline + 12 new Milestone 4B.1 tests).
- **Failures / Errors**: **0**.
- **Regressions**: **0**.

---

## 8. Conclusion

Milestone 4B.1 is complete. The integration layer `src/assistant/api_client.py` is robust, fully tested, and ready to serve as the unified data backbone for Milestone 4B.2 UI component development.

**PRESERVATION BOUNDARIES CONFIRMATION:**
- Only `src/assistant/api_client.py`, `tests/test_api_client.py`, and documentation were created.
- No existing backend formulas, priority-band rules, datasets, or models were modified.
- No Git operations were performed.
