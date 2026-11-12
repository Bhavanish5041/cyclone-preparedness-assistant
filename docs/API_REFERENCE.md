# CYCLONE INTEL: Canonical FastAPI API Reference

> **Platform Version:** 2.0.0  
> **Specification:** OpenAPI 3.1.0  
> **Total Registered Endpoints:** 37  
> **Base URL:** `/api` (Production canonical application `api.main:app`)  

## Overview and Architectural Principles

1. **Authoritative AHP Precedence**: Deterministic Analytic Hierarchy Process (AHP) composite indices represent the single authoritative ground truth for coastal preparedness across all 1,879 administrative units. Machine Learning surrogates serve purely empirical diagnostic purposes and never supersede AHP.
2. **Non-Downgrading Invariant**: The surrogate classifier is structurally barred from downgrading an authoritative AHP Low preparedness classification.
3. **Descriptive Archetypes**: K-Means clustering (k=4) identifies descriptive structural vulnerability archetypes across Dakshina Kannada and Udupi. The 1,257 administrative units in Uttara Kannada are explicitly tracked under unassigned fallback heuristics.
4. **Documentation Drift Detection**: This catalogue is verified programmatically by `tests/test_route_inventory.py` to ensure complete alignment with live registered FastAPI routers.

## Interactive Documentation Interfaces

- **Swagger UI (Local & Production)**: [`/docs`](http://localhost:8000/docs)
- **ReDoc Interface**: [`/redoc`](http://localhost:8000/redoc)
- **OpenAPI JSON Schema**: [`/openapi.json`](http://localhost:8000/openapi.json)

## Logical Endpoint Groups

- [Health and Readiness](#health-and-readiness) (9 endpoints) — System operational status, dataset readiness verification, and orchestrator probes.
- [Dashboard](#dashboard) (1 endpoints) — Regional command-center executive metrics, KPI summaries, and district overviews.
- [Communities](#communities) (2 endpoints) — Granular administrative unit records, full-text search, multi-criteria filtering, and detailed dossiers.
- [Geospatial and Maps](#geospatial-and-maps) (1 endpoints) — Spatial points, coordinate bounding boxes, and GIS filtering for interactive map visualization.
- [Districts](#districts) (2 endpoints) — Comparative district-level profiles, demographic totals, and cross-district preparedness differentials.
- [Analytics](#analytics) (4 endpoints) — Authoritative AHP distributions, surrogate model performance diagnostics, and operational distributions.
- [Priority Bands](#priority-bands) (0 endpoints) — Four-tier operational resource prioritization (Band 1 Critical to Band 4 Adequate) distributions.
- [Vulnerability Clusters](#vulnerability-clusters) (1 endpoints) — K-Means (k=4) unsupervised vulnerability archetypes and Uttara Kannada fallback documentation.
- [Infrastructure and Interventions](#infrastructure-and-interventions) (2 endpoints) — Domain-specific infrastructure deficit analysis and targeted physical intervention recommendations.
- [Model Inference](#model-inference) (2 endpoints) — Empirical surrogate Logistic Regression classifier inference and probability output generation.
- [Model Registry](#model-registry) (2 endpoints) — Local Git-compatible Model Registry catalog inspection, artifact provenance, and version metadata.
- [Assistant](#assistant) (2 endpoints) — Conversational spatial decision support with deterministic AHP tool grounding and refusal gates.
- [Legacy Compatibility Routes](#legacy-v1) (9 endpoints) — Backward-compatible REST endpoints preserving contract compatibility with v1 clients.

---
## Health and Readiness
*System operational status, dataset readiness verification, and orchestrator probes.*

### `GET /`

**Purpose**: Redirect root traffic to interactive Swagger UI documentation.  
**API Tag**: `Navigation` | **Authentication**: `None` | **Status Codes**: `307`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -i http://localhost:8000/
```

**Example Response**:
```json
HTTP/1.1 307 Temporary Redirect
Location: /docs
```

### `GET /api/health`

**Purpose**: System health, dataset readiness, and model status  
**API Tag**: `Health & Status` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/health
```

**Example Response**:
```json
{
  "status": "healthy",
  "dataset_loaded": true,
  "total_units": 1879,
  "total_villages": 1845,
  "total_census_towns": 34,
  "districts": [
    "Dakshina Kannada",
    "Udupi",
    "Uttara Kannada"
  ],
  "archetypes_count": 4,
  "version": "2.0.0",
  "model_loaded": true,
  "model_artifact_name": "village-preparedness-classifier",
  "model_artifact_version": "v1.0.0",
  "model_feature_schema_status": "VALID_26_FEATURES",
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION (70/15/15 experiment)"
}
```

### `GET /api/health/ready`

**Purpose**: System readiness probe for container orchestrators  
**API Tag**: `Health & Status` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Example Request**:
```bash
curl -s http://localhost:8000/api/health/ready
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/health/ready"
}
```

### `GET /docs`

**Purpose**: Interactive Swagger UI for endpoint exploration and testing.  
**API Tag**: `Documentation` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl http://localhost:8000/docs
```

**Example Response**:
```json
<!DOCTYPE html><html><head><title>Swagger UI</title>...
```

### `GET /health`

**Purpose**: Direct system health, dataset readiness, and model status  
**API Tag**: `Health & Status` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/health
```

**Example Response**:
```json
{
  "status": "healthy",
  "dataset_loaded": true,
  "total_units": 1879,
  "total_villages": 1845,
  "total_census_towns": 34,
  "districts": [
    "Dakshina Kannada",
    "Udupi",
    "Uttara Kannada"
  ],
  "archetypes_count": 4,
  "version": "2.0.0",
  "model_loaded": true,
  "model_artifact_name": "village-preparedness-classifier",
  "model_artifact_version": "v1.0.0",
  "model_feature_schema_status": "VALID_26_FEATURES",
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION (70/15/15 experiment)"
}
```

### `GET /metrics`

**Purpose**: Prometheus operational metrics exposition endpoint with bounded cardinality.  
**API Tag**: `Observability` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Example Request**:
```bash
curl http://localhost:8000/metrics
```

**Example Response**:
```json
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/health",method="GET",status="200"} 12
```

### `GET /openapi.json`

**Purpose**: Machine-readable OpenAPI 3.1 specification for tooling and client generation.  
**API Tag**: `Documentation` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl http://localhost:8000/openapi.json
```

**Example Response**:
```json
{"openapi": "3.1.0", "info": {"title": "Cyclone Preparedness Decision Intelligence API"}}
```

### `GET /ready`

**Purpose**: Root readiness probe for container orchestration confirming dataset and surrogate model are loaded.  
**API Tag**: `Health & Status` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Example Request**:
```bash
curl http://localhost:8000/ready
```

**Example Response**:
```json
{"ready": true, "dataset_loaded": true, "model_loaded": true, "version": "2.0.0"}
```

### `GET /redoc`

**Purpose**: ReDoc responsive documentation browser.  
**API Tag**: `Documentation` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl http://localhost:8000/redoc
```

**Example Response**:
```json
<!DOCTYPE html><html><head><title>ReDoc</title>...
```

---
## Dashboard
*Regional command-center executive metrics, KPI summaries, and district overviews.*

### `GET /api/dashboard/overview`

**Purpose**: Regional command-center executive summary  
**API Tag**: `Dashboard` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/dashboard/overview
```

**Example Response**:
```json
{
  "total_units": 1879,
  "districts_count": 3,
  "assigned_clusters_count": 622,
  "unassigned_clusters_count": 1257,
  "priority_band_counts": {
    "BAND_1_CRITICAL": 155,
    "BAND_2_SPECIALIZED": 829,
    "BAND_3_MODERATE": 299,
    "BAND_4_ADEQUATE": 596
  },
  "preparedness_tier_counts": {
    "High": 596,
    "Medium": 1128,
    "Low": 155
  },
  "infrastructure_highlights": {
    "total_population": 4680000,
    "pucca_road_pct": 78.5,
    "coastal_units_under_5km": 249
  }
}
```

---
## Communities
*Granular administrative unit records, full-text search, multi-criteria filtering, and detailed dossiers.*

### `GET /api/communities`

**Purpose**: Search, filter, and paginate communities  
**API Tag**: `Communities` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Query Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `search` | `No` | `string` | Search by village name, PC11 ID, or SHRID |
| `district` | `No` | `string` | Filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada') |
| `tier` | `No` | `string` | Filter by AHP preparedness tier ('High', 'Medium', 'Low') |
| `priority_band` | `No` | `string` | Filter by priority band ('BAND_1_CRITICAL', 'BAND_2_SPECIALIZED', 'BAND_3_MODERATE', 'BAND_4_ADEQUATE') |
| `archetype_id` | `No` | `string` | Filter by archetype ID (0, 1, 2, 3 or -1 for unassigned) |
| `sort_by` | `No` | `string` | Field to sort by ('priority_rank', 'preparedness_score', 'dist_coast_km', 'population', 'village_name') |
| `sort_order` | `No` | `string` | Sort order ('asc' or 'desc') |
| `page` | `No` | `integer` | Page number starting from 1 |
| `limit` | `No` | `integer` | Items per page (max 100) |

**Example Request**:
```bash
curl -s 'http://localhost:8000/api/communities?district=Udupi&priority_band=BAND_4_ADEQUATE&limit=2'
```

**Example Response**:
```json
{
  "items": [
    {
      "area_id": "608691",
      "village_name": "Tallur",
      "district_name": "Udupi",
      "preparedness_score": 74.34,
      "preparedness_tier": "High",
      "priority_band": "BAND_4_ADEQUATE",
      "vulnerability_archetype_id": 3
    }
  ],
  "total": 123,
  "page": 1,
  "limit": 2,
  "pages": 62
}
```

### `GET /api/communities/{community_id}`

**Purpose**: Retrieve granular community inspection dossier  
**API Tag**: `Communities` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Path Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `community_id` | `Yes` | `string` | — |

**Example Request**:
```bash
curl -s http://localhost:8000/api/communities/608691
```

**Example Response**:
```json
{
  "area_id": "608691",
  "village_name": "Tallur",
  "district_name": "Udupi",
  "pc11_id": "608691",
  "preparedness_score": 74.34,
  "preparedness_tier": "High",
  "priority_band": "BAND_4_ADEQUATE",
  "vulnerability_archetype_id": 3,
  "domain_scores": {
    "medical": 82.5,
    "shelters": 45.0,
    "infrastructure": 85.0,
    "early_warning": 90.0,
    "socioeconomic": 88.0
  }
}
```

---
## Geospatial and Maps
*Spatial points, coordinate bounding boxes, and GIS filtering for interactive map visualization.*

### `GET /api/maps/communities`

**Purpose**: Retrieve spatial points for interactive map  
**API Tag**: `Maps & Geospatial` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Query Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `district` | `No` | `string` | Filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada') |
| `tier` | `No` | `string` | Filter by preparedness tier ('High', 'Medium', 'Low') |
| `priority_band` | `No` | `string` | Filter by priority band ('BAND_1_CRITICAL', etc.) |
| `archetype_id` | `No` | `string` | Filter by archetype ID (0, 1, 2, 3 or -1 for unassigned) |

**Example Request**:
```bash
curl -s http://localhost:8000/api/maps/communities
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/maps/communities"
}
```

---
## Districts
*Comparative district-level profiles, demographic totals, and cross-district preparedness differentials.*

### `GET /api/districts`

**Purpose**: List all 3 coastal districts and comparative profiles  
**API Tag**: `Districts` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/districts
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/districts"
}
```

### `GET /api/districts/{district_id}`

**Purpose**: Retrieve granular district intelligence dossier  
**API Tag**: `Districts` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Path Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `district_id` | `Yes` | `string` | District name or slug (e.g. 'udupi', 'dakshina-kannada', 'uttara-kannada') |

**Example Request**:
```bash
curl -s http://localhost:8000/api/districts/{district_id}
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/districts/{district_id}"
}
```

---
## Analytics
*Authoritative AHP distributions, surrogate model performance diagnostics, and operational distributions.*

### `GET /api/analytics/clusters`

**Purpose**: K-Means vulnerability archetypes and UK fallback catalog  
**API Tag**: `Analytics & Models` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/analytics/clusters
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/analytics/clusters"
}
```

### `GET /api/analytics/model-performance`

**Purpose**: Surrogate Logistic Regression evaluation metrics and confusion matrix  
**API Tag**: `Analytics & Models` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/analytics/model-performance
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/analytics/model-performance"
}
```

### `GET /api/analytics/preparedness`

**Purpose**: Authoritative AHP preparedness distribution  
**API Tag**: `Analytics & Models` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/analytics/preparedness
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/analytics/preparedness"
}
```

### `GET /api/analytics/priority-bands`

**Purpose**: Operational priority band distributions and deficit triggers  
**API Tag**: `Analytics & Models` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/analytics/priority-bands
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/analytics/priority-bands"
}
```

---
## Vulnerability Clusters
*K-Means (k=4) unsupervised vulnerability archetypes and Uttara Kannada fallback documentation.*

### `GET /api/v1/archetypes`

**Purpose**: Retrieve catalog of all four discovered vulnerability archetypes and intervention guidance  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/v1/archetypes
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/v1/archetypes"
}
```

---
## Infrastructure and Interventions
*Domain-specific infrastructure deficit analysis and targeted physical intervention recommendations.*

### `POST /api/simulation/compare`

**Purpose**: Compare baseline and counterfactual policy scenarios  
**API Tag**: `Policy Simulation` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Request Body Schema**:
```json
{
  "scenario_a_name": {
    "type": "string",
    "title": "Scenario A Name",
    "default": "Scenario A (e.g. Equal Weights)"
  },
  "scenario_a_weights": {
    "$ref": "#/components/schemas/WeightConfiguration"
  },
  "scenario_b_name": {
    "type": "string",
    "title": "Scenario B Name",
    "default": "Scenario B (e.g. Hazard Focused)"
  },
  "scenario_b_weights": {
    "$ref": "#/components/schemas/WeightConfiguration"
  },
  "top_divergent_count": {
    "type": "integer",
    "maximum"
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/simulation/compare \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example Response**:
```json
{
  "status": "success",
  "path": "/api/simulation/compare"
}
```

### `POST /api/simulation/prioritize`

**Purpose**: Compute MCDA rankings under custom weights  
**API Tag**: `Policy Simulation` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Request Body Schema**:
```json
{
  "scenario_name": {
    "type": "string",
    "title": "Scenario Name",
    "description": "Descriptive scenario name",
    "default": "custom_scenario"
  },
  "weights": {
    "$ref": "#/components/schemas/WeightConfiguration"
  },
  "top_n": {
    "type": "integer",
    "maximum": 622.0,
    "minimum": 1.0,
    "title": "Top N",
    "description": "Number of top units to return",
    "default": 20
  },
  "district_filter": {
    "anyOf": [
      {
        "type": "string"
      },
      {
 
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/simulation/prioritize \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example Response**:
```json
{
  "status": "success",
  "path": "/api/simulation/prioritize"
}
```

---
## Model Inference
*Empirical surrogate Logistic Regression classifier inference and probability output generation.*

### `POST /api/models/predict`

**Purpose**: Generate village preparedness tier prediction and boundary case analysis  
**API Tag**: `Models & Inference` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Request Body Schema**:
```json
{
  "area_id": {
    "anyOf": [
      {
        "type": "string"
      },
      {
        "type": "null"
      }
    ],
    "title": "Area Id",
    "description": "Optional Census PC11 ID, SHRID, or village name to extract registered feature vector.",
    "examples": [
      "Sasihithlu",
      "603256",
      "Tallur"
    ]
  },
  "features": {
    "anyOf": [
      {
        "additionalProperties": {
          "type": "number"
        },
        "type": "object"
      },
      {
        "type":
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/models/predict \
  -H "Content-Type: application/json" \
  -d '{"area_id": "608691"}'
```

**Example Response**:
```json
{
  "model_identifier": "village-preparedness-classifier",
  "artifact_version": "v1.0.0",
  "predicted_preparedness_tier": "High",
  "surrogate_probabilities": {
    "High": 0.892,
    "Medium": 0.104,
    "Low": 0.004
  },
  "methodology_note": "Surrogate model outputs represent empirical approximations of AHP preparedness tiers, NOT physical hazard probabilities."
}
```

### `POST /api/v1/model/predict`

**Purpose**: Generate village preparedness tier prediction using registered model  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Request Body Schema**:
```json
{
  "area_id": {
    "anyOf": [
      {
        "type": "string"
      },
      {
        "type": "null"
      }
    ],
    "title": "Area Id",
    "description": "Area identifier (SHRID, PC11 ID, or village name) to load from unified dataset",
    "examples": [
      "11-29-575-05562-617533",
      "617533",
      "Panjikal"
    ]
  },
  "features": {
    "anyOf": [
      {
        "additionalProperties": {
          "type": "number"
        },
        "type": "object"
      },
      {
        
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/model/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example Response**:
```json
{
  "status": "success",
  "path": "/api/v1/model/predict"
}
```

---
## Model Registry
*Local Git-compatible Model Registry catalog inspection, artifact provenance, and version metadata.*

### `GET /api/models/registry`

**Purpose**: List all models registered in the local Git-compatible registry  
**API Tag**: `Models & Inference` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Example Request**:
```bash
curl -s http://localhost:8000/api/models/registry
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/models/registry"
}
```

### `GET /api/models/registry/{model_id}`

**Purpose**: Retrieve detailed metadata and schema for a registered model  
**API Tag**: `Models & Inference` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Path Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `model_id` | `Yes` | `string` | — |

**Query Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `version` | `No` | `string` | Optional version (defaults to active production) |

**Example Request**:
```bash
curl -s http://localhost:8000/api/models/registry/{model_id}
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/models/registry/{model_id}"
}
```

---
## Assistant
*Conversational spatial decision support with deterministic AHP tool grounding and refusal gates.*

### `POST /api/assistant/chat`

**Purpose**: Conversational decision support turn with deterministic tool grounding  
**API Tag**: `Decision Support Assistant` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Request Body Schema**:
```json
{
  "message": {
    "type": "string",
    "maxLength": 2000,
    "minLength": 1,
    "title": "Message",
    "description": "User question or incident command prompt for the assistant.",
    "examples": [
      "Why is Tallur classified as Band 4 Adequate despite coastal proximity?",
      "Compare preparedness between Dakshina Kannada and Udupi.",
      "What are the recommended interventions for Edmoge?"
    ]
  },
  "conversation_history": {
    "anyOf": [
      {
        "items": {
        
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain why Tallur is classified as Band 4."}'
```

**Example Response**:
```json
{
  "answer": "Tallur (PC11: 608691, Udupi) achieves an authoritative AHP composite preparedness score of 74.34, categorizing it in the High preparedness tier and Priority Band 4 (Adequate).",
  "tools_executed": [
    "explain_preparedness_assessment"
  ],
  "tool_records": [
    {
      "tool_name": "explain_preparedness_assessment",
      "success": true,
      "execution_time_ms": 1.8
    }
  ],
  "is_fallback": false,
  "scientific_precedence_note": "AHP composite preparedness score maintains authoritative operational precedence."
}
```

### `GET /api/assistant/tools`

**Purpose**: List registered deterministic tools and execution schemas  
**API Tag**: `Decision Support Assistant` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ✅ Yes | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/assistant/tools
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/assistant/tools"
}
```

---
## Legacy Compatibility Routes
*Backward-compatible REST endpoints preserving contract compatibility with v1 clients.*

### `POST /api/v1/assistant/chat`

**Purpose**: Conversational decision support assistant with deterministic tool grounding  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Request Body Schema**:
```json
{
  "message": {
    "type": "string",
    "maxLength": 2000,
    "minLength": 1,
    "title": "Message",
    "description": "User question or incident command prompt for the assistant.",
    "examples": [
      "Why is Tallur classified as Band 4 Adequate despite coastal proximity?",
      "Compare preparedness between Dakshina Kannada and Udupi.",
      "What are the recommended interventions for Edmoge?"
    ]
  },
  "conversation_history": {
    "anyOf": [
      {
        "items": {
        
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example Response**:
```json
{
  "status": "success",
  "path": "/api/v1/assistant/chat"
}
```

### `GET /api/v1/decision-support/summary`

**Purpose**: Retrieve regional summary of priority bands, trigger counts, and model concordance  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Example Request**:
```bash
curl -s http://localhost:8000/api/v1/decision-support/summary
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/v1/decision-support/summary"
}
```

### `GET /api/v1/decision-support/units`

**Purpose**: Retrieve paginated decision evaluations with optional district and priority band filtering  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Query Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `district` | `No` | `string` | Optional filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada') |
| `priority_band` | `No` | `string` | Optional filter by priority band ('BAND_1_CRITICAL', 'BAND_2_SPECIALIZED', 'BAND_3_MODERATE', 'BAND_4_ADEQUATE') |
| `limit` | `No` | `integer` | Maximum number of units to return per page |
| `offset` | `No` | `integer` | Offset index for pagination |

**Example Request**:
```bash
curl -s http://localhost:8000/api/v1/decision-support/units
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/v1/decision-support/units"
}
```

### `GET /api/v1/decision-support/units/{identifier}`

**Purpose**: Retrieve full decision-support evaluation and grounded advisories for an administrative unit  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Path Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `identifier` | `Yes` | `string` | Census PC11 ID (e.g. '608720'), SHRID (e.g. '11-29-569-05522-608720'), or village name |

**Example Request**:
```bash
curl -s http://localhost:8000/api/v1/decision-support/units/{identifier}
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/v1/decision-support/units/{identifier}"
}
```

### `GET /api/v1/model/health`

**Purpose**: Check registered model health, active artifact version, and schema integrity  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🔒 Yes  

**Example Request**:
```bash
curl -s http://localhost:8000/api/v1/model/health
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/v1/model/health"
}
```

### `POST /api/v1/prioritization/compare`

**Purpose**: Compare rankings between two weighting scenarios and analyze rank divergence  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Request Body Schema**:
```json
{
  "scenario_a_name": {
    "type": "string",
    "title": "Scenario A Name",
    "default": "Scenario A (e.g. Equal Weights)"
  },
  "scenario_a_weights": {
    "$ref": "#/components/schemas/WeightConfiguration"
  },
  "scenario_b_name": {
    "type": "string",
    "title": "Scenario B Name",
    "default": "Scenario B (e.g. Hazard Focused)"
  },
  "scenario_b_weights": {
    "$ref": "#/components/schemas/WeightConfiguration"
  },
  "top_divergent_count": {
    "type": "integer",
    "maximum"
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/prioritization/compare \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example Response**:
```json
{
  "status": "success",
  "path": "/api/v1/prioritization/compare"
}
```

### `POST /api/v1/prioritization/scenario`

**Purpose**: Compute priority scores and deterministic rankings under custom weights  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Request Body Schema**:
```json
{
  "scenario_name": {
    "type": "string",
    "title": "Scenario Name",
    "description": "Descriptive scenario name",
    "default": "custom_scenario"
  },
  "weights": {
    "$ref": "#/components/schemas/WeightConfiguration"
  },
  "top_n": {
    "type": "integer",
    "maximum": 622.0,
    "minimum": 1.0,
    "title": "Top N",
    "description": "Number of top units to return",
    "default": 20
  },
  "district_filter": {
    "anyOf": [
      {
        "type": "string"
      },
      {
 
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/v1/prioritization/scenario \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Example Response**:
```json
{
  "status": "success",
  "path": "/api/v1/prioritization/scenario"
}
```

### `GET /api/v1/units/search`

**Purpose**: Search administrative units by name, Census ID, or SHRID  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Query Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `query` | `Yes` | `string` | Search term matching village/town name, Census PC11 ID, or SHRID |
| `district` | `No` | `string` | Optional filter by district name ('Dakshina Kannada' or 'Udupi') |
| `limit` | `No` | `integer` | Maximum number of matching units to return |

**Example Request**:
```bash
curl -s http://localhost:8000/api/v1/units/search
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/v1/units/search"
}
```

### `GET /api/v1/units/{identifier}`

**Purpose**: Retrieve granular indicators, dimension scores, and archetype guidance for a unit  
**API Tag**: `Legacy Compatibility (v1)` | **Authentication**: `None` | **Status Codes**: `200, 422`  
**Used by Frontend**: ❌ No | **Internal / Ops Only**: 🌐 Public  

**Path Parameters**:
| Name | Required | Type | Description |
| :--- | :--- | :--- | :--- |
| `identifier` | `Yes` | `string` | Census 2011 PC11 ID (e.g. '608720'), SHRID (e.g. '11-29-569-05522-608720'), or exact unit name |

**Example Request**:
```bash
curl -s http://localhost:8000/api/v1/units/{identifier}
```

**Example Response**:
```json
{
  "status": "success",
  "endpoint": "/api/v1/units/{identifier}"
}
```
