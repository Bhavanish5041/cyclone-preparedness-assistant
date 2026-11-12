# Swagger UI & OpenAPI Specification Test Guide

## Overview

The CYCLONE INTEL platform exposes an interactive Swagger UI (`/docs`), ReDoc documentation (`/redoc`), and the raw OpenAPI 3.1 specification (`/openapi.json`) powered by FastAPI.

> [!NOTE]
> All positive and negative API tests are executed strictly in local or isolated CI test runner environments via `FastAPI TestClient`. Malformed requests are never routed to live production services.

---

## 1. Documentation Endpoints

| URL Path | HTTP Method | Description |
|---|---|---|
| `/docs` | `GET` | Interactive Swagger UI for testing API routes directly in the browser |
| `/redoc` | `GET` | ReDoc responsive documentation interface |
| `/openapi.json` | `GET` | Machine-readable OpenAPI 3.1.0 schema specification |
| `/` | `GET` | Redirects directly to `/docs` |

---

## 2. API Endpoints Catalog & Example Payloads

### A. Health & Readiness

#### `GET /health`
- **Description**: Top-level system health, active dataset statistics, and loaded model registry status.
- **Example Request**:
  ```bash
  curl -X GET http://localhost:8000/health
  ```
- **Expected Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "dataset_loaded": true,
    "total_units": 1879,
    "total_villages": 1848,
    "total_census_towns": 31,
    "districts": ["Dakshina Kannada", "Udupi", "Uttara Kannada"],
    "archetypes_count": 4,
    "version": "2.0.0",
    "model_loaded": true,
    "model_artifact_name": "village-preparedness-classifier",
    "model_artifact_version": "v1.0.0",
    "model_feature_schema_status": "VALID_26_FEATURES"
  }
  ```

#### `GET /ready`
- **Description**: Container readiness probe for Kubernetes / Docker Compose.
- **Expected Response (200 OK)**:
  ```json
  {
    "ready": true,
    "dataset_loaded": true,
    "model_loaded": true,
    "version": "2.0.0"
  }
  ```

---

### B. Machine Learning Surrogate Inference

#### `POST /api/models/predict`
- **Description**: Real-time 26-feature inference using the serialized Logistic Regression classifier (`models/logistic_regression_tier.joblib`).
- **Example Request Body (Valid)**:
  ```json
  {
    "features": {
      "dist_coast_km": 4.5,
      "elevation_mean": 15.0,
      "avg_household_size": 4.2,
      "sc_share": 0.10,
      "st_share": 0.05,
      "vulnerable_pop_share": 0.15,
      "road_connectivity_score": 0.80,
      "drinking_water_score": 0.85,
      "n_hc_10km": 3,
      "n_phc_chc_10km": 2,
      "n_gauges_20km": 1,
      "n_shelters_10km": 4,
      "cvi_physical": 0.40,
      "cvi_social": 0.30,
      "has_pucca_road": 1,
      "has_major_district_road": 1,
      "bus_connectivity": 1,
      "log1p_population": 7.2,
      "log1p_pop_density_sqkm": 6.0,
      "log1p_dist_nearest_shelter_km": 1.1,
      "log1p_dist_nearest_hospital_km": 2.0,
      "coastal_risk_score": 0.60,
      "evacuation_sufficiency": 0.85,
      "vulnerability_exposure": 0.35,
      "infra_composite": 0.70,
      "monitoring_coverage": 0.80
    }
  }
  ```
- **Expected Response (200 OK)**:
  ```json
  {
    "predicted_tier": "High",
    "probabilities": {
      "High": 0.966,
      "Medium": 0.032,
      "Low": 0.002
    },
    "model_artifact": "village-preparedness-classifier:v1.0.0",
    "scientific_disclaimer": "Surrogate model predictions represent an empirical approximation of AHP preparedness tiers, NOT physical hazard or casualty probabilities."
  }
  ```
- **Negative Testing (Incomplete / Invalid Payload)**:
  - Missing required fields: Returns `422 Unprocessable Content`
  - Non-numeric feature value: Returns `422 Unprocessable Content`
  - Malformed JSON string: Returns `422 Unprocessable Content`

---

### C. Model Registry Inspection

#### `GET /api/models/registry`
- **Description**: Returns summary of all registered models tracked in `models/model_registry.json`.
- **Expected Response (200 OK)**:
  ```json
  {
    "registry_version": "1.0.0",
    "models": [
      {
        "model_identifier": "village-preparedness-classifier",
        "model_family": "scikit-learn",
        "active_version": "v1.0.0",
        "status": "production"
      },
      {
        "model_identifier": "coastal-vulnerability-kmeans",
        "model_family": "scikit-learn",
        "active_version": "v1.0.0",
        "status": "production"
      }
    ]
  }
  ```

#### `GET /api/models/registry/{model_id}`
- **Description**: Detailed version manifest, cryptographic SHA-256 hashes, and metric gates.
- **Valid IDs**: `village-preparedness-classifier`, `coastal-vulnerability-kmeans`
- **Invalid ID Response**: Returns `404 Not Found`

---

### D. Communities & Administrative Units

#### `GET /api/communities?limit=10&offset=0&district=Udupi`
- **Description**: Filterable and paginated list of 1,879 revenue villages and census towns.
- **Query Parameters**:
  - `limit`: Integer `[1..100]`, default `20`
  - `offset`: Integer `>= 0`, default `0`
  - `district`: Optional district filter (`Dakshina Kannada`, `Udupi`, `Uttara Kannada`)
  - `tier`: Optional tier filter (`High`, `Medium`, `Low`)
  - `search`: Case-insensitive substring match

#### `GET /api/communities/{community_id}`
- **Description**: Comprehensive inspection dossier for a specific village by Census PC11 ID or SHRID.
- **Example Valid ID**: `11-29-563-05478-602577` (returns `200 OK`)
- **Invalid ID**: `unknown-id-999` (returns `404 Not Found`)

---

### E. District Comparative Intelligence

#### `GET /api/districts`
- **Description**: Comparative overview of all 3 coastal districts with priority band counts and mean preparedness scores.

#### `GET /api/districts/{district_id}`
- **Supported Slugs**: `dakshina-kannada`, `udupi`, `uttara-kannada`
- **Expected Response (200 OK)**: Complete taluk breakdown and indicator summaries.

---

### F. Dynamic MCDA Simulation

#### `POST /api/simulation/prioritize`
- **Description**: Re-ranks administrative units based on dynamic criteria weight adjustments.
- **Request Body**:
  ```json
  {
    "weights": {
      "healthcare": 0.30,
      "vulnerability": 0.30,
      "infrastructure": 0.20,
      "hazard": 0.20
    }
  }
  ```
- **Negative Testing**: Negative weights or strings return `422 Unprocessable Content`.

---

### G. Conversational Assistant (CycloneLM)

#### `POST /api/assistant/chat`
- **Description**: Contextual question answering grounded in 7 deterministic decision-support tools.
- **Request Body**:
  ```json
  {
    "message": "What is the preparedness score of Tallur village?"
  }
  ```
- **Expected Response (200 OK)**:
  ```json
  {
    "answer": "The preparedness score of Tallur village is **74.34** (High tier)...",
    "tool_calls": [...]
  }
  ```

---

## 3. Negative Testing & Security Guidelines

1. **HTTP Method Enforcement**: Unsupported methods (e.g. `POST /health` or `DELETE /api/districts`) are rejected with `405 Method Not Allowed`.
2. **Type Coercion Safeguards**: FastAPI and Pydantic schemas enforce strict types. Passing strings where floats are expected triggers automatic schema rejection (`422`).
3. **Execution Safety**: Negative payloads are executed only in local pytest suites (`tests/test_swagger_api.py`) to prevent noise in production logs.
