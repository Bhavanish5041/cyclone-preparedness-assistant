# Phase 2.4 — FastAPI Production Inference Integration

**Document ID:** `docs/phase2_4_fastapi_model_inference.md`  
**Phase:** 2.4 (FastAPI Model Inference & Feature Engineering Integration)  
**Status:** Complete & Verified  
**Date:** November 2026  
**Repository:** `cyclone-preparedness-assistant`  
**Test Suite:** 258 passing tests (expanded from 234 in Phase 2.3)  

---

## 1. Executive Summary

Phase 2.4 completes the production integration of the **Weights & Biases (W&B)-registered Logistic Regression village preparedness-tier classifier** into the existing FastAPI decision-support service.

The implementation strictly operationalizes the existing fitted and serialized model (`village-preparedness-classifier:v0`). **No new ML models were introduced, trained, or refitted.**

Key capabilities delivered:
1. **Faithful 26-Feature Engineering Pipeline:** Traced, audited, and reproduced the 9 missing engineered features from raw baseline indicators with $< 10^{-12}$ discrepancy across all 1,879 study area records.
2. **Lifecycle Model Loading:** Loaded once during FastAPI application startup (`lifespan`) via the Phase 2.3 model loader (`src/models/load_registered_model.py`), supporting local offline artifact caching and explicit version configuration (`WANDB_MODEL_ARTIFACT`).
3. **Dedicated Inference Endpoint:** Added `POST /api/v1/model/predict`, supporting village lookups by area identifier (SHRID, PC11 ID, or village name) as well as arbitrary validated feature payloads.
4. **Calibrated Probabilities & Boundary-Case Flagging:** Delivers simplex-normalized class probabilities ($\sum P = 1.0$), probability margins, and automated boundary-case ambiguity flags (`margin < 0.10` or `max_prob < 0.60`).
5. **Absolute Deterministic AHP Precedence:** Enforces strict operational guardrails ensuring ML surrogate predictions cannot downgrade an authoritative AHP `Low` preparedness classification.
6. **Health and Readiness Monitoring:** Extended `/health` and added `/api/v1/model/health` exposing artifact version, 26-feature schema status, and lineage without leaking credentials or local paths.

---

## 2. Architecture & Inference Flow

```
                     +---------------------------------------+
                     |         Incoming HTTP Request         |
                     +---------------------------------------+
                                        |
               +------------------------+------------------------+
               |                                                 |
      [Area ID Query]                                   [Feature Payload]
  (e.g. '617533', Panjikal)                       (26 features or raw proxies)
               |                                                 |
               v                                                 v
  +--------------------------+                      +--------------------------+
  |  Lookup in Unified v2    |                      | Validate Input Structure |
  |  Dataset (1,879 Units)   |                      | (Non-empty, No Null Keys)|
  +--------------------------+                      +--------------------------+
               |                                                 |
               +------------------------+------------------------+
                                        |
                                        v
                     +---------------------------------------+
                     |    Feature Engineering & Validation   |
                     | (src/features/inference_features.py)  |
                     |  - Check Base 17 & Source Indicators  |
                     |  - Compute 9 Missing Features         |
                     |  - Validate dtypes, NaNs, and Infs    |
                     |  - Align strictly to 26 canonical     |
                     +---------------------------------------+
                                        |
                                        v
                     +---------------------------------------+
                     |   RegisteredModelContainer (W&B:v0)   |
                     |  - RobustScaler Transformation        |
                     |  - Multinomial Logistic Regression    |
                     |  - Simplex Normalization (Sum = 1.0)  |
                     |  - Boundary-Case Flagging             |
                     +---------------------------------------+
                                        |
                                        v
                     +---------------------------------------+
                     |       Deterministic Safeguards        |
                     |  - Check Baseline AHP Concordance     |
                     |  - Enforce Low-Tier Precedence        |
                     |  - Attach Operational Disclaimers     |
                     +---------------------------------------+
                                        |
                                        v
                     +---------------------------------------+
                     |       ModelPredictionResponse         |
                     +---------------------------------------+
```

---

## 3. Registered Model Loading & Configuration

The model is instantiated during the FastAPI `lifespan` application lifecycle manager in `src/api/main.py`.

### 3.1 Configuration Environment Variable
The service checks the `WANDB_MODEL_ARTIFACT` environment variable to determine which versioned artifact to load:
```bash
export WANDB_MODEL_ARTIFACT="village-preparedness-classifier:v0"
```
If unset, the service defaults to `"village-preparedness-classifier:v0"`.

### 3.2 Loading Principles
- **Single-Time Initialization:** Loaded once during `lifespan` startup and attached to `app.state.model_container`. Route handlers access it via FastAPI dependency injection (`Depends(get_model)`).
- **Fast Startup via Offline Caching:** Resolves artifacts through `.wandb_artifacts_cache/`. If the artifact bundle is cached locally, startup completes in $< 50$ ms without external network calls.
- **Fail-Fast Error Handling:** If the configured artifact is unresolvable or fails schema verification, FastAPI immediately aborts startup with a descriptive `RuntimeError`. It never silently substitutes an arbitrary local model.
- **Zero Credential Exposure:** Neither the API responses nor the service logs print W&B API keys, tokens, or local filesystem absolute paths.

---

## 4. Faithful Feature Engineering & Validation

The unified preparedness dataset (`unified_village_preparedness_v2.parquet`) contains 17 base features and 14 raw indicator columns, but lacks 9 engineered features required by the Logistic Regression classifier.

A dedicated engineering module (`src/features/inference_features.py`) was created. It faithfully reproduces the mathematical transformations from `notebooks/01_EDA.ipynb`.

### 4.1 Canonical 26-Feature Schema
The registered model expects exactly 26 features in strict fit order:

| Index | Feature Name | Origin | Transformation Formula / Source |
|:-----:|:-------------|:-------|:--------------------------------|
| 0 | `dist_coast_km` | Base | Straight-line distance to coast |
| 1 | `elevation_mean` | Base | Mean elevation above sea level |
| 2 | `avg_household_size` | Base | Census 2011 persons per household |
| 3 | `sc_share` | Base | Scheduled Caste population share |
| 4 | `st_share` | Base | Scheduled Tribe population share |
| 5 | `vulnerable_pop_share`| Base | Combined vulnerable demographic share |
| 6 | `road_connectivity_score` | Base | Road accessibility composite score |
| 7 | `drinking_water_score` | Base | Potable water access score |
| 8 | `n_hc_10km` | Base | Healthcare centers within 10 km |
| 9 | `n_phc_chc_10km` | Base | PHC/CHC facilities within 10 km |
| 10 | `n_gauges_20km` | Base | Telemetric rain gauges within 20 km |
| 11 | `n_shelters_10km` | Base | Multi-purpose cyclone shelters within 10 km |
| 12 | `cvi_physical` | Base | Physical Coastal Vulnerability Index |
| 13 | `cvi_social` | Base | Social Coastal Vulnerability Index |
| 14 | `has_pucca_road` | Base (Cat) | Binary indicator: all-weather pucca road (0/1) |
| 15 | `has_major_district_road` | Base (Cat) | Binary indicator: major district road (0/1) |
| 16 | `bus_connectivity` | Base (Cat) | Binary indicator: public bus transit access (0/1) |
| 17 | `log1p_population` | **Engineered** | $\ln(1 + \max(\text{population}, 0))$ |
| 18 | `log1p_pop_density_sqkm` | **Engineered** | $\ln(1 + \max(\text{pop\_density\_sqkm}, 0))$ |
| 19 | `log1p_dist_nearest_shelter_km` | **Engineered** | $\ln(1 + \max(\text{dist\_nearest\_shelter\_km}, 0))$ |
| 20 | `log1p_dist_nearest_hospital_km` | **Engineered** | $\ln(1 + \max(\text{dist\_nearest\_hospital\_km}, 0))$ |
| 21 | `coastal_risk_score` | **Engineered** | $\left[\exp(-\text{dist\_coast\_km} / 10.0) \times \exp(-\text{elevation\_mean} / 30.0)\right].\text{round}(4)$ |
| 22 | `evacuation_sufficiency` | **Engineered** | $\text{clip}\left(\frac{\text{shelter\_capacity\_nearby}}{\max(\text{population}, 1)}, 0, 10\right).\text{round}(4)$ |
| 23 | `vulnerability_exposure` | **Engineered** | $\left[0.5 \times \frac{\text{cvi\_integrated}}{773.4} + 0.3 \times \left(1 - \frac{\text{dist\_coast\_km}}{90.38}\right) + 0.2 \times \text{vulnerable\_pop\_share}\right].\text{round}(4)$ |
| 24 | `infra_composite` | **Engineered** | $\left[\frac{\text{road\_connectivity\_score}}{3} + \frac{\text{drinking\_water\_score}}{2} + \text{bus\_connectivity}\right].\text{round}(4) / 3.0$ |
| 25 | `monitoring_coverage` | **Engineered** | $\left[\left(1 - \frac{\text{dist\_nearest\_gauge\_km}}{15.43}\right) \times 0.6 + \left(\frac{\text{n\_gauges\_20km}}{77.0}\right) \times 0.4\right].\text{round}(4)$ |

### 4.2 Mathematical Equivalence Verification
Across all 1,879 units in the study area, the engineered features produced by `src/features/inference_features.py` were compared against `modelling_dataset.parquet`:
- **Discrepancy:** Maximum absolute difference $\le 1.77 \times 10^{-15}$ across all 26 features.
- **Model Output Concordance:** Model predictions across all 1,879 records match the offline baseline with **100.00% exact class equality**.

---

## 5. API Reference & Endpoints

### 5.1 System Health (`GET /health`)
Extended to report model readiness and active artifact version.

**Request:**
```http
GET /health HTTP/1.1
Host: localhost:8000
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "dataset_loaded": true,
  "total_units": 622,
  "total_villages": 563,
  "total_census_towns": 59,
  "districts": [
    "Dakshina Kannada",
    "Udupi"
  ],
  "archetypes_count": 4,
  "version": "1.0.0",
  "model_loaded": true,
  "model_artifact_name": "village-preparedness-classifier",
  "model_artifact_version": "v0",
  "model_feature_schema_status": "VALID_26_FEATURES",
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. It is NOT the original unpersisted in-memory estimator."
}
```

---

### 5.2 Model Health (`GET /api/v1/model/health`)
Dedicated inspection route for active ML artifact metadata.

**Request:**
```http
GET /api/v1/model/health HTTP/1.1
Host: localhost:8000
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "artifact_name": "village-preparedness-classifier",
  "artifact_version": "v0",
  "artifact_source": "bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:v0",
  "features_count": 26,
  "feature_schema_status": "VALID_26_FEATURES",
  "target_classes": [
    "High",
    "Low",
    "Medium"
  ],
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. It is NOT the original unpersisted in-memory estimator.",
  "target_nature": "AHP_CAPACITY_TIER_SURROGATE"
}
```

---

### 5.3 Model Prediction (`POST /api/v1/model/predict`)

#### Example 1: High Preparedness Village Lookup (Panjikal)
**Request:**
```json
{
  "area_id": "11-29-575-05562-617533"
}
```

**Response (200 OK):**
```json
{
  "area_id": "11-29-575-05562-617533",
  "village_name": "Panjikal",
  "district_name": "Dakshina Kannada",
  "predicted_preparedness_tier": "High",
  "class_probabilities": {
    "prob_high": 0.999642,
    "prob_low": 0.0,
    "prob_medium": 0.000358
  },
  "probability_margin": 0.999284,
  "max_probability": 0.999642,
  "is_boundary_case": false,
  "deterministic_ahp_tier": "High",
  "deterministic_ahp_score": 67.86,
  "ahp_agreement_status": "FULL_AGREEMENT",
  "ahp_precedence_note": null,
  "model_artifact_name": "village-preparedness-classifier",
  "model_artifact_version": "v0",
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. It is NOT the original unpersisted in-memory estimator.",
  "target_nature": "AHP_CAPACITY_TIER_SURROGATE",
  "disclaimer": "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
}
```

---

#### Example 2: Boundary Case Ambiguity Detection (Ghadasai)
Top probabilities are close ($P(\text{Medium}) = 0.5305$, $P(\text{High}) = 0.4695$, margin $= 0.0609 < 0.10$). Automatically flagged as a boundary case.

**Request:**
```json
{
  "area_id": "11-29-563-05478-602593"
}
```

**Response (200 OK):**
```json
{
  "area_id": "11-29-563-05478-602593",
  "village_name": "Ghadasai",
  "district_name": "Uttara Kannada",
  "predicted_preparedness_tier": "Medium",
  "class_probabilities": {
    "prob_high": 0.469548,
    "prob_low": 0.0,
    "prob_medium": 0.530452
  },
  "probability_margin": 0.060904,
  "max_probability": 0.530452,
  "is_boundary_case": true,
  "deterministic_ahp_tier": "Medium",
  "deterministic_ahp_score": 58.81,
  "ahp_agreement_status": "FULL_AGREEMENT",
  "ahp_precedence_note": null,
  "model_artifact_name": "village-preparedness-classifier",
  "model_artifact_version": "v0",
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. It is NOT the original unpersisted in-memory estimator.",
  "target_nature": "AHP_CAPACITY_TIER_SURROGATE",
  "disclaimer": "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
}
```

---

#### Example 3: Low-Tier Precedence Safeguard (Kamargaon)
Authoritative AHP tier is `Low` (score: 39.91), but ML model predicted `Medium`. The service triggers an explicit safeguard mandate reinforcing that AHP holds precedence and priority cannot be downgraded.

**Request:**
```json
{
  "area_id": "11-29-563-05478-602600"
}
```

**Response (200 OK):**
```json
{
  "area_id": "11-29-563-05478-602600",
  "village_name": "Kamargaon",
  "district_name": "Uttara Kannada",
  "predicted_preparedness_tier": "Medium",
  "class_probabilities": {
    "prob_high": 0.0,
    "prob_low": 0.013501,
    "prob_medium": 0.986499
  },
  "probability_margin": 0.972998,
  "max_probability": 0.986499,
  "is_boundary_case": false,
  "deterministic_ahp_tier": "Low",
  "deterministic_ahp_score": 39.91,
  "ahp_agreement_status": "BORDERLINE_DISAGREEMENT",
  "ahp_precedence_note": "AHP PRECEDENCE MANDATE: Authoritative AHP capacity tier is 'Low'. Per operational safety rules, ML model prediction cannot downgrade emergency priority.",
  "model_artifact_name": "village-preparedness-classifier",
  "model_artifact_version": "v0",
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. It is NOT the original unpersisted in-memory estimator.",
  "target_nature": "AHP_CAPACITY_TIER_SURROGATE",
  "disclaimer": "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
}
```

---

#### Example 4: Direct Feature Payload Inference
Users can submit custom feature sets for hypothetical planning or uncatalogued settlements.

**Request:**
```json
{
  "features": {
    "dist_coast_km": 14.5,
    "elevation_mean": 82.0,
    "avg_household_size": 4.6,
    "sc_share": 0.08,
    "st_share": 0.02,
    "vulnerable_pop_share": 0.10,
    "road_connectivity_score": 2.0,
    "drinking_water_score": 1.0,
    "n_hc_10km": 3.0,
    "n_phc_chc_10km": 1.0,
    "n_gauges_20km": 4.0,
    "n_shelters_10km": 1.0,
    "cvi_physical": 2.5,
    "cvi_social": 1.8,
    "has_pucca_road": 1.0,
    "has_major_district_road": 0.0,
    "bus_connectivity": 1.0,
    "population": 2100.0,
    "pop_density_sqkm": 280.0,
    "dist_nearest_shelter_km": 6.5,
    "dist_nearest_hospital_km": 12.0,
    "shelter_capacity_nearby": 500.0,
    "cvi_integrated": 142.0,
    "dist_nearest_gauge_km": 8.0
  }
}
```

**Response (200 OK):**
```json
{
  "area_id": null,
  "village_name": null,
  "district_name": null,
  "predicted_preparedness_tier": "Medium",
  "class_probabilities": {
    "prob_high": 0.042188,
    "prob_low": 0.000312,
    "prob_medium": 0.9575
  },
  "probability_margin": 0.915312,
  "max_probability": 0.9575,
  "is_boundary_case": false,
  "deterministic_ahp_tier": null,
  "deterministic_ahp_score": null,
  "ahp_agreement_status": null,
  "ahp_precedence_note": null,
  "model_artifact_name": "village-preparedness-classifier",
  "model_artifact_version": "v0",
  "model_provenance": "REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. It is NOT the original unpersisted in-memory estimator.",
  "target_nature": "AHP_CAPACITY_TIER_SURROGATE",
  "disclaimer": "OPERATIONAL NOTICE: Model predictions and class probabilities represent an empirical surrogate approximating AHP-derived village preparedness tiers (High, Medium, Low). They are NOT calibrated physical cyclone-risk probabilities, storm-surge inundation forecasts, or casualty predictions. Deterministic AHP assessments maintain absolute operational precedence for emergency resource allocation."
}
```

---

## 6. Decision Support Integration & Deterministic Precedence

The decision-support architecture enforces a strict hierarchy:

1. **Analytical Hierarchy Process (AHP) Authoritative Status:**  
   The deterministic AHP multi-criteria score ($0.0 - 100.0$) and tier (`High`, `Medium`, `Low`) represent official administrative capacity. Priority bands (`BAND_1_CRITICAL` through `BAND_4_ADEQUATE`) are derived deterministically from AHP scores and physical deficit triggers.
2. **ML Model as Diagnostic Surrogate:**  
   The Logistic Regression model predicts empirical tier probabilities. It serves secondary diagnostic functions: identifying borderline ambiguity and cross-checking stability.
3. **No Downgrading Rule:**  
   If a village has an AHP score $< 40.0$ (`Low` tier, `BAND_1_CRITICAL`), it is strictly ineligible for priority downgrading, even if the ML classifier predicts `Medium` or `High`.
4. **Boundary-Case Surface:**  
   Villages with `is_boundary_case = True` ($\text{margin} < 0.10$ or $P_{max} < 0.60$) are highlighted for district disaster management manual review.

---

## 7. Testing & Quality Verification

A dedicated test suite was implemented at `tests/test_api_model_inference.py` covering 24 comprehensive test conditions.

### 7.1 Test Matrix Breakdown
1. **Feature Engineering Equivalence (3 tests):**
   - Verified all 1,879 villages match `modelling_dataset.parquet` to $\le 10^{-12}$.
   - Verified single-row dictionary feature engineering.
   - Verified precomputed 26-feature alignment and re-ordering invariance.
2. **Input Validation & Corrupted Data Handling (6 tests):**
   - Verified rejection of missing raw indicators (e.g. `population`).
   - Verified rejection of missing base features (e.g. `dist_coast_km`).
   - Verified rejection of `NaN` / null values.
   - Verified rejection of `+/-Inf` values.
   - Verified rejection of non-numeric string values.
   - Verified rejection of empty input payloads.
3. **Lifespan Startup & Failure Handling (2 tests):**
   - Verified successful model loading and attachment during startup.
   - Verified startup aborts with `RuntimeError` if artifact cannot be retrieved.
4. **Health & Readiness Endpoints (2 tests):**
   - Verified `/health` exposes model loaded status, artifact version, and schema integrity without leaking credentials.
   - Verified `/api/v1/model/health` endpoint schema and target nature.
5. **Inference Execution Contracts (5 tests):**
   - Verified inference by SHRID area identifier (Panjikal).
   - Verified inference by PC11 census ID (Bellare).
   - Verified 404 response on unknown area identifier.
   - Verified inference by precomputed 26-feature dictionary.
   - Verified inference by raw indicator dictionary (automatic 9-feature engineering).
6. **Boundary Case & Ambiguity Flagging (1 test):**
   - Verified `is_boundary_case` flag when probability margin is $< 0.10$ or max probability $< 0.60$.
7. **AHP Precedence & Low-Tier Safeguards (2 tests):**
   - Verified AHP Low-tier village retains `BAND_1_CRITICAL` and attaches `AHP PRECEDENCE MANDATE` note when ML prediction disagrees.
   - Verified existing decision-support evaluation route preserves `BAND_1_CRITICAL`.
8. **Existing Endpoint Invariance (1 test):**
   - Verified all existing search, prioritization scenario, and archetype endpoints remain fully operational and unaltered.

### 7.2 Full Repository Test Execution
```bash
======================== 258 passed, 13 warnings in 18.70s =========================
```
- **Test Baseline (Phase 2.3):** 234 passing tests
- **Phase 2.4 Tests Added:** 24 passing tests
- **Current Total:** **258 passing tests** (100% pass rate, zero regressions)

---

## 8. Operational Safeguards & Model Limitations

1. **Surrogate Preparedness Nature:**  
   Predictions represent an empirical approximation of AHP preparedness capacity bands. They do **not** represent physical cyclone landfall probabilities, storm-surge inundation depths, or casualty rates.
2. **Administrative Precedence:**  
   Deterministic AHP scoring holds absolute precedence for disaster relief resource allocation.
3. **Low-Tier Recall Caveat (73.91%):**  
   On held-out testing, 6 of 23 Low-preparedness villages were misclassified as Medium (e.g. Kamargaon). Operational disaster authorities must never downgrade villages based solely on ML predictions.
4. **Boundary Case Protocol:**  
   Villages with `is_boundary_case = True` represent high classification uncertainty and must undergo manual field inspection prior to altering resource allocations.
