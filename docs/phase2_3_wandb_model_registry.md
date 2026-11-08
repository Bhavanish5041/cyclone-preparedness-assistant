# Phase 2.3 — Weights & Biases Model Registry and Model Versioning

**Document ID:** `docs/phase2_3_wandb_model_registry.md`  
**Phase:** 2.3 (W&B Model Registry and Model Versioning)  
**Status:** Complete & Verified  
**Date:** November 2026  
**Repository:** `cyclone-preparedness-assistant`  
**Test Suite:** 234 passing tests (expanded from 209 baseline tests)  

---

## 1. Executive Summary

Phase 2.3 establishes formal enterprise model lifecycle management, artifact versioning, and cloud-backed model registration for the **multinomial Logistic Regression village preparedness-tier classifier** using **Weights & Biases (W&B SDK 0.31.0)**.

The fitted model and its comprehensive evaluation metadata have been bundled into a standardized 7-file artifact package, logged to W&B under the versioned artifact `village-preparedness-classifier`, and linked to the W&B Model Registry collection `wandb-registry-model/village-preparedness-classifier` under the justified operational aliases `candidate` and `validated`.

A production-grade model retrieval and inference container module (`src/models/load_registered_model.py`) was implemented. It strictly enforces explicit version/alias addressing, verifies the canonical 26-feature input schema, enforces simplex probability normalization, and provides full backwards-compatible inference with 100% prediction equivalence against the local serialized baseline.

> [!IMPORTANT]
> **Deterministic AHP Precedence & Target Definition:**  
> The classification target represents **AHP-derived village preparedness capacity tiers (`High`, `Medium`, `Low`)**, reflecting composite physical, social, and infrastructural readiness. It is **NOT** a physical cyclone hazard, wind speed, or casualty predictor. Deterministic Analytical Hierarchy Process (AHP) assessments maintain absolute operational precedence for emergency resource allocation. The Logistic Regression model provides calibrated tier probabilities for granular boundary-case ambiguity detection.

> [!WARNING]
> **Operational Governance: 'production' Alias Prohibition:**  
> Per Phase 2.3 safety boundaries, the `production` alias was **NOT** assigned. The model artifact is tagged exclusively with `candidate` and `validated` aliases. Formal assignment of `production` requires multi-stakeholder operational sign-off and subsequent API deployment milestones.

---

## 2. Weights & Biases Configuration & Authentication Audit

Before executing registration, a comprehensive audit of the W&B environment was conducted to ensure reliable authentication without leaking secrets.

### 2.1 Environment Baseline
- **W&B SDK Version:** `wandb==0.31.0` (installed in `.venv`)
- **Python Interpreter:** `/home/bhavanish/cyclone-preparedness-assistant/.venv/bin/python` (Python 3.14.4)
- **Authentication Method:** Authenticated via local configuration at `/home/bhavanish/.config/wandb/settings`
- **W&B Entity:** `bhavanishreddy69-na`
- **W&B Project:** `cyclone-preparedness-assistant`
- **Online Registry Run URL:** [https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/i3nj4r4y](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/i3nj4r4y)

### 2.2 W&B Model Registry Architecture (SDK 0.31.0)
In `wandb>=0.30`, individual user accounts and non-enterprise teams link artifacts to the Model Registry using the native run linkage API:
```python
run.link_artifact(
    artifact=model_artifact,
    target_path="wandb-registry-model/village-preparedness-classifier",
    aliases=["candidate", "validated"],
)
```
This registers the artifact in the centralized registry collection while preserving full lineage back to the originating project run, code execution environment, and performance evaluation metrics.

---

## 3. Model Artifact Integrity & Provenance Audit

The registered artifact represents the exact reproducible reconstruction of the teammate's baseline 70/15/15 experiment from `notebooks/02_Baseline_Models.ipynb`. **No retraining or refitting was performed.**

### 3.1 Serialized Model Components
- **Pipeline Artifact:** `models/logistic_regression_tier.joblib`
  - Architecture: `sklearn.pipeline.Pipeline`
  - Step 1: `RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))`
  - Step 2: `LogisticRegression(C=1.0, max_iter=500, multi_class='multinomial', solver='lbfgs', random_state=42)`
- **Label Encoder:** `models/label_encoder.joblib`
  - Classes: `['High', 'Low', 'Medium']`
  - Encoding: `0 -> High`, `1 -> Low`, `2 -> Medium`
- **Metadata:** `models/logistic_regression_metadata.json`
  - Framework: `scikit-learn 1.9.1`
  - Split configuration: 1,315 train (70%), 282 validation (15%), 282 test (15%)

### 3.2 Canonical 26-Feature Schema
The model expects exactly 26 engineered spatial, demographic, infrastructural, and hazard features in strict fit order:

| Index | Feature Name | Type | Description |
|:-----:|:-------------|:-----|:------------|
| 0 | `dist_coast_km` | Numerical | Distance to coastline (km) |
| 1 | `elevation_mean` | Numerical | Mean elevation above sea level (m) |
| 2 | `avg_household_size` | Numerical | Average persons per household |
| 3 | `sc_share` | Numerical | Scheduled Caste population share [0, 1] |
| 4 | `st_share` | Numerical | Scheduled Tribe population share [0, 1] |
| 5 | `vulnerable_pop_share` | Numerical | Combined vulnerable population share [0, 1] |
| 6 | `has_pucca_road` | Categorical | Binary indicator: all-weather pucca road (0/1) |
| 7 | `has_major_district_road` | Categorical | Binary indicator: major district road access (0/1) |
| 8 | `bus_connectivity` | Categorical | Binary indicator: public bus transit access (0/1) |
| 9 | `power_supply_domestic_hrs` | Numerical | Daily domestic power availability (hours) |
| 10 | `power_supply_agricultural_hrs` | Numerical | Daily agricultural power availability (hours) |
| 11 | `nearest_hospital_distance_km` | Numerical | Distance to nearest secondary/tertiary hospital (km) |
| 12 | `all_weather_road_distance_km` | Numerical | Distance to nearest all-weather road (km) |
| 13 | `nearest_town_distance_km` | Numerical | Distance to nearest major urban center (km) |
| 14 | `nearest_mandi_distance_km` | Numerical | Distance to nearest agricultural mandi (km) |
| 15 | `cyclone_hazard_risk_score` | Numerical | Baseline physical cyclone hazard index |
| 16 | `cyclone_inundation_risk_score`| Numerical | Coastal storm surge & inundation score |
| 17 | `landslide_hazard_score` | Numerical | Western Ghats slope landslide susceptibility |
| 18 | `flood_risk_score` | Numerical | Riverine & localized flash flood score |
| 19 | `combined_hazard_score` | Numerical | Composite multi-hazard vulnerability index |
| 20 | `infrastructure_readiness_score`| Numerical | Physical infrastructure composite score |
| 21 | `evacuation_road_density` | Numerical | Evacuation road network density |
| 22 | `evacuation_sufficiency` | Numerical | Ratio of evacuation route capacity to population |
| 23 | `vulnerability_exposure` | Numerical | Composite vulnerable population exposure |
| 24 | `infra_composite` | Numerical | Aggregated infrastructure readiness index |
| 25 | `monitoring_coverage` | Numerical | Early warning and sensor coverage score |

### 3.3 Evaluation Summary (Held-Out Test Split, N=282)
- **Accuracy:** `94.33%` (266 / 282 correct)
- **Macro F1-Score:** `0.9060`
- **Weighted F1-Score:** `0.9422`
- **High Preparedness Recall:** `94.44%` (51 / 54 correct)
- **Medium Preparedness Recall:** `97.04%` (197 / 203 correct)
- **Low Preparedness Recall:** `73.91%` (17 / 23 correct; 6 misclassified as Medium)

---

## 4. W&B Artifact Packaging Architecture

To guarantee complete self-containment, reproducibility, and compliance with data governance, the registered artifact packages 7 standardized files via `build_artifact_staging_package()`:

```
village-preparedness-classifier/
├── logistic_regression_tier.joblib         # Fitted Scikit-learn Pipeline (RobustScaler + LogReg)
├── label_encoder.joblib                    # Fitted LabelEncoder for ['High', 'Low', 'Medium']
├── logistic_regression_metadata.json       # Metadata (sanitized of raw split indices)
├── feature_schema.json                     # Canonical 26 features, ordering, and types
├── evaluation_summary.json                 # Test split performance metrics and confusion matrix
├── model_card.md                           # Formal model documentation & ethics card
└── intended_use_and_limitations.md         # Operational guidelines & safety boundaries
```

### Data Minimization & Privacy Protection
The raw `models/logistic_regression_metadata.json` contained large integer arrays of dataset row indices under `"split_indices"`. In accordance with data minimization best practices:
- The staging pipeline strips `"split_indices"` from the registered metadata file.
- Only the split sizes (`1315 / 282 / 282`) and evaluation metrics are persisted to W&B.
- The original local `models/logistic_regression_metadata.json` remains completely intact and unmodified.

---

## 5. Model Registration Workflow

A reproducible CLI registration script was implemented at `scripts/register_model_wandb.py`.

### 5.1 Registration Execution
```bash
/home/bhavanish/cyclone-preparedness-assistant/.venv/bin/python scripts/register_model_wandb.py \
  --project cyclone-preparedness-assistant \
  --entity bhavanishreddy69-na \
  --artifact-name village-preparedness-classifier \
  --collection-name village-preparedness-classifier \
  --aliases "candidate,validated" \
  --mode online
```

### 5.2 Registration Verification Details
- **W&B Run ID:** `i3nj4r4y`
- **Registered Artifact Version:** `village-preparedness-classifier:v0`
- **Registry Target:** `wandb-registry-model/village-preparedness-classifier`
- **Assigned Aliases:** `candidate`, `validated`, `latest`
- **Files Staged & Uploaded:** 7 files (total size: 55.4 KB)
- **Metadata Tracked:**
  - `model_type`: Scikit-learn Pipeline (RobustScaler + LogisticRegression)
  - `scikit_learn_version`: 1.9.1
  - `features_count`: 26
  - `evaluation_metrics`: Accuracy 0.9433, Macro F1 0.9060, Low recall 0.7391
  - `deterministic_precedence`: True
  - `target_definition`: AHP capacity surrogate

---

## 6. Model Retrieval and Inference Architecture

The retrieval module (`src/models/load_registered_model.py`) provides a robust, production-grade interface for downloading, caching, and serving registered model artifacts.

### 6.1 Architectural Principles
1. **Strict Version Specification:**  
   Addresses without explicit version or alias tags (e.g. bare `"village-preparedness-classifier"`) are rejected with `ValueError`. This eliminates silent drift toward untested latest artifacts.
2. **Local Caching:**  
   Retrieved artifacts are cached under `.wandb_artifacts_cache/`. Subsequent loads execute instantaneously offline without redundant network round-trips.
3. **Resilient Error Handling:**  
   Authentication and network failures raise a specialized `WandbRetrievalError` with actionable remediation guidance. Corrupted or incomplete artifact downloads raise descriptive `ValueError` exceptions.
4. **Input Schema Alignment & Invariance:**  
   Prior to inference, the container automatically validates that all 26 canonical features are present, non-empty, and finite numeric values. It re-aligns input columns to the exact training fit order, guaranteeing invariance to arbitrary input column ordering.
5. **Simplex Probability Normalization:**  
   Predicted probabilities are strictly normalized ($\sum_{c} P(c) = 1.0$) across all rows, preventing floating-point precision warnings in downstream evaluation libraries.

### 6.2 Inference Interface (`RegisteredModelContainer`)

```python
from src.models.load_registered_model import load_registered_model

# Retrieve specific validated version
model = load_registered_model("village-preparedness-classifier:candidate")

# 1. Discrete tier predictions (1D array of strings)
tiers = model.predict(X_villages)  # -> array(['High', 'Medium', 'Low', ...])

# 2. Calibrated class probabilities (simplex normalized)
probs = model.predict_proba(X_villages)
# -> {'prob_high': array([...]), 'prob_low': array([...]), 'prob_medium': array([...])}

# 3. Comprehensive decision-support dataframe
df_detailed = model.predict_detailed(X_villages)
# Contains: predicted_preparedness_tier, prob_high, prob_low, prob_medium,
#           max_probability, probability_margin, is_boundary_case, target_nature
```

---

## 7. Demonstration and Verification

An end-to-end integration demonstration script was created at `scripts/demo_registered_model.py`.

### 7.1 Execution Command
```bash
/home/bhavanish/cyclone-preparedness-assistant/.venv/bin/python scripts/demo_registered_model.py
```

### 7.2 Demonstration Output
```
================================================================================
CYCLONE PREPAREDNESS ASSISTANT — W&B MODEL RETRIEVAL & INFERENCE DEMONSTRATION
================================================================================
OPERATIONAL BOUNDARY & TARGET NOTICE:
  [TARGET: AHP CAPACITY TIER SURROGATE]
  AHP-derived village preparedness tiers (High, Medium, Low) reflecting composite physical,
  social, and infrastructural readiness. Deterministic AHP assessments serve as the baseline ground truth.
  Logistic Regression outputs provide calibrated tier probabilities for granular boundary case detection,
  NOT physical cyclone-risk or hazard prediction.
  Deterministic AHP assessments maintain operational precedence for emergency resource allocation.
================================================================================

[1/4] Retrieving Model Artifact from W&B: 'village-preparedness-classifier:candidate'...
2026-11-08 11:36:38,097 [INFO] src.models.load_registered_model - Using locally cached artifact from /home/bhavanish/cyclone-preparedness-assistant/.wandb_artifacts_cache/bhavanishreddy69-na_cyclone-preparedness-assistant_village-preparedness-classifier_candidate
2026-11-08 11:36:38,635 [INFO] src.models.load_registered_model - Successfully loaded and verified registered model artifact: bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:candidate
  • Successfully retrieved: bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:candidate
  • Resolved Version:      candidate
  • Local Cache Location:  /home/bhavanish/cyclone-preparedness-assistant/.wandb_artifacts_cache/bhavanishreddy69-na_cyclone-preparedness-assistant_village-preparedness-classifier_candidate
  • Model Type:            sklearn.pipeline.Pipeline (RobustScaler + LogisticRegression)
  • Framework:             scikit-learn 1.9.1
  • Expected Features:     26 features in canonical order
  • Target Classes:        High, Low, Medium

[2/4] Retrieved Model Provenance & Benchmark Performance:
  • Model Provenance:      REPRODUCIBLE RECONSTRUCTION: This model is a faithful, verified reconstruction of the teammate's documented 70/15/15 experiment from notebooks/02_Baseline_Models.ipynb. It is NOT the original unpersisted in-memory estimator.
  • Test Accuracy:         0.9433
  • Test Macro F1:         0.9060
  • Low-tier Recall:       0.7391

[3/4] Loading Test Set Observations for Inference:
  • Selected 5 representative held-out test villages:
    - [11-29-575-05562-617533] Panjikal (Dakshina Kannada)
    - [11-29-563-05479-602696] Chapakhand (Uttara Kannada)
    - [11-29-575-05565-617756] Bellare (Dakshina Kannada)
    - [11-29-563-05486-603608] Mudhalli (Uttara Kannada)
    - [11-29-563-05483-603302] Madralli  (Uttara Kannada)

[4/4] Performing Inference using Retrieved Registered Model:
---------------------------------------------------------------------------------------------------------
Area ID / Village                   | Predicted Tier  | P(High)  | P(Low)   | P(Med)   | Margin  | Boundary?
---------------------------------------------------------------------------------------------------------
Panjikal (Dakshina Kannada)         | High            |  0.9996  |  0.0000  |  0.0004  | 0.9993 | No       
Chapakhand (Uttara Kannada)         | Medium          |  0.0000  |  0.0052  |  0.9948  | 0.9895 | No       
Bellare (Dakshina Kannada)          | High            |  0.6164  |  0.0000  |  0.3836  | 0.2328 | No       
Mudhalli (Uttara Kannada)           | Low             |  0.0000  |  0.6222  |  0.3778  | 0.2443 | No       
Madralli  (Uttara Kannada)          | Medium          |  0.0037  |  0.0002  |  0.9961  | 0.9924 | No       
---------------------------------------------------------------------------------------------------------

SUMMARY:
  • Model successfully retrieved, validated, and executed.
  • 26-feature input schema verified with zero discrepancies.
  • Probabilities strictly adhere to simplex constraints (sum = 1.0).
  • Ready for downstream integration into decision-support services.
================================================================================
```

---

## 8. Test Suite & Verification Results

A dedicated automated test suite was developed at `tests/test_wandb_model_registry.py` covering 25 test conditions. Tests use complete mocking for W&B cloud interactions, ensuring 100% offline test execution without requiring internet access or active credentials.

### 8.1 Registry Test Suite Breakdown (`tests/test_wandb_model_registry.py`)
1. **Packaging & Sanitization:**
   - `test_build_artifact_staging_package_contents`: Verifies presence and non-zero size of all 7 files.
   - `test_build_artifact_staging_package_sanitizes_split_indices`: Confirms bulky split index arrays are stripped from metadata.
   - `test_build_artifact_staging_package_missing_source_files`: Confirms `FileNotFoundError` if source models are missing.
2. **Feature Schema Integrity:**
   - `test_feature_schema_integrity`: Verifies exactly 26 features, strict order flag, 3 categorical and 23 numerical features.
3. **Evaluation Summary & Alerts:**
   - `test_evaluation_summary_metrics`: Verifies test accuracy 0.9433, macro F1 0.9060, Low recall 0.7391, and operational alerts.
4. **Model Card & Limitations:**
   - `test_model_card_and_limitations_operational_language`: Verifies mandatory AHP-surrogate notices and deterministic precedence language.
5. **Strict Versioning & Governance:**
   - `test_load_registered_model_rejects_unversioned_address`: Verifies unversioned addresses are rejected with `ValueError`.
   - `test_register_model_artifact_forbids_production_alias`: Verifies assigning `production` alias raises `ValueError`.
6. **Local Caching & Artifact Recovery:**
   - `test_load_registered_model_from_cache`: Verifies offline loading from cached directories.
   - `test_load_registered_model_corrupted_artifact_error`: Verifies `ValueError` when downloaded artifacts miss critical files.
   - `test_load_registered_model_corrupted_metadata_json`: Verifies `ValueError` when metadata JSON is corrupted.
7. **Resilience & Network Failure:**
   - `test_wandb_api_initialization_failure`: Verifies `WandbRetrievalError` when W&B API init fails.
   - `test_wandb_artifact_download_network_failure`: Verifies `WandbRetrievalError` when download fails.
8. **Input Schema Validation:**
   - `test_container_rejects_non_dataframe`: Verifies non-DataFrame inputs raise `ValueError`.
   - `test_container_rejects_missing_features`: Verifies missing features raise `ValueError`.
   - `test_container_rejects_non_numeric_types`: Verifies string/object columns raise `ValueError`.
   - `test_container_rejects_nans`: Verifies NaN inputs raise `ValueError`.
   - `test_container_rejects_infinities`: Verifies +/-Inf inputs raise `ValueError`.
   - `test_container_reordering_invariance`: Verifies input columns in scrambled order are correctly aligned.
9. **Inference Contracts & Simplex Constraint:**
   - `test_container_predict_contract`: Verifies discrete predictions contract.
   - `test_container_predict_proba_simplex`: Verifies probability simplex constraint ($\sum P = 1.0$).
   - `test_container_predict_detailed_contract`: Verifies boundary case detection and required schema.
10. **Prediction Equivalence:**
    - `test_prediction_equivalence_with_local_baseline`: Confirms 100% discrete prediction match and probability difference $< 10^{-6}$ against local model on all 282 held-out test observations.
11. **Offline & Online Registration Workflows:**
    - `test_register_model_artifact_offline_mocked`: Confirms offline registration runs cleanly without network calls.
    - `test_register_model_artifact_online_mocked`: Confirms online registration calls `link_artifact` with designated aliases.

### 8.2 Full Repository Test Execution
```bash
======================== 234 passed, 15 warnings in 35.19s =========================
```
- **Previous Baseline:** 209 passing tests
- **Phase 2.3 Tests Added:** 25 passing tests
- **Current Total:** **234 passing tests** (100% pass rate, zero regressions)

---

## 9. Operational Guardrails & Limitations Summary

1. **AHP Capacity Surrogate Target:**  
   Predictions represent village preparedness capacity tiers (`High`, `Medium`, `Low`) based on AHP scoring. They do not predict meteorological storm tracks, flood depths, or structural failure.
2. **Deterministic Precedence:**  
   In emergency response scenarios, deterministic AHP scoring remains authoritative. ML predictions serve for decision support and boundary-case detection.
3. **Low-Tier Recall Caveat (73.91%):**  
   On held-out evaluation, 6 of 23 Low-preparedness villages were misclassified as Medium. In relief allocation, under-estimating vulnerability carries severe consequences. All villages predicted as Low, or where $P(\text{Low}) > 0.20$, or where deterministic AHP score is Low, must receive priority on-site validation.
4. **Boundary-Case Ambiguity:**  
   Villages with top probability margin $< 10\%$ or maximum probability $< 60\%$ are flagged as boundary cases (`is_boundary_case = True`) and require manual review.
