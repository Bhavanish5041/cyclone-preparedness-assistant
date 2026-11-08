# Phase 2.2 — Evidently AI Integration and Model Monitoring

**Document ID:** `docs/phase2_2_evidently_monitoring.md`  
**Phase:** 2.2 (Model Quality Monitoring & Evidently AI Integration)  
**Status:** Complete & Verified  
**Date:** November 2026  
**Repository:** `cyclone-preparedness-assistant`  
**Test Baseline:** 209 passing tests (expanded from 197 passing tests in Phase 2.1)

---

## 1. Executive Summary

Phase 2.2 extends the production monitoring and auditing framework developed in Phase 2.1 by integrating **Evidently AI (`evidently==0.7.23`)** as an optional, modular visualization and reporting layer alongside the built-in, lightweight native SciPy/NumPy statistical engine.

The monitoring suite audits the 26 canonical features of the multinomial Logistic Regression tier classifier across four essential pillars:
1. **Data Quality:** Missing values, empty records, column data types, constant features, and feature schema compliance.
2. **Feature Drift:** Covariate shift detection across all 26 model features comparing reference (training) and comparison (held-out test) datasets.
3. **Prediction Drift:** Output stability auditing for predicted tier classes (`High`, `Low`, `Medium`) and calibrated probability distributions (`prob_high`, `prob_low`, `prob_medium`).
4. **Classification Performance:** Empirical accuracy, precision, recall, F1-scores, and confusion matrix against verified ground-truth Analytical Hierarchy Process (AHP) labels.

> [!IMPORTANT]
> **Deterministic AHP Precedence & Target Definition:**  
> The classification target in this monitoring framework strictly represents **AHP-derived village preparedness tiers (`High`, `Medium`, `Low`)**, reflecting composite physical, social, and infrastructural readiness. It is **NOT** a physical cyclone hazard or meteorological damage predictor. Deterministic AHP assessments maintain absolute precedence for resource prioritization; Logistic Regression provides calibrated probabilities for boundary-case ambiguity detection.

---

## 2. Environment & Dependency Audit

Prior to installation, a rigorous compatibility audit of the project virtual environment (`.venv`) was performed to avoid breaking core scientific dependencies.

### 2.1 Core Environment Baseline
- **Python Interpreter:** `/home/bhavanish/cyclone-preparedness-assistant/.venv/bin/python` (Python 3.14.4)
- **NumPy:** `2.5.4`
- **Pandas:** `3.0.6`
- **SciPy:** `1.18.1`
- **Scikit-learn:** `1.9.1`
- **Pydantic:** `2.14.1`

### 2.2 Evidently Version Selection
An inspection of legacy Evidently releases (e.g. `0.4.38` through `0.6.7`) revealed a rigid constraint `numpy<2.1,>=1.22.0`. Attempting to install older releases caused pip to attempt compiling NumPy 2.0.2 from source, which failed under Python 3.14 and risked downgrading core dependencies.

In contrast, **`evidently==0.7.23`** updated its requirements to `numpy>=1.23.0`, providing direct, native compatibility with NumPy 2.5.4, Pandas 3.0.6, Scikit-learn 1.9.1, and Python 3.14.4 without downgrading any existing package.

### 2.3 Installation Details
- **Installation Command:**
  ```bash
  /home/bhavanish/cyclone-preparedness-assistant/.venv/bin/pip install "evidently==0.7.23"
  ```
- **Manifest Updates:**
  - `pyproject.toml`: Added `dependencies = ["evidently==0.7.23"]` under `[project]`.
  - `requirements.txt`: Created pinned reference manifest `evidently==0.7.23`.

---

## 3. Architecture & Integration Design

The monitoring architecture maintains a **dual-engine design** that couples zero-dependency native statistical monitoring with rich Evidently AI dashboards.

```
                                  +------------------------------------+
                                  |     ModelQualityMonitor            |
                                  +------------------------------------+
                                            |               |
                    +-----------------------+               +-----------------------+
                    |                                                               |
                    v                                                               v
   +--------------------------------+                             +-----------------------------------+
   |   Native Statistical Engine    |                             |      Evidently AI Suite           |
   |   (Built-in SciPy / NumPy)     |                             |   (EvidentlyReportGenerator)      |
   +--------------------------------+                             +-----------------------------------+
   | • 26-Feature Schema Validation |                             | • Data Quality Preset Report      |
   | • KS-Test (Feature Drift)      |                             | • Feature Drift Preset Report     |
   | • Wasserstein Distance         |                             | • Prediction Drift Preset Report  |
   | • Population Stability Index   |                             | • Classification Performance      |
   | • Multi-class Brier & Log Loss |                             |   Preset Report (vs AHP Ground)   |
   | • Boundary Case Ratio (<10%)   |                             +-----------------------------------+
   +--------------------------------+                                               |
                    |                                                               v
                    |                                             +-----------------------------------+
                    |                                             | Output Dashboards & Manifests     |
                    +-------------------------------------------->| reports/model_monitoring/         |
                                                                  |  • *.html (Interactive Visuals)   |
                                                                  |  • *.json (Machine-Readable APIs) |
                                                                  +-----------------------------------+
```

### 3.1 Key Modules & Classes (`src/monitoring/model_quality.py`)
- `FeatureSchemaValidator`: Strict input validation ensuring all 26 model features are present, strictly ordered, numeric, and finite.
- `ModelQualityEvaluator`: Empirical classification metric calculator (accuracy, macro/weighted precision/recall/F1, confusion matrix, Brier scores, log loss, boundary cases).
- `EvidentlyReportGenerator`: Modular generator wrapping Evidently AI presets with dual-path import resolution (supporting both modern and legacy Evidently namespaces).
- `ModelQualityMonitor`: Top-level coordinator supporting execution of native checks and optional Evidently reports (`run_evidently=True`).
- `EvidentlyReportResult` & `EvidentlySuiteResult`: Pydantic models structuring metadata, summary metrics, and file paths.

---

## 4. Reference and Comparison Data Integrity

All monitoring evaluations use **only existing project data**. No synthetic or simulated inference data was fabricated.

| Dataset Role | Source Split | Sample Size | Description |
| :--- | :--- | :--- | :--- |
| **Reference Baseline** | 70% Stratified Training Split | 1,315 villages | Original training observations from `modelling_dataset.parquet`. |
| **Comparison Target** | 15% Stratified Test Split | 282 villages | Held-out evaluation split never seen during training. |

### 4.1 Feature Ordering & Preprocessing
Features are extracted strictly in the canonical 26-feature order determined during model training:
1. `dist_coast_km`
2. `elevation_mean`
3. `avg_household_size`
4. `sc_share`
5. `st_share`
6. `vulnerable_pop_share`
7. `road_connectivity_score`
8. `drinking_water_score`
9. `n_hc_10km`
10. `n_phc_chc_10km`
11. `n_gauges_20km`
12. `n_shelters_10km`
13. `cvi_physical`
14. `cvi_social`
15. `has_pucca_road` *(Categorical indicator)*
16. `has_major_district_road` *(Categorical indicator)*
17. `bus_connectivity` *(Categorical indicator)*
18. `log1p_population`
19. `log1p_pop_density_sqkm`
20. `log1p_dist_nearest_shelter_km`
21. `log1p_dist_nearest_hospital_km`
22. `coastal_risk_score`
23. `evacuation_sufficiency`
24. `vulnerability_exposure`
25. `infra_composite`
26. `monitoring_coverage`

### 4.2 Explicit Label Distinctions
The monitoring framework strictly isolates four separate concepts to prevent label confusion:
1. **AHP Target Tier (`actual_preparedness_tier`):** Ground-truth composite index derived via multi-criteria decision analysis (`High`, `Medium`, `Low`).
2. **Logistic Regression Prediction (`predicted_preparedness_tier`):** Probabilistic model prediction trained to approximate AHP tiers.
3. **K-Means Archetype (`cluster`):** Unsupervised vulnerability cluster (0 to 3) representing socio-spatial patterns.
4. **Decision-Support Priority Bands (Bands 1–4):** Rule-based administrative intervention tiers combining AHP, hazard exposure, and infrastructure buffers.

---

## 5. Drift Metrics & Alert Interpretation

The monitoring system distinguishes between four separate operational phenomena:

### 5.1 The Four Monitoring Pillars

| Pillar | Condition Audited | Ground-Truth Required? | Primary Metrics |
| :--- | :--- | :---: | :--- |
| **1. Data Quality Issues** | Corrupted inputs, missing values, schema sequence mismatches, infinite values. | No | Null counts, infinite counts, dtype compliance. |
| **2. Feature Distribution Drift** | Covariate shift $P(X)$ in raw features between baseline and evaluation sets. | No | Kolmogorov-Smirnov test ($D$, $p$-value), Wasserstein distance. |
| **3. Prediction Distribution Drift** | Shifts in predicted class proportions or probability simplex $P(\hat{Y} \mid X)$. | No | Population Stability Index (PSI), Wasserstein distance on probabilities. |
| **4. Model Performance Degradation** | Drop in classification accuracy or recall (concept drift $P(Y \mid X)$). | **Yes** | Accuracy, Low-tier recall, Macro F1, Brier loss, boundary cases. |

> [!CAUTION]
> **Data Drift vs. Concept Drift:**  
> A feature distribution shift (Data Drift) does **NOT** automatically imply that model predictions are degraded (Concept Drift). If the conditional relationship $P(Y \mid X)$ remains invariant, predictions remain valid. Conversely, true concept drift can **only** be verified when ground-truth labels are available. Unlabeled production batches must never claim concept drift.

### 5.2 Statistical Drift Metrics Explained
- **Kolmogorov-Smirnov (KS) Test:** Non-parametric test evaluating the maximum vertical distance $D$ between two empirical cumulative distribution functions. A $p$-value $< 0.05$ flags statistically significant distribution difference.
- **Wasserstein Distance (Earth Mover's Distance):** Measures the minimum "work" required to transform one distribution into another. Retains physical units (e.g., kilometers, elevation meters), making it intuitive for geographic features.
- **Population Stability Index (PSI):** Quantifies shift in binned probability or feature distributions:
  - $\text{PSI} < 0.10$: Negligible change; distribution is stable.
  - $0.10 \le \text{PSI} < 0.25$: Moderate shift; warrants monitoring.
  - $\text{PSI} \ge 0.25$: Significant distribution shift; requires operational review.

### 5.3 Provisional Engineering Thresholds
The following alert thresholds are implemented as **provisional engineering heuristics** based on disaster management domain constraints. They are **not** regulatory or validated industry constants:
- **Low-Tier Recall Alert (< 80%):** Flags risk of misclassifying highly vulnerable villages as medium or high preparedness, preventing dangerous false negatives.
- **Boundary Case Ratio Alert (> 15%):** Flags when more than 15% of predictions have a probability margin $< 10\%$ or maximum class probability $< 60\%$.
- **Dataset-Level Drift Alert (> 33%):** Flags when more than one-third of model features exhibit statistically significant drift.

---

## 6. Generated Reports and Demonstration

### 6.1 Generated Monitoring Artifacts
All reports are saved in `reports/model_monitoring/` (isolated from source code):

| Artifact Name | Format | Size | Purpose |
| :--- | :--- | :--- | :--- |
| `evidently_data_quality.html` | HTML | ~3.8 MB | Interactive distribution plots, missing values, column statistics. |
| `evidently_data_quality.json` | JSON | ~33 KB | Machine-readable data quality metrics for automated pipelines. |
| `evidently_feature_drift.html` | HTML | ~4.2 MB | Interactive 26-feature distribution comparison overlays. |
| `evidently_feature_drift.json` | JSON | ~23 KB | Per-feature KS/Wasserstein statistics and drift flags. |
| `evidently_prediction_drift.html` | HTML | ~3.7 MB | Predicted tier distribution and probability shifts. |
| `evidently_prediction_drift.json` | JSON | ~5 KB | Class balance and probability drift metrics. |
| `evidently_classification_performance.html` | HTML | ~3.9 MB | Confusion matrix, ROC curves, and precision-recall graphs vs AHP labels. |
| `evidently_classification_performance.json` | JSON | ~8 KB | Class-wise precision, recall, F1, and support metrics. |
| `comprehensive_monitoring_report.json` | JSON | ~17 KB | Unified manifest uniting schema, native metrics, drift, and Evidently metadata. |

### 6.2 Running the Demonstration Script
To reproduce the monitoring execution without altering any model or data:

```bash
/home/bhavanish/cyclone-preparedness-assistant/.venv/bin/python scripts/run_model_monitoring.py
```

Optional arguments:
```bash
/home/bhavanish/cyclone-preparedness-assistant/.venv/bin/python scripts/run_model_monitoring.py \
    --output-dir reports/model_monitoring \
    --metadata-path models/logistic_regression_metadata.json \
    --modelling-path /home/bhavanish/cyclone-preparedness/data/processed/modelling_dataset.parquet \
    --probabilities-path data/processed/village_preparedness_probabilities.parquet
```

---

## 7. Test Results

### 7.1 Test Execution Summary
- **Previous Baseline (Phase 2.1):** 197 passing tests.
- **New Tests Added (Phase 2.2):** 12 comprehensive unit and integration tests.
- **Current Test Suite Total:** **209 passed**, 0 failures, 0 errors.

### 7.2 Summary of New Test Coverage (`tests/test_model_quality_monitoring.py`)
1. `test_evidently_availability_check`: Verifies Evidently version detection and preset availability.
2. `test_evidently_data_quality_report_generation`: Verifies HTML/JSON generation for data quality.
3. `test_evidently_feature_drift_report_generation`: Verifies drift calculation across the 26 features.
4. `test_evidently_prediction_drift_report_generation`: Verifies prediction and probability drift reporting.
5. `test_evidently_classification_performance_reporting_with_labels`: Verifies classification performance reporting vs AHP labels.
6. `test_evidently_handling_of_missing_values`: Verifies accurate reporting of NaN values.
7. `test_evidently_26_feature_schema_and_ordering`: Verifies automatic re-alignment of scrambled column orders.
8. `test_evidently_html_report_generation_meaningful_content`: Verifies non-empty HTML containing domain features.
9. `test_evidently_json_structured_output_generation`: Verifies structured JSON schemas and metrics list.
10. `test_evidently_invalid_incompatible_inputs`: Verifies explicit `ValueError` raises on empty datasets and mismatched lengths.
11. `test_evidently_report_generation_errors_in_suite`: Verifies graceful partial status reporting on individual report failures.
12. `test_native_monitoring_fallback_when_evidently_uninstalled`: Verifies fallback to native SciPy/NumPy when Evidently is uninstalled.
13. `test_comprehensive_monitoring_cycle_with_evidently_suite`: End-to-end integration test of `run_monitoring_cycle(run_evidently=True)`.

---

## 8. Known Limitations & Safe Operational Boundaries

1. **Offline Evaluation vs. Streaming Production:**  
   The reference and comparison datasets represent historical train and test splits from the same time period. True production monitoring requires logging ongoing inference batches from field assessments.
2. **Deterministic AHP Precedence:**  
   In case of disagreement between deterministic AHP scoring and probabilistic Logistic Regression predictions, the deterministic AHP score remains authoritative.
3. **Provisional Engineering Thresholds:**  
   Alert levels (such as 80% low-tier recall, 15% boundary cases, 33% feature drift) should be calibrated empirically with coastal district disaster management authorities during field trials.
4. **Computational Footprint:**  
   Generating standalone interactive HTML reports with embedded JavaScript plots takes approximately 2–4 seconds per report. For high-frequency automated batch jobs, JSON reports should be favored.
