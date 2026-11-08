# Model Card: Village Preparedness Tier Classifier

## Model Details
- **Model Name:** Village Preparedness Tier Classifier
- **Model Artifact:** `logistic_regression_tier.joblib`
- **Architecture:** Scikit-learn Pipeline (RobustScaler + multinomial LogisticRegression)
- **Framework:** scikit-learn 1.9.1
- **Date Registered:** 2026-11-08
- **Provenance:** Reproducible reconstruction of teammate's documented 70/15/15 experiment (`notebooks/02_Baseline_Models.ipynb`).

## Intended Use
- **Primary Use:** Decision support for disaster preparedness authorities in Coastal Karnataka.
- **Classification Target:** Village preparedness capacity tiers (`High`, `Medium`, `Low`).
- **Target Definition:** AHP-derived village preparedness tiers (High, Medium, Low) reflecting composite physical, social, and infrastructural readiness. Deterministic AHP assessments serve as the baseline ground truth. Logistic Regression outputs provide calibrated tier probabilities for granular boundary case detection, NOT physical cyclone-risk or hazard prediction.
- **Deterministic Precedence:** Deterministic AHP assessments hold operational precedence over model predictions.

## Features & Schema
- **Input Dimension:** Exactly 26 engineered spatial, infrastructural, demographic, and hazard features.
- **Strict Ordering:** Feature columns must match fit order exactly.
- **Categorical Indicators:** `has_pucca_road`, `has_major_district_road`, `bus_connectivity`.

## Evaluation Summary (Held-Out Test Split, N=282)
- **Overall Accuracy:** 0.9433 (94.33%)
- **Macro F1-Score:** 0.9060
- **High Preparedness Recall:** 0.9444
- **Medium Preparedness Recall:** 0.9704
- **Low Preparedness Recall:** 0.7391 (Safety Alert: 6/23 Low-tier villages predicted as Medium)

## Ethical & Safety Considerations
- **Human-in-the-Loop:** Predictions serve as secondary verification and boundary-case detection; they do NOT replace local ground-truth administrative judgment.
- **Vulnerability Sensitivity:** Low-tier villages must receive safety-buffer inspections even if classified as Medium.
