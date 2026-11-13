# Model Risk and Quality Gate Report

**Execution Timestamp**: `2026-11-13T01:31:00.698854+00:00`  
**Overall Pipeline Status**: `✅ PASS`  
**Model Registry Promotion**: `🚀 ELIGIBLE` (`PROMOTION_ELIGIBLE`)

---

## 1. Executive Summary

| Dimension | Checked Model | Metric / Scope | Measured | Gate Threshold | Status |
|---|---|---|---|---|---|
| **Classification** | `village-preparedness-classifier` | Held-Out Test Accuracy | `0.9433` | `≥ 0.85` | ✅ |
| **Classification** | `village-preparedness-classifier` | Held-Out Macro F1 | `0.906` | `≥ 0.80` | ✅ |
| **Classification** | `village-preparedness-classifier` | Low-Tier Recall | `0.7391` | `≥ 0.70` | ✅ |
| **Clustering** | `coastal-vulnerability-kmeans` | Silhouette Score (k=4) | `0.3348` | `≥ 0.30` | ✅ |
| **Clustering** | `coastal-vulnerability-kmeans` | Davies-Bouldin Index | `1.1771` | `≤ 1.30` | ✅ |
| **Clustering** | `coastal-vulnerability-kmeans` | Calinski-Harabasz Index | `298.59` | `≥ 250.0` | ✅ |
| **Clustering** | `coastal-vulnerability-kmeans` | Cluster Balance Ratio | `0.3293` | `≥ 0.25` | ✅ |
| **Clustering** | `coastal-vulnerability-kmeans` | Multi-Seed Mean ARI (10 seeds) | `0.9912` | `≥ 0.95` | ✅ |
| **Clustering** | `coastal-vulnerability-kmeans` | Multi-Seed Min ARI (10 seeds) | `0.9635` | `≥ 0.90` | ✅ |

---

## 2. Scientific Limitations & Domain Safeguards

> [!IMPORTANT]
> **Logistic Regression Classification**: Logistic Regression is an AHP-derived preparedness-tier surrogate, not a cyclone hazard predictor. Its probabilities must not be described as calibrated without calibration evaluation.

> [!NOTE]
> **K-Means Clustering**: Discovered archetypes represent empirical similarity groupings derived from four normalized spatial indicators. They are NOT predetermined hazard categories, validated disaster risk classes, or casualty predictions.

---

## 3. Promotion & Registry Audit Rules

- **Champion / Production Promotion Gate**: Only models that pass **all 8 dimensions** receive eligibility for production registry promotion.
- **Fail-Safe Policy**: Any quality gate breach results in immediate pipeline termination (exit code 1) and blocks deployment.
- **Artifact Immutability**: Production models in `models/` are never overwritten during exploratory evaluation.
