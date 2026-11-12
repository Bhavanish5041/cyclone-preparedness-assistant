# 09. Verified Metrics, Test Results & Model Limitations

## 1. Verified Metrics Ledger

All metrics listed below reflect live evaluations against the repository datasets, verified models, and automated test suites.

### 1.1 Logistic Regression Surrogate Classifier
Evaluated on stratified held-out test partitions:

| Metric | Measured Value | Benchmark Threshold | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **94.33%** | $\ge 90.0\%$ | PASS ✅ |
| **Test Macro F1 Score** | **90.60%** | $\ge 85.0\%$ | PASS ✅ |
| **Test Weighted F1 Score** | **94.22%** | $\ge 90.0\%$ | PASS ✅ |
| **Overall AHP Concordance** | **95.37%** (1,792 / 1,879) | $\ge 90.0\%$ | PASS ✅ |
| **Low-Tier Recall** | **88.39%** | $\ge 70.0\%$ | PASS ✅ |
| **High-Tier Recall** | **97.15%** | $\ge 90.0\%$ | PASS ✅ |

#### Split Provenance Breakdown:
- **`TRAIN_IN_SAMPLE` (1,315 units):** 1,261 agreeing (95.89% agreement)
- **`VAL_OUT_OF_SAMPLE` (282 units):** 265 agreeing (93.97% agreement)
- **`TEST_OUT_OF_SAMPLE` (282 units):** 266 agreeing (94.33% agreement)
- *Total Disagreements:* Exactly **87 units** out of 1,879. In all 87 disagreement cases, the deterministic AHP tier is strictly preserved as the canonical assessment.

---

### 1.2 K-Means Vulnerability Archetyping ($K = 4$)
Evaluated across 622 administrative units in Dakshina Kannada (369) and Udupi (253):

| Evaluation Metric | Measured Value | Methodological Interpretation |
| :--- | :---: | :--- |
| **Silhouette Score** | **0.3348** | Reasonable cluster cohesion in 4D spatial feature space. |
| **Davies–Bouldin Index** | **1.1771** | Low cluster overlap ($<1.2$). |
| **Calinski–Harabasz Score** | **298.59** | Dense clusters relative to separation distance. |
| **Cluster Balance Ratio** | **0.3293** | Smallest cluster ($n=82$) to largest ($n=249$); prevents single-unit micro-clusters. |
| **Multi-Seed Stability (ARI)** | **0.9902** | Mean Adjusted Rand Index across 10 random seeds (42, 123, 456, 789, 999, 1024, 2024, 7, 13, 99). |
| **Inter-Algorithm Agreement** | **0.7575** | Adjusted Rand Index comparing K-Means ($k=4$) vs. Agglomerative (Ward linkage, $k=4$). |

---

### 1.3 Operational Priority Band Census ($N = 1,879$)

| Priority Band | Total Units | Dakshina Kannada | Udupi | Uttara Kannada |
| :--- | :---: | :---: | :---: | :---: |
| **`BAND_1_CRITICAL`** | **155** | 2 | 0 | 153 |
| **`BAND_2_SPECIALIZED`** | **829** | 123 | 20 | 686 |
| **`BAND_3_MODERATE`** | **299** | 45 | 43 | 211 |
| **`BAND_4_ADEQUATE`** | **596** | 199 | 190 | 207 |
| **TOTAL** | **1,879** | **369** | **253** | **1,257** |

---

### 1.4 Test Suite Totals Across 10 CI Stages

```
================================================================================
                    CYCLONE INTEL: AUTOMATED TEST SUITE SUMMARY
================================================================================
Stage / Test Suite                                Items      Passing    Status
--------------------------------------------------------------------------------
Stage 2: Model Integrity & Cryptographic Hashes      45           45      PASS ✅
Stage 3: Model Quality Gates Regression              14           14      PASS ✅
Stage 4: Static Linting & Vulnerability Scan      Files        0 err      PASS ✅
Stage 5: Full Analytical Backend Tests              940          940      PASS ✅
Stage 6: Frontend Vitest Integration Suite           31           31      PASS ✅
Stage 7: Qwen & Ollama Offline Mocks                 22           22      PASS ✅
Stage 8: Docker Stack Validation Suite               11           11      PASS ✅
Stage 9: End-to-End Regression & Tallur Invariant    20           20      PASS ✅
CD Pipeline: Deployment Engine & Rollback Tests      23           23      PASS ✅
CD Pipeline: Security & Approval Gate Tests           8            8      PASS ✅
--------------------------------------------------------------------------------
TOTAL TEST COVERAGE                               1,114        1,114      PASS ✅
================================================================================
```

---

### 1.5 System Latency & Resource Utilization

| Component | Benchmark Metric | Measured Performance |
| :--- | :--- | :--- |
| **Deterministic Data Tools** | Execution Latency | $< 10\text{ ms}$ per tool call |
| **Deterministic Data Tools** | Payload Size | $< 6.5\text{ KB}$ (strictly bounded JSON) |
| **Local LLM Inference** | Qwen 2.5:0.5B Turn Latency | $250 - 650\text{ ms}$ on standard 16-core CPU |
| **Local LLM Memory** | Ollama Host RAM Footprint | $\approx 618\text{ MB}$ |
| **Container Memory** | `cyclone-api` Container RAM | Peak $285\text{ MiB}$ (Limit: $512\text{ MiB}$) |
| **Container Memory** | `cyclone-frontend` Container RAM | Peak $35\text{ MiB}$ (Limit: $256\text{ MiB}$) |

---

## 2. Explicit Scientific Limitations & Boundaries

To preserve academic integrity, always state these four limitations upfront:

1. **Census Baseline Indicators:**  
   Infrastructural indicators (pucca roads, drinking water, electricity) are anchored in the **Census of India 2011** and the Socioeconomic High-resolution Rural-Urban Geographic Platform (SHRUG). While geodesic hospital and coastline distances have been recalculated using contemporary OpenStreetMap layers, intra-village infrastructure reflects the 2011 baseline.
2. **Surrogate Model Probabilities are Uncalibrated:**  
   Softmax probability scores from the Logistic Regression model measure alignment with AHP decision tiers. They do **not** represent physical probabilities of cyclone occurrence, structural roof failure, or human injury.
3. **Spatial Scope of K-Means Clustering:**  
   The 4 K-Means archetypes apply strictly to the 622 units of Dakshina Kannada and Udupi. The 1,257 units of Uttara Kannada are marked as out-of-study-area and evaluated through a deterministic sector deficit fallback engine.
4. **Decision Support, Not Dispatch Decrees:**  
   All outputs are advisory planning heuristics for District Disaster Management Authorities. They do not constitute statutory evacuation orders or automated emergency dispatch commands.
