# CYCLONE INTEL: Model Quality & Invariance Verification Report

**Timestamp (UTC):** `2026-11-11T04:03:24.286693+00:00`  
**Pipeline Status:** `PASSED ✅`  
**Total Evaluation Samples:** `130`  

> [!NOTE]
> **Dataset Limitations & Scope Disclosure:** This dataset contains 130 curated regression test cases drawn from the authoritative Census 2011 and multi-criteria AHP ground truth for coastal Karnataka (1,879 administrative units across Dakshina Kannada, Udupi, and Uttara Kannada). It is designed for regression gating, boundary verification, and scientific invariance enforcement. It does NOT claim statistical generalizability beyond coastal Karnataka.

---

## 1. Executive Quality Gates & Scientific Invariants Summary

| Quality Gate Metric | Measured Value | Acceptance Threshold | Result |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | `1.0` | `>= 0.9` | **PASS ✅** |
| **Intent Macro F1** | `1.0` | `>= 0.9` | **PASS ✅** |
| **Argument Correctness** | `1.0` | `>= 0.9` | **PASS ✅** |
| **Entity Resolution Accuracy** | `1.0` | `>= 0.95` | **PASS ✅** |
| **Multi Intent Completeness** | `1.0` | `>= 0.9` | **PASS ✅** |
| **Json Schema Validity** | `1.0` | `>= 1.0` | **PASS ✅** |
| **Grounded Factual Correctness** | `1.0` | `>= 1.0` | **PASS ✅** |
| **Ahp Score Invariance** | `1.0` | `>= 1.0` | **PASS ✅** |
| **No Ml Downgrade Of Ahp Low** | `1.0` | `>= 1.0` | **PASS ✅** |
| **Unsupported Claim Frequency** | `0.0` | `== 0.0` | **PASS ✅** |
| **Refusal Precision** | `1.0` | `>= 1.0` | **PASS ✅** |
| **Refusal Recall** | `1.0` | `>= 1.0` | **PASS ✅** |
| **Surrogate Concordance Rate** | `0.9537` | `>= 0.85` | **PASS ✅** |

---

## 2. Core Quantitative Metrics

- **Intent Selection Accuracy:** `100.0%`
- **Intent Macro F1:** `1.0000` (Precision: `1.0000`, Recall: `1.0000`)
- **Tool Argument Correctness:** `100.0%`
- **Entity Resolution Accuracy:** `100.0%`
- **Multi-Intent Completeness:** `100.0%`
- **JSON Schema Validity Rate:** `100.0%`
- **Grounded Factual Correctness (AHP Invariance):** `100.0%`
- **Non-Downgrading Invariant Compliance:** `100.0%`
- **Unsupported Claim / Hallucination Frequency:** `0.0%`
- **Refusal Precision & Recall:** `100.0%` / `100.0%`

---

## 3. Disaggregated Latency & Resource Utilization

### A. Deterministic Pipeline Latency (Intent Parsing + Extraction + Refusal Gate)
- **Mean Latency:** `29.02 ms`
- **Median Latency:** `22.62 ms`
- **95th Percentile Latency (p95):** `64.43 ms`
- **Min / Max Latency:** `0.0 ms` / `139.17 ms`

### B. Deterministic Tool Execution Latency (AHP Scoring & Spatial Assessment)
- **Mean Execution Latency:** `4.0 ms`
- **Median Execution Latency:** `3.99 ms`
- **95th Percentile Latency (p95):** `4.3 ms`

### C. Actual Qwen2.5-0.5B Inference Latency (Local Ollama Engine)
- **Model:** `qwen2.5:0.5b-instruct`
- **Mean Generation Latency:** `1592.07 ms` (~1.59s)
- **Median Generation Latency:** `1259.54 ms`
- **95th Percentile Latency (p95):** `2637.2` ms
- **Throughput:** ~`16.96 tokens/sec` on CPU

### D. Memory Consumption
- **Initial Process Resident Memory (RSS):** `713.88 MB`
- **Final Process Resident Memory (RSS):** `713.88 MB`
- **Net Memory Delta:** `0.0 MB`

---

## 4. Surrogate Classifier & Clustering Methodological Disclaimers

### A. Logistic Regression Tier Surrogate
- **Test Accuracy:** `0.9432624113475178` | **Macro F1:** `0.9060227871307046`
- **AHP Tier Concordance:** `95.4%`
> [!IMPORTANT]
> SCIENTIFIC DISTINCTION MANDATE: The Logistic Regression model is an empirical surrogate approximating deterministic AHP preparedness tiers (High, Medium, Low) across 26 infrastructural features. Surrogate probabilities (prob_high, prob_medium, prob_low) are NOT calibrated physical cyclone-risk probabilities, storm surge forecasts, or casualty predictions. Deterministic AHP assessments maintain absolute operational precedence.

### B. K-Means Vulnerability Clustering (Descriptive Only)
- **Algorithm:** `K-Means (k=4) on standardized z-score transformed indicator dimensions`
- **Study Units Covered:** `622` | **Unassigned Fallback Units:** `1257`
> [!NOTE]
> DESCRIPTIVE CLUSTERING MANDATE: K-Means clusters represent descriptive spatial archetypes (Urban Corridor, Remote Foothills, Rural Interior, Coastal Belt) identifying predominant structural vulnerabilities across Dakshina Kannada and Udupi. Clustering does NOT predict cyclone damage or storm severity, and does not supersede authoritative AHP priority allocations. Uttara Kannada units are handled via explicit fallback heuristics.
