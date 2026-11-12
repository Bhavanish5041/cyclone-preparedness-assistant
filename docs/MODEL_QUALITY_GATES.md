# CYCLONE INTEL: Model Quality Gates & Scientific Safeguards

## 1. Overview & Quality Governance Framework

The CYCLONE INTEL decision intelligence platform relies on multiple machine learning and analytical models:
1. **Multi-Criteria Decision Analysis (AHP)**: Deterministic, authoritative composite preparedness scoring and 4-tier operational priority band allocation.
2. **Multinomial Logistic Regression Surrogate**: Empirical surrogate model approximating AHP preparedness tiers across 26 infrastructural features to provide continuous boundary ambiguity detection.
3. **K-Means Spatial Clustering ($k=4$)**: Descriptive structural vulnerability archetypes across 622 coastal units (with 1,257 unassigned Uttara Kannada units handled via fallback guidance).
4. **Qwen2.5-0.5B / Tool-Calling Orchestrator**: Conversational natural language interface with deterministic refusal gating and tool-grounded factual answers.

To ensure patient and community safety in disaster risk management, automated quality gates run on every pull request and release build.

---

## 2. Complete Quality Gate Specification & Thresholds

| Quality Gate Identifier | Category | Measured Value | Acceptance Threshold | Baseline Justification | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`intent_accuracy`** | Assistant | `1.000` (100%) | $\ge 0.90$ (90%) | Prevents executing incorrect tools during critical incident queries. | **PASS ✅** |
| **`intent_macro_f1`** | Assistant | `1.000` (1.00) | $\ge 0.90$ (0.90) | Ensures balanced intent sensitivity across frequent and rare tools. | **PASS ✅** |
| **`argument_correctness`** | Assistant | `1.000` (100%) | $\ge 0.90$ (90%) | Prevents querying wrong identifiers or malformed parameters. | **PASS ✅** |
| **`entity_resolution_accuracy`** | Assistant | `1.000` (100%) | $\ge 0.95$ (95%) | Resolves colloquial village names, aliases, and Census codes. | **PASS ✅** |
| **`multi_intent_completeness`** | Assistant | `1.000` (100%) | $\ge 0.90$ (90%) | Completely executes compound inquiries (e.g., lookup + recommend). | **PASS ✅** |
| **`json_schema_validity`** | Assistant | `1.000` (100%) | $== 1.00$ (100%) | Enforces strict Pydantic parsing with zero malformed JSON. | **PASS ✅** |
| **`grounded_factual_correctness`** | Scientific | `1.000` (100%) | $== 1.00$ (100%) | Verifies 100% fidelity to authoritative AHP ground truth scores. | **PASS ✅** |
| **`ahp_score_invariance`** | Scientific | `1.000` (100%) | $== 1.00$ (100%) | Asserts AHP score stability (e.g., Tallur is invariant at 74.34). | **PASS ✅** |
| **`no_ml_downgrade_of_ahp_low`** | Scientific | `1.000` (100%) | $== 1.00$ (100%) | **Non-negotiable invariant**: ML predictions never downgrade AHP Low. | **PASS ✅** |
| **`unsupported_claim_frequency`** | Safety | `0.000` (0.0%) | $== 0.00$ (0.0%) | Zero tolerance for conversational hallucinations or fabricated metrics. | **PASS ✅** |
| **`refusal_precision`** | Safety | `1.000` (100%) | $== 1.00$ (100%) | Immediately refuses dynamic/live meteorological telemetry requests. | **PASS ✅** |
| **`refusal_recall`** | Safety | `1.000` (100%) | $== 1.00$ (100%) | Catches 100% of out-of-scope operational queries. | **PASS ✅** |
| **`surrogate_concordance_rate`** | ML Surrogate | `0.9537` (95.4%)| $\ge 0.85$ (85%) | Surrogate model agrees with authoritative AHP tier on $\ge 85\%$ of units. | **PASS ✅** |
| **`surrogate_accuracy`** | ML Surrogate | `0.9433` (94.3%)| $\ge 0.90$ (90%) | Held-out test accuracy on 282 units exceeds verified baseline. | **PASS ✅** |
| **`surrogate_macro_f1`** | ML Surrogate | `0.9060` (0.91) | $\ge 0.85$ (0.85) | High, Medium, and Low classes all maintain balanced F1 scores. | **PASS ✅** |
| **`surrogate_low_tier_recall`** | ML Surrogate | `0.7391` (73.9%)| $\ge 0.70$ (70%) | Guarantees sensitive detection of vulnerable Low preparedness units. | **PASS ✅** |
| **`surrogate_failure_rate`** | ML Surrogate | `0.0000` (0.0%) | $== 0.00$ (0.0%) | Zero unhandled exceptions or NaN probability vectors. | **PASS ✅** |
| **`surrogate_schema_validity`** | ML Surrogate | `1.0000` (100%)| $== 1.00$ (100%) | Strictly adheres to the canonical 26-feature ordering. | **PASS ✅** |

---

## 3. Disaggregated Latency Performance Limits

All deterministic decision support routines execute on local commodity CPU without hardware accelerators:

| Subsystem Component | Mean Latency | Median Latency | 95th Percentile (p95) | Performance Limit |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic Pipeline** (Intent parse, regex extractor, refusal gate) | `15.94 ms` | `12.33 ms` | `36.42 ms` | $< 250.0\text{ ms}$ |
| **Deterministic Tool Execution** (AHP score retrieval, distance calculations) | `2.21 ms` | `2.23 ms` | `2.48 ms` | $< 50.0\text{ ms}$ |
| **Local Qwen2.5-0.5B Generation** (Ollama CPU engine) | `378.5 ms` | `367.0 ms` | `510.2 ms` | $< 1500.0\text{ ms}$ |

---

## 4. Evaluation Datasets & Methodological Disclosures

### A. Evaluation Dataset Composition
The automated evaluation pipeline utilizes two distinct evaluation datasets:
1. **Curated Regression Dataset (`tests/fixtures/model_test_dataset.json`)**: Contains 130 meticulously verified test cases designed to stress-test intent parsing, entity resolution, multi-intent compound queries, refusal gates, and edge-case syntax.
2. **Held-Out Test Split (`data/processed/modelling_dataset.parquet`)**: Contains 282 held-out village records (20% of the modeling universe) completely separated from model training, used to benchmark surrogate accuracy, class F1-scores, and confusion matrices.

> [!NOTE]
> **Dataset Scope Disclosure:** The dataset covers coastal Karnataka (Dakshina Kannada, Udupi, and Uttara Kannada) based on Census 2011, SECC economic indicators, and MoHFW healthcare registries. The models do NOT claim statistical generalizability to non-coastal or other Indian states.

### B. Scientific Invariance Mandate
- **AHP Operational Authority:** Multi-criteria AHP scores (0–100) and 4-tier priority bands (Band 1 Critical to Band 4 Adequate) maintain absolute precedence in operational decision-making.
- **Surrogate Distinction:** Logistic Regression probabilities ($P(\text{High}), P(\text{Medium}), P(\text{Low})$) reflect classifier confidence approximating the AHP score. They are **NOT** calibrated physical cyclone-risk probabilities, storm-surge damage forecasts, or casualty predictions.
- **Descriptive Clustering:** K-Means clusters identify shared structural deficit patterns across coastal communities. They are purely descriptive and must not be used to predict disaster impact severity.
- **Uttara Kannada Preservation:** All 1,257 Uttara Kannada units are deliberately preserved as unassigned with explicit district fallback guidance rather than forcing them into ill-fitting cluster boundaries.

---

## 5. Verification Commands

To execute the model quality pipeline and generate reports locally:

```bash
# 1. Run complete model quality pipeline and generate JSON/Markdown reports
python scripts/run_model_evaluation.py

# 2. Run automated model quality regression gates
pytest tests/test_model_quality_gates.py -v

# 3. Run model integrity, checksum, and tampering checks
pytest tests/test_model_integrity_and_safeguards.py -v
```
