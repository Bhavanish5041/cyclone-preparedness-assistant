# CycloneLM v0.1: Comprehensive Evaluation Report

## 1. Evaluation Methodology

The CycloneLM evaluation suite is automated via `scripts/cyclonelm/evaluate.py` and unit tested under `tests/cyclonelm/`. The evaluation framework decouples deterministic tool validation from autoregressive language generation to ensure rigorous scientific governance and verifiable reproducibility.

---

## 2. Quantitative Evaluation Results

The results below reflect the official benchmark run recorded in `reports/cyclonelm_evaluation_report.json`:

### 2.1 Model Architecture & Resource Footprint

| Metric | Measured Value | Threshold / Standard | Status |
|---|---|---|:---:|
| **Trainable Parameter Count** | 239,360 | $< 1,000,000$ | Passed |
| **Non-Embedding Parameters** | 165,632 | $< 700,000$ | Passed |
| **Model Size on Disk** | 2,977.8 KB (~2.91 MB) | $< 50$ MB | Passed |
| **Peak Resident Set Size (RSS)** | 1,264.2 MB | $< 4,000$ MB | Passed |
| **Device Execution** | CPU (Intel Core i5-13500H) | CPU-compatible | Passed |

### 2.2 Checkpoint Reload Invariance

| Verification Step | Result | Precision Tolerance | Status |
|---|---|---|:---:|
| **Weight Tensor Match** | Identical | Max abs difference = $0.00\text{e}+00$ | Passed |
| **State Dict Keys** | 100% Match | Zero missing / unexpected keys | Passed |
| **Deterministic Generation Output** | Bit-for-bit identical | Identical token sequence on fixed seed | Passed |

### 2.3 CPU Inference Latency & Throughput

Measured over 5 trials generating 32 tokens with sequence length 16:

| Benchmark Dimension | Measured Performance | Operational Significance |
|---|---:|---|
| **Average Latency per Token** | **1.22 ms** | Real-time interactive response |
| **Generation Throughput** | **821.8 tokens / sec** | Ultra-fast local execution on CPU |
| **Time to First Token (TTFT)** | **~2.8 ms** | Negligible UI blocking overhead |

---

## 3. Tool-Use Protocol Evaluation

The structured tool protocol orchestrator was evaluated against the platform's 7 registered deterministic tools:

| Tool Name | Test Query / Payload | Pydantic Schema Status | Execution Result |
|---|---|:---:|:---:|
| `lookup_community` | `{"query": "Ullal", "district": "Dakshina Kannada"}` | Validated | **Success** (Retrieved 1 unit, AHP score 38.4) |
| `compare_districts` | `{"district_a": "Dakshina Kannada", "district_b": "Udupi"}` | Validated | **Success** (Aggregated comparative profile) |
| `explain_vulnerability_archetype` | `{"archetype_id": 0}` | Validated | **Success** (High Exposure Coastal Littoral) |
| `analyze_infrastructure_and_exposure` | `{"identifier": "Ullal"}` | Validated | **Success** (Exposure & shelter deficit data) |
| `explain_preparedness_assessment` | `{"identifier": "Ullal"}` | Validated | **Success** (5 domain scores & criteria weights) |
| `get_intervention_recommendations` | `{"identifier": "Ullal"}` | Validated | **Success** (Prioritized engineering interventions) |
| `filter_and_rank_communities` | `{"district": "Dakshina Kannada", "priority_band": "Band 1"}` | Validated | **Success** (Filtered 42 critical habitations) |

### 3.1 Security & Guardrail Verification

- **Unrecognized Tool Execution**: Attempted execution of `execute_arbitrary_shell_cmd` was intercepted and safely rejected with `ToolNotFoundError`.
- **Session Duplicate Detection**: Duplicate calls in the same conversational turn were intercepted and tagged with `DuplicateCallError`.
- **Arbitrary Code Execution Safeguard**: Model strings are parsed via `json.loads` within regex boundaries; `eval()`, `exec()`, and shell subshells are completely prohibited.

---

## 4. Scientific Governance Adherence Matrix

| Governance Requirement | Enforcement Mechanism | Evaluation Outcome |
|---|---|:---:|
| **AHP Absolute Precedence** | Prompt guardrail + Orchestration context | **Passed** |
| **Surrogate ML Demarcation** | Required "empirical surrogate" label | **Passed** |
| **Non-Downgrading Invariant** | AHP Low communities locked to Band 1 Critical | **Passed** |
| **Descriptive Archetypes** | Structural profiling only, no loss forecast | **Passed** |
| **Refusal of Dynamic Hallucination** | Refusal to synthesize live wind or bed occupancy | **Passed** |
| **Ground Truthing Requirement** | Mandatory disclaimer for DDMA field engineering | **Passed** |

---

## 5. Known Limitations of CycloneLM v0.1

1. **Model Parameter Scale (239K Parameters)**:
   CycloneLM v0.1 is an ultra-compact Transformer engineered to prove end-to-end architectural integrity, local training pipelines, and deterministic tool dispatching. It is **not** a general-purpose fluent conversationalist and must not be expected to produce unconstrained, multi-paragraph essays without structured scaffolding.
2. **Scaffolded Tool Dispatching**:
   While the tool orchestration and validation layer is fully robust, TinyGPT models benefit from few-shot priming or template scaffolding to ensure 100% syntactically perfect `<tool_call>` generation across ambiguous user inputs.
3. **Experimental Deployment Status**:
   In compliance with project safety mandates, CycloneLM v0.1 is deployed **strictly in experimental mode**. The production FastAPI assistant route continues to use the existing LLM provider until CycloneLM passes complete human-in-the-loop operational validation by disaster management authorities.
