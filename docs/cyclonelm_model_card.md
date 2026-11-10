# CycloneLM v0.1: Model Card

## 1. Model Details

- **Model Identifier**: CycloneLM v0.1 (Experimental)
- **Developer**: CYCLONE INTEL Engineering & Research Team
- **Model Type**: Causal Decoder-Only Transformer (TinyGPT architecture)
- **Base Architecture**: 2 Layers, 4 Attention Heads, $d_{\text{model}} = 64$, $d_{\text{ff}} = 256$, Pre-LN
- **Vocabulary**: 1,024 subword tokens (Byte-Pair Encoding)
- **Weight Initialization**: From scratch (Gaussian $\mathcal{N}(0, 0.02)$, no pretrained checkpoint)
- **Parameter Count**: 239,360 trainable parameters
- **License**: Internal Research Model / Project Academic Use

---

## 2. Intended Use & Operational Boundary

### 2.1 Intended Use Cases
- Assisting District Disaster Management Authorities (DDMAs) in Coastal Karnataka (Dakshina Kannada, Udupi, Uttara Kannada) with structured queries.
- Generating deterministic tool-call structures (`lookup_community`, `compare_districts`, etc.).
- Explaining the methodology of the Analytic Hierarchy Process (AHP), Consistency Ratios ($CR < 0.10$), and 5 domain criteria.
- Synthesizing tool evidence into objective advisory briefing notes.

### 2.2 Out-of-Scope & Prohibited Uses
- **Direct Real-Time Emergency Commands**: CycloneLM is an analytical decision support aid, not an autonomous emergency dispatch controller.
- **Overriding AHP Baselines**: The model must never alter or recalculate authoritative AHP preparedness scores or priority bands.
- **Meteorological Forecasting**: The model must not forecast cyclone genesis, landfall locations, or storm surge heights.
- **Dynamic Telemetry Fabrication**: The model is strictly prohibited from fabricating live wind speeds, road trafficability, or hospital bed occupancy.

---

## 3. Quantitative Evaluation Summary

| Benchmark Metric | Measured Result | Evaluation Target | Compliance |
|---|---|---|:---:|
| **Total Parameters** | 239,360 | $< 1,000,000$ | Passed |
| **Non-Embedding Parameters** | 165,632 | $< 700,000$ | Passed |
| **Checkpoint Size (Disk)** | 2.91 MB | $< 50$ MB | Passed |
| **Peak Training RSS Memory** | 1,264 MB | $< 4,000$ MB (Available: 4.6 GB) | Passed |
| **CPU Training Duration (250 steps)** | 8.99 seconds | $< 300$ seconds | Passed |
| **CPU Inference Latency** | 1.22 ms / token | $< 25$ ms / token | Passed |
| **CPU Generation Throughput** | 821.8 tokens / sec | $> 50$ tokens / sec | Passed |
| **Checkpoint Reload Invariance** | Identical ($0.00\text{e}+00$ diff) | $< 10^{-6}$ | Passed |
| **Final Validation Loss** | 3.0714 | $< 4.0$ | Passed |
| **Final Validation Perplexity** | 21.57 | $< 50.0$ | Passed |

---

## 4. Pretrained Model Feasibility Assessment (Phase 7 Study)

In addition to building CycloneLM from scratch, we conducted a rigorous feasibility analysis of small open-weight models for potential secondary fine-tuning on our available hardware:
- **Processor**: Intel Core i5-13500H (12 Cores, 16 Threads, up to 4.7 GHz).
- **Available System Memory**: ~4.6 GB available out of 12.0 GB installed.
- **Dedicated GPU**: None (Intel Iris Xe integrated graphics only).

### 4.1 Comparative Model Matrix

| Candidate Model | Parameters | License | Disk Size (FP16 / Q4_K_M) | RAM Needed (Inference) | RAM Needed (LoRA Fine-Tuning) | CPU Feasibility Status |
|---|---:|---|---|---|---|:---:|
| **SmolLM-135M** (Hugging Face) | 135M | Apache 2.0 | 270 MB / 85 MB | ~0.4 GB | ~1.8 GB | **Highly Feasible** |
| **Qwen2.5-0.5B** (Alibaba) | 490M | Apache 2.0 | 980 MB / 320 MB | ~0.8 GB | ~3.8 GB | **Feasible (Tight RAM)** |
| **TinyLlama-1.1B** | 1.1B | Apache 2.0 | 2.2 GB / 680 MB | ~1.6 GB | $> 6.5$ GB (OOM Risk) | **Inference Only** |
| **Llama-3.2-1B** | 1.2B | Llama 3.2 Community | 2.4 GB / 750 MB | ~1.8 GB | $> 7.0$ GB (OOM Risk) | **Inference Only** |

### 4.2 Architectural Conclusions
1. **SmolLM-135M** is the most viable candidate for future LoRA experiments on our CPU setup, requiring under 2 GB of RAM during parameter-efficient fine-tuning with gradient checkpointing.
2. Models $\ge 1.0\text{B}$ parameters risk system Out-Of-Memory (OOM) crashes on our 12 GB RAM setup (which currently has 4.6 GB available) if fine-tuning is attempted without external GPU offload.
3. **Primary Recommendation**: Retain our custom-built **CycloneLM (239K params)** as the core native model, and explore SmolLM-135M as an experimental secondary comparator only if explicit external approval is granted.

---

## 5. Ethical Considerations & Safety Boundaries

1. **Closed-World Grounding**: The assistant must synthesize verified outputs from the 7 deterministic tools.
2. **Scientific Governance Precedence**: If a surrogate model prediction conflicts with an AHP tier, the deterministic AHP tier governs operational allocation.
3. **Transparent Uncertainty**: Historical baseline limitations (Census 2011, MoHFW directory) must be explicitly disclosed when briefing disaster management authorities.
