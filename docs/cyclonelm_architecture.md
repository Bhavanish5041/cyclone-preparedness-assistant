# CycloneLM v0.1: Architecture Specification

## 1. Executive Overview

**CycloneLM** is an internally developed, domain-specialised Transformer language model engineered specifically for the **CYCLONE INTEL** Coastal Karnataka Decision Support System. Developed natively inside `/home/bhavanish/cyclone-preparedness-assistant`, CycloneLM serves as the foundational milestone toward eliminating third-party external LLM API dependencies while upholding strict scientific governance, deterministic tool execution, and local CPU efficiency.

```
+-----------------------------------------------------------------------------------+
|                           CYCLONE INTEL PLATFORM                                  |
|                                                                                   |
|  +--------------------+        +---------------------+      +------------------+  |
|  |  React Frontend    | <----> |  FastAPI Backend    | <--> | Analytic Engine  |  |
|  |  (Espresso Theme)  |        |  (API Contracts)    |      | (AHP + ML + Clust)| |
|  +--------------------+        +----------+----------+      +--------+---------+  |
|                                           |                          |            |
|                                           v                          |            |
|                         +---------------------------------+          |            |
|                         |  CycloneLM Assistant Adapter    |          |            |
|                         |  (src/cyclonelm/adapters/)      |          |            |
|                         +----------------+----------------+          |            |
|                                          |                           |            |
|                                          v                           v            |
|                         +---------------------------------+  Deterministic Tools  |
|                         |    Tool Orchestration Layer     | <-------------------+ |
|                         |    (Pydantic Validation, Dedup) |  (7 Reg Tools)        |
|                         +----------------+----------------+                       |
|                                          |                                        |
|                                          v                                        |
|                         +---------------------------------+                       |
|                         |    CycloneLM Inference Engine   |                       |
|                         |    (Autoregressive Generation)  |                       |
|                         +----------------+----------------+                       |
|                                          |                                        |
|                         +----------------+----------------+                       |
|                         |    CycloneLM Transformer        |                       |
|                         |    (Decoder-Only TinyGPT)       |                       |
|                         |    - 2 Layers, 4 Heads          |                       |
|                         |    - d_model: 64, d_ff: 256     |                       |
|                         |    - 239,360 Parameters (< 1M)  |                       |
|                         +---------------------------------+                       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Model Architecture & Hyperparameters

CycloneLM v0.1 implements an autoregressive, causal decoder-only Transformer constructed in pure PyTorch, starting from randomly initialised weights.

### 2.1 Hyperparameter Specifications

| Dimension / Hyperparameter | Specification | Rationale & Architectural Design |
|---|---:|---|
| **Model Type** | Decoder-Only Transformer | Causal language modeling for instruction synthesis |
| **Vocab Size ($V$)** | 1,024 | Compact domain BPE vocabulary preserving special tokens |
| **Hidden Dimension ($d_{\text{model}}$)** | 64 | Tuned for CPU cache residency and low latency |
| **Transformer Layers ($L$)** | 2 | Sufficient representational capacity for domain routing |
| **Attention Heads ($H$)** | 4 | Multi-head diversification ($d_k = 64 / 4 = 16$) |
| **Feedforward Dimension ($d_{\text{ff}}$)** | 256 | $4 \times d_{\text{model}}$ expansion with GELU activation |
| **Context Window ($T_{\text{max}}$)** | 128 tokens | Accommodates tool calls and structured responses |
| **Layer Normalization** | Pre-LN ($\epsilon = 10^{-5}$) | Enhanced gradient stability during training |
| **Weight Initialization** | $\mathcal{N}(0, 0.02)$ | Residual projection scaled by $1/\sqrt{2L}$ |
| **Tied Word Embeddings** | False | Independent token embedding and LM head projection |
| **Total Parameters** | **239,360** | **Well below the 1,000,000 maximum ceiling** |

---

## 3. Detailed Component Breakdown

### 3.1 Embeddings
The model utilizes dual embedding layers:
1. **Token Embeddings**: Maps discrete token indices $i \in \{0, \dots, V-1\}$ to continuous representations in $\mathbb{R}^{d_{\text{model}}}$.
2. **Positional Embeddings**: Learned positional embeddings $p \in \{0, \dots, T_{\text{max}}-1\}$ in $\mathbb{R}^{d_{\text{model}}}$.

$$\mathbf{h}_0 = \mathbf{E}_{\text{token}}(\mathbf{x}) + \mathbf{E}_{\text{pos}}(\mathbf{p})$$

### 3.2 Causal Multi-Head Self-Attention
Attention heads project hidden states into Query, Key, and Value spaces:
$$\mathbf{Q} = \mathbf{x} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{x} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{x} \mathbf{W}_V$$

Scaled dot-product attention computes attention distributions subject to a lower-triangular causal mask $\mathbf{M} \in \mathbb{R}^{T \times T}$:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$
where $M_{i,j} = 0$ for $j \le i$, and $M_{i,j} = -\infty$ for $j > i$.

### 3.3 Transformer Block (Pre-LayerNorm)
To maintain gradient propagation across layers without vanishing gradients, CycloneLM employs the modern Pre-LN configuration:
$$\mathbf{x}^{(l)\prime} = \mathbf{x}^{(l-1)} + \text{Dropout}(\text{Attention}(\text{LN}_1(\mathbf{x}^{(l-1)})))$$
$$\mathbf{x}^{(l)} = \mathbf{x}^{(l)\prime} + \text{Dropout}(\text{FFN}(\text{LN}_2(\mathbf{x}^{(l)\prime})))$$

### 3.4 Parameter Count Breakdown

$$\text{Params} = \underbrace{V \cdot d_{\text{model}}}_{\text{Token Emb}} + \underbrace{T_{\text{max}} \cdot d_{\text{model}}}_{\text{Pos Emb}} + L \cdot \underbrace{\left(4 d_{\text{model}}^2 + 4 d_{\text{model}} + 2 d_{\text{model}} \cdot d_{\text{ff}} + d_{\text{ff}} + d_{\text{model}} + 4 d_{\text{model}}\right)}_{\text{Per-Layer Blocks}} + \underbrace{2 d_{\text{model}}}_{\text{Final LN}} + \underbrace{V \cdot d_{\text{model}}}_{\text{LM Head}}$$

For $V=1024, d_{\text{model}}=64, L=2, d_{\text{ff}}=256, T_{\text{max}}=128$:
- Token Embeddings: $1,024 \times 64 = 65,536$
- Positional Embeddings: $128 \times 64 = 8,192$
- Layer 1 + Layer 2: $2 \times 49,856 = 99,712$
- Final LayerNorm: $2 \times 64 = 128$
- LM Output Head: $64 \times 1,024 = 65,536$
- **Total Trainable Parameters**: **239,360**

---

## 4. Structured Tool-Use Architecture

CycloneLM interacts with the platform's analytical repository via a secure, deterministic protocol:

1. **Extraction**: Structured regex intercepts `<tool_call>{"tool": "...", "parameters": {...}}</tool_call>`.
2. **Safety Boundary**: Model-generated strings never interact with Python `eval()`, `exec()`, or raw SQL.
3. **Pydantic Validation**: All parameters are strongly validated against the schemas of the seven registered tools in `src.assistant.tools.registry`.
4. **Deduplication**: Calls with identical tool names and parameters within the same session are trapped and prevented from entering infinite feedback loops.
5. **Round Limit**: Capped at `max_rounds=3` to guarantee execution termination.
6. **Evidence Context**: Valid tool results are formatted into `<evidence>` sections for final grounded synthesis.

---

## 5. Scientific Governance Boundaries

CycloneLM operates under strict scientific constraints:
- **AHP Absolute Precedence**: Deterministic composite preparedness scores (0.0–100.0) and 4-band priority classifications govern all operational allocations.
- **ML Surrogate Demarcation**: Multinomial Logistic Regression is an empirical surrogate trained on 26 static features approximating AHP boundaries, NOT a physical cyclone hazard or casualty forecast model.
- **Non-Downgrading Invariant**: The ML surrogate can NEVER downgrade an authoritative AHP Low preparedness classification.
- **Zero-Hallucination Policy**: Refuses to generate synthetic dynamic telemetry (live wind speeds, barometric pressure, live shelter occupancy).
