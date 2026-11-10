# CycloneLM v0.1: Training Specification & Dynamics

## 1. Overview

CycloneLM v0.1 is trained from scratch using an end-to-end PyTorch pipeline designed for CPU execution. The pipeline consumes a curated domain corpus spanning tropical cyclone meteorology, Coastal Karnataka geography, MCDA/AHP methodology, machine learning surrogate demarcation, and structured tool interactions.

---

## 2. Dataset Provenance and Licensing

### 2.1 Provenance Matrix

| Domain Sector | Data Sources & Lineage | License & Rights |
|---|---|---|
| **Meteorology & Basin Dynamics** | India Meteorological Department (IMD) cyclone classification bulletins, World Meteorological Organization (WMO) operational definitions. | Public Sector Information / Open Academic Reference |
| **Regional Geography & Exposure** | Government of Karnataka Coastal Disaster Management documentation, Coastal Regulation Zone (CRZ) reports. | Government Open Access / Public Record |
| **MCDA & AHP Methodology** | Thomas L. Saaty's Analytic Hierarchy Process academic formulations, consistency ratio thresholds. | Open Academic Reference |
| **ML Surrogate & Safety Invariant** | CYCLONE INTEL Milestone 3A/3B technical specifications (26 features, non-downgrading safety rule). | Internal Project IP / CC0 1.0 Universal |
| **Vulnerability Archetypes** | K-Means clustering documentation for Dakshina Kannada and Udupi ($N=622$). | Internal Project IP / CC0 1.0 Universal |
| **Scientific Governance Rules** | Strict prohibition against dynamic telemetry hallucination. | Internal Project Governance |

> [!IMPORTANT]
> **Zero Memorization Requirement**: Community-specific numerical values (e.g., individual village population, exact AHP scores) are **NOT** memorized in model weights. Numerical evidence is strictly retrieved at runtime via deterministic tools (`lookup_community`, etc.).

### 2.2 Dataset Statistics

```
data/cyclonelm/
├── pretraining/
│   ├── cyclonelm_domain_corpus.txt   # 9 curated multi-paragraph technical sections
│   ├── train.txt                     # 85% train split (4,949 tokens)
│   └── val.txt                       # 15% validation split (1,420 tokens)
├── instructions/
│   ├── cyclonelm_instructions.jsonl  # 160 curated instruction-response pairs
│   ├── train.jsonl                   # 136 training pairs
│   └── val.jsonl                     # 24 validation pairs
└── processed/
    ├── cyclonelm_combined_corpus.txt # 39,665 total tokens (pretraining + instructions)
    ├── train_combined.txt            # 34,424 tokens
    └── val_combined.txt              # 5,241 tokens
```

---

## 3. Tokenizer Architecture

CycloneLM uses a pure-Python Byte-Pair Encoding (**BPETokenizer**) constructed from scratch:
- **Base Alphabet**: Seeded with all 100 printable ASCII characters + whitespace + punctuation to guarantee zero unhandled character exceptions.
- **Special Tokens**:
  - `<pad>` (ID: 0) — Context padding
  - `<unk>` (ID: 1) — Unknown token fallback
  - `<bos>` (ID: 2) — Beginning of sequence
  - `<eos>` (ID: 3) — End of sequence
  - `<tool_call>` (ID: 4) — Structured tool invocation start
  - `<tool_result>` (ID: 5) — Tool execution result delimiter
  - `<user>` (ID: 6) — User prompt tag
  - `<assistant>` (ID: 7) — Assistant response tag
  - `<system>` (ID: 8) — System governance instructions
- **Vocab Size**: 1,024 subword tokens.
- **Serialization**: Fully self-contained JSON schema at `models/cyclonelm/tokenizer/vocab.json`.

---

## 4. Training Dynamics and Optimization

### 4.1 Optimization Parameters

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$)
- **Weight Decay**: 0.01 applied selectively to 2D tensor weights; 0.0 applied to biases and LayerNorm parameters.
- **Learning Rate Schedule**: Linear warmup for 50 steps from 0.0 to $1 \times 10^{-3}$, followed by Cosine Annealing decay down to $1 \times 10^{-4}$.
- **Gradient Clipping**: Maximum norm $\le 1.0$.
- **Batch Size**: 8 sequences of length 128 tokens.
- **Hardware**: CPU (Intel Core i5-13500H, 16 vCPUs).

### 4.2 Loss Convergence Progression

The table below illustrates loss progression over 250 global steps:

| Step | Epoch | Learning Rate | Train Loss | Val Loss | Val Perplexity |
|---:|---:|---:|---:|---:|---:|
| **10** | 1 | 0.000220 | 6.7389 | — | — |
| **25** | 1 | 0.000520 | 6.2841 | 6.2633 | 524.97 |
| **50** | 2 | 0.001000 | 5.3422 | 5.2896 | 198.26 |
| **75** | 3 | 0.000965 | 4.6012 | 4.4162 | 82.78 |
| **100** | 4 | 0.000868 | 4.0080 | 3.8308 | 46.10 |
| **125** | 4 | 0.000725 | 3.9011 | 3.5056 | 33.30 |
| **150** | 5 | 0.000550 | 3.7802 | 3.3281 | 27.89 |
| **175** | 6 | 0.000378 | 3.6914 | 3.2037 | 24.62 |
| **200** | 7 | 0.000232 | 3.3842 | 3.1310 | 22.90 |
| **225** | 7 | 0.000135 | 3.5120 | 3.0967 | 22.12 |
| **250** | 8 | 0.000100 | 3.4642 | **3.0714** | **21.57** |

---

## 5. Checkpointing & Reproducibility

Checkpoints are serialized to `models/cyclonelm/checkpoints/` containing:
- `model_state_dict`: Weight matrices.
- `optimizer_state_dict`: AdamW momentum and variance accumulators.
- `config`: Architecture hyperparameters.
- `train_config`: Optimization settings.
- `global_step`: Exact iteration index.
- `history`: Metric logging logs.

Reproducibility is strictly enforced by seeding Python `random`, `numpy`, and PyTorch with seed `42`.

---

## 6. Execution Commands

```bash
# 1. Prepare data and curate domain instructions
.venv/bin/python scripts/cyclonelm/prepare_data.py

# 2. Train CycloneLM from scratch
.venv/bin/python scripts/cyclonelm/train.py --epochs 30 --max_steps 250

# 3. Generate sample text
.venv/bin/python scripts/cyclonelm/generate.py --prompt "AHP preparedness score" --temperature 0.6

# 4. Run comprehensive benchmark evaluation
.venv/bin/python scripts/cyclonelm/evaluate.py
```
