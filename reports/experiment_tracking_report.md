# Weights & Biases (W&B) Experiment Tracking Report

**Project**: `cyclone-preparedness-assistant`  
**W&B Workspace URL**: [https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant)  
**Entity**: `bhavanishreddy69-na`  
**Execution Timestamp**: `2026-10-22 23:57:30 IST`  
**Online Synchronization**: **Verified & Synchronized (17 Runs)**  

---

## 1. Executive Summary

This report documents the online experiment tracking implementation for the **Cyclone Preparedness Assistant** using Weights & Biases (W&B). All analytical formulations, feature matrices, and evaluation metrics from Phase 2 were strictly preserved and synchronized to the project workspace.

### Key Tracking Achievements:
1. **Unsupervised Vulnerability Clustering Exploration (`clustering-exploration`)**:
   - Logged **10 candidate model runs** across $k \in \{2, 3, 4, 5, 6\}$ for both **K-Means** (Euclidean, $n_{\text{init}}=20$) and **Agglomerative Clustering** (Ward linkage).
   - Logged **1 comprehensive benchmark summary run** with interactive `wandb.Table` artifacts capturing:
     - Full multi-algorithm comparison grid (Silhouette, Davies–Bouldin, Calinski–Harabasz, Inertia, Balance ratio)
     - Inter-algorithm partition agreement (Adjusted Rand Index and Adjusted Mutual Information across $k$)
     - Multi-seed stability for the selected $k=4$ baseline across 10 distinct random seeds (Mean $\text{ARI} = 0.9902$)
     - Discovered archetype profile centroids and unit counts
2. **Prioritization Sensitivity & Rank Stability Analysis (`prioritization-sensitivity`)**:
   - Logged **5 distinct policy weighting scenarios** evaluated on all 622 administrative units.
   - Logged **1 sensitivity summary run** with interactive `wandb.Table` artifacts capturing:
     - The $5 \times 5$ pairwise Spearman rank correlation matrix ($r_s$)
     - The 14 robust high-priority communities consistently appearing in the top-100 across all 5 weighting paradigms.
3. **Engineering Rigor & Security**:
   - Zero API key leakage (authenticated via local `.netrc` without exposing keys in code, git, or command history).
   - Strict CLI control (`--wandb-mode online|offline|disabled`) preventing accidental run duplication during routine testing.
   - Comprehensive test suite with mocked W&B calls (**87 tests passing**) ensuring 100% offline verification.

---

## 2. Experiment Group 1: Unsupervised Vulnerability Clustering (`clustering-exploration`)

- **Group**: `clustering-exploration`
- **Job Type**: `model-evaluation` & `benchmark-summary`
- **Input Features**: 4 standardized z-score transformed dimension scores:
  1. $D_{\text{health}}$: Healthcare Access Deficit
  2. $D_{\text{demo}}$: Demographic Vulnerability (SC/ST Socioeconomic Disadvantage)
  3. $D_{\text{infra}}$: Infrastructure Deficit (Water, Road, Drainage Deficits)
  4. $D_{\text{coast}}$: Coastal Hazard Proximity (Inverted Distance to Coastline)

### Table 1: Synchronized Clustering Experiment Runs
| Run Name | Algorithm | $k$ | Silhouette | Davies–Bouldin | Calinski–Harabasz | Balance Ratio | Selected? | W&B Run URL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `kmeans-k2` | K-Means | 2 | 0.3004 | 1.3453 | 297.13 | 0.7182 | No | [Run 2yfmhtfi](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/2yfmhtfi) |
| `agglomerative-k2` | Agglomerative | 2 | 0.2673 | 1.4689 | 238.61 | 0.6631 | No | [Run f54cs4ei](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/f54cs4ei) |
| `kmeans-k3` | K-Means | 3 | 0.3360 | 1.1134 | 311.30 | 0.4564 | No | [Run 4an56mbk](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/4an56mbk) |
| `agglomerative-k3` | Agglomerative | 3 | 0.2986 | 1.1482 | 260.70 | 0.5968 | No | [Run l5dhfk14](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/l5dhfk14) |
| **`kmeans-k4-baseline`** | **K-Means** | **4** | **0.3348** | **1.1771** | **298.59** | **0.3293** | **Selected Baseline** | [Run fdiiep48](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/fdiiep48) |
| `agglomerative-k4` | Agglomerative | 4 | 0.3045 | 1.1904 | 250.41 | 0.2832 | No | [Run rhwgaf8e](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/rhwgaf8e) |
| `kmeans-k5` | K-Means | 5 | 0.2935 | 1.1889 | 273.93 | 0.3707 | No | [Run k0nqd1ms](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/k0nqd1ms) |
| `agglomerative-k5` | Agglomerative | 5 | 0.3107 | 1.2542 | 234.86 | 0.2832 | No | [Run x6rz7gi4](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/x6rz7gi4) |
| `kmeans-k6` | K-Means | 6 | 0.2797 | 1.1611 | 258.10 | 0.4375 | No | [Run rnnyzh2i](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/rnnyzh2i) |
| `agglomerative-k6` | Agglomerative | 6 | 0.2791 | 1.2449 | 225.37 | 0.2832 | No | [Run 4irpcg9t](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/4irpcg9t) |
| **`clustering-benchmark-summary`** | **Summary / Tables** | - | - | - | - | - | **Artifact Hub** | [Summary Run roue0z51](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/roue0z51) |

---

## 3. Experiment Group 2: Prioritization Sensitivity & Rank Stability (`prioritization-sensitivity`)

- **Group**: `prioritization-sensitivity`
- **Job Type**: `scenario-evaluation` & `sensitivity-summary`
- **Unit Universe**: 622 verified administrative units (563 rural revenue villages, 59 Census towns).

### Table 2: Synchronized Prioritization Sensitivity Runs
| Run Name | Weights ($w_h, w_d, w_i, w_c$) | Pop Scale | Score Min | Score Mean | Score Max | Spearman $r_s$ vs. Baseline | Max Rank Shift | W&B Run URL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`prioritization-equal-weights`** | $(0.25, 0.25, 0.25, 0.25)$ | False | 0.1691 | 0.4623 | 0.7018 | 1.000 | 0 | [Run krmjba30](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/krmjba30) |
| **`prioritization-hazard-focused`** | $(0.15, 0.15, 0.20, 0.50)$ | False | 0.2077 | 0.5477 | 0.7853 | 0.704 | 499 | [Run su9ccalx](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/su9ccalx) |
| **`prioritization-access-focused`** | $(0.40, 0.15, 0.35, 0.10)$ | False | 0.1617 | 0.4299 | 0.7467 | 0.829 | 431 | [Run 371o6i3k](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/371o6i3k) |
| **`prioritization-no-coastal`** | $(0.333, 0.333, 0.334, 0.000)$ | False | 0.0755 | 0.3843 | 0.7364 | 0.656 | 430 | [Run o60jfq0a](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/o60jfq0a) |
| **`prioritization-with-population-scale`** | $(0.25, 0.25, 0.25, 0.25)$ | True | 0.2484 | 0.4619 | 0.6674 | 0.980 | 209 | [Run ncz256qp](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/ncz256qp) |
| **`prioritization-sensitivity-summary`** | **Summary / Tables** | - | - | - | - | - | - | [Summary Run emvjpzbp](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/emvjpzbp) |

### Key Policy Insight Recorded in W&B:
The rank correlation between `hazard_focused` (50% coastal weight) and `no_coastal` (0% coastal weight) is **$-0.025$** (virtually zero). This confirms that prioritizing acute storm-surge evacuation along the coast versus reinforcing remote interior healthcare clinics in the Western Ghats foothills represents two orthogonal policy goals that cannot be conflated into an uncalibrated composite score.

---

## 4. How to Reproduce Experiment Synchronization

The experiment tracking pipeline provides modular CLI commands supporting `online`, `offline`, and `disabled` modes:

### 1. Unified Synchronization (All Experiments)
```bash
# Online cloud synchronization (when authenticated)
python scripts/run_wandb_tracking.py --mode online

# Offline local recording (stored in ./wandb/offline-run-* for later sync)
python scripts/run_wandb_tracking.py --mode offline
```

### 2. Individual Pipeline Scripts
```bash
# Run clustering experiments with online tracking
python scripts/run_clustering_experiments.py --wandb-mode online

# Run prioritization evaluation with online tracking
python scripts/evaluate_prioritization_baseline.py --wandb-mode online
```

### 3. Automated Test Suite (100% Offline with Mocks)
```bash
# Execute entire test suite (87 tests, including W&B tracking unit tests)
pytest -v
```

---

## 5. Security & Verification Guardrails

1. **API Key Security**: The API key is stored securely in `~/.netrc` with standard `0600` permissions and was never printed to terminal logs, written to source files, or added to Git staging.
2. **Offline Fallback Guarantee**: If network drops or credentials expire, scripts automatically fall back to saving results to local JSON (`reports/clustering_experiments_manifest.json` and `reports/sensitivity_analysis_summary.json`) without interrupting execution.
3. **No MLflow / No Silent Mutations**: All raw datasets in `data/raw/` remain untouched.
