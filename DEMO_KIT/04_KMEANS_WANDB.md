# 04. K-Means Vulnerability Archetyping & W&B Tracking

## 1. Descriptive Archetyping vs. Disaster Prediction

> [!IMPORTANT]
> **Core Scientific Demarcation:**  
> K-Means clustering is implemented strictly as an **unsupervised descriptive vulnerability archetyping tool**. It discovers empirical similarity clusters across four normalized spatial indicator dimensions. It does **NOT**:
> - Predict cyclone landfall locations or storm surge heights.
> - Forecast physical structural damages or building collapse rates.
> - Predict casualty figures or evacuation compliance rates.
>
> Clusters serve as operational planning archetypes to help emergency planners design specialized regional intervention packages.

---

## 2. Weights & Biases (W&B) Experiment Workspace

All unsupervised clustering runs, hyperparameter sweeps, and multidimensional sensitivity experiments are systematically tracked and versioned in Weights & Biases.

- **W&B Entity:** `bhavanishreddy69-na`
- **W&B Project:** `cyclone-preparedness-assistant`
- **W&B Cloud Workspace:** [https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant)
- **Synchronized Artifact Total:** 17 Verified Cloud Runs

---

## 3. Experiment Group 1: Clustering Exploration (`clustering-exploration`)

### 3.1 Input Feature Space
Clustering operates across 622 administrative revenue units in Dakshina Kannada (369) and Udupi (253) on 4 standardized dimensions:
1. $D_{\text{health}}$: Healthcare Access Deficit ($z$-score)
2. $D_{\text{demo}}$: Demographic Vulnerability ($z$-score)
3. $D_{\text{infra}}$: Infrastructure Deficit ($z$-score)
4. $D_{\text{coast}}$: Coastal Hazard Proximity ($z$-score)

### 3.2 Synchronized Clustering Runs & Evaluation Metrics
Both **K-Means** ($n_{\text{init}}=20$) and **Agglomerative Clustering** (Ward linkage) were benchmarked across $k \in \{2, 3, 4, 5, 6\}$:

| Run Name | Algorithm | $k$ | Silhouette | Davies–Bouldin | Calinski–Harabasz | Balance Ratio | Selected Status | W&B Cloud Run URL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `kmeans-k2` | K-Means | 2 | 0.3004 | 1.3453 | 297.13 | 0.7182 | Candidate | [Run 2yfmhtfi](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/2yfmhtfi) |
| `agglomerative-k2` | Agglomerative | 2 | 0.2673 | 1.4689 | 238.61 | 0.6631 | Candidate | [Run f54cs4ei](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/f54cs4ei) |
| `kmeans-k3` | K-Means | 3 | 0.3360 | 1.1134 | 311.30 | 0.4564 | Candidate | [Run 4an56mbk](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/4an56mbk) |
| `agglomerative-k3` | Agglomerative | 3 | 0.2986 | 1.1482 | 260.70 | 0.5968 | Candidate | [Run l5dhfk14](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/l5dhfk14) |
| **`kmeans-k4-baseline`** | **K-Means** | **4** | **0.3348** | **1.1771** | **298.59** | **0.3293** | **Selected Production Baseline** | [Run fdiiep48](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/fdiiep48) |
| `agglomerative-k4` | Agglomerative | 4 | 0.3045 | 1.1904 | 250.41 | 0.2832 | Candidate | [Run rhwgaf8e](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/rhwgaf8e) |
| `kmeans-k5` | K-Means | 5 | 0.2935 | 1.1889 | 273.93 | 0.3707 | Candidate | [Run k0nqd1ms](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/k0nqd1ms) |
| `agglomerative-k5` | Agglomerative | 5 | 0.3107 | 1.2542 | 234.86 | 0.2832 | Candidate | [Run x6rz7gi4](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/x6rz7gi4) |
| `kmeans-k6` | K-Means | 6 | 0.2797 | 1.1611 | 258.10 | 0.4375 | Candidate | [Run rnnyzh2i](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/rnnyzh2i) |
| `agglomerative-k6` | Agglomerative | 6 | 0.2791 | 1.2449 | 225.37 | 0.2832 | Candidate | [Run 4irpcg9t](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/4irpcg9t) |
| **`clustering-benchmark-summary`** | **Interactive Summary Table** | — | — | — | — | — | **Artifact Hub** | [Summary Run roue0z51](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/roue0z51) |

---

## 4. Experiment Group 2: Prioritization Sensitivity (`prioritization-sensitivity`)

To verify whether policy weighting changes induce rank volatility, 5 distinct policy weighting scenarios were logged to W&B across all 622 units:

| Scenario Name | Weights ($w_h, w_d, w_i, w_c$) | Pop Scale | Min Score | Mean Score | Max Score | Spearman $r_s$ vs. Baseline | Max Shift | W&B Cloud Run URL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`prioritization-equal-weights`** | $(0.25, 0.25, 0.25, 0.25)$ | False | 0.1691 | 0.4623 | 0.7018 | 1.000 | 0 | [Run krmjba30](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/krmjba30) |
| **`prioritization-hazard-focused`** | $(0.15, 0.15, 0.20, 0.50)$ | False | 0.2077 | 0.5477 | 0.7853 | 0.704 | 499 | [Run su9ccalx](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/su9ccalx) |
| **`prioritization-access-focused`** | $(0.40, 0.15, 0.35, 0.10)$ | False | 0.1617 | 0.4299 | 0.7467 | 0.829 | 431 | [Run 371o6i3k](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/371o6i3k) |
| **`prioritization-no-coastal`** | $(0.333, 0.333, 0.334, 0.000)$ | False | 0.0755 | 0.3843 | 0.7364 | 0.656 | 430 | [Run o60jfq0a](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/o60jfq0a) |
| **`prioritization-with-population-scale`** | $(0.25, 0.25, 0.25, 0.25)$ | True | 0.2484 | 0.4619 | 0.6674 | 0.980 | 209 | [Run ncz256qp](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/ncz256qp) |
| **`prioritization-sensitivity-summary`** | **Interactive Summary Table** | — | — | — | — | — | — | [Summary Run emvjpzbp](https://forge.coreweave.com/wandb/bhavanishreddy69-na/cyclone-preparedness-assistant/runs/emvjpzbp) |

### Key Policy Insight Recorded in W&B:
The Spearman rank correlation between `hazard_focused` (50% coastal hazard weight) and `no_coastal` (0% coastal hazard weight) is **$-0.025$** (virtually orthogonal).  
*Significance:* Evacuating immediate coastal storm-surge communities vs. reinforcing remote inland healthcare clinics in the Western Ghats foothills represents two entirely distinct operational priorities. They must be managed as separate intervention tracks rather than blended into an uncalibrated composite rank.

---

## 5. Registered Model Artifacts in W&B

The supervised classifier is registered as a versioned artifact in the W&B Model Registry:

- **Model Identifier:** `village-preparedness-classifier`
- **Active Production Tag:** `v1.0.0` (or `v0`)
- **Artifact Address:** `bhavanishreddy69-na/cyclone-preparedness-assistant/village-preparedness-classifier:v0`
- **Offline Cached Directory:** `.wandb_artifacts_cache/bhavanishreddy69-na_cyclone-preparedness-assistant_village-preparedness-classifier_v0`
- **Cryptographic File Hash:** `logistic_regression_tier.joblib` (`sha256: c7a4a48beeb326b328c59115b331542038cba3323a2c08e0eceba7f67888e67e`)

---

## 6. Offline Experiment Reproduction

The tracking framework seamlessly falls back to offline local storage when network access is absent:

```bash
# 1. Run full W&B sync in online cloud mode (requires active authentication)
python scripts/run_wandb_tracking.py --mode online

# 2. Run offline local logging (records to ./wandb/offline-run-* for later sync)
python scripts/run_wandb_tracking.py --mode offline

# 3. Completely disable W&B tracking during rapid development
python scripts/run_wandb_tracking.py --mode disabled
```
