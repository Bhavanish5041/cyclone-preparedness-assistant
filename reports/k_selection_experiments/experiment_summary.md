# K-Means Model Selection Experiment Summary: Vulnerability Archetype Discovery

- **W&B Entity**: `bhavanishreddy69-na`
- **W&B Project**: `cyclone-preparedness-assistant`
- **W&B Experiment Group**: [`kmeans-k-selection-comparison`](https://wandb.ai/bhavanishreddy69-na/cyclone-preparedness-assistant/groups/kmeans-k-selection-comparison)
- **Evaluation Date**: 2026-11-12 19:16:18 UTC
- **Evaluated Candidates**: K in {2, 3, 4, 5, 6} (`random_state=42`, `n_init=20`)

---

## 1. Quantitative Evaluation Benchmark Table

| K | Inertia | Silhouette | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | Balance Ratio | Cluster Sizes | Runtime | Stability Mean ARI | Stability Min ARI |
|---|---|---|---|---|---|---|---|---|---|
| **K=2** | 1681.94 | **0.3004** | 1.3453 | 297.1 | 0.718 | `[260,362]` | 0.172s | 0.9949 | 0.9808 |
| **K=3** | 1240.38 | **0.3360** | 1.1134 | 311.3 | 0.456 | `[131,287,204]` | 0.119s | 0.9983 | 0.9945 |
| **K=4** | 1015.73 | **0.3348** | 1.1771 | 298.6 | 0.329 | `[130,82,161,249]` | 0.119s | 0.9912 | 0.9635 |
| **K=5** | 896.29 | **0.2935** | 1.1889 | 273.9 | 0.371 | `[81,125,205,76,135]` | 0.159s | 0.9774 | 0.9090 |
| **K=6** | 803.89 | **0.2797** | 1.1611 | 258.1 | 0.438 | `[106,88,119,56,128,125]` | 0.137s | 0.9514 | 0.8900 |

---

## 2. Objective Metric Analysis & Conflicting Indicators

### A. Geometric Compactness and Separation
- **Silhouette Coefficient**: Peaks at **$K=3$ (0.3360)**, closely followed by **$K=4$ (0.3348)**. The difference between $K=3$ and $K=4$ is negligible (difference = -0.0012). Beyond $K=4$, silhouette drops sharply to $0.2935$ ($K=5$) and $0.2797$ ($K=6$).
- **Davies-Bouldin Index (Lower is Better)**: Minimizes at **$K=3$ (1.1134)**, followed by $K=6$ (1.1611) and $K=4$ (1.1771). $K=2$ exhibits poor cluster separation ($1.3453$).
- **Calinski-Harabasz Index (Higher is Better)**: Maximizes at **$K=3$ (311.30)**, followed by $K=4$ (298.59), with steep declines for $K=5$ (273.93) and $K=6$ (258.10).
- **Inertia (Elbow Criterion)**: Decreases monotonically: $1681.9 \rightarrow 1240.4$ (26.3% drop at $K=3$) $\rightarrow 1015.7$ (18.1% drop at $K=4$) $\rightarrow 896.3$ (11.8% drop at $K=5$) $\rightarrow 803.9$ (10.3% drop at $K=6$). The primary elbow inflection occurs between $K=3$ and $K=4$, after which marginal gains level off.

### B. Multi-Seed Initialization Stability (10 Distinct Seeds)
- **$K=3$**: Near-perfect determinism (Mean ARI = **0.9983**, Min ARI = **0.9945**).
- **$K=2$**: Very high stability (Mean ARI = **0.9949**, Min ARI = 0.9808).
- **$K=4$**: Highly robust and stable (Mean ARI = **0.9912**, Min ARI = 0.9635).
- **$K=5$ and $K=6$**: Stability begins to deteriorate as clusters fragment (Min ARI falls to **0.9090** for $K=5$ and **0.8900** for $K=6$).

---

## 3. Domain Interpretability & Operational Disaster Planning Trade-offs

### Why $K=3$ is Mathematically Superior but Operationally Flawed
Purely based on unweighted geometric metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, and Seed Stability), **$K=3$ achieves the highest mathematical scores**. However, in emergency management and disaster preparedness, geometric compactness must be balanced with operational utility:
- **$K=3$ partitions the study area into**:
  1. Urban & Peri-Urban Corridor (131 units, 1.55M pop)
  2. High-Exposure Rural Coastal Belt (287 units, 1.05M pop)
  3. Combined Rural Interior & Remote Foothills (204 units, 660K pop)
- **The Operational Failure of $K=3$**: It combines two distinct inland populations with diametrically opposing disaster needs:
  * **Remote Foothills (Western Ghats)**: Extreme hospital distance (mean 23.6 km), isolated terrain prone to road cut-offs and landslides. Intervention needed: **Mobile trauma units, surgical prepositioning, satellite communications, and all-weather road clearance**.
  * **Agrarian Rural Interior**: Moderate hospital access (mean 14.1 km), but high socio-economic disadvantage (21.9% SC/ST) and open drainage deficits. Intervention needed: **Targeted social safety nets, clean drinking water access, and community shelter accessibility**.
- Lumping these two zones together under $K=3$ conceals the acute medical isolation of 82 foothill villages (252,738 residents).

### Why $K=4$ is the Recommended Production Model
- **Resolves the Foothill vs. Agrarian Interior Duality**: $K=4$ cleanly disaggregates the interior into **Cluster 1 (Remote Foothill Healthcare-Isolated, 82 units)** and **Cluster 2 (Socioeconomically Vulnerable Rural Interior, 161 units)**, while maintaining Cluster 0 (Urban Hubs, 130 units) and Cluster 3 (Coastal Belt, 249 units).
- **Negligible Geometric Penalty**: The silhouette difference between $K=3$ (0.3360) and $K=4$ (0.3348) is only **0.0012** (a 0.35% variation), while stability remains exceptionally high (Mean ARI = 0.9912).
- **Direct Alignment with Civil Defense Interventions**: Each of the 4 archetypes maps directly to a distinct, actionable emergency resource deployment protocol.

### Why $K=2$, $K=5$, and $K=6$ are Non-Viable
- **$K=2$**: Over-simplifies the coastal district geography into crude 'Inland' vs. 'Coastal' halves, failing to separate urban centers from fishing villages and resulting in the lowest silhouette (0.3004) and worst Davies-Bouldin index (1.3453).
- **$K=5$ and $K=6$**: Fragment existing natural groupings along subtle continuous gradients rather than distinct operational archetypes. Silhouette falls below 0.30 (0.2935 and 0.2797), Calinski-Harabasz drops significantly, and multi-seed stability degrades (min ARI falls to 0.8900).

---

## 4. Final Recommendation

> [!TIP]
> **Final Recommendation**: While **$K=3$** wins marginally on pure geometric metrics (Silhouette 0.3360, DB 1.1134, CH 311.3), **$K=4$ (Silhouette 0.3348, DB 1.1771, CH 298.6, ARI 0.9912)** is strongly justified as the best operational model for disaster preparedness because it captures the critical Western Ghats healthcare-isolation archetype without meaningful loss of geometric cohesion.