# Model Card: Coastal Vulnerability Archetypes (`kmeans-candidate-k6`)

## Model Details
- **Model Identifier**: `kmeans-candidate-k6`
- **Model Family**: scikit-learn Unsupervised Clustering
- **Algorithm**: K-Means ($K=6$, `random_state=42`, `n_init=20`)
- **Framework**: scikit-learn
- **Preprocessing**: `StandardScaler` (zero mean, unit variance)
- **Canonical Features**: 4 Standardized Dimension Scores
- **Status**: Experimental Model Selection Candidate

## Intended Use & Non-Causal Planning Archetype Disclaimer
> [!IMPORTANT]
> DISCLAIMER: Discovered archetypes represent empirical similarity groupings derived from four normalized spatial indicators. They are NOT predetermined hazard categories, validated disaster risk classes, or casualty predictions. They serve as descriptive planning archetypes to identify communities with shared intervention requirements.

## Quantitative Evaluation Metrics (Primary Seed=42)
- **Inertia**: 803.89
- **Silhouette Score**: 0.2797
- **Davies-Bouldin Index**: 1.1611 (lower is better)
- **Calinski-Harabasz Index**: 258.1 (higher is better)
- **Cluster Balance Ratio**: 0.4375
- **Min Cluster Size**: 56
- **Max Cluster Size**: 128
- **Fit & Eval Runtime**: 0.1372s

## Multi-Seed Stability (10 Random Seeds)
- **Mean ARI vs. Seed 42**: 0.9514
- **Min ARI vs. Seed 42**: 0.89
- **Std Dev ARI**: 0.0332
- **Mean AMI vs. Seed 42**: 0.9456

## Discovered Archetype Breakdown
### Cluster 0: Coastal Belt with Elevated Healthcare Isolation
- **Revenue Units**: 106 (17.0%)
- **Population**: 366,935 (11.3% of total study population)
- **District Breakdown**: 41.5% Dakshina Kannada | 58.5% Udupi
- **Healthcare Deficit Mean**: 0.4031
- **Demographic Vulnerability Mean**: 0.208
- **Infrastructure Deficit Mean**: 0.6424
- **Coastal Exposure Mean**: 0.8346
- **Empirical Description**: Coastal settlements with high marine exposure (0.8346) and high infrastructure deficit (0.6424), with elevated distance to hospitals (mean 19.0 km, deficit 0.4031). Predominantly Udupi (58.5%).
- **Primary Intervention Need**: Coastal emergency clinics, cyclone shelters with medical triage, and road clearance.

### Cluster 1: Extreme Demographic Vulnerability Interior
- **Revenue Units**: 88 (14.1%)
- **Population**: 276,332 (8.5% of total study population)
- **District Breakdown**: 73.9% Dakshina Kannada | 26.1% Udupi
- **Healthcare Deficit Mean**: 0.3651
- **Demographic Vulnerability Mean**: 0.613
- **Infrastructure Deficit Mean**: 0.6153
- **Coastal Exposure Mean**: 0.5024
- **Empirical Description**: Interior communities with peak demographic vulnerability (mean 0.6130, SC/ST disadvantage) and high infrastructure deficit (0.6153). Predominantly Dakshina Kannada (73.9%).
- **Primary Intervention Need**: Targeted social safety nets, clean drinking water distribution, and accessible community shelters.

### Cluster 2: Urban & Peri-Urban Well-Resourced Corridor
- **Revenue Units**: 119 (19.1%)
- **Population**: 1,517,463 (46.6% of total study population)
- **District Breakdown**: 74.8% Dakshina Kannada | 25.2% Udupi
- **Healthcare Deficit Mean**: 0.2469
- **Demographic Vulnerability Mean**: 0.1984
- **Infrastructure Deficit Mean**: 0.1681
- **Coastal Exposure Mean**: 0.8302
- **Empirical Description**: Primary urban corridor with low infrastructure deficit (0.1681), low healthcare deficit (0.2469), and the largest population share (1.52M). Strong hospital and infrastructure baseline.
- **Primary Intervention Need**: Stormwater pump maintenance, power grid backup, and hospital surge readiness.

### Cluster 3: Deep Foothill Healthcare-Isolated Frontier
- **Revenue Units**: 56 (9.0%)
- **Population**: 169,800 (5.2% of total study population)
- **District Breakdown**: 73.2% Dakshina Kannada | 26.8% Udupi
- **Healthcare Deficit Mean**: 0.5984
- **Demographic Vulnerability Mean**: 0.3439
- **Infrastructure Deficit Mean**: 0.5779
- **Coastal Exposure Mean**: 0.2786
- **Empirical Description**: Most remote Western Ghats foothill settlements with maximum hospital distance (mean 25.2 km, deficit 0.5984) and lowest coastal exposure (mean 51.8 km to coast).
- **Primary Intervention Need**: Pre-positioned trauma supplies, satellite communication links, and emergency evacuation access.

### Cluster 4: Transitional Interior Agrarian Belt
- **Revenue Units**: 128 (20.6%)
- **Population**: 465,928 (14.3% of total study population)
- **District Breakdown**: 77.3% Dakshina Kannada | 22.7% Udupi
- **Healthcare Deficit Mean**: 0.2704
- **Demographic Vulnerability Mean**: 0.3402
- **Infrastructure Deficit Mean**: 0.6127
- **Coastal Exposure Mean**: 0.5601
- **Empirical Description**: Intermediate interior zone with moderate demographic vulnerability (0.3402) and intermediate coastal distance (31.6 km).
- **Primary Intervention Need**: Agricultural drainage clearing, drinking water purification, and community shelter upgrades.

### Cluster 5: Immediate Shoreline Fishing Communities
- **Revenue Units**: 125 (20.1%)
- **Population**: 458,399 (14.1% of total study population)
- **District Breakdown**: 24.8% Dakshina Kannada | 75.2% Udupi
- **Healthcare Deficit Mean**: 0.2201
- **Demographic Vulnerability Mean**: 0.1768
- **Infrastructure Deficit Mean**: 0.6381
- **Coastal Exposure Mean**: 0.9134
- **Empirical Description**: Ultra-close shoreline settlements (mean 6.2 km to coast, coastal exposure 0.9134) with moderate healthcare access (10.2 km to hospital) but open drainage and well water reliance.
- **Primary Intervention Need**: Early warning sirens, sea-wall maintenance, fishing vessel harbor protection, and saline barrier reinforcement.

## Operational Limitations
1. **Descriptive Empirical Clusters**: Clusters reflect statistical spatial groupings, not validated casualty forecasts.
2. **AHP Precedence**: Clustering does not re-rank authoritative multi-criteria prioritization.
3. **Local Verification**: Rapid field assessment is required prior to prepositioning assets.
