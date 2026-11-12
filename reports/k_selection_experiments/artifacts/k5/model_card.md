# Model Card: Coastal Vulnerability Archetypes (`kmeans-candidate-k5`)

## Model Details
- **Model Identifier**: `kmeans-candidate-k5`
- **Model Family**: scikit-learn Unsupervised Clustering
- **Algorithm**: K-Means ($K=5$, `random_state=42`, `n_init=20`)
- **Framework**: scikit-learn
- **Preprocessing**: `StandardScaler` (zero mean, unit variance)
- **Canonical Features**: 4 Standardized Dimension Scores
- **Status**: Experimental Model Selection Candidate

## Intended Use & Non-Causal Planning Archetype Disclaimer
> [!IMPORTANT]
> DISCLAIMER: Discovered archetypes represent empirical similarity groupings derived from four normalized spatial indicators. They are NOT predetermined hazard categories, validated disaster risk classes, or casualty predictions. They serve as descriptive planning archetypes to identify communities with shared intervention requirements.

## Quantitative Evaluation Metrics (Primary Seed=42)
- **Inertia**: 896.29
- **Silhouette Score**: 0.2935
- **Davies-Bouldin Index**: 1.1889 (lower is better)
- **Calinski-Harabasz Index**: 273.93 (higher is better)
- **Cluster Balance Ratio**: 0.3707
- **Min Cluster Size**: 76
- **Max Cluster Size**: 205
- **Fit & Eval Runtime**: 0.159s

## Multi-Seed Stability (10 Random Seeds)
- **Mean ARI vs. Seed 42**: 0.9774
- **Min ARI vs. Seed 42**: 0.909
- **Std Dev ARI**: 0.024
- **Mean AMI vs. Seed 42**: 0.9673

## Discovered Archetype Breakdown
### Cluster 0: Extreme Demographic Vulnerability Interior
- **Revenue Units**: 81 (13.0%)
- **Population**: 256,421 (7.9% of total study population)
- **District Breakdown**: 77.8% Dakshina Kannada | 22.2% Udupi
- **Healthcare Deficit Mean**: 0.3527
- **Demographic Vulnerability Mean**: 0.6213
- **Infrastructure Deficit Mean**: 0.6195
- **Coastal Exposure Mean**: 0.4891
- **Empirical Description**: Agrarian interior communities with the highest demographic disadvantage (SC/ST vulnerability mean 0.6213) and high infrastructure deficit (0.6195). Predominantly Dakshina Kannada (77.8%).
- **Primary Intervention Need**: Targeted social safety nets, clean drinking water access, and community-based shelter facilities.

### Cluster 1: Urban & Peri-Urban Well-Resourced Corridor
- **Revenue Units**: 125 (20.1%)
- **Population**: 1,532,742 (47.1% of total study population)
- **District Breakdown**: 75.2% Dakshina Kannada | 24.8% Udupi
- **Healthcare Deficit Mean**: 0.2517
- **Demographic Vulnerability Mean**: 0.1962
- **Infrastructure Deficit Mean**: 0.1776
- **Coastal Exposure Mean**: 0.8238
- **Empirical Description**: Urban and peri-urban settlements with low infrastructure deficit (0.1776) and high population density (1.53M population). Concentrated around Mangalore City and taluk hubs.
- **Primary Intervention Need**: Urban flood drainage, hospital surge preparedness, and electrical substation protection.

### Cluster 2: Immediate Coastal Shoreline Belt
- **Revenue Units**: 205 (33.0%)
- **Population**: 738,542 (22.7% of total study population)
- **District Breakdown**: 30.7% Dakshina Kannada | 69.3% Udupi
- **Healthcare Deficit Mean**: 0.2902
- **Demographic Vulnerability Mean**: 0.1801
- **Infrastructure Deficit Mean**: 0.6446
- **Coastal Exposure Mean**: 0.8941
- **Empirical Description**: Communities closest to the Arabian Sea (mean 7.6 km to coast, coastal exposure 0.8941) with high infrastructure deficit (0.6446). Heavily concentrated in Udupi (69.3%).
- **Primary Intervention Need**: Marine storm-surge barriers, cyclone evacuation shelters, and saline intrusion defenses.

### Cluster 3: Remote Foothill & Healthcare-Isolated Communities
- **Revenue Units**: 76 (12.2%)
- **Population**: 228,775 (7.0% of total study population)
- **District Breakdown**: 59.2% Dakshina Kannada | 40.8% Udupi
- **Healthcare Deficit Mean**: 0.5768
- **Demographic Vulnerability Mean**: 0.3487
- **Infrastructure Deficit Mean**: 0.591
- **Coastal Exposure Mean**: 0.3683
- **Empirical Description**: Deep Western Ghats foothill communities with severe healthcare distance (mean 24.2 km to hospital, healthcare deficit 0.5768) and low coastal exposure (mean 45.4 km to coast). Sheltered from storm surge but vulnerable to isolation.
- **Primary Intervention Need**: Mobile medical units, pre-positioned trauma supplies, all-weather road maintenance, and satellite communication.

### Cluster 4: Transitional Interior Agrarian Belt
- **Revenue Units**: 135 (21.7%)
- **Population**: 498,377 (15.3% of total study population)
- **District Breakdown**: 77.0% Dakshina Kannada | 23.0% Udupi
- **Healthcare Deficit Mean**: 0.2706
- **Demographic Vulnerability Mean**: 0.3424
- **Infrastructure Deficit Mean**: 0.6174
- **Coastal Exposure Mean**: 0.5857
- **Empirical Description**: Sub-coastal agrarian interior communities with moderate demographic vulnerability (0.3424), intermediate coastal exposure (0.5857), and high infrastructure deficit (0.6174).
- **Primary Intervention Need**: Drainage canal de-silting, emergency drinking water purification, and all-weather road access.

## Operational Limitations
1. **Descriptive Empirical Clusters**: Clusters reflect statistical spatial groupings, not validated casualty forecasts.
2. **AHP Precedence**: Clustering does not re-rank authoritative multi-criteria prioritization.
3. **Local Verification**: Rapid field assessment is required prior to prepositioning assets.
