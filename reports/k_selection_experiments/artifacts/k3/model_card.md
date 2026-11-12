# Model Card: Coastal Vulnerability Archetypes (`kmeans-candidate-k3`)

## Model Details
- **Model Identifier**: `kmeans-candidate-k3`
- **Model Family**: scikit-learn Unsupervised Clustering
- **Algorithm**: K-Means ($K=3$, `random_state=42`, `n_init=20`)
- **Framework**: scikit-learn
- **Preprocessing**: `StandardScaler` (zero mean, unit variance)
- **Canonical Features**: 4 Standardized Dimension Scores
- **Status**: Experimental Model Selection Candidate

## Intended Use & Non-Causal Planning Archetype Disclaimer
> [!IMPORTANT]
> DISCLAIMER: Discovered archetypes represent empirical similarity groupings derived from four normalized spatial indicators. They are NOT predetermined hazard categories, validated disaster risk classes, or casualty predictions. They serve as descriptive planning archetypes to identify communities with shared intervention requirements.

## Quantitative Evaluation Metrics (Primary Seed=42)
- **Inertia**: 1240.38
- **Silhouette Score**: 0.336
- **Davies-Bouldin Index**: 1.1134 (lower is better)
- **Calinski-Harabasz Index**: 311.3 (higher is better)
- **Cluster Balance Ratio**: 0.4564
- **Min Cluster Size**: 131
- **Max Cluster Size**: 287
- **Fit & Eval Runtime**: 0.1192s

## Multi-Seed Stability (10 Random Seeds)
- **Mean ARI vs. Seed 42**: 0.9983
- **Min ARI vs. Seed 42**: 0.9945
- **Std Dev ARI**: 0.0025
- **Mean AMI vs. Seed 42**: 0.997

## Discovered Archetype Breakdown
### Cluster 0: Urban & Peri-Urban Well-Resourced Corridor
- **Revenue Units**: 131 (21.1%)
- **Population**: 1,549,683 (47.6% of total study population)
- **District Breakdown**: 76.3% Dakshina Kannada | 23.7% Udupi
- **Healthcare Deficit Mean**: 0.252
- **Demographic Vulnerability Mean**: 0.1994
- **Infrastructure Deficit Mean**: 0.1878
- **Coastal Exposure Mean**: 0.8149
- **Empirical Description**: Well-resourced urban and peri-urban corridor with low healthcare deficit, low demographic vulnerability, and the lowest infrastructure deficit (closed drainage, piped water). High population concentration (1.55M people, 76.3% DK).
- **Primary Intervention Need**: Stormwater pump maintenance, urban grid power resilience, and hospital surge capacity.

### Cluster 1: High-Exposure Rural Coastal Belt
- **Revenue Units**: 287 (46.1%)
- **Population**: 1,045,240 (32.1% of total study population)
- **District Breakdown**: 39.7% Dakshina Kannada | 60.3% Udupi
- **Healthcare Deficit Mean**: 0.282
- **Demographic Vulnerability Mean**: 0.224
- **Infrastructure Deficit Mean**: 0.6436
- **Coastal Exposure Mean**: 0.8287
- **Empirical Description**: High-exposure coastal settlements with heavy infrastructure deficits (untreated water, open drains prone to saline flooding) and proximity to the Arabian Sea (mean 12.3 km to coast). Predominantly Udupi (60.3%).
- **Primary Intervention Need**: Cyclone evacuation shelters, saline water intrusion barriers, and coastal embankment reinforcement.

### Cluster 2: Combined Rural Interior & Remote Foothills
- **Revenue Units**: 204 (32.8%)
- **Population**: 659,934 (20.3% of total study population)
- **District Breakdown**: 76.0% Dakshina Kannada | 24.0% Udupi
- **Healthcare Deficit Mean**: 0.4212
- **Demographic Vulnerability Mean**: 0.4613
- **Infrastructure Deficit Mean**: 0.6052
- **Coastal Exposure Mean**: 0.4331
- **Empirical Description**: Agrarian interior and foothill communities combining elevated healthcare deficit (mean 17.2 km to hospital) and high demographic vulnerability (mean 46.1% score). Predominantly Dakshina Kannada (76.0%).
- **Primary Intervention Need**: Rural road connectivity, decentralized primary health clinics, and targeted socio-economic safety nets.

## Operational Limitations
1. **Descriptive Empirical Clusters**: Clusters reflect statistical spatial groupings, not validated casualty forecasts.
2. **AHP Precedence**: Clustering does not re-rank authoritative multi-criteria prioritization.
3. **Local Verification**: Rapid field assessment is required prior to prepositioning assets.
