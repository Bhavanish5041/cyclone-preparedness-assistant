# Model Card: Coastal Vulnerability Archetypes (`kmeans-candidate-k2`)

## Model Details
- **Model Identifier**: `kmeans-candidate-k2`
- **Model Family**: scikit-learn Unsupervised Clustering
- **Algorithm**: K-Means ($K=2$, `random_state=42`, `n_init=20`)
- **Framework**: scikit-learn
- **Preprocessing**: `StandardScaler` (zero mean, unit variance)
- **Canonical Features**: 4 Standardized Dimension Scores
- **Status**: Experimental Model Selection Candidate

## Intended Use & Non-Causal Planning Archetype Disclaimer
> [!IMPORTANT]
> DISCLAIMER: Discovered archetypes represent empirical similarity groupings derived from four normalized spatial indicators. They are NOT predetermined hazard categories, validated disaster risk classes, or casualty predictions. They serve as descriptive planning archetypes to identify communities with shared intervention requirements.

## Quantitative Evaluation Metrics (Primary Seed=42)
- **Inertia**: 1681.94
- **Silhouette Score**: 0.3004
- **Davies-Bouldin Index**: 1.3453 (lower is better)
- **Calinski-Harabasz Index**: 297.13 (higher is better)
- **Cluster Balance Ratio**: 0.7182
- **Min Cluster Size**: 260
- **Max Cluster Size**: 362
- **Fit & Eval Runtime**: 0.172s

## Multi-Seed Stability (10 Random Seeds)
- **Mean ARI vs. Seed 42**: 0.9949
- **Min ARI vs. Seed 42**: 0.9808
- **Std Dev ARI**: 0.008
- **Mean AMI vs. Seed 42**: 0.9891

## Discovered Archetype Breakdown
### Cluster 0: Inland & Foothill Vulnerability Zone
- **Revenue Units**: 260 (41.8%)
- **Population**: 890,908 (27.4% of total study population)
- **District Breakdown**: 73.5% Dakshina Kannada | 26.5% Udupi
- **Healthcare Deficit Mean**: 0.3966
- **Demographic Vulnerability Mean**: 0.4373
- **Infrastructure Deficit Mean**: 0.5982
- **Coastal Exposure Mean**: 0.4735
- **Empirical Description**: Inland communities characterized by elevated healthcare isolation (mean 16.2 km to hospital), elevated demographic vulnerability (21.3% SC/ST), and high infrastructure deficit (open drains and unpaved roads). Predominantly Dakshina Kannada (73.5%).
- **Primary Intervention Need**: Decentralized primary health clinics, all-weather transport corridors, and social welfare support.

### Cluster 1: Coastal Shoreline & Urban-Adjacent Corridor
- **Revenue Units**: 362 (58.2%)
- **Population**: 2,363,949 (72.6% of total study population)
- **District Breakdown**: 49.2% Dakshina Kannada | 50.8% Udupi
- **Healthcare Deficit Mean**: 0.2673
- **Demographic Vulnerability Mean**: 0.1956
- **Infrastructure Deficit Mean**: 0.4897
- **Coastal Exposure Mean**: 0.8559
- **Empirical Description**: Coastal shoreline and urban-adjacent communities with high marine storm-surge and wind exposure (mean 10.4 km to coast), lower healthcare deficit, and lower demographic vulnerability. Balanced across DK and Udupi.
- **Primary Intervention Need**: Storm surge defenses, early warning dissemination, urban drainage resilience, and coastal embankment reinforcement.

## Operational Limitations
1. **Descriptive Empirical Clusters**: Clusters reflect statistical spatial groupings, not validated casualty forecasts.
2. **AHP Precedence**: Clustering does not re-rank authoritative multi-criteria prioritization.
3. **Local Verification**: Rapid field assessment is required prior to prepositioning assets.
