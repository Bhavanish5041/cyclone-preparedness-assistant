# Model Card: Coastal Vulnerability Archetypes (`kmeans-candidate-k4`)

## Model Details
- **Model Identifier**: `kmeans-candidate-k4`
- **Model Family**: scikit-learn Unsupervised Clustering
- **Algorithm**: K-Means ($K=4$, `random_state=42`, `n_init=20`)
- **Framework**: scikit-learn
- **Preprocessing**: `StandardScaler` (zero mean, unit variance)
- **Canonical Features**: 4 Standardized Dimension Scores
- **Status**: Experimental Model Selection Candidate

## Intended Use & Non-Causal Planning Archetype Disclaimer
> [!IMPORTANT]
> DISCLAIMER: Discovered archetypes represent empirical similarity groupings derived from four normalized spatial indicators. They are NOT predetermined hazard categories, validated disaster risk classes, or casualty predictions. They serve as descriptive planning archetypes to identify communities with shared intervention requirements.

## Quantitative Evaluation Metrics (Primary Seed=42)
- **Inertia**: 1015.73
- **Silhouette Score**: 0.3348
- **Davies-Bouldin Index**: 1.1771 (lower is better)
- **Calinski-Harabasz Index**: 298.59 (higher is better)
- **Cluster Balance Ratio**: 0.3293
- **Min Cluster Size**: 82
- **Max Cluster Size**: 249
- **Fit & Eval Runtime**: 0.1186s

## Multi-Seed Stability (10 Random Seeds)
- **Mean ARI vs. Seed 42**: 0.9912
- **Min ARI vs. Seed 42**: 0.9635
- **Std Dev ARI**: 0.0114
- **Mean AMI vs. Seed 42**: 0.9882

## Discovered Archetype Breakdown
### Cluster 0: Urban & Peri-Urban Well-Resourced Corridor
- **Revenue Units**: 130 (20.9%)
- **Population**: 1,543,843 (47.4% of total study population)
- **District Breakdown**: 76.2% Dakshina Kannada | 23.8% Udupi
- **Healthcare Deficit Mean**: 0.2525
- **Demographic Vulnerability Mean**: 0.1977
- **Infrastructure Deficit Mean**: 0.1862
- **Coastal Exposure Mean**: 0.8157
- **Empirical Description**: Communities with low healthcare deficit, low demographic vulnerability, and the lowest infrastructure deficit (treated piped tap water and closed drainage). Highly concentrated in Mangalore City Corporation and taluk headquarters.
- **Primary Intervention Need**: Stormwater pump maintenance, urban grid power resilience, and hospital surge preparedness.

### Cluster 1: Remote Foothill & Healthcare-Isolated Communities
- **Revenue Units**: 82 (13.2%)
- **Population**: 252,738 (7.8% of total study population)
- **District Breakdown**: 59.8% Dakshina Kannada | 40.2% Udupi
- **Healthcare Deficit Mean**: 0.5729
- **Demographic Vulnerability Mean**: 0.3685
- **Infrastructure Deficit Mean**: 0.5966
- **Coastal Exposure Mean**: 0.3767
- **Empirical Description**: Communities with extreme distance to apex hospitals (mean 23.6 km) and PHCs (mean 6.1 km), located deep in the Western Ghats foothills of Sullia, Belthangady, and Karkala. Sheltered from direct marine storm surge but vulnerable to medical isolation and flash flooding.
- **Primary Intervention Need**: Mobile medical units, pre-positioned trauma supplies, all-weather road clearance, and satellite communications.

### Cluster 2: Socioeconomically Vulnerable Rural Interior
- **Revenue Units**: 161 (25.9%)
- **Population**: 553,690 (17.0% of total study population)
- **District Breakdown**: 80.7% Dakshina Kannada | 19.3% Udupi
- **Healthcare Deficit Mean**: 0.3031
- **Demographic Vulnerability Mean**: 0.487
- **Infrastructure Deficit Mean**: 0.618
- **Coastal Exposure Mean**: 0.5107
- **Empirical Description**: Agrarian interior communities with elevated SC/ST population ratios (mean 21.9%), moderate healthcare access, and widespread reliance on untreated local wells and open drainage.
- **Primary Intervention Need**: Targeted social safety net distribution, community shelter accessibility, and emergency water purification tablets.

### Cluster 3: High-Exposure Coastal Rural Belt
- **Revenue Units**: 249 (40.0%)
- **Population**: 904,586 (27.8% of total study population)
- **District Breakdown**: 36.5% Dakshina Kannada | 63.5% Udupi
- **Healthcare Deficit Mean**: 0.2863
- **Demographic Vulnerability Mean**: 0.2015
- **Infrastructure Deficit Mean**: 0.6433
- **Coastal Exposure Mean**: 0.8586
- **Empirical Description**: Coastal shoreline and estuarine settlements within 10 km of the Arabian Sea with high marine storm-surge and wind exposure, combined with heavy infrastructure deficits (untreated well water and open drains prone to saline flooding).
- **Primary Intervention Need**: Cyclone evacuation shelters, saline water intrusion barriers, coastal embankment reinforcement, and early warning sirens.

## Operational Limitations
1. **Descriptive Empirical Clusters**: Clusters reflect statistical spatial groupings, not validated casualty forecasts.
2. **AHP Precedence**: Clustering does not re-rank authoritative multi-criteria prioritization.
3. **Local Verification**: Rapid field assessment is required prior to prepositioning assets.
