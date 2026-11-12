# CYCLONE INTEL
## Machine Learning Methodology and Model Explainability

### 1. Overview

CYCLONE INTEL uses two machine learning approaches to analyse community-level cyclone preparedness and vulnerability:

1. **Logistic Regression:** A supervised classification model that estimates a community's preparedness category.
2. **K-Means Clustering:** An unsupervised learning algorithm that groups communities into vulnerability archetypes based on similarities in their characteristics.

These models serve different purposes. Logistic Regression provides an approximation of preparedness classifications derived from the Analytic Hierarchy Process (AHP), while K-Means identifies groups of communities with similar vulnerability characteristics.

Neither model predicts cyclone formation, cyclone trajectories, casualties or future disaster events.

---

# 2. Logistic Regression: Preparedness Classification

### 2.1 Objective

The objective of Logistic Regression is to classify communities into one of three preparedness categories:

- Low
- Medium
- High

The model learns the relationship between community-level indicators and preparedness categories established through the AHP methodology.

It acts as a computational surrogate for AHP-derived classifications. It does not independently establish ground-truth disaster preparedness.

### 2.2 Dataset

The documented modelling dataset contains 1,879 administrative units.

The dataset contains 26 input features representing geographical exposure, infrastructure, healthcare accessibility, demographic characteristics and monitoring availability.

### 2.3 Input features

The model uses the following 26 features:

| Category | Features |
|---|---|
| Geographic exposure | `dist_coast_km`, `elevation_mean` |
| Demographics | `avg_household_size`, `sc_share`, `st_share`, `vulnerable_pop_share` |
| Infrastructure | `road_connectivity_score`, `drinking_water_score`, `has_pucca_road`, `has_major_district_road`, `bus_connectivity` |
| Healthcare and monitoring | `n_hc_10km`, `n_phc_chc_10km`, `n_gauges_20km`, `n_shelters_10km` |
| Composite indicators | `cvi_physical`, `cvi_social`, `coastal_risk_score`, `evacuation_sufficiency`, `vulnerability_exposure`, `infra_composite`, `monitoring_coverage` |
| Engineered features | `log1p_population`, `log1p_pop_density_sqkm`, `log1p_dist_nearest_shelter_km`, `log1p_dist_nearest_hospital_km` |

All 26 features are required, with a strict feature order.

### 2.4 Data preprocessing

The model uses a RobustScaler before classification.

RobustScaler transforms features using their median and interquartile range:

\[
x' = \frac{x-\operatorname{median}(x)}{Q_{75}-Q_{25}}
\]

This reduces the influence of extreme values compared with standard scaling based on mean and standard deviation.

The preprocessing and classifier are combined into one Scikit-learn Pipeline, ensuring that the same transformations are applied during training and inference.

### 2.5 Train-validation-test split

The documented experimental design uses stratified sampling:

| Dataset partition | Percentage | Approximate records |
|---|---:|---:|
| Training | 70% | 1,315 |
| Validation | 15% | 282 |
| Testing | 15% | 282 |
| Total | 100% | 1,879 |

Random state: 42.

Stratification preserves approximately the same preparedness-class proportions across the three partitions.

### 2.6 Model configuration

The model uses Scikit-learn LogisticRegression with the following configuration:

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Regularisation | L2 |
| C | 1.0 |
| Solver | lbfgs |
| Maximum iterations | 500 |
| Random state | 42 |
| Input scaler | RobustScaler |

The model performs multiclass classification. With the documented Scikit-learn version, lbfgs uses multinomial cross-entropy automatically.

### 2.7 How Logistic Regression works

Logistic Regression calculates a score for each preparedness category using a weighted combination of the input features.

For multiclass classification, the model converts these scores into class probabilities using the softmax function:

\[
P(y=k\mid x)=
\frac{e^{z_k}}{\sum_j e^{z_j}}
\]

Here:

- \(x\) represents the community's feature vector.
- \(z_k\) is the learned score for class \(k\).
- \(P(y=k\mid x)\) is the estimated probability assigned to that class.

The model selects the class with the highest estimated probability.

These probabilities are model outputs, not necessarily calibrated real-world probabilities of preparedness.

### 2.8 Evaluation

The documented reproduction script specifies the following expected test metrics:

| Metric | Expected value |
|---|---:|
| Accuracy | 94.3262% |
| Macro F1 | 90.6023% |
| Weighted F1 | 94.2247% |

The reproduction script checks whether the newly trained model matches these recorded results within a tolerance of 0.0001.

The metrics should be described as agreement with AHP-derived labels, not as independently measured cyclone preparedness accuracy.

### 2.9 Limitations and safeguards

- The target classifications originate from AHP, so the model learns to reproduce AHP-derived decisions.
- The model cannot establish that AHP classifications correspond to actual disaster outcomes.
- Predicted probabilities should not be presented as calibrated risk probabilities without calibration testing.
- The authoritative AHP assessment must not be silently replaced or downgraded by the surrogate model.
- The model requires all 26 expected input features.

---

# 3. K-Means: Vulnerability Archetype Discovery

### 3.1 Objective

K-Means is used to discover groups of communities with similar vulnerability characteristics.

Unlike Logistic Regression, K-Means is an unsupervised learning algorithm. It does not require predefined preparedness labels.

Its purpose is to identify descriptive vulnerability archetypes that can support planning and intervention prioritisation.

### 3.2 Input features

The clustering implementation deliberately selects four dimensions:

1. Healthcare Accessibility Deficit
2. Demographic Vulnerability
3. Infrastructure Deficit
4. Coastal Hazard Exposure

The corresponding feature columns are:

- `dim_healthcare_deficit`
- `dim_demographic_vulnerability`
- `dim_infrastructure_deficit`
- `dim_coastal_exposure`

Administrative identifiers, names, geographic coordinates, polygons and combined priority scores are excluded from the clustering feature matrix.

### 3.3 Feature standardisation

The four dimensions are standardised using Scikit-learn StandardScaler.

\[
z = \frac{x-\mu}{\sigma}
\]

This gives each dimension a comparable scale before calculating distances.

### 3.4 Why K-Means?

K-Means partitions observations into K groups by minimising the within-cluster sum of squared distances to cluster centroids.

Its objective function is:

\[
J = \sum_{i=1}^{n}\sum_{k=1}^{K}
r_{ik}\|x_i-\mu_k\|^2
\]

Where:

- \(K\) is the number of clusters.
- \(x_i\) is an observation.
- \(\mu_k\) is the centroid of cluster \(k\).
- \(r_{ik}\) indicates whether observation \(i\) belongs to cluster \(k\).

Communities with similar dimension scores are assigned to the same cluster.

### 3.5 Selecting the number of clusters

The experiment evaluates K-Means for:

\[
K \in \{2,3,4,5,6\}
\]

It also evaluates Agglomerative clustering using Ward linkage for the same values of K.

The evaluation framework includes:

- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score
- Cluster balance ratio
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
- Random-seed stability

The final implementation explicitly selects K-Means with K=4.

The code documents the following evidence for the selected solution:

| Evaluation | K=4 result |
|---|---:|
| Silhouette score | 0.3348 |
| K-Means vs Agglomerative ARI | 0.7575 |
| Mean random-seed ARI | 0.9902 |

The selection also considers whether the resulting clusters have meaningful domain interpretations.

The complete K=5 comparison should be retained in the experiment manifest before claiming that K=4 outperformed K=5 on every individual metric.

### 3.6 Random-seed stability

The K=4 solution is tested against ten random seeds:

42, 123, 456, 789, 999, 1024, 2024, 7, 13 and 99.

The resulting assignments are compared against the seed-42 solution using ARI and AMI.

The documented mean ARI is 0.9902, indicating highly consistent assignments across the tested seeds.

### 3.7 The four vulnerability archetypes

The final K=4 solution produces four descriptive archetypes:

**Cluster 0: Urban & Peri-Urban Well-Resourced Corridor**

Characteristics:
- Relatively low healthcare deficit.
- Lower demographic vulnerability.
- Lower infrastructure deficit.

Suggested intervention focus:
- Stormwater pump maintenance.
- Urban grid power resilience.
- Hospital surge preparedness.

**Cluster 1: Remote Foothill & Healthcare-Isolated Communities**

Characteristics:
- Greater distance from apex hospitals and PHCs.
- Communities in Western Ghats foothill areas.
- Healthcare accessibility challenges.

Suggested intervention focus:
- Mobile medical units.
- Pre-positioned trauma supplies.
- All-weather road clearance.
- Satellite communications.

**Cluster 2: Socioeconomically Vulnerable Rural Interior**

Characteristics:
- Higher SC/ST population ratios.
- Moderate healthcare accessibility.
- Infrastructure weaknesses involving water and drainage.

Suggested intervention focus:
- Targeted social safety-net distribution.
- Community shelter accessibility.
- Emergency water purification.

**Cluster 3: High-Exposure Coastal Rural Belt**

Characteristics:
- Communities near the Arabian Sea coastline.
- Coastal and estuarine exposure.
- Infrastructure weaknesses involving water and drainage.

Suggested intervention focus:
- Cyclone evacuation shelters.
- Saline water intrusion barriers.
- Coastal embankment reinforcement.
- Early-warning sirens.

These labels are descriptive interpretations of the cluster profiles, not predetermined hazard classes.

### 3.8 Limitations

- K-Means is sensitive to feature scaling and the selected number of clusters.
- Cluster membership indicates similarity, not a probability of disaster.
- Clusters do not establish that every community within a group has identical needs.
- The results depend on the quality and coverage of the underlying dataset.
- The archetypes should be used as planning aids, not predictions of casualties or cyclone impacts.

---

# 4. How the Two Models Work Together

The models answer different questions.

| Aspect | Logistic Regression | K-Means |
|---|---|---|
| Learning type | Supervised | Unsupervised |
| Objective | Preparedness classification | Vulnerability grouping |
| Input | 26 features | 4 dimensions |
| Output | Low, Medium or High | One of four clusters |
| Reference | AHP-derived labels | No predefined labels |
| Evaluation | Accuracy, precision, recall, F1 | Silhouette, DBI, CH, ARI, AMI |
| Main use | Approximate preparedness classification | Identify shared intervention needs |

### Combined decision-support workflow

1. Community-level data is collected and transformed into model features.
2. Logistic Regression estimates the AHP-derived preparedness category.
3. K-Means assigns a descriptive vulnerability archetype.
4. The authoritative AHP assessment and operational priority band remain distinct outputs.
5. Deterministic intervention logic uses the available evidence to support planning recommendations.
6. The dashboard presents the results to authorities and citizens.

The models complement each other but do not replace the underlying AHP methodology or authoritative priority allocation.

---

# 5. Important Scientific Distinctions

CYCLONE INTEL distinguishes four concepts:

- **Preparedness:** How a community scores under the AHP assessment.
- **Vulnerability:** Structural characteristics that may create difficulties during a cyclone.
- **Priority:** The operational allocation band assigned by the prioritisation methodology.
- **Intervention:** Measures that can address identified weaknesses.

A high preparedness category does not automatically imply a low operational priority band. These are separate measures and must be interpreted independently.

---

# 6. Further Evaluation Recommended

The following evaluations should be completed or their existing evidence retrieved before making stronger claims:

### Logistic Regression
- Majority-class baseline.
- DummyClassifier baseline.
- Decision Tree and Random Forest comparisons.
- Per-class precision, recall and F1.
- Confusion matrix.
- Probability calibration analysis.
- Feature importance and SHAP explanations.

### K-Means
- Complete K=2 through K=6 metric comparison.
- Elbow/inertia analysis.
- Silhouette comparison across K values.
- Davies-Bouldin and Calinski-Harabasz comparisons.
- Cluster stability analysis.
- Cluster profile validation.

### Reproducibility
- Preserve source datasets and feature-generation scripts.
- Record Python and library versions.
- Record random seeds and hyperparameters.
- Version model artifacts and feature schemas.
- Preserve experiment manifests and evaluation reports.

---

# 7. Conclusion

CYCLONE INTEL combines supervised and unsupervised machine learning to improve community-level cyclone preparedness analysis.

Logistic Regression provides a reproducible approximation of AHP-derived preparedness classifications, while K-Means identifies four interpretable vulnerability archetypes from four standardised community dimensions.

Together, they provide complementary analytical perspectives for preparedness assessment and intervention planning.

The system is a decision-support platform based on available community-level indicators. It is not a real-time cyclone forecasting or disaster-outcome prediction system.