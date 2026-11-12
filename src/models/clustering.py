"""Unsupervised vulnerability archetype discovery for cyclone preparedness.

Implements clustering on four standardized preparedness dimension scores:
1. Healthcare Accessibility Deficit
2. Demographic Vulnerability (SC/ST Socioeconomic Disadvantage)
3. Infrastructure Deficit (Water, Road, Drainage Deficits)
4. Coastal Hazard Exposure (Inverted Distance to Arabian Sea Coastline)

Strictly excludes administrative identifiers, geographic coordinates, polygons,
and combined priority scores from the feature matrix X.

Provides robust multi-algorithm evaluation (K-Means and Agglomerative Ward),
evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz), agreement metrics
(Adjusted Rand Index, Adjusted Mutual Information), and safe Weights & Biases (W&B) logging.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import wandb

logger = logging.getLogger(__name__)

# The four verified, non-redundant dimension score columns
CLUSTERING_DIMENSION_COLUMNS: List[str] = [
    "dim_healthcare_deficit",
    "dim_demographic_vulnerability",
    "dim_infrastructure_deficit",
    "dim_coastal_exposure",
]

ARCHETYPE_DISCLAIMER = (
    "DISCLAIMER: Discovered archetypes represent empirical similarity groupings derived from "
    "four normalized spatial indicators. They are NOT predetermined hazard categories, validated "
    "disaster risk classes, or casualty predictions. They serve as descriptive planning archetypes "
    "to identify communities with shared intervention requirements."
)

# Empirically verified descriptive labels for the k=4 solution
ARCHETYPE_LABELS_K4: Dict[int, Dict[str, str]] = {
    0: {
        "short_label": "Urban & Peri-Urban Well-Resourced Corridor",
        "detailed_description": (
            "Communities with low healthcare deficit, low demographic vulnerability, and the lowest "
            "infrastructure deficit (treated piped tap water and closed drainage). Highly concentrated in "
            "Mangalore City Corporation and taluk headquarters."
        ),
        "primary_intervention_need": "Stormwater pump maintenance, urban grid power resilience, and hospital surge preparedness.",
    },
    1: {
        "short_label": "Remote Foothill & Healthcare-Isolated Communities",
        "detailed_description": (
            "Communities with extreme distance to apex hospitals (mean 23.6 km) and PHCs (mean 6.1 km), "
            "located deep in the Western Ghats foothills of Sullia, Belthangady, and Karkala. Sheltered from direct "
            "marine storm surge but vulnerable to medical isolation and flash flooding."
        ),
        "primary_intervention_need": "Mobile medical units, pre-positioned trauma supplies, all-weather road clearance, and satellite communications.",
    },
    2: {
        "short_label": "Socioeconomically Vulnerable Rural Interior",
        "detailed_description": (
            "Agrarian interior communities with elevated SC/ST population ratios (mean 21.9%), moderate "
            "healthcare access, and widespread reliance on untreated local wells and open drainage."
        ),
        "primary_intervention_need": "Targeted social safety net distribution, community shelter accessibility, and emergency water purification tablets.",
    },
    3: {
        "short_label": "High-Exposure Coastal Rural Belt",
        "detailed_description": (
            "Coastal shoreline and estuarine settlements within 10 km of the Arabian Sea with high marine storm-surge "
            "and wind exposure, combined with heavy infrastructure deficits (untreated well water and open drains prone to saline flooding)."
        ),
        "primary_intervention_need": "Cyclone evacuation shelters, saline water intrusion barriers, coastal embankment reinforcement, and early warning sirens.",
    },
}


def extract_clustering_features(
    df: pd.DataFrame,
    dimension_cols: Optional[List[str]] = None,
    standardize: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """Extract and validate the clustering feature matrix X from input DataFrame.

    Strictly ensures that only numeric dimension scores are included, and no
    administrative identifiers, names, coordinates, or composite scores leak in.

    Args:
        df: Input DataFrame containing dimension score columns.
        dimension_cols: List of column names to extract (defaults to CLUSTERING_DIMENSION_COLUMNS).
        standardize: If True, applies StandardScaler (zero mean, unit variance).

    Returns:
        Tuple of (X_matrix, scaler_instance).

    Raises:
        KeyError: If any required dimension column is missing.
        ValueError: If X contains NaNs or infinite values.
    """
    if dimension_cols is None:
        dimension_cols = CLUSTERING_DIMENSION_COLUMNS

    for col in dimension_cols:
        if col not in df.columns:
            raise KeyError(f"Required dimension column '{col}' missing from dataframe.")

    raw_X = df[dimension_cols].to_numpy(dtype=float)

    if np.isnan(raw_X).any():
        raise ValueError("Clustering feature matrix X contains NaN values.")
    if np.isinf(raw_X).any():
        raise ValueError("Clustering feature matrix X contains infinite values.")

    # Verify input values are bounded in [0.0, 1.0]
    if (raw_X < -0.01).any() or (raw_X > 1.01).any():
        logger.warning("Some dimension score values fall outside expected [0.0, 1.0] interval.")

    scaler = StandardScaler()
    if standardize:
        X = scaler.fit_transform(raw_X)
    else:
        X = raw_X.copy()
        scaler.fit(raw_X)

    return X, scaler


@dataclass
class ClusteringMetrics:
    """Evaluation metrics for a single clustering model run."""

    algorithm: str
    n_clusters: int
    random_state: Optional[int]
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    inertia: Optional[float]
    cluster_sizes: List[int]
    cluster_balance_ratio: float  # min(size) / max(size)


def fit_and_evaluate_clustering(
    X: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: int = 4,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, ClusteringMetrics, Any]:
    """Fit a clustering model and evaluate intrinsic geometric metrics.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        algorithm: 'kmeans' or 'agglomerative'.
        n_clusters: Number of clusters (k).
        random_state: Random seed for stochastic algorithms.
        **kwargs: Additional parameters passed to sklearn estimator.

    Returns:
        Tuple of (cluster_labels, ClusteringMetrics, fitted_model).
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2.")
    if n_clusters >= len(X):
        raise ValueError("n_clusters must be strictly less than n_samples.")

    if algorithm.lower() == "kmeans":
        n_init = kwargs.get("n_init", 20)
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            **kwargs,
        )
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
    elif algorithm.lower() == "agglomerative":
        linkage = kwargs.get("linkage", "ward")
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        labels = model.fit_predict(X)
        inertia = None
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'kmeans' or 'agglomerative'.")

    # Evaluate metrics
    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    sizes = pd.Series(labels).value_counts().sort_index().tolist()
    balance = float(min(sizes) / max(sizes))

    metrics = ClusteringMetrics(
        algorithm=algorithm.lower(),
        n_clusters=n_clusters,
        random_state=random_state if algorithm.lower() == "kmeans" else None,
        silhouette=round(sil, 4),
        davies_bouldin=round(db, 4),
        calinski_harabasz=round(ch, 2),
        inertia=round(inertia, 2) if inertia is not None else None,
        cluster_sizes=sizes,
        cluster_balance_ratio=round(balance, 4),
    )

    return labels, metrics, model


def compare_clustering_agreement(
    labels1: np.ndarray,
    labels2: np.ndarray,
) -> Dict[str, float]:
    """Compute partition agreement metrics between two cluster label assignments."""
    ari = float(adjusted_rand_score(labels1, labels2))
    ami = float(adjusted_mutual_info_score(labels1, labels2))
    return {
        "adjusted_rand_index": round(ari, 4),
        "adjusted_mutual_info": round(ami, 4),
    }


def compute_archetype_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    dimension_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute comprehensive statistical profiles for each discovered cluster archetype.

    Calculates:
    - Cluster size and percentage of total units
    - District composition (Dakshina Kannada vs Udupi breakdown)
    - Mean and median for each of the four dimension scores
    - Raw geographic proxies (distance to coast, distance to hospital, distance to PHC, SC/ST ratio)

    Args:
        df: Input DataFrame with dimension scores and context columns.
        cluster_labels: Array of cluster assignments of length len(df).
        dimension_cols: List of dimension column names.

    Returns:
        DataFrame with cluster profiles indexed by cluster ID.
    """
    if dimension_cols is None:
        dimension_cols = CLUSTERING_DIMENSION_COLUMNS

    temp_df = df.copy()
    temp_df["cluster_id"] = cluster_labels
    n_total = len(temp_df)

    profile_records = []
    for c_id in sorted(temp_df["cluster_id"].unique()):
        c_sub = temp_df[temp_df["cluster_id"] == c_id]
        c_size = len(c_sub)
        pct = round(c_size / n_total * 100, 1)

        # District breakdown
        n_dk = int((c_sub["district_name"] == "Dakshina Kannada").sum())
        n_ud = int((c_sub["district_name"] == "Udupi").sum())
        pct_dk = round(n_dk / c_size * 100, 1)
        pct_ud = round(n_ud / c_size * 100, 1)

        record: Dict[str, Any] = {
            "cluster_id": c_id,
            "unit_count": c_size,
            "pct_of_total": pct,
            "count_dk": n_dk,
            "count_udupi": n_ud,
            "pct_dk": pct_dk,
            "pct_udupi": pct_ud,
        }

        # Population statistics where available
        if "total_population" in temp_df.columns:
            tot_pop_all = float(temp_df["total_population"].sum())
            c_pop = int(c_sub["total_population"].sum())
            record["total_population"] = c_pop
            record["total_population_mean"] = round(float(c_sub["total_population"].mean()), 1)
            record["pct_of_total_population"] = round(c_pop / tot_pop_all * 100, 1) if tot_pop_all > 0 else 0.0

        if "sc_population" in c_sub.columns and "st_population" in c_sub.columns:
            sc_pop = int(c_sub["sc_population"].sum())
            st_pop = int(c_sub["st_population"].sum())
            sc_st_pop = sc_pop + st_pop
            record["sc_population"] = sc_pop
            record["st_population"] = st_pop
            record["sc_st_population"] = sc_st_pop
            if "total_population" in c_sub.columns and c_pop > 0:
                record["pct_sc_st_population"] = round(sc_st_pop / c_pop * 100, 1)

        # Dimension score statistics
        for dim in dimension_cols:
            record[f"{dim}_mean"] = round(float(c_sub[dim].mean()), 4)
            record[f"{dim}_median"] = round(float(c_sub[dim].median()), 4)
            record[f"{dim}_std"] = round(float(c_sub[dim].std()), 4)

        # Raw feature context where available
        if "dist_to_coastline_km" in c_sub.columns:
            record["dist_to_coastline_km_mean"] = round(float(c_sub["dist_to_coastline_km"].mean()), 1)
            record["dist_to_coastline_km_median"] = round(float(c_sub["dist_to_coastline_km"].median()), 1)
        if "dist_to_nearest_hospital_km" in c_sub.columns:
            record["dist_to_hospital_km_mean"] = round(float(c_sub["dist_to_nearest_hospital_km"].mean()), 1)
        if "dist_to_nearest_phc_chc_km" in c_sub.columns:
            record["dist_to_phc_chc_km_mean"] = round(float(c_sub["dist_to_nearest_phc_chc_km"].mean()), 1)
        if "sc_st_ratio" in c_sub.columns:
            record["sc_st_ratio_mean"] = round(float(c_sub["sc_st_ratio"].mean()), 3)

        # Descriptive label if k=4
        if c_id in ARCHETYPE_LABELS_K4:
            record["archetype_name"] = ARCHETYPE_LABELS_K4[c_id]["short_label"]
            record["primary_intervention_need"] = ARCHETYPE_LABELS_K4[c_id]["primary_intervention_need"]

        profile_records.append(record)

    return pd.DataFrame(profile_records).set_index("cluster_id")


def compute_kmeans_pca(
    X_scaled: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on standardized clustering feature matrix X.

    Args:
        X_scaled: Standardized feature matrix (n_samples, n_features).
        n_components: Number of principal components.
        random_state: Seed for reproducible PCA decomposition.

    Returns:
        Tuple of (fitted_pca_model, X_pca_transformed, explained_variance_ratios).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, pca.explained_variance_ratio_


def plot_kmeans_pca_projection(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    cluster_centers: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[PCA, np.ndarray, plt.Figure]:
    """Generate 2D PCA projection scatter plot with cluster centroids.

    Args:
        X_scaled: Standardized feature matrix.
        labels: Cluster assignment vector.
        cluster_centers: Array of cluster center coordinates in standardized feature space.
        output_path: Path to save the generated PNG plot.

    Returns:
        Tuple of (pca_model, X_pca, fig).
    """
    pca, X_pca, var_exp = compute_kmeans_pca(X_scaled, n_components=2, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors = {
        0: "#2ca02c",  # Green: Urban Well-Resourced Corridor
        1: "#d62728",  # Red: Foothill Healthcare-Isolated
        2: "#9467bd",  # Purple: Vulnerable Rural Interior
        3: "#1f77b4",  # Blue: High-Exposure Coastal Belt
    }
    names = {
        0: "Cluster 0: Urban Well-Resourced Corridor",
        1: "Cluster 1: Foothill Healthcare-Isolated",
        2: "Cluster 2: Vulnerable Rural Interior",
        3: "Cluster 3: High-Exposure Coastal Belt",
    }

    for c_id in sorted(np.unique(labels)):
        mask = (labels == c_id)
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors.get(c_id, f"C{c_id}"),
            label=names.get(c_id, f"Cluster {c_id}"),
            alpha=0.75,
            edgecolors="none",
            s=45,
        )

    if cluster_centers is not None:
        centers_pca = pca.transform(cluster_centers)
        ax.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c="#FFD700",
            marker="*",
            s=240,
            edgecolors="black",
            linewidths=1.5,
            label="Cluster Centroids (k=4)",
            zorder=10,
        )

    ax.set_title(
        f"K-Means (k=4) Vulnerability Archetypes — 2D PCA Projection\n"
        f"Total Explained Variance: {(var_exp[0] + var_exp[1]) * 100:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0] * 100:.1f}% Variance)", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1] * 100:.1f}% Variance)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", framealpha=0.95, fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PCA projection figure to {p}")

    return pca, X_pca, fig


def plot_cluster_population_distributions(
    archetype_profiles_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot revenue unit counts and total population distribution per archetype.

    Args:
        archetype_profiles_df: Summary archetype profile DataFrame.
        output_path: Path to save the generated PNG plot.

    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    clusters = sorted(archetype_profiles_df.index)
    colors = ["#2ca02c", "#d62728", "#9467bd", "#1f77b4"]
    names = [
        ARCHETYPE_LABELS_K4.get(c, {}).get("short_label", f"Cluster {c}")
        for c in clusters
    ]

    units = [int(archetype_profiles_df.loc[c, "unit_count"]) for c in clusters]
    tot_units = sum(units)
    bars1 = ax1.bar(range(len(clusters)), units, color=colors, edgecolor="#333333", linewidth=0.7)
    ax1.set_title(f"Revenue Units per Archetype (Total: {tot_units})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Revenue Units", fontsize=10)
    ax1.set_xticks(range(len(clusters)))
    ax1.set_xticklabels([f"C{c}" for c in clusters], fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h}\n({h / tot_units * 100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5)

    if "total_population" in archetype_profiles_df.columns:
        pops = [int(archetype_profiles_df.loc[c, "total_population"]) for c in clusters]
        tot_pop = sum(pops)
        bars2 = ax2.bar(range(len(clusters)), pops, color=colors, edgecolor="#333333", linewidth=0.7)
        ax2.set_title(f"Total Population per Archetype (Total: {tot_pop:,})", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Total Census 2011 Population", fontsize=10)
        ax2.set_xticks(range(len(clusters)))
        ax2.set_xticklabels([f"C{c}" for c in clusters], fontsize=10, fontweight="bold")
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
        for bar in bars2:
            h = bar.get_height()
            ax2.annotate(f"{h:,}\n({h / tot_pop * 100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)
    else:
        pcts = [float(archetype_profiles_df.loc[c, "pct_of_total"]) for c in clusters]
        ax2.bar(range(len(clusters)), pcts, color=colors, edgecolor="#333333", linewidth=0.7)
        ax2.set_title("Percentage of Units per Archetype", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Percentage (%)", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(clusters))]
    fig.legend(handles, [f"C{c}: {names[i]}" for i, c in enumerate(clusters)],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cluster distribution figure to {p}")

    return fig


def stage_kmeans_artifacts(
    model: Any,
    scaler: Any,
    metadata: Dict[str, Any],
    profiles_df: pd.DataFrame,
    staging_dir: Path,
) -> List[Path]:
    """Serialize and stage all standardized K-Means artifact files for W&B model packaging.

    Args:
        model: Fitted KMeans model object.
        scaler: Fitted StandardScaler preprocessor.
        metadata: Comprehensive metadata dictionary.
        profiles_df: Archetype summary profiles DataFrame.
        staging_dir: Target directory to stage files into.

    Returns:
        List of staged artifact file paths.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_files = []

    # 1. Fitted K-Means model
    model_path = staging_dir / "kmeans_k4_model.joblib"
    joblib.dump(model, model_path)
    staged_files.append(model_path)

    # 2. Fitted StandardScaler
    scaler_path = staging_dir / "kmeans_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    staged_files.append(scaler_path)

    # 3. Serialized Metadata JSON
    meta_path = staging_dir / "kmeans_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    staged_files.append(meta_path)

    # 4. Feature Schema JSON
    schema_path = staging_dir / "feature_schema.json"
    schema = {
        "dimensions": len(CLUSTERING_DIMENSION_COLUMNS),
        "dimension_columns": CLUSTERING_DIMENSION_COLUMNS,
        "scaling": "sklearn.preprocessing.StandardScaler (mean=0, std=1)",
        "feature_descriptions": {
            "dim_healthcare_deficit": "Normalized straight-line distance to nearest PHC/CHC and Hospital",
            "dim_demographic_vulnerability": "Proportion of scheduled caste and scheduled tribe population (Census 2011)",
            "dim_infrastructure_deficit": "Untreated water source, unpaved road, and open/no drainage deficits (Census 2011)",
            "dim_coastal_exposure": "Inverted distance to the Arabian Sea coastline proxy",
        },
        "disallowed_columns": [
            "geometry",
            "shrid2",
            "pc11_id",
            "priority_score",
            "priority_rank",
            "rep_point_latitude",
            "rep_point_longitude",
        ],
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    staged_files.append(schema_path)

    # 5. Archetype Profiles JSON
    profiles_path = staging_dir / "archetype_profiles.json"
    profiles_records = profiles_df.reset_index().to_dict(orient="records")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles_records, f, indent=2)
    staged_files.append(profiles_path)

    # 6. Model Card Markdown
    card_path = staging_dir / "model_card.md"
    card_content = f"""# Model Card: Coastal Vulnerability K-Means Clustering (`coastal-vulnerability-kmeans`)

## Model Details
- **Model Identifier**: coastal-vulnerability-kmeans
- **Model Family**: scikit-learn Unsupervised Clustering
- **Algorithm**: K-Means ($k=4$, `random_state=42`, `n_init=20`)
- **Framework**: scikit-learn
- **Preprocessing**: StandardScaler (zero mean, unit variance)

## Intended Use & Non-Causal Planning Archetype Disclaimer
> [!IMPORTANT]
> {ARCHETYPE_DISCLAIMER}

## Quantitative Metrics
- **Silhouette Coefficient**: {metadata.get("evaluation_metrics", {}).get("silhouette", 0.3348)}
- **Davies-Bouldin Index**: {metadata.get("evaluation_metrics", {}).get("davies_bouldin", 1.1771)}
- **Calinski-Harabasz Variance Ratio**: {metadata.get("evaluation_metrics", {}).get("calinski_harabasz", 298.59)}
- **Inertia**: {metadata.get("evaluation_metrics", {}).get("inertia", 1015.73)}
- **Cluster Balance Ratio**: {metadata.get("evaluation_metrics", {}).get("cluster_balance_ratio", 0.3293)}
- **Total Revenue Units**: {metadata.get("total_units", 622)}
- **Total Population Represented**: {metadata.get("total_population", 3254857):,}

## Discovered Archetype Definitions
- **Cluster 0**: Urban & Peri-Urban Well-Resourced Corridor (130 units, 1,543,843 pop)
- **Cluster 1**: Remote Foothill & Healthcare-Isolated Communities (82 units, 252,738 pop)
- **Cluster 2**: Socioeconomically Vulnerable Rural Interior (161 units, 553,690 pop)
- **Cluster 3**: High-Exposure Coastal Rural Belt (249 units, 904,586 pop)
"""
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    staged_files.append(card_path)

    # 7. Intended Use and Limitations Markdown
    limitations_path = staging_dir / "intended_use_and_limitations.md"
    limitations_content = f"""# Intended Use and Operational Limitations

## Operational Scope
Discovered clusters identify shared disaster planning challenges across coastal Karnataka (Dakshina Kannada and Udupi).

## Operational Boundaries & Safeguards
1. **Descriptive Clustering Only**: These clusters represent empirical groupings of spatial attributes, not hazard or casualty predictions.
2. **Never Overwrite AHP Priorities**: Cluster IDs describe intervention types, but do not replace or re-order authoritative AHP preparedness rankings.
3. **No Ground Damage Claims**: Ground validation is strictly required before directing critical physical capital.
"""
    with open(limitations_path, "w", encoding="utf-8") as f:
        f.write(limitations_content)
    staged_files.append(limitations_path)

    return staged_files


def save_kmeans_local_artifacts(
    model: Any,
    scaler: Any,
    metadata: Dict[str, Any],
    profiles_df: pd.DataFrame,
    output_dir: Union[str, Path] = Path("models"),
) -> Dict[str, Path]:
    """Persist local model and preprocessing artifacts into models directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_file = out_path / "kmeans_k4_model.joblib"
    scaler_file = out_path / "kmeans_scaler.joblib"
    meta_file = out_path / "kmeans_metadata.json"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model": model_file,
        "scaler": scaler_file,
        "metadata": meta_file,
    }


def log_clustering_experiments_wandb(
    experiments_data: List[Dict[str, Any]],
    project_name: str = "cyclone-preparedness-assistant",
    job_type: str = "clustering-experiments",
    fallback_json_path: Optional[Path] = None,
    mode: Optional[str] = None,
    entity: Optional[str] = None,
    agreement_data: Optional[List[Dict[str, Any]]] = None,
    seed_stability_data: Optional[List[Dict[str, Any]]] = None,
    archetype_profiles_df: Optional[pd.DataFrame] = None,
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    X_scaled: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    model_dir: Optional[Union[str, Path]] = None,
    artifact_name: str = "coastal-vulnerability-kmeans",
    artifact_aliases: Optional[List[str]] = None,
) -> bool:
    """Log clustering experiment configurations and metrics to Weights & Biases if configured.

    Gracefully detects whether W&B is authenticated and configured.
    Supports individual model run logging, benchmark tables, artifact registration, and local fallback.

    Args:
        experiments_data: List of experiment dictionaries containing config and metrics.
        project_name: W&B project identifier.
        job_type: W&B job category.
        fallback_json_path: Path to save local JSON run manifest.
        mode: Explicit mode ('online', 'offline', or 'disabled').
        entity: W&B entity/team username.
        agreement_data: Inter-algorithm agreement metrics.
        seed_stability_data: Seed stability metrics for k=4.
        archetype_profiles_df: Summary archetype profile DataFrame.
        model: Fitted KMeans model object.
        scaler: Fitted StandardScaler preprocessor.
        X_scaled: Standardized feature matrix.
        df: Units DataFrame with identifiers and attributes.
        figures_dir: Directory containing visualization figures.
        model_dir: Directory to save local model artifacts.
        artifact_name: W&B Model Artifact name.
        artifact_aliases: Aliases to assign to registered artifact.

    Returns:
        True if logged online to W&B; False if logged locally in offline/disabled fallback mode.
    """
    from src.tracking.wandb_logger import log_clustering_experiments, resolve_wandb_mode

    effective_mode = resolve_wandb_mode(mode)
    res = log_clustering_experiments(
        experiments_data=experiments_data,
        agreement_data=agreement_data,
        seed_stability_data=seed_stability_data,
        archetype_profiles_df=archetype_profiles_df,
        project_name=project_name,
        entity=entity,
        mode=effective_mode,
        fallback_json_path=fallback_json_path,
        model=model,
        scaler=scaler,
        X_scaled=X_scaled,
        df=df,
        figures_dir=figures_dir,
        model_dir=model_dir,
        artifact_name=artifact_name,
        artifact_aliases=artifact_aliases,
    )
    return bool(res.get("synced", False))

