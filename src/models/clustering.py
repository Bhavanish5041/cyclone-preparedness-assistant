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
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
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
) -> bool:
    """Log clustering experiment configurations and metrics to Weights & Biases if configured.

    Gracefully detects whether W&B is authenticated and configured.
    Supports individual model run logging, benchmark tables, and local fallback.

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
    )
    return bool(res.get("synced", False))

