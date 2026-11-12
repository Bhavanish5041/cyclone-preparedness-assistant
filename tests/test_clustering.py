"""Unit tests for unsupervised vulnerability archetype discovery and experiment tracking."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.clustering import (
    CLUSTERING_DIMENSION_COLUMNS,
    ClusteringMetrics,
    compare_clustering_agreement,
    compute_archetype_profiles,
    compute_kmeans_pca,
    extract_clustering_features,
    fit_and_evaluate_clustering,
    log_clustering_experiments_wandb,
    plot_cluster_population_distributions,
    plot_kmeans_pca_projection,
    save_kmeans_local_artifacts,
    stage_kmeans_artifacts,
)


@pytest.fixture
def synthetic_clustering_df() -> pd.DataFrame:
    """Create synthetic DataFrame with 4 normalized dimension scores and context."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "shrid2": [f"shrid_{i}" for i in range(n)],
            "unit_name": [f"Unit_{i}" for i in range(n)],
            "district_name": ["Dakshina Kannada" if i % 2 == 0 else "Udupi" for i in range(n)],
            "dim_healthcare_deficit": np.random.uniform(0.1, 0.9, n),
            "dim_demographic_vulnerability": np.random.uniform(0.0, 0.8, n),
            "dim_infrastructure_deficit": np.random.uniform(0.2, 1.0, n),
            "dim_coastal_exposure": np.random.uniform(0.0, 1.0, n),
            "dist_to_coastline_km": np.random.uniform(0.5, 65.0, n),
            "dist_to_nearest_hospital_km": np.random.uniform(2.0, 35.0, n),
            "dist_to_nearest_phc_chc_km": np.random.uniform(0.5, 10.0, n),
            "sc_st_ratio": np.random.uniform(0.02, 0.40, n),
        }
    )


def test_extract_clustering_features_valid(synthetic_clustering_df: pd.DataFrame):
    """Verify feature extraction produces correct shape and standardizes features."""
    X, scaler = extract_clustering_features(synthetic_clustering_df, standardize=True)
    assert X.shape == (len(synthetic_clustering_df), 4)
    # Check standardization: mean ~ 0, std ~ 1
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-5)


def test_extract_clustering_features_missing_col(synthetic_clustering_df: pd.DataFrame):
    """Verify extract_clustering_features raises KeyError if a dimension is missing."""
    df_missing = synthetic_clustering_df.drop(columns=["dim_coastal_exposure"])
    with pytest.raises(KeyError, match="dim_coastal_exposure"):
        extract_clustering_features(df_missing)


def test_fit_and_evaluate_clustering_kmeans_determinism(synthetic_clustering_df: pd.DataFrame):
    """Verify K-Means produces identical results given the same random_state."""
    X, _ = extract_clustering_features(synthetic_clustering_df, standardize=True)

    labels1, metrics1, _ = fit_and_evaluate_clustering(X, algorithm="kmeans", n_clusters=3, random_state=42)
    labels2, metrics2, _ = fit_and_evaluate_clustering(X, algorithm="kmeans", n_clusters=3, random_state=42)

    assert np.array_equal(labels1, labels2)
    assert metrics1.silhouette == metrics2.silhouette
    assert metrics1.davies_bouldin == metrics2.davies_bouldin
    assert metrics1.cluster_sizes == metrics2.cluster_sizes
    assert -1.0 <= metrics1.silhouette <= 1.0
    assert metrics1.davies_bouldin >= 0.0


def test_fit_and_evaluate_clustering_agglomerative(synthetic_clustering_df: pd.DataFrame):
    """Verify Agglomerative clustering runs cleanly with Ward linkage."""
    X, _ = extract_clustering_features(synthetic_clustering_df, standardize=True)
    labels, metrics, _ = fit_and_evaluate_clustering(X, algorithm="agglomerative", n_clusters=3, linkage="ward")

    assert len(labels) == len(synthetic_clustering_df)
    assert len(set(labels)) == 3
    assert metrics.algorithm == "agglomerative"
    assert -1.0 <= metrics.silhouette <= 1.0


def test_fit_and_evaluate_clustering_invalid_k(synthetic_clustering_df: pd.DataFrame):
    """Verify error raised when k < 2 or k >= n_samples."""
    X, _ = extract_clustering_features(synthetic_clustering_df, standardize=True)
    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        fit_and_evaluate_clustering(X, n_clusters=1)

    with pytest.raises(ValueError, match="strictly less than n_samples"):
        fit_and_evaluate_clustering(X, n_clusters=len(X))


def test_compare_clustering_agreement_identical():
    """Verify agreement between identical partitions returns ARI=1.0 and AMI=1.0."""
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    agr = compare_clustering_agreement(labels, labels)
    assert np.isclose(agr["adjusted_rand_index"], 1.0)
    assert np.isclose(agr["adjusted_mutual_info"], 1.0)


def test_compute_archetype_profiles(synthetic_clustering_df: pd.DataFrame):
    """Verify archetype profiles table aggregates correctly across clusters."""
    labels = np.array([i % 4 for i in range(len(synthetic_clustering_df))])
    profiles = compute_archetype_profiles(synthetic_clustering_df, labels)

    assert len(profiles) == 4
    assert profiles["unit_count"].sum() == len(synthetic_clustering_df)
    for dim in CLUSTERING_DIMENSION_COLUMNS:
        assert f"{dim}_mean" in profiles.columns
        assert f"{dim}_median" in profiles.columns
    assert "pct_dk" in profiles.columns
    assert "pct_udupi" in profiles.columns


def test_log_clustering_experiments_wandb_offline_fallback(tmp_path: Path):
    """Verify graceful offline fallback when W&B is not configured."""
    manifest_file = tmp_path / "test_manifest.json"
    dummy_experiments = [
        {"algorithm": "kmeans", "n_clusters": 4, "silhouette": 0.3348},
        {"algorithm": "agglomerative", "n_clusters": 4, "silhouette": 0.3045},
    ]

    # Without WANDB_API_KEY, should return False, log locally, and not crash
    logged_online = log_clustering_experiments_wandb(
        experiments_data=dummy_experiments,
        fallback_json_path=manifest_file,
    )

    assert logged_online is False
    assert manifest_file.exists()
    with open(manifest_file) as f:
        loaded = json.load(f)
    assert len(loaded) == 2
    assert loaded[0]["algorithm"] == "kmeans"


def test_compute_archetype_profiles_with_population(synthetic_clustering_df: pd.DataFrame):
    """Verify archetype profiles calculate population totals and SC/ST shares when columns exist."""
    df_pop = synthetic_clustering_df.copy()
    df_pop["total_population"] = [1000 + i * 50 for i in range(len(df_pop))]
    df_pop["sc_population"] = [50 + i * 2 for i in range(len(df_pop))]
    df_pop["st_population"] = [25 + i for i in range(len(df_pop))]

    labels = np.array([i % 4 for i in range(len(df_pop))])
    profiles = compute_archetype_profiles(df_pop, labels)

    assert "total_population" in profiles.columns
    assert "pct_of_total_population" in profiles.columns
    assert "sc_st_population" in profiles.columns
    assert "pct_sc_st_population" in profiles.columns
    assert profiles["total_population"].sum() == df_pop["total_population"].sum()
    assert np.isclose(profiles["pct_of_total_population"].sum(), 100.0, atol=0.5)


def test_compute_kmeans_pca(synthetic_clustering_df: pd.DataFrame):
    """Verify PCA computation returns 2 components and substantial explained variance."""
    X, _ = extract_clustering_features(synthetic_clustering_df, standardize=True)
    pca, X_pca, var_exp = compute_kmeans_pca(X, n_components=2, random_state=42)

    assert X_pca.shape == (len(synthetic_clustering_df), 2)
    assert len(var_exp) == 2
    assert 0.0 < sum(var_exp) <= 1.0


def test_plot_kmeans_pca_projection(synthetic_clustering_df: pd.DataFrame, tmp_path: Path):
    """Verify 2D PCA projection plot is saved as a valid non-empty PNG."""
    X, _ = extract_clustering_features(synthetic_clustering_df, standardize=True)
    labels, _, model = fit_and_evaluate_clustering(X, algorithm="kmeans", n_clusters=4, random_state=42)
    out_png = tmp_path / "test_pca.png"

    pca, X_pca, fig = plot_kmeans_pca_projection(
        X_scaled=X,
        labels=labels,
        cluster_centers=model.cluster_centers_,
        output_path=out_png,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_plot_cluster_population_distributions(synthetic_clustering_df: pd.DataFrame, tmp_path: Path):
    """Verify cluster distribution plot generates cleanly."""
    df_pop = synthetic_clustering_df.copy()
    df_pop["total_population"] = [1000 + i * 50 for i in range(len(df_pop))]
    labels = np.array([i % 4 for i in range(len(df_pop))])
    profiles = compute_archetype_profiles(df_pop, labels)

    out_png = tmp_path / "test_dist.png"
    fig = plot_cluster_population_distributions(profiles, output_path=out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_stage_and_save_kmeans_artifacts(synthetic_clustering_df: pd.DataFrame, tmp_path: Path):
    """Verify serialization, staging, and local saving of all standardized K-Means artifacts."""
    X, scaler = extract_clustering_features(synthetic_clustering_df, standardize=True)
    labels, metrics, model = fit_and_evaluate_clustering(X, algorithm="kmeans", n_clusters=4, random_state=42)
    profiles = compute_archetype_profiles(synthetic_clustering_df, labels)

    metadata = {
        "model_identifier": "test-kmeans",
        "algorithm": "KMeans",
        "n_clusters": 4,
        "features": CLUSTERING_DIMENSION_COLUMNS,
        "evaluation_metrics": {
            "silhouette": metrics.silhouette,
            "inertia": metrics.inertia,
        },
    }

    staging_dir = tmp_path / "staging"
    staged_files = stage_kmeans_artifacts(model, scaler, metadata, profiles, staging_dir)

    expected_names = {
        "kmeans_k4_model.joblib",
        "kmeans_scaler.joblib",
        "kmeans_metadata.json",
        "feature_schema.json",
        "archetype_profiles.json",
        "model_card.md",
        "intended_use_and_limitations.md",
    }
    actual_names = {f.name for f in staged_files}
    assert expected_names.issubset(actual_names)
    for f in staged_files:
        assert f.exists()
        assert f.stat().st_size > 0

    # Local save
    local_dir = tmp_path / "models"
    saved = save_kmeans_local_artifacts(model, scaler, metadata, profiles, local_dir)
    assert (local_dir / "kmeans_k4_model.joblib").exists()
    assert (local_dir / "kmeans_scaler.joblib").exists()
    assert (local_dir / "kmeans_metadata.json").exists()
