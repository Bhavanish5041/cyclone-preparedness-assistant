"""Unit tests for the complete K-Means model selection and W&B tracking workflow."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scripts.evaluate_kmeans_wandb import (
    evaluate_multi_seed_stability,
    generate_experiment_summary,
    plot_combined_evaluation_curves,
    plot_elbow_curve,
    plot_metric_curve,
    plot_cluster_size_comparison,
    plot_cluster_profile_radar_and_bars,
    plot_cluster_population_barchart,
    run_kmeans_model_selection_experiment,
    stage_candidate_artifact,
)
from src.models.clustering import (
    CLUSTERING_DIMENSION_COLUMNS,
    compute_archetype_profiles,
    extract_clustering_features,
    fit_and_evaluate_clustering,
)


@pytest.fixture
def synthetic_features():
    """Generate synthetic clustering dataset for rapid test execution."""
    np.random.seed(42)
    n = 80
    df = pd.DataFrame(
        {
            "shrid2": [f"shrid_{i}" for i in range(n)],
            "unit_name": [f"Unit_{i}" for i in range(n)],
            "district_name": ["Dakshina Kannada" if i % 2 == 0 else "Udupi" for i in range(n)],
            "total_population": np.random.randint(1000, 20000, n),
            "sc_population": np.random.randint(50, 1000, n),
            "st_population": np.random.randint(20, 500, n),
            "dim_healthcare_deficit": np.random.uniform(0.1, 0.9, n),
            "dim_demographic_vulnerability": np.random.uniform(0.0, 0.8, n),
            "dim_infrastructure_deficit": np.random.uniform(0.2, 1.0, n),
            "dim_coastal_exposure": np.random.uniform(0.0, 1.0, n),
        }
    )
    X, scaler = extract_clustering_features(df, standardize=True)
    return X, scaler, df


def test_evaluate_multi_seed_stability(synthetic_features):
    """Verify multi-seed stability produces valid bounded metrics and seed records."""
    X, _, _ = synthetic_features
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    base_labels = km.fit_predict(X)
    seeds = [42, 123, 456]

    summary, records = evaluate_multi_seed_stability(
        X_scaled=X,
        k=3,
        base_labels=base_labels,
        test_seeds=seeds,
        n_init=10,
    )

    assert len(records) == len(seeds)
    assert records[0]["seed"] == 42
    assert records[0]["ari_vs_seed_42"] == 1.0
    assert records[0]["ami_vs_seed_42"] == 1.0
    assert 0.0 <= summary["stability_mean_ari"] <= 1.0
    assert 0.0 <= summary["stability_min_ari"] <= 1.0
    assert summary["stability_seed_count"] == 3


def test_stage_candidate_artifact(synthetic_features, tmp_path: Path):
    """Verify artifact staging creates all 7 required files with valid contents."""
    X, scaler, df = synthetic_features
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    profiles_df = compute_archetype_profiles(df, labels, k=3)

    metrics = {
        "inertia": float(km.inertia_),
        "silhouette": 0.35,
        "davies_bouldin": 1.12,
        "calinski_harabasz": 250.0,
        "cluster_balance_ratio": 0.6,
        "min_cluster_size": 20,
        "max_cluster_size": 35,
        "runtime_seconds": 0.05,
    }
    stability = {
        "stability_mean_ari": 0.99,
        "stability_min_ari": 0.97,
        "stability_std_ari": 0.01,
        "stability_mean_ami": 0.98,
    }

    staged_files = stage_candidate_artifact(
        k=3,
        model=km,
        scaler=scaler,
        metrics=metrics,
        stability=stability,
        profiles_df=profiles_df,
        staging_dir=tmp_path,
    )

    expected_names = {
        "kmeans_k3_model.joblib",
        "kmeans_scaler.joblib",
        "kmeans_metadata.json",
        "feature_schema.json",
        "archetype_profiles.json",
        "model_card.md",
        "intended_use_and_limitations.md",
    }
    created_names = {p.name for p in staged_files}
    assert expected_names == created_names

    # Check valid JSON
    with open(tmp_path / "kmeans_metadata.json") as f:
        meta = json.load(f)
        assert meta["model_identifier"] == "kmeans-candidate-k3"
        assert meta["n_clusters"] == 3

    with open(tmp_path / "feature_schema.json") as f:
        schema = json.load(f)
        assert schema["dimensions"] == 4

    # Check Model Card
    card_text = (tmp_path / "model_card.md").read_text()
    assert "kmeans-candidate-k3" in card_text
    assert "Quantitative Evaluation Metrics" in card_text


def test_visualizations_generation(synthetic_features, tmp_path: Path):
    """Verify all visualization routines generate valid PNG files."""
    X, _, df = synthetic_features
    labels, _, model = fit_and_evaluate_clustering(X, algorithm="kmeans", n_clusters=3, random_state=42)
    profiles_df = compute_archetype_profiles(df, labels, k=3)

    p_elbow = tmp_path / "elbow.png"
    p_metric = tmp_path / "metric.png"
    p_combined = tmp_path / "combined.png"
    p_radar = tmp_path / "radar.png"
    p_pop = tmp_path / "pop.png"
    p_size = tmp_path / "sizes.png"

    plot_elbow_curve([2, 3, 4], [150.0, 100.0, 80.0], p_elbow)
    plot_metric_curve([2, 3, 4], [0.3, 0.4, 0.35], "Silhouette", "Score", "#2ca02c", p_metric, higher_is_better=True)
    plot_combined_evaluation_curves([2, 3, 4], [150.0, 100.0, 80.0], [0.3, 0.4, 0.35], [1.3, 1.1, 1.2], [200.0, 300.0, 280.0], p_combined)
    plot_cluster_profile_radar_and_bars(profiles_df, k=3, output_path=p_radar)
    plot_cluster_population_barchart(profiles_df, k=3, output_path=p_pop)
    plot_cluster_size_comparison(
        {2: {"cluster_sizes": [30, 50], "cluster_balance_ratio": 0.6}, 3: {"cluster_sizes": [20, 30, 30], "cluster_balance_ratio": 0.66}},
        p_size,
    )

    for p in [p_elbow, p_metric, p_combined, p_radar, p_pop, p_size]:
        assert p.exists()
        assert p.stat().st_size > 1000
