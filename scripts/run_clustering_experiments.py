#!/usr/bin/env python3
"""CLI script to run unsupervised vulnerability archetype discovery experiments and W&B logging.

Reads:
- data/processed/coastal_prioritization_baseline.parquet

Generates:
- data/processed/coastal_vulnerability_archetypes.parquet
- data/processed/coastal_vulnerability_archetypes.csv
- reports/clustering_experiments_manifest.json

Prints:
- Algorithm comparison tables, silhouette/Davies-Bouldin scores, inter-algorithm agreement,
  random-seed stability, and descriptive archetype profiles.
"""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from src.models.clustering import (
    ARCHETYPE_DISCLAIMER,
    ARCHETYPE_LABELS_K4,
    CLUSTERING_DIMENSION_COLUMNS,
    ClusteringMetrics,
    compare_clustering_agreement,
    compute_archetype_profiles,
    extract_clustering_features,
    fit_and_evaluate_clustering,
    log_clustering_experiments_wandb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_clustering_experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unsupervised vulnerability clustering experiments.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/processed/coastal_prioritization_baseline.parquet"),
        help="Path to baseline prioritization parquet.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/processed/coastal_vulnerability_archetypes.parquet"),
        help="Path to save enriched archetypes GeoParquet.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/coastal_vulnerability_archetypes.csv"),
        help="Path to save tabular CSV copy.",
    )
    parser.add_argument(
        "--experiments-json",
        type=Path,
        default=Path("reports/clustering_experiments_manifest.json"),
        help="Path to save experiment run manifest.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="Weights & Biases mode (default: checks WANDB_MODE env, falls back to offline/disabled).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cyclone-preparedness-assistant",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name.",
    )
    args = parser.parse_args()

    logger.info(f"Loading input baseline dataset from {args.input_parquet}...")
    gdf = gpd.read_parquet(args.input_parquet)

    # 1. Extract and validate feature matrix X
    X_scaled, scaler = extract_clustering_features(gdf, standardize=True)
    logger.info(f"Extracted standardized feature matrix X with shape: {X_scaled.shape}")

    # 2. Run Experiment Grid across algorithms and cluster counts (k=2..6)
    experiment_records = []
    kmeans_models = {}
    agg_models = {}

    for k in range(2, 7):
        # K-Means
        km_labels, km_metrics, km_model = fit_and_evaluate_clustering(
            X_scaled, algorithm="kmeans", n_clusters=k, random_state=42
        )
        kmeans_models[k] = (km_labels, km_metrics, km_model)
        experiment_records.append(asdict(km_metrics))

        # Agglomerative (Ward linkage)
        agg_labels, agg_metrics, agg_model = fit_and_evaluate_clustering(
            X_scaled, algorithm="agglomerative", n_clusters=k, linkage="ward"
        )
        agg_models[k] = (agg_labels, agg_metrics, agg_model)
        experiment_records.append(asdict(agg_metrics))

    # 3. Algorithm Agreement Comparison (ARI and AMI between KMeans and Agglomerative)
    agreement_records = []
    for k in range(2, 7):
        km_labels = kmeans_models[k][0]
        agg_labels = agg_models[k][0]
        agr = compare_clustering_agreement(km_labels, agg_labels)
        agr["k"] = k
        agr["km_silhouette"] = kmeans_models[k][1].silhouette
        agr["agg_silhouette"] = agg_models[k][1].silhouette
        agreement_records.append(agr)
    agreement_df = pd.DataFrame(agreement_records)

    # 4. Multi-seed stability for KMeans k=4
    seed_stability_records = []
    base_km_k4_labels = kmeans_models[4][0]
    test_seeds = [42, 123, 456, 789, 999, 1024, 2024, 7, 13, 99]

    for seed in test_seeds:
        km_s_labels, km_s_metrics, _ = fit_and_evaluate_clustering(
            X_scaled, algorithm="kmeans", n_clusters=4, random_state=seed
        )
        agr_seed = compare_clustering_agreement(base_km_k4_labels, km_s_labels)
        seed_stability_records.append(
            {
                "seed": seed,
                "silhouette": km_s_metrics.silhouette,
                "ari_vs_seed_42": agr_seed["adjusted_rand_index"],
                "ami_vs_seed_42": agr_seed["adjusted_mutual_info"],
            }
        )
    seed_stability_df = pd.DataFrame(seed_stability_records)

    # 5. Select Best-Supported Solution: K-Means k=4 (Standardized, random_state=42)
    # Highest inter-algorithm agreement (ARI=0.7575), excellent silhouette (0.3348),
    # near-perfect seed stability (mean ARI=0.9902), and clear domain interpretability.
    selected_k = 4
    final_labels = kmeans_models[selected_k][0]

    # Assign archetypes to dataset
    enriched_gdf = gdf.copy()
    enriched_gdf["vulnerability_archetype_id"] = final_labels
    enriched_gdf["vulnerability_archetype_label"] = [
        ARCHETYPE_LABELS_K4[c]["short_label"] for c in final_labels
    ]
    enriched_gdf["archetype_intervention_need"] = [
        ARCHETYPE_LABELS_K4[c]["primary_intervention_need"] for c in final_labels
    ]
    enriched_gdf["clustering_disclaimer"] = ARCHETYPE_DISCLAIMER

    # 6. Compute Comprehensive Archetype Profiles
    archetype_profiles_df = compute_archetype_profiles(enriched_gdf, final_labels)

    # 7. Log Experiments to W&B / Local JSON
    from src.tracking.wandb_logger import log_clustering_experiments, resolve_wandb_mode
    effective_wandb_mode = resolve_wandb_mode(args.wandb_mode)
    wandb_res = log_clustering_experiments(
        experiments_data=experiment_records,
        agreement_data=agreement_records,
        seed_stability_data=seed_stability_records,
        archetype_profiles_df=archetype_profiles_df,
        project_name=args.wandb_project,
        entity=args.wandb_entity,
        mode=effective_wandb_mode,
        fallback_json_path=args.experiments_json,
    )

    # 8. Save Enriched Output Datasets
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    enriched_gdf.to_parquet(args.output_parquet)
    logger.info(f"Saved {len(enriched_gdf)} archetype-labeled records to {args.output_parquet}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_df = pd.DataFrame(enriched_gdf.drop(columns=["geometry"]))
        csv_df.to_csv(args.output_csv, index=False)
        logger.info(f"Saved tabular CSV to {args.output_csv}")

    # 9. Print Formatted Summary Tables
    print("\n" + "=" * 95)
    print("UNSUPERVISED VULNERABILITY ARCHETYPE DISCOVERY — EXPERIMENT REPORT")
    print("=" * 95)
    print("CLUSTERING BENCHMARK RESULTS (k=2..6, Features: 4 Standardized Dimensions):")
    exp_df = pd.DataFrame(experiment_records)
    cols_display = ["algorithm", "n_clusters", "silhouette", "davies_bouldin", "calinski_harabasz", "cluster_balance_ratio", "cluster_sizes"]
    print(exp_df[cols_display].to_string(index=False))

    print("\n" + "-" * 95)
    print("INTER-ALGORITHM AGREEMENT (K-Means vs. Agglomerative Ward):")
    print(agreement_df[["k", "adjusted_rand_index", "adjusted_mutual_info", "km_silhouette", "agg_silhouette"]].to_string(index=False))

    print("\n" + "-" * 95)
    print("K-MEANS k=4 RANDOM SEED STABILITY (10 Distinct Seeds):")
    print(f"Mean Adjusted Rand Index (ARI) across seeds: {seed_stability_df['ari_vs_seed_42'].mean():.4f}")
    print(seed_stability_df.head(5).to_string(index=False))

    print("\n" + "-" * 95)
    print("DISCOVERED VULNERABILITY ARCHETYPES (k=4 Solution Profiles):")
    profile_cols = [
        "archetype_name",
        "unit_count",
        "pct_of_total",
        "pct_dk",
        "dim_healthcare_deficit_mean",
        "dim_demographic_vulnerability_mean",
        "dim_infrastructure_deficit_mean",
        "dim_coastal_exposure_mean",
        "dist_to_coastline_km_mean",
        "dist_to_hospital_km_mean",
    ]
    print(archetype_profiles_df[profile_cols].to_string())

    print("\n" + "-" * 95)
    print("WEIGHTS & BIASES (W&B) TRACKING REPORT:")
    print(f"Tracking Status: {wandb_res.get('status', 'unknown').upper()} (Synchronized: {wandb_res.get('synced', False)})")
    print(f"Project: {wandb_res.get('project', args.wandb_project)}")
    print(f"Total Runs Logged: {wandb_res.get('run_count', 0)}")
    if wandb_res.get("summary_run_url"):
        print(f"Summary Workspace URL: {wandb_res['summary_run_url']}")
    for r in wandb_res.get("runs", []):
        if r.get("url"):
            print(f"  - {r['name']}: {r['url']}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
