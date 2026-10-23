#!/usr/bin/env python3
"""Unified CLI runner for online/offline Weights & Biases experiment tracking.

Synchronizes:
1. Vulnerability Clustering Models (K-Means & Agglomerative, k=2..6, agreement, seed stability)
2. Multidimensional Prioritization Sensitivity Scenarios (5 Weighting Paradigms)

Usage:
    # Offline mode (default, no credentials required)
    python scripts/run_wandb_tracking.py --mode offline

    # Online mode (requires W&B authentication via `wandb login` or WANDB_API_KEY)
    python scripts/run_wandb_tracking.py --mode online

    # Run only clustering or prioritization
    python scripts/run_wandb_tracking.py --experiments clustering --mode online
    python scripts/run_wandb_tracking.py --experiments prioritization --mode online
"""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

from src.features.prioritization import (
    DEFAULT_SCENARIOS,
    calculate_prioritization_baseline,
    evaluate_sensitivity_scenarios,
)
from src.models.clustering import (
    ARCHETYPE_DISCLAIMER,
    ARCHETYPE_LABELS_K4,
    compare_clustering_agreement,
    compute_archetype_profiles,
    extract_clustering_features,
    fit_and_evaluate_clustering,
)
from src.tracking.wandb_logger import (
    DEFAULT_PROJECT_NAME,
    is_wandb_authenticated,
    log_clustering_experiments,
    log_prioritization_experiments,
    resolve_wandb_mode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_wandb_tracking")


def run_clustering_tracking(
    input_parquet: Path,
    project_name: str,
    entity: str,
    mode: str,
    manifest_path: Path,
) -> dict:
    """Run clustering experiments and log to W&B."""
    logger.info(f"Loading input baseline dataset from {input_parquet}...")
    gdf = gpd.read_parquet(input_parquet)

    # 1. Feature extraction
    X_scaled, _ = extract_clustering_features(gdf, standardize=True)

    # 2. Benchmark grid
    experiment_records = []
    kmeans_models = {}
    agg_models = {}

    for k in range(2, 7):
        km_labels, km_metrics, km_model = fit_and_evaluate_clustering(
            X_scaled, algorithm="kmeans", n_clusters=k, random_state=42
        )
        kmeans_models[k] = (km_labels, km_metrics, km_model)
        experiment_records.append(asdict(km_metrics))

        agg_labels, agg_metrics, agg_model = fit_and_evaluate_clustering(
            X_scaled, algorithm="agglomerative", n_clusters=k, linkage="ward"
        )
        agg_models[k] = (agg_labels, agg_metrics, agg_model)
        experiment_records.append(asdict(agg_metrics))

    # 3. Inter-algorithm agreement
    agreement_records = []
    for k in range(2, 7):
        agr = compare_clustering_agreement(kmeans_models[k][0], agg_models[k][0])
        agr["k"] = k
        agr["km_silhouette"] = kmeans_models[k][1].silhouette
        agr["agg_silhouette"] = agg_models[k][1].silhouette
        agreement_records.append(agr)

    # 4. Multi-seed stability
    base_labels = kmeans_models[4][0]
    seed_stability_records = []
    for seed in [42, 123, 456, 789, 999, 1024, 2024, 7, 13, 99]:
        km_s_labels, km_s_metrics, _ = fit_and_evaluate_clustering(
            X_scaled, algorithm="kmeans", n_clusters=4, random_state=seed
        )
        agr_seed = compare_clustering_agreement(base_labels, km_s_labels)
        seed_stability_records.append(
            {
                "seed": seed,
                "silhouette": km_s_metrics.silhouette,
                "ari_vs_seed_42": agr_seed["adjusted_rand_index"],
                "ami_vs_seed_42": agr_seed["adjusted_mutual_info"],
            }
        )

    # 5. Selected solution profiles (k=4)
    final_labels = kmeans_models[4][0]
    enriched_gdf = gdf.copy()
    enriched_gdf["vulnerability_archetype_id"] = final_labels
    enriched_gdf["vulnerability_archetype_label"] = [
        ARCHETYPE_LABELS_K4[c]["short_label"] for c in final_labels
    ]
    archetype_profiles_df = compute_archetype_profiles(enriched_gdf, final_labels)

    # 6. Log to W&B
    return log_clustering_experiments(
        experiments_data=experiment_records,
        agreement_data=agreement_records,
        seed_stability_data=seed_stability_records,
        archetype_profiles_df=archetype_profiles_df,
        project_name=project_name,
        entity=entity,
        mode=mode,
        fallback_json_path=manifest_path,
    )


def run_prioritization_tracking(
    input_parquet: Path,
    project_name: str,
    entity: str,
    mode: str,
    summary_path: Path,
) -> dict:
    """Run prioritization sensitivity experiments and log to W&B."""
    logger.info(f"Loading input baseline dataset from {input_parquet}...")
    gdf = gpd.read_parquet(input_parquet)

    # Calculate baseline and sensitivity
    scored_gdf, _ = calculate_prioritization_baseline(gdf, DEFAULT_SCENARIOS["equal_weights"])
    results_df, _, sensitivity_metrics = evaluate_sensitivity_scenarios(gdf, DEFAULT_SCENARIOS)

    for col in [c for c in results_df.columns if c.startswith("score_") or c.startswith("rank_")]:
        scored_gdf[col] = results_df[col].values

    return log_prioritization_experiments(
        sensitivity_metrics=sensitivity_metrics,
        scenarios_dict=DEFAULT_SCENARIOS,
        scored_gdf=scored_gdf,
        project_name=project_name,
        entity=entity,
        mode=mode,
        fallback_json_path=summary_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize Cyclone Preparedness experiments to Weights & Biases.")
    parser.add_argument(
        "--experiments",
        type=str,
        choices=["all", "clustering", "prioritization"],
        default="all",
        help="Which experiment groups to execute and log (default: all).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="W&B sync mode: online (upload to cloud), offline (local wandb runs), disabled (skip wandb).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help=f"Weights & Biases project name (default: {DEFAULT_PROJECT_NAME}).",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Weights & Biases username or team entity.",
    )
    parser.add_argument(
        "--clustering-input",
        type=Path,
        default=Path("data/processed/coastal_prioritization_baseline.parquet"),
        help="Input dataset for clustering experiments.",
    )
    parser.add_argument(
        "--prioritization-input",
        type=Path,
        default=Path("data/processed/coastal_exposure_features.parquet"),
        help="Input dataset for prioritization experiments.",
    )
    args = parser.parse_args()

    effective_mode = resolve_wandb_mode(args.mode)
    logger.info(
        f"Starting W&B Experiment Tracking Runner. Mode: {effective_mode.upper()} | "
        f"Project: {args.project} | Authenticated: {is_wandb_authenticated()}"
    )

    results = {}

    if args.experiments in {"all", "clustering"}:
        logger.info("\n>>> Executing Unsupervised Clustering Experiments Group...")
        results["clustering"] = run_clustering_tracking(
            input_parquet=args.clustering_input,
            project_name=args.project,
            entity=args.entity,
            mode=effective_mode,
            manifest_path=Path("reports/clustering_experiments_manifest.json"),
        )

    if args.experiments in {"all", "prioritization"}:
        logger.info("\n>>> Executing Prioritization Sensitivity Experiments Group...")
        results["prioritization"] = run_prioritization_tracking(
            input_parquet=args.prioritization_input,
            project_name=args.project,
            entity=args.entity,
            mode=effective_mode,
            summary_path=Path("reports/sensitivity_analysis_summary.json"),
        )

    # Print Final Verification Report
    print("\n" + "=" * 95)
    print("WEIGHTS & BIASES (W&B) EXPERIMENT TRACKING SYNCHRONIZATION REPORT")
    print("=" * 95)
    print(f"Project: {args.project}")
    print(f"Operating Mode: {effective_mode.upper()}")
    print(f"Online Synchronized: {any(r.get('synced', False) for r in results.values())}")
    print("-" * 95)

    for group_name, r in results.items():
        print(f"\nGroup: {group_name.upper()}")
        print(f"  - Status: {r.get('status', 'unknown').upper()}")
        print(f"  - Runs Logged: {r.get('run_count', 0)}")
        if r.get("summary_run_url"):
            print(f"  - Summary Run URL: {r['summary_run_url']}")
        runs = r.get("runs", [])
        if runs:
            print(f"  - Individual Runs ({len(runs)}):")
            for item in runs[:5]:
                print(f"    * {item['name']}: {item.get('url') or 'Stored locally in offline mode'}")
            if len(runs) > 5:
                print(f"    * ... and {len(runs) - 5} additional runs.")

    print("\n" + "=" * 95)
    if effective_mode == "offline":
        print(
            "NOTE: Runs were recorded locally in offline mode. To synchronize these runs online when authenticated, run:\n"
            "  wandb sync wandb/offline-run-*\n"
            "Or run directly with online mode:\n"
            "  python scripts/run_wandb_tracking.py --mode online"
        )
        print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
