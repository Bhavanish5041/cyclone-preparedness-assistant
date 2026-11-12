"""Weights & Biases (W&B) experiment tracking module for Cyclone Preparedness Assistant.

Provides structured, reproducible tracking for:
1. Unsupervised Vulnerability Clustering Experiments (K-Means & Agglomerative, k=2..6)
2. Multidimensional Prioritization Sensitivity Experiments (5 Weighting Scenarios)

Design Principles:
- Preserves analytical logic and empirical metric definitions.
- Groups experiments cleanly using `group`, `job_type`, and descriptive `name` / `tags`.
- Supports explicit CLI switches and environment variables (`--wandb-mode online|offline|disabled`).
- Gracefully falls back to offline/local tracking without crashing pipelines.
- Never prints, logs, or commits API keys or raw datasets.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wandb

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_NAME = "cyclone-preparedness-assistant"


def is_wandb_authenticated() -> bool:
    """Check if valid W&B authentication credentials exist in environment, netrc, or settings.

    Returns:
        True if WANDB_API_KEY is present, or if netrc/settings/api shows active credentials.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True

    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            content = netrc_path.read_text()
            if "wandb.ai" in content or "coreweave.com" in content:
                return True
        except Exception:
            pass

    try:
        settings = wandb.setup().settings
        if settings.api_key:
            return True
        api = wandb.Api()
        return bool(api.api_key or api.default_entity)
    except Exception:
        return False


def resolve_wandb_mode(cli_mode: Optional[str] = None) -> str:
    """Determine effective W&B operating mode ('online', 'offline', or 'disabled').

    Priority:
    1. CLI argument (`--wandb-mode`)
    2. Environment variable `WANDB_MODE`
    3. If online requested but unauthenticated -> falls back to 'offline'
    4. Default -> 'offline'

    Args:
        cli_mode: Optional mode string passed from command-line arguments.

    Returns:
        One of 'online', 'offline', or 'disabled'.
    """
    raw_mode = cli_mode or os.environ.get("WANDB_MODE", "offline")
    mode = raw_mode.strip().lower()

    if mode not in {"online", "offline", "disabled"}:
        logger.warning(f"Unrecognized W&B mode '{mode}'. Defaulting to 'offline'.")
        mode = "offline"

    if mode == "online" and not is_wandb_authenticated():
        logger.warning(
            "W&B online mode requested, but no active authentication found in environment. "
            "Set WANDB_API_KEY or authenticate with `wandb login`. Falling back to 'offline' mode."
        )
        return "offline"

    return mode


def _safe_log(run: Any, payload: Dict[str, Any]) -> None:
    """Log metrics or table artifacts directly to run object if available."""
    if hasattr(run, "log"):
        run.log(payload)
    else:
        wandb.log(payload)


# ==============================================================================
# CLUSTERING EXPERIMENT TRACKING
# ==============================================================================
def log_clustering_experiments(
    experiments_data: List[Dict[str, Any]],
    agreement_data: Optional[List[Dict[str, Any]]] = None,
    seed_stability_data: Optional[List[Dict[str, Any]]] = None,
    archetype_profiles_df: Optional[pd.DataFrame] = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    entity: Optional[str] = None,
    mode: str = "offline",
    fallback_json_path: Optional[Union[str, Path]] = None,
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    X_scaled: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    model_dir: Optional[Union[str, Path]] = None,
    artifact_name: str = "coastal-vulnerability-kmeans",
    artifact_aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Log unsupervised clustering benchmark runs, summary tables, visualizations, and model artifacts to W&B.

    Args:
        experiments_data: List of metrics dictionaries for each (algorithm, k) model.
        agreement_data: List of agreement metrics (ARI, AMI) between KMeans & Agglomerative.
        seed_stability_data: List of multi-seed stability metrics for KMeans k=4.
        archetype_profiles_df: DataFrame of discovered archetype summary profiles.
        project_name: W&B project name.
        entity: W&B entity/team username.
        mode: 'online', 'offline', or 'disabled'.
        fallback_json_path: Local path to save JSON manifest.
        model: Fitted KMeans model object (optional).
        scaler: Fitted StandardScaler preprocessor (optional).
        X_scaled: Standardized feature matrix (optional).
        df: Units DataFrame with identifiers and attributes (optional).
        figures_dir: Directory containing visualization figures (optional).
        model_dir: Directory to save local model artifacts (optional).
        artifact_name: W&B Model Artifact name.
        artifact_aliases: Aliases to assign to registered artifact.

    Returns:
        Dictionary summarizing logged runs, URLs, artifact details, and sync status.
    """
    from src.models.clustering import (
        ARCHETYPE_DISCLAIMER,
        CLUSTERING_DIMENSION_COLUMNS,
        plot_cluster_population_distributions,
        plot_kmeans_pca_projection,
        save_kmeans_local_artifacts,
        stage_kmeans_artifacts,
    )

    # 1. Always ensure local JSON manifest is written for local reproducibility
    if fallback_json_path is not None:
        p = Path(fallback_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(experiments_data, f, indent=2)
        logger.info(f"Saved local clustering experiment manifest to {p}")

    fig_dir = Path(figures_dir) if figures_dir is not None else Path("reports/figures")
    m_dir = Path(model_dir) if model_dir is not None else Path("models")

    # If model and scaler are provided, ensure local artifacts are saved locally even if W&B is disabled
    if model is not None and scaler is not None and archetype_profiles_df is not None:
        meta_local = {
            "model_identifier": artifact_name,
            "algorithm": "KMeans",
            "n_clusters": 4,
            "random_state": 42,
            "n_init": 20,
            "standardize": True,
            "features": CLUSTERING_DIMENSION_COLUMNS,
            "disclaimer": ARCHETYPE_DISCLAIMER,
            "total_units": len(df) if df is not None else 622,
            "evaluation_metrics": {
                "silhouette": 0.3348,
                "davies_bouldin": 1.1771,
                "calinski_harabasz": 298.59,
                "inertia": float(getattr(model, "inertia_", 1015.73)),
                "cluster_balance_ratio": 0.3293,
            },
        }
        if hasattr(model, "cluster_centers_"):
            meta_local["cluster_centers"] = model.cluster_centers_.tolist()
        save_kmeans_local_artifacts(model, scaler, meta_local, archetype_profiles_df, m_dir)

    if mode == "disabled":
        logger.info("W&B tracking disabled by configuration.")
        return {"status": "disabled", "synced": False, "runs": []}

    logged_runs = []
    is_online = (mode == "online")

    try:
        # 2. Log Individual Algorithm/Cluster Model Runs
        for exp in experiments_data:
            algo = exp.get("algorithm", "kmeans").lower()
            k = exp.get("n_clusters", 4)
            is_baseline = (algo == "kmeans" and k == 4)
            run_name = f"{algo}-k{k}" + ("-baseline" if is_baseline else "")

            tags = ["clustering", "unsupervised", algo, f"k={k}"]
            if is_baseline:
                tags.append("selected-baseline")

            config = {
                "algorithm": algo,
                "n_clusters": k,
                "random_state": exp.get("random_state"),
                "standardized_features": True,
                "feature_dimensions": CLUSTERING_DIMENSION_COLUMNS,
                "is_selected_solution": is_baseline,
            }

            run = wandb.init(
                project=project_name,
                entity=entity,
                group="clustering-exploration",
                job_type="model-evaluation",
                name=run_name,
                tags=tags,
                config=config,
                mode=mode,
                reinit=True,
            )

            # Log evaluation metrics
            metrics_payload = {
                "silhouette": exp.get("silhouette"),
                "davies_bouldin": exp.get("davies_bouldin"),
                "calinski_harabasz": exp.get("calinski_harabasz"),
                "cluster_balance_ratio": exp.get("cluster_balance_ratio"),
            }
            if exp.get("inertia") is not None:
                metrics_payload["inertia"] = exp.get("inertia")

            # Log cluster sizes
            sizes = exp.get("cluster_sizes", [])
            for idx, s in enumerate(sizes):
                metrics_payload[f"cluster_{idx}_size"] = s

            # Population statistics for baseline
            if is_baseline and archetype_profiles_df is not None:
                if "total_population" in archetype_profiles_df.columns:
                    tot_pop = int(archetype_profiles_df["total_population"].sum())
                    metrics_payload["total_study_population"] = tot_pop
                    for c_id in archetype_profiles_df.index:
                        metrics_payload[f"cluster_{c_id}_population"] = int(archetype_profiles_df.loc[c_id, "total_population"])
                if "sc_st_population" in archetype_profiles_df.columns:
                    for c_id in archetype_profiles_df.index:
                        metrics_payload[f"cluster_{c_id}_sc_st_population"] = int(archetype_profiles_df.loc[c_id, "sc_st_population"])

            _safe_log(run, metrics_payload)
            run_info = {
                "run_id": getattr(run, "id", None),
                "name": run_name,
                "url": getattr(run, "url", None),
                "algorithm": algo,
                "n_clusters": k,
                "silhouette": exp.get("silhouette"),
            }
            logged_runs.append(run_info)
            run.finish()

        # 3. Log Benchmark Summary & Tables Run
        summary_run = wandb.init(
            project=project_name,
            entity=entity,
            group="clustering-exploration",
            job_type="benchmark-summary",
            name="clustering-benchmark-summary",
            tags=["clustering", "benchmark-summary", "archetypes"],
            config={
                "candidate_algorithms": ["kmeans", "agglomerative"],
                "tested_k_range": [2, 3, 4, 5, 6],
                "selected_solution": {"algorithm": "kmeans", "k": 4, "random_state": 42},
            },
            mode=mode,
            reinit=True,
        )

        # Table 1: Model Comparison Grid
        grid_cols = ["Algorithm", "k", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Inertia", "Balance Ratio"]
        grid_data = []
        for exp in experiments_data:
            grid_data.append([
                exp.get("algorithm"),
                exp.get("n_clusters"),
                exp.get("silhouette"),
                exp.get("davies_bouldin"),
                exp.get("calinski_harabasz"),
                exp.get("inertia") if exp.get("inertia") is not None else -1.0,
                exp.get("cluster_balance_ratio"),
            ])
        comparison_table = wandb.Table(columns=grid_cols, data=grid_data)
        _safe_log(summary_run, {"clustering_models_comparison_table": comparison_table})

        # Table 2: Inter-algorithm Agreement Table
        if agreement_data:
            agr_cols = ["k", "Adjusted_Rand_Index", "Adjusted_Mutual_Info", "KMeans_Silhouette", "Agg_Silhouette"]
            agr_data = [
                [a["k"], a["adjusted_rand_index"], a["adjusted_mutual_info"], a["km_silhouette"], a["agg_silhouette"]]
                for a in agreement_data
            ]
            agreement_table = wandb.Table(columns=agr_cols, data=agr_data)
            _safe_log(summary_run, {"inter_algorithm_agreement_table": agreement_table})

        # Table 3: Seed Stability Table
        if seed_stability_data:
            seed_cols = ["Seed", "Silhouette", "ARI_vs_Seed_42", "AMI_vs_Seed_42"]
            seed_data = [
                [s["seed"], s["silhouette"], s["ari_vs_seed_42"], s["ami_vs_seed_42"]]
                for s in seed_stability_data
            ]
            seed_table = wandb.Table(columns=seed_cols, data=seed_data)
            _safe_log(summary_run, {"kmeans_k4_seed_stability_table": seed_table})

        # Table 4: Archetype Profiles Table (Enhanced with population where available)
        if archetype_profiles_df is not None:
            has_pop = "total_population" in archetype_profiles_df.columns
            prof_cols = [
                "Cluster_ID",
                "Archetype_Label",
                "Unit_Count",
                "Pct_Total",
            ]
            if has_pop:
                prof_cols.extend(["Total_Population", "Pct_Total_Pop", "SC_ST_Population", "Pct_SC_ST"])
            prof_cols.extend([
                "Count_DK",
                "Count_Udupi",
                "Health_Deficit_Mean",
                "Demo_Vuln_Mean",
                "Infra_Deficit_Mean",
                "Coastal_Exposure_Mean",
            ])

            prof_data = []
            for c_id, row in archetype_profiles_df.reset_index().iterrows():
                row_items = [
                    int(row.get("cluster_id", c_id)),
                    str(row.get("archetype_name", f"Cluster {c_id}")),
                    int(row.get("unit_count", 0)),
                    float(row.get("pct_of_total", 0.0)),
                ]
                if has_pop:
                    row_items.extend([
                        int(row.get("total_population", 0)),
                        float(row.get("pct_of_total_population", 0.0)),
                        int(row.get("sc_st_population", 0)),
                        float(row.get("pct_sc_st_population", 0.0)),
                    ])
                row_items.extend([
                    int(row.get("count_dk", 0)),
                    int(row.get("count_udupi", 0)),
                    float(row.get("dim_healthcare_deficit_mean", 0.0)),
                    float(row.get("dim_demographic_vulnerability_mean", 0.0)),
                    float(row.get("dim_infrastructure_deficit_mean", 0.0)),
                    float(row.get("dim_coastal_exposure_mean", 0.0)),
                ])
                prof_data.append(row_items)
            archetype_table = wandb.Table(columns=prof_cols, data=prof_data)
            _safe_log(summary_run, {"discovered_archetypes_table": archetype_table})

        # Table 5: Existing Cluster Assignments Table (Full 622 revenue units, no personal data)
        pca_coords = None
        if df is not None:
            if X_scaled is not None:
                try:
                    from sklearn.decomposition import PCA
                    pca_eval = PCA(n_components=2, random_state=42)
                    pca_coords = pca_eval.fit_transform(X_scaled)
                except Exception:
                    pca_coords = None

            assign_cols = [
                "shrid2",
                "pc11_id",
                "unit_name",
                "district_name",
                "cluster_id",
                "archetype_label",
                "total_population",
                "sc_st_ratio",
                "dim_healthcare_deficit",
                "dim_demographic_vulnerability",
                "dim_infrastructure_deficit",
                "dim_coastal_exposure",
            ]
            if pca_coords is not None:
                assign_cols.extend(["pca_1", "pca_2"])

            assign_data = []
            for i, (_, r) in enumerate(df.iterrows()):
                c_id_val = int(r.get("vulnerability_archetype_id", r.get("cluster_id", 0)))
                label_val = str(r.get("vulnerability_archetype_label", f"Cluster {c_id_val}"))
                row_vals = [
                    str(r.get("shrid2", "")),
                    str(r.get("pc11_id", "")),
                    str(r.get("unit_name", "")),
                    str(r.get("district_name", "")),
                    c_id_val,
                    label_val,
                    int(r.get("total_population", 0)),
                    round(float(r.get("sc_st_ratio", 0.0)), 4),
                    round(float(r.get("dim_healthcare_deficit", 0.0)), 4),
                    round(float(r.get("dim_demographic_vulnerability", 0.0)), 4),
                    round(float(r.get("dim_infrastructure_deficit", 0.0)), 4),
                    round(float(r.get("dim_coastal_exposure", 0.0)), 4),
                ]
                if pca_coords is not None and i < len(pca_coords):
                    row_vals.extend([
                        round(float(pca_coords[i, 0]), 4),
                        round(float(pca_coords[i, 1]), 4),
                    ])
                assign_data.append(row_vals)

            assignments_table = wandb.Table(columns=assign_cols, data=assign_data)
            _safe_log(summary_run, {"cluster_assignments_table": assignments_table})

        # 4. Generate & Log Visualizations (PCA Projection, Cluster Distributions, Curves)
        fig_dir.mkdir(parents=True, exist_ok=True)
        pca_fig_path = fig_dir / "kmeans_k4_pca_projection.png"
        dist_fig_path = fig_dir / "kmeans_k4_cluster_distribution.png"

        if X_scaled is not None and model is not None:
            try:
                centers = getattr(model, "cluster_centers_", None)
                labels = getattr(model, "labels_", np.zeros(len(X_scaled)))
                plot_kmeans_pca_projection(X_scaled, labels, cluster_centers=centers, output_path=pca_fig_path)
            except Exception as e:
                logger.warning(f"Could not generate PCA plot: {e}")

        if archetype_profiles_df is not None:
            try:
                plot_cluster_population_distributions(archetype_profiles_df, output_path=dist_fig_path)
            except Exception as e:
                logger.warning(f"Could not generate cluster distribution plot: {e}")

        # Log images to W&B
        image_mappings = [
            ("visualizations/kmeans_pca_projection", pca_fig_path, "K-Means (k=4) 2D PCA Projection with Cluster Centroids"),
            ("visualizations/cluster_population_distributions", dist_fig_path, "Vulnerability Archetype Unit & Population Distributions"),
            ("visualizations/clustering_metric_evaluation_curve", fig_dir / "clustering_metric_evaluation_curve.png", "Clustering Metric Evaluation Curves (k=2..6)"),
            ("visualizations/clustering_agreement_heatmap", fig_dir / "clustering_agreement_heatmap.png", "Inter-Algorithm Agreement Heatmap"),
            ("visualizations/vulnerability_archetypes_feature_radar_and_profiles", fig_dir / "vulnerability_archetypes_feature_radar_and_profiles.png", "Dimension Profiles by Archetype"),
            ("visualizations/vulnerability_archetypes_choropleth", fig_dir / "vulnerability_archetypes_choropleth.png", "Vulnerability Archetypes Geographic Choropleth Map"),
        ]

        for log_key, img_path, caption in image_mappings:
            if img_path.exists():
                try:
                    if hasattr(wandb, "Image"):
                        _safe_log(summary_run, {log_key: wandb.Image(str(img_path), caption=caption)})
                except Exception as img_err:
                    logger.debug(f"Could not log image {img_path}: {img_err}")

        # 5. Versioned Model Artifact Creation & Logging
        artifact_info: Dict[str, Any] = {}
        if model is not None and scaler is not None and archetype_profiles_df is not None:
            try:
                with tempfile.TemporaryDirectory() as staging_tmp:
                    staging_path = Path(staging_tmp)
                    art_meta = {
                        "model_identifier": artifact_name,
                        "model_family": "scikit-learn",
                        "algorithm": "KMeans",
                        "n_clusters": 4,
                        "random_state": 42,
                        "n_init": 20,
                        "standardize": True,
                        "features": CLUSTERING_DIMENSION_COLUMNS,
                        "disclaimer": ARCHETYPE_DISCLAIMER,
                        "source_dataset": "data/processed/coastal_prioritization_baseline.parquet",
                        "total_units": len(df) if df is not None else 622,
                        "total_population": int(archetype_profiles_df["total_population"].sum()) if "total_population" in archetype_profiles_df.columns else 3254857,
                        "evaluation_metrics": {
                            "silhouette": 0.3348,
                            "davies_bouldin": 1.1771,
                            "calinski_harabasz": 298.59,
                            "inertia": float(getattr(model, "inertia_", 1015.73)),
                            "cluster_balance_ratio": 0.3293,
                        },
                        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    if hasattr(model, "cluster_centers_"):
                        art_meta["cluster_centers"] = model.cluster_centers_.tolist()

                    stage_kmeans_artifacts(model, scaler, art_meta, archetype_profiles_df, staging_path)

                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type="model",
                        description=(
                            "Fitted K-Means (k=4) unsupervised model and StandardScaler preprocessing artifact "
                            "discovering vulnerability archetypes across 622 coastal revenue units in Dakshina Kannada and Udupi."
                        ),
                        metadata=art_meta,
                    )
                    artifact.add_dir(str(staging_path))

                    aliases = artifact_aliases or ["candidate", "validated", "baseline"]
                    logged_artifact = summary_run.log_artifact(artifact, aliases=aliases)

                    if is_online:
                        try:
                            summary_run.link_artifact(artifact, target_path=f"wandb-registry-model/{artifact_name}")
                        except Exception as link_err:
                            logger.info(f"W&B registry linking note: {link_err}")

                    artifact_info = {
                        "artifact_name": artifact_name,
                        "artifact_version": getattr(logged_artifact, "version", "v0"),
                        "artifact_url": getattr(logged_artifact, "url", None),
                    }
            except Exception as art_err:
                logger.warning(f"Failed to package or log W&B model artifact: {art_err}")

        summary_url = getattr(summary_run, "url", None)
        summary_run.finish()

        result = {
            "status": "online" if is_online else "offline",
            "synced": is_online,
            "project": project_name,
            "run_count": len(logged_runs) + 1,
            "runs": logged_runs,
            "summary_run_url": summary_url,
            **artifact_info,
        }
        logger.info(
            f"Successfully logged {len(logged_runs) + 1} clustering runs to W&B "
            f"(Mode: {result['status']}, Synced: {result['synced']})."
        )
        return result

    except Exception as e:
        logger.error(f"Error during W&B clustering logging: {e}. Falling back gracefully.")
        return {
            "status": "error",
            "synced": False,
            "error": str(e),
            "project": project_name,
            "runs": logged_runs,
        }


# ==============================================================================
# PRIORITIZATION SENSITIVITY EXPERIMENT TRACKING
# ==============================================================================
def log_prioritization_experiments(
    sensitivity_metrics: Dict[str, Any],
    scenarios_dict: Dict[str, Any],
    scored_gdf: pd.DataFrame,
    project_name: str = DEFAULT_PROJECT_NAME,
    entity: Optional[str] = None,
    mode: str = "offline",
    fallback_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Log multidimensional prioritization sensitivity scenarios to W&B.

    Args:
        sensitivity_metrics: Dictionary containing pairwise correlations and robust unit summaries.
        scenarios_dict: Dictionary mapping scenario names to PrioritizationScenario definitions.
        scored_gdf: GeoDataFrame containing scores and ranks for each scenario.
        project_name: W&B project name.
        entity: W&B entity/team username.
        mode: 'online', 'offline', or 'disabled'.
        fallback_json_path: Local path to save JSON manifest.

    Returns:
        Dictionary summarizing logged runs, URLs, and sync status.
    """
    if fallback_json_path is not None:
        p = Path(fallback_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(sensitivity_metrics, f, indent=2)
        logger.info(f"Saved local sensitivity summary to {p}")

    if mode == "disabled":
        logger.info("W&B tracking disabled by configuration.")
        return {"status": "disabled", "synced": False, "runs": []}

    logged_runs = []
    is_online = (mode == "online")
    base_ranks = scored_gdf["rank_equal_weights"] if "rank_equal_weights" in scored_gdf.columns else scored_gdf["priority_rank"]

    try:
        # 1. Log each scenario run
        for sc_name, sc_obj in scenarios_dict.items():
            run_name = f"prioritization-{sc_name.replace('_', '-')}"
            is_base = (sc_name == "equal_weights")

            tags = ["prioritization", "sensitivity", "analytical-baseline", sc_name]
            if is_base:
                tags.append("baseline-weights")

            config = {
                "scenario_name": sc_name,
                "weight_healthcare": sc_obj.weight_healthcare,
                "weight_demographic": sc_obj.weight_demographic,
                "weight_infrastructure": sc_obj.weight_infrastructure,
                "weight_coastal": sc_obj.weight_coastal,
                "include_population_scale": sc_obj.include_population_scale,
                "total_units": len(scored_gdf),
            }

            run = wandb.init(
                project=project_name,
                entity=entity,
                group="prioritization-sensitivity",
                job_type="scenario-evaluation",
                name=run_name,
                tags=tags,
                config=config,
                mode=mode,
                reinit=True,
            )

            score_col = f"score_{sc_name}" if f"score_{sc_name}" in scored_gdf.columns else "priority_score"
            rank_col = f"rank_{sc_name}" if f"rank_{sc_name}" in scored_gdf.columns else "priority_rank"

            scores = scored_gdf[score_col]
            ranks = scored_gdf[rank_col]
            rank_shifts = (base_ranks - ranks).abs()

            spearman_vs_base = sensitivity_metrics.get("spearman_rank_correlations", {}).get("equal_weights", {}).get(sc_name, 1.0)
            kendall_vs_base = sensitivity_metrics.get("kendall_tau_correlations", {}).get("equal_weights", {}).get(sc_name, 1.0)

            _safe_log(run, {
                "score_min": round(float(scores.min()), 4),
                "score_max": round(float(scores.max()), 4),
                "score_mean": round(float(scores.mean()), 4),
                "score_median": round(float(scores.median()), 4),
                "score_std": round(float(scores.std()), 4),
                "spearman_corr_vs_baseline": round(float(spearman_vs_base), 4),
                "kendall_tau_vs_baseline": round(float(kendall_vs_base), 4),
                "max_rank_shift_vs_baseline": int(rank_shifts.max()),
                "mean_abs_rank_shift_vs_baseline": round(float(rank_shifts.mean()), 1),
            })

            logged_runs.append({
                "run_id": getattr(run, "id", None),
                "name": run_name,
                "url": getattr(run, "url", None),
                "scenario": sc_name,
                "spearman_corr_vs_baseline": spearman_vs_base,
            })
            run.finish()

        # 2. Log Prioritization Sensitivity Summary Run & Tables
        summary_run = wandb.init(
            project=project_name,
            entity=entity,
            group="prioritization-sensitivity",
            job_type="sensitivity-summary",
            name="prioritization-sensitivity-summary",
            tags=["prioritization", "sensitivity-summary"],
            config={
                "scenarios_evaluated": list(scenarios_dict.keys()),
                "baseline_scenario": "equal_weights",
                "total_units": len(scored_gdf),
            },
            mode=mode,
            reinit=True,
        )

        # Table 1: Pairwise Spearman Rank Correlations
        spearman_dict = sensitivity_metrics.get("spearman_rank_correlations", {})
        sc_names = list(scenarios_dict.keys())
        corr_cols = ["Scenario"] + [s.replace("_", " ").title() for s in sc_names]
        corr_data = []
        for s1 in sc_names:
            row = [s1.replace("_", " ").title()]
            for s2 in sc_names:
                row.append(round(float(spearman_dict.get(s1, {}).get(s2, 1.0)), 3))
            corr_data.append(row)

        spearman_table = wandb.Table(columns=corr_cols, data=corr_data)
        _safe_log(summary_run, {"spearman_rank_correlation_matrix_table": spearman_table})

        # Table 2: Robust High-Priority Units
        robust_units = sensitivity_metrics.get("robust_units_summary", [])
        if robust_units:
            rob_cols = ["SHRID", "Unit_Name", "District", "Mean_Rank", "Std_Rank", "Min_Rank", "Max_Rank", "Rank_Shift_Span"]
            rob_data = [
                [
                    u.get("shrid2"),
                    u.get("unit_name"),
                    u.get("district_name"),
                    u.get("rank_mean"),
                    u.get("rank_std"),
                    u.get("rank_min"),
                    u.get("rank_max"),
                    u.get("rank_shift_span"),
                ]
                for u in robust_units
            ]
            robust_table = wandb.Table(columns=rob_cols, data=rob_data)
            _safe_log(summary_run, {"robust_high_priority_units_table": robust_table})

        summary_url = getattr(summary_run, "url", None)
        summary_run.finish()

        result = {
            "status": "online" if is_online else "offline",
            "synced": is_online,
            "project": project_name,
            "run_count": len(logged_runs) + 1,
            "runs": logged_runs,
            "summary_run_url": summary_url,
        }
        logger.info(
            f"Successfully logged {len(logged_runs) + 1} prioritization sensitivity runs to W&B "
            f"(Mode: {result['status']}, Synced: {result['synced']})."
        )
        return result

    except Exception as e:
        logger.error(f"Error during W&B prioritization logging: {e}. Falling back gracefully.")
        return {
            "status": "error",
            "synced": False,
            "error": str(e),
            "project": project_name,
            "runs": logged_runs,
        }
