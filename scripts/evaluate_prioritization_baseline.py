#!/usr/bin/env python3
"""CLI script to evaluate the transparent multidimensional prioritization baseline and sensitivity analysis.

Reads:
- data/processed/coastal_exposure_features.parquet

Outputs:
- data/processed/coastal_prioritization_baseline.parquet
- data/processed/coastal_prioritization_baseline.csv
- reports/sensitivity_analysis_summary.json

Prints:
- Detailed dimension score distributions, baseline rankings, rank correlations, and robust unit analysis.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

from src.features.prioritization import (
    DEFAULT_SCENARIOS,
    PRIORITIZATION_DISCLAIMER,
    calculate_prioritization_baseline,
    evaluate_sensitivity_scenarios,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_prioritization")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multidimensional prioritization baseline.")
    parser.add_argument(
        "--input-features",
        type=Path,
        default=Path("data/processed/coastal_exposure_features.parquet"),
        help="Path to input coastal exposure features parquet.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/processed/coastal_prioritization_baseline.parquet"),
        help="Output GeoParquet path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/coastal_prioritization_baseline.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/sensitivity_analysis_summary.json"),
        help="Output sensitivity metrics JSON path.",
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

    logger.info(f"Loading input features from {args.input_features}...")
    gdf = gpd.read_parquet(args.input_features)

    # 1. Calculate Baseline (Equal Weights)
    baseline_scenario = DEFAULT_SCENARIOS["equal_weights"]
    scored_gdf, baseline_audit = calculate_prioritization_baseline(gdf, baseline_scenario)

    # 2. Run Sensitivity Analysis across all scenarios
    results_df, spearman_mat, sensitivity_metrics = evaluate_sensitivity_scenarios(gdf, DEFAULT_SCENARIOS)

    # Merge scenario scores and ranks into scored_gdf
    scenario_cols = [c for c in results_df.columns if c.startswith("score_") or c.startswith("rank_")]
    for col in scenario_cols:
        scored_gdf[col] = results_df[col].values

    scored_gdf["prioritization_disclaimer"] = PRIORITIZATION_DISCLAIMER

    # 3. Save Output Artifacts
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    scored_gdf.to_parquet(args.output_parquet)
    logger.info(f"Saved {len(scored_gdf)} baseline records to {args.output_parquet}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_df = pd.DataFrame(scored_gdf.drop(columns=["geometry"]))
        csv_df.to_csv(args.output_csv, index=False)
        logger.info(f"Saved tabular CSV to {args.output_csv}")

    from src.tracking.wandb_logger import log_prioritization_experiments, resolve_wandb_mode
    effective_wandb_mode = resolve_wandb_mode(args.wandb_mode)
    wandb_res = log_prioritization_experiments(
        sensitivity_metrics=sensitivity_metrics,
        scenarios_dict=DEFAULT_SCENARIOS,
        scored_gdf=scored_gdf,
        project_name=args.wandb_project,
        entity=args.wandb_entity,
        mode=effective_wandb_mode,
        fallback_json_path=args.output_json,
    )

    # 4. Print Summary Tables
    print("\n" + "=" * 95)
    print("MULTIDIMENSIONAL PRIORITIZATION BASELINE — AUDIT SUMMARY (n=622)")
    print("=" * 95)
    print(f"Baseline Weighting: Equal 25% across Healthcare, Demographics, Infrastructure, and Coastal Proximity")
    print(f"Priority Score Range: {baseline_audit['score_summary']['min']:.4f} to {baseline_audit['score_summary']['max']:.4f} "
          f"(Mean: {baseline_audit['score_summary']['mean']:.4f}, Median: {baseline_audit['score_summary']['median']:.4f})")
    print("-" * 95)
    print("DIMENSION SCORE DISTRIBUTIONS:")
    dim_cols = [
        "dim_healthcare_deficit",
        "dim_demographic_vulnerability",
        "dim_infrastructure_deficit",
        "dim_coastal_exposure",
        "priority_score",
    ]
    dim_summary = scored_gdf[dim_cols].describe().T[["min", "25%", "50%", "mean", "75%", "max", "std"]]
    print(dim_summary.to_string())

    print("\n" + "-" * 95)
    print("TOP 15 HIGHEST PRIORITY UNITS (EQUAL WEIGHTS BASELINE):")
    top_cols = [
        "shrid2",
        "unit_name",
        "district_name",
        "priority_score",
        "dim_healthcare_deficit",
        "dim_demographic_vulnerability",
        "dim_infrastructure_deficit",
        "dim_coastal_exposure",
    ]
    top_15 = scored_gdf.sort_values("priority_score", ascending=False)[top_cols].head(15)
    print(top_15.to_string(index=False))

    print("\n" + "-" * 95)
    print("PAIRWISE SPEARMAN RANK CORRELATIONS ACROSS SENSITIVITY SCENARIOS:")
    print(spearman_mat.to_string())

    print("\n" + "-" * 95)
    print(f"ROBUST HIGH-PRIORITY UNITS (Consistently in Top-100 across ALL 5 Scenarios, n={len(sensitivity_metrics['robust_units_summary'])}):")
    robust_df = pd.DataFrame(sensitivity_metrics["robust_units_summary"])
    if not robust_df.empty:
        print(robust_df[["shrid2", "unit_name", "district_name", "rank_mean", "rank_std", "rank_shift_span"]].to_string(index=False))

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
