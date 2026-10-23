#!/usr/bin/env python3
"""CLI script to generate publication-quality figures for the prioritization baseline and sensitivity analysis.

Reads:
- data/processed/coastal_prioritization_baseline.parquet
- data/interim/coastal_karnataka_ne_10m_coastline.geojson

Generates:
- reports/figures/prioritization_dimension_choropleths.png
- reports/figures/prioritization_baseline_and_sensitivity_maps.png
- reports/figures/prioritization_sensitivity_rank_stability.png
"""

import argparse
import logging
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("plot_prioritization_figures")


def plot_dimension_choropleths(
    gdf: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot 2x2 grid of choropleths for the four separate dimension scores."""
    logger.info("Generating 2x2 dimension choropleth grid...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    dims = [
        ("dim_healthcare_deficit", "Healthcare Accessibility Deficit", "Reds", axes[0, 0]),
        ("dim_demographic_vulnerability", "Demographic Vulnerability (SC/ST Ratio)", "Purples", axes[0, 1]),
        ("dim_infrastructure_deficit", "Infrastructure Deficit (Water/Road/Drainage)", "YlOrBr", axes[1, 0]),
        ("dim_coastal_exposure", "Coastal Hazard Exposure (Inverted Proximity)", "Blues", axes[1, 1]),
    ]

    for col, title, cmap, ax in dims:
        gdf.plot(
            column=col,
            cmap=cmap,
            linewidth=0.3,
            edgecolor="#444444",
            vmin=0.0,
            vmax=1.0,
            legend=True,
            legend_kwds={"label": f"{title} (0 to 1)", "orientation": "horizontal", "shrink": 0.75, "pad": 0.05},
            ax=ax,
        )
        coastline_gdf.plot(ax=ax, color="#0055ff", linewidth=1.8, label="Coastline")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle(
        "Four Independent Preparedness Prioritization Dimensions — Coastal Karnataka (n=622)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved dimension choropleths to {output_path}")


def plot_baseline_and_sensitivity_maps(
    gdf: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot comparative choropleths showing priority scores under three weighting paradigms."""
    logger.info("Generating comparative priority score maps...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8.5))

    maps = [
        ("priority_score", "Equal Weights Baseline (25% each)", "viridis", ax1),
        ("score_hazard_focused", "Hazard-Dominant (50% Coastal Proximity)", "magma", ax2),
        ("score_access_focused", "Access-Dominant (40% Health, 35% Infra)", "cividis", ax3),
    ]

    for col, title, cmap, ax in maps:
        gdf.plot(
            column=col,
            cmap=cmap,
            linewidth=0.3,
            edgecolor="#333333",
            vmin=0.2,
            vmax=0.8,
            legend=True,
            legend_kwds={"label": f"Priority Score [0.0 - 1.0]", "orientation": "horizontal", "shrink": 0.8, "pad": 0.05},
            ax=ax,
        )
        coastline_gdf.plot(ax=ax, color="#00ffff" if cmap != "cividis" else "#0055ff", linewidth=1.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle(
        "Sensitivity of Spatial Prioritization to Alternative Weighting Paradigms\n"
        "(Highlighting the Divergence Between Coastal Surge Exposure vs. Foothill Healthcare Isolation)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparative maps to {output_path}")


def plot_sensitivity_rank_stability(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot correlation matrix across scenario ranks and rank shift span distribution."""
    logger.info("Generating rank stability and correlation figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    scenario_cols = [
        "rank_equal_weights",
        "rank_hazard_focused",
        "rank_access_focused",
        "rank_no_coastal",
        "rank_with_population_scale",
    ]
    labels = [
        "Equal (25% ea)",
        "Hazard-Heavy (50% Coast)",
        "Access-Heavy (Health+Infra)",
        "No-Coastal (Inland)",
        "With Pop Scale",
    ]

    # 1. Rank correlation heatmap
    corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, c1 in enumerate(scenario_cols):
        for j, c2 in enumerate(scenario_cols):
            sr, _ = spearmanr(gdf[c1], gdf[c2])
            corr.iloc[i, j] = sr

    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        center=0.5,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Spearman Rank Correlation ($r_s$)"},
        ax=ax1,
    )
    ax1.set_title("Rank Correlation Across Weighting Schemes ($r_s$)", fontsize=12, fontweight="bold")

    # 2. Rank shift span histogram
    rank_matrix = gdf[scenario_cols]
    shift_span = rank_matrix.max(axis=1) - rank_matrix.min(axis=1)

    sns.histplot(shift_span, bins=30, kde=True, color="#2b5c8f", ax=ax2)
    ax2.axvline(shift_span.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median Shift: {shift_span.median():.0f} ranks")
    ax2.axvline(shift_span.mean(), color="orange", linestyle=":", linewidth=1.5, label=f"Mean Shift: {shift_span.mean():.0f} ranks")
    ax2.set_title("Unit Rank Volatility (Max Rank − Min Rank Across 5 Schemes)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Rank Shift Span (positions)", fontsize=10)
    ax2.set_ylabel("Number of Administrative Units", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved rank stability plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prioritization figures.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/processed/coastal_prioritization_baseline.parquet"),
        help="Path to baseline parquet.",
    )
    parser.add_argument(
        "--coastline-path",
        type=Path,
        default=Path("data/interim/coastal_karnataka_ne_10m_coastline.geojson"),
        help="Path to coastline GeoJSON.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    gdf = gpd.read_parquet(args.input_parquet)
    coastline_gdf = gpd.read_file(args.coastline_path)

    dim_grid_path = args.figures_dir / "prioritization_dimension_choropleths.png"
    comp_maps_path = args.figures_dir / "prioritization_baseline_and_sensitivity_maps.png"
    stability_path = args.figures_dir / "prioritization_sensitivity_rank_stability.png"

    plot_dimension_choropleths(gdf, coastline_gdf, dim_grid_path)
    plot_baseline_and_sensitivity_maps(gdf, coastline_gdf, comp_maps_path)
    plot_sensitivity_rank_stability(gdf, stability_path)


if __name__ == "__main__":
    main()
