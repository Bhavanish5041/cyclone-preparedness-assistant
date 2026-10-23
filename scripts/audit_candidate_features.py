#!/usr/bin/env python3
"""CLI script to audit candidate feature quality, evaluate correlations, and generate figures.

Reads:
- data/processed/coastal_exposure_features.parquet
- data/interim/coastal_karnataka_ne_10m_coastline.geojson

Generates:
- reports/figures/coastal_proximity_choropleth.png
- reports/figures/coastal_proximity_distribution.png
- reports/figures/candidate_feature_correlation_matrix.png

Outputs:
- Console tables for feature dictionary, missingness, distributions, and collinearity.
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

from src.features.dictionary import (
    CANDIDATE_FEATURE_DICTIONARY,
    audit_feature_quality,
    detect_collinear_pairs,
    get_feature_dictionary_df,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("audit_candidate_features")


def plot_coastal_proximity_choropleth(
    gdf: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot publication-quality choropleth of distance to coastline."""
    logger.info("Generating coastal proximity choropleth map...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.1, 1]})

    # Subplot 1: Overview of full coastal Karnataka
    gdf.plot(
        column="dist_to_coastline_km",
        cmap="mako_r",
        linewidth=0.3,
        edgecolor="#333333",
        legend=True,
        legend_kwds={"label": "Straight-line Distance to Coastline (km)", "orientation": "horizontal", "shrink": 0.8},
        ax=ax1,
    )
    coastline_gdf.plot(ax=ax1, color="#00ffff", linewidth=2.5, label="Natural Earth 10m Coastline")
    ax1.set_title("Coastal Karnataka — Distance to Coastline Proxy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Longitude (°E)", fontsize=10)
    ax1.set_ylabel("Latitude (°N)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right")

    # Subplot 2: Immediate coastal zone (< 15 km)
    coastal_zone = gdf[gdf["dist_to_coastline_km"] <= 15.0]
    inland_zone = gdf[gdf["dist_to_coastline_km"] > 15.0]

    inland_zone.plot(ax=ax2, color="#e0e0e0", linewidth=0.2, edgecolor="#ffffff")
    coastal_zone.plot(
        column="dist_to_coastline_km",
        cmap="mako_r",
        linewidth=0.4,
        edgecolor="#222222",
        vmin=0,
        vmax=15,
        ax=ax2,
    )
    coastline_gdf.plot(ax=ax2, color="#0055ff", linewidth=2.5, label="Natural Earth 10m Coastline")
    ax2.set_xlim(74.55, 75.15)
    ax2.set_ylim(12.7, 13.95)
    ax2.set_title("Near-Coast Buffer Zone (≤ 15 km from Sea)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Longitude (°E)", fontsize=10)
    ax2.set_ylabel("Latitude (°N)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved choropleth map to {output_path}")


def plot_coastal_proximity_distribution(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot district-stratified distribution and boxplot for distance to coastline."""
    logger.info("Generating coastal proximity distribution figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Map district codes
    gdf_plot = gdf.copy()
    gdf_plot["district_name"] = gdf_plot["shrid2"].apply(
        lambda s: "Dakshina Kannada" if "575" in str(s) else "Udupi"
    )

    # Subplot 1: KDE and Histograms
    sns.histplot(
        data=gdf_plot,
        x="dist_to_coastline_km",
        hue="district_name",
        kde=True,
        bins=25,
        palette={"Dakshina Kannada": "#2b5c8f", "Udupi": "#d95f02"},
        alpha=0.45,
        element="step",
        ax=ax1,
    )
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="1 km: Shoreline Fringe")
    ax1.axvline(5.0, color="orange", linestyle="--", linewidth=1.5, label="5 km: Near-Coast Buffer")
    ax1.axvline(10.0, color="gold", linestyle="--", linewidth=1.5, label="10 km: Transition Zone")
    ax1.set_title("Distribution of Distance to Coastline by District", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Straight-line Distance to Coastline (km)", fontsize=10)
    ax1.set_ylabel("Number of Administrative Units", fontsize=10)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Boxplot with annotations
    sns.boxplot(
        data=gdf_plot,
        x="district_name",
        y="dist_to_coastline_km",
        palette={"Dakshina Kannada": "#2b5c8f", "Udupi": "#d95f02"},
        width=0.4,
        ax=ax2,
    )
    sns.stripplot(
        data=gdf_plot,
        x="district_name",
        y="dist_to_coastline_km",
        color="black",
        alpha=0.2,
        jitter=0.2,
        size=3,
        ax=ax2,
    )

    # Annotate district medians
    dk_med = gdf_plot[gdf_plot["district_name"] == "Dakshina Kannada"]["dist_to_coastline_km"].median()
    ud_med = gdf_plot[gdf_plot["district_name"] == "Udupi"]["dist_to_coastline_km"].median()
    ax2.text(0, dk_med + 1.5, f"Median: {dk_med:.1f} km", horizontalalignment="center", fontweight="bold")
    ax2.text(1, ud_med + 1.5, f"Median: {ud_med:.1f} km", horizontalalignment="center", fontweight="bold")

    ax2.set_title("District Comparison (Median & IQR)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("District", fontsize=10)
    ax2.set_ylabel("Straight-line Distance to Coastline (km)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def plot_candidate_feature_correlation_matrix(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot correlation matrix across numeric candidate features."""
    logger.info("Generating candidate feature correlation matrix...")
    numeric_features = [
        "dist_to_coastline_km",
        "dist_to_nearest_phc_chc_km",
        "dist_to_nearest_hospital_km",
        "dist_to_nearest_facility_any_km",
        "facilities_in_unit_total",
        "phc_chc_in_unit_count",
        "hospitals_in_unit_count",
        "total_population",
        "total_households",
        "sc_st_ratio",
        "sex_ratio_female_per_1000_male",
        "has_treated_tap_water",
        "has_all_weather_road",
        "has_pucca_road",
        "has_domestic_power",
    ]

    valid_cols = [c for c in numeric_features if c in df.columns]
    corr = df[valid_cols].corr(method="spearman")

    # Clean display labels
    clean_labels = [
        c.replace("dist_to_nearest_", "dist_")
        .replace("_km", "")
        .replace("_in_unit_", "_in_")
        .replace("has_", "")
        for c in valid_cols
    ]

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.6,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={"label": "Spearman Rank Correlation ($r_s$)", "shrink": 0.8},
    )
    plt.title(
        "Candidate Preparedness Features — Pairwise Spearman Correlation Matrix\n"
        "(Auditing Multicollinearity & Feature Leakage)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation heatmap to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate features and generate visualizations.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/processed/coastal_exposure_features.parquet"),
        help="Path to coastal exposure features parquet.",
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

    gdf = gpd.read_parquet(args.features_path)
    coastline_gdf = gpd.read_file(args.coastline_path)

    # 1. Generate Figures
    choropleth_path = args.figures_dir / "coastal_proximity_choropleth.png"
    dist_plot_path = args.figures_dir / "coastal_proximity_distribution.png"
    corr_plot_path = args.figures_dir / "candidate_feature_correlation_matrix.png"

    plot_coastal_proximity_choropleth(gdf, coastline_gdf, choropleth_path)
    plot_coastal_proximity_distribution(gdf, dist_plot_path)
    plot_candidate_feature_correlation_matrix(gdf, corr_plot_path)

    # 2. Quality Audit Table
    quality_df = audit_feature_quality(gdf)
    print("\n" + "=" * 90)
    print("CANDIDATE FEATURE QUALITY & DISTRIBUTION AUDIT (n=622)")
    print("=" * 90)
    display_cols = ["feature_name", "domain", "missing_pct", "min", "median", "mean", "max", "skewness"]
    print(quality_df[display_cols].to_string(index=False))

    # 3. Multicollinearity & Feature Redundancy Audit
    collinear_df = detect_collinear_pairs(gdf, threshold=0.60, method="spearman")
    print("\n" + "=" * 90)
    print("MULTICOLLINEARITY & REDUNDANCY WARNINGS (|r_s| >= 0.60)")
    print("=" * 90)
    if len(collinear_df) > 0:
        for _, row in collinear_df.iterrows():
            severity = "[SEVERE REDUNDANCY]" if row["is_severe"] else "[MODERATE REDUNDANCY]"
            print(f"{severity} {row['feature_1']} <--> {row['feature_2']} (r = {row['correlation']:+.2f})")
            print(f"   Note: {row['leakage_risk_note']}\n")
    else:
        print("No collinear pairs detected above threshold.")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
