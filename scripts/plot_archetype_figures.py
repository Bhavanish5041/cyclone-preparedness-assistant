#!/usr/bin/env python3
"""CLI script to generate publication-grade figures for unsupervised vulnerability archetype discovery.

Reads:
- data/processed/coastal_vulnerability_archetypes.parquet
- data/interim/coastal_karnataka_ne_10m_coastline.geojson
- reports/clustering_experiments_manifest.json

Generates:
- reports/figures/clustering_metric_evaluation_curve.png
- reports/figures/vulnerability_archetypes_choropleth.png
- reports/figures/vulnerability_archetypes_feature_radar_and_profiles.png
- reports/figures/clustering_agreement_heatmap.png
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("plot_archetype_figures")

ARCHETYPE_COLORS = {
    0: "#2ca02c",  # Green: Well-resourced urban corridor
    1: "#d62728",  # Red: Foothill healthcare-isolated
    2: "#9467bd",  # Purple: Socioeconomically vulnerable interior
    3: "#1f77b4",  # Blue: High-exposure coastal rural belt
}

ARCHETYPE_NAMES = {
    0: "Cluster 0: Urban Well-Resourced Corridor (n=130, 20.9%)",
    1: "Cluster 1: Foothill Healthcare-Isolated (n=82, 13.2%)",
    2: "Cluster 2: Vulnerable Rural Interior (n=161, 25.9%)",
    3: "Cluster 3: High-Exposure Coastal Belt (n=249, 40.0%)",
}


def plot_metric_evaluation_curve(
    manifest_path: Path,
    output_path: Path,
) -> None:
    """Plot Silhouette, Davies-Bouldin, and Calinski-Harabasz curves for k=2..6."""
    logger.info("Generating clustering metric evaluation curves...")
    with open(manifest_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    km_df = df[df["algorithm"] == "kmeans"].sort_values("n_clusters")
    agg_df = df[df["algorithm"] == "agglomerative"].sort_values("n_clusters")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

    # 1. Silhouette score (higher = better)
    ax1.plot(km_df["n_clusters"], km_df["silhouette"], "o-", color="#1f77b4", linewidth=2, label="K-Means")
    ax1.plot(agg_df["n_clusters"], agg_df["silhouette"], "s--", color="#ff7f0e", linewidth=2, label="Agglomerative (Ward)")
    ax1.axvline(4, color="gray", linestyle=":", label="Selected k=4")
    ax1.set_title("Silhouette Score (Higher = Better)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Silhouette Coefficient", fontsize=10)
    ax1.set_xticks(range(2, 7))
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left")

    # 2. Davies-Bouldin index (lower = better)
    ax2.plot(km_df["n_clusters"], km_df["davies_bouldin"], "o-", color="#1f77b4", linewidth=2, label="K-Means")
    ax2.plot(agg_df["n_clusters"], agg_df["davies_bouldin"], "s--", color="#ff7f0e", linewidth=2, label="Agglomerative (Ward)")
    ax2.axvline(4, color="gray", linestyle=":", label="Selected k=4")
    ax2.set_title("Davies-Bouldin Index (Lower = Better)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax2.set_ylabel("Davies-Bouldin Score", fontsize=10)
    ax2.set_xticks(range(2, 7))
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right")

    # 3. Calinski-Harabasz score (higher = better)
    ax3.plot(km_df["n_clusters"], km_df["calinski_harabasz"], "o-", color="#1f77b4", linewidth=2, label="K-Means")
    ax3.plot(agg_df["n_clusters"], agg_df["calinski_harabasz"], "s--", color="#ff7f0e", linewidth=2, label="Agglomerative (Ward)")
    ax3.axvline(4, color="gray", linestyle=":", label="Selected k=4")
    ax3.set_title("Calinski-Harabasz Score (Higher = Better)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax3.set_ylabel("Calinski-Harabasz Variance Ratio", fontsize=10)
    ax3.set_xticks(range(2, 7))
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right")

    plt.suptitle("Clustering Metric Evaluation Curves Across Cluster Counts (k=2..6)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metric curves to {output_path}")


def plot_vulnerability_archetypes_choropleth(
    gdf: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot geographic distribution of the 4 discovered vulnerability archetypes."""
    logger.info("Generating vulnerability archetypes choropleth map...")
    fig, ax = plt.subplots(figsize=(12, 14))

    # Map cluster colors
    gdf_plot = gdf.copy()
    gdf_plot["display_label"] = gdf_plot["vulnerability_archetype_id"].map(ARCHETYPE_NAMES)

    # Plot each cluster with custom colors
    for c_id in sorted(ARCHETYPE_COLORS.keys()):
        c_sub = gdf_plot[gdf_plot["vulnerability_archetype_id"] == c_id]
        c_sub.plot(
            ax=ax,
            color=ARCHETYPE_COLORS[c_id],
            edgecolor="#333333",
            linewidth=0.35,
            label=ARCHETYPE_NAMES[c_id],
        )

    # Overlay coastline
    coastline_gdf.plot(ax=ax, color="#000000", linewidth=2.0, linestyle="-", label="Arabian Sea Coastline")

    ax.set_title(
        "Unsupervised Vulnerability Archetypes — Coastal Karnataka (n=622)\n"
        "(Partitioned via K-Means k=4 on Four Standardized Preparedness Dimensions)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved archetype choropleth to {output_path}")


def plot_archetype_feature_profiles(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot grouped bar chart comparing the four dimension means across the four archetypes."""
    logger.info("Generating archetype feature profiles bar chart...")
    feature_cols = [
        "dim_healthcare_deficit",
        "dim_demographic_vulnerability",
        "dim_infrastructure_deficit",
        "dim_coastal_exposure",
    ]
    dim_labels = [
        "Healthcare\nDeficit",
        "Demographic\nVulnerability",
        "Infrastructure\nDeficit",
        "Coastal\nExposure",
    ]

    means = gdf.groupby("vulnerability_archetype_id")[feature_cols].mean()

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(feature_cols))
    width = 0.18

    for c_id in range(4):
        vals = means.loc[c_id].values
        offset = (c_id - 1.5) * width
        rects = ax.bar(
            x + offset,
            vals,
            width,
            label=ARCHETYPE_NAMES[c_id],
            color=ARCHETYPE_COLORS[c_id],
            edgecolor="#333333",
            linewidth=0.6,
        )
        ax.bar_label(rects, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title("Vulnerability Dimension Profiles by Archetype (Mean Dimension Scores [0, 1])", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Normalized Deficit Score [0.0 = Min Deficit, 1.0 = Max Deficit]", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature profiles chart to {output_path}")


def plot_clustering_agreement_heatmap(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot agreement heatmap comparing K-Means, Agglomerative, and multi-seed partitions."""
    logger.info("Generating clustering agreement heatmap...")
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    feature_cols = [
        "dim_healthcare_deficit",
        "dim_demographic_vulnerability",
        "dim_infrastructure_deficit",
        "dim_coastal_exposure",
    ]
    X_scaled = StandardScaler().fit_transform(gdf[feature_cols].values)

    models = {
        "KMeans k=3": KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(X_scaled),
        "KMeans k=4 (Seed 42)": KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X_scaled),
        "KMeans k=4 (Seed 123)": KMeans(n_clusters=4, random_state=123, n_init=20).fit_predict(X_scaled),
        "KMeans k=4 (Seed 999)": KMeans(n_clusters=4, random_state=999, n_init=20).fit_predict(X_scaled),
        "Agglom k=4 (Ward)": AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X_scaled),
        "KMeans k=5": KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(X_scaled),
    }

    names = list(models.keys())
    n = len(names)
    ari_mat = pd.DataFrame(index=names, columns=names, dtype=float)

    for i in range(n):
        for j in range(n):
            ari_mat.iloc[i, j] = adjusted_rand_score(models[names[i]], models[names[j]])

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        ari_mat,
        cmap="Blues",
        vmin=0.5,
        vmax=1.0,
        annot=True,
        fmt=".3f",
        square=True,
        cbar_kws={"label": "Adjusted Rand Index (ARI)"},
        ax=ax,
    )
    ax.set_title("Clustering Solution Agreement & Seed Stability (ARI)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved agreement heatmap to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot archetype discovery figures.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/processed/coastal_vulnerability_archetypes.parquet"),
        help="Path to archetypes parquet.",
    )
    parser.add_argument(
        "--coastline-path",
        type=Path,
        default=Path("data/interim/coastal_karnataka_ne_10m_coastline.geojson"),
        help="Path to coastline GeoJSON.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("reports/clustering_experiments_manifest.json"),
        help="Path to experiment manifest JSON.",
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

    metric_path = args.figures_dir / "clustering_metric_evaluation_curve.png"
    choropleth_path = args.figures_dir / "vulnerability_archetypes_choropleth.png"
    profiles_path = args.figures_dir / "vulnerability_archetypes_feature_radar_and_profiles.png"
    agreement_path = args.figures_dir / "clustering_agreement_heatmap.png"

    plot_metric_evaluation_curve(args.manifest_path, metric_path)
    plot_vulnerability_archetypes_choropleth(gdf, coastline_gdf, choropleth_path)
    plot_archetype_feature_profiles(gdf, profiles_path)
    plot_clustering_agreement_heatmap(gdf, agreement_path)


if __name__ == "__main__":
    main()
