"""Altair and Matplotlib chart generators in warm chocolate editorial aesthetic.

Provides:
- District priority band distribution stacked bar charts (counts and normalized).
- AHP preparedness tier distribution charts (overall and clustered by district).
- Vulnerability archetype distribution charts (assigned clusters vs fallback).
- Multidimensional radar charts for community deep dives.
- Altair scatter plots for scenario rank divergence and mover audits.
- Archetype comparative dimension deficit profiles.
- Coordinated warm tones: #C18B62, #D6B28C, #A76F4E, #8D9A78, #E0CBB3.
- Strictly zero emojis and zero em dashes in chart titles and axis labels.
"""

from typing import Any, Dict, List, Optional
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.assistant.theme import ARCHETYPE_COLORS, BAND_COLORS, TIER_COLORS


def generate_district_band_chart(
    df: pd.DataFrame,
    normalized: bool = False,
    height: int = 340,
) -> alt.Chart:
    """Generate Altair stacked bar chart comparing priority band distribution across districts.

    Args:
        df: DataFrame containing 'district_name' and 'provisional_priority_band'.
        normalized: If True, normalizes bars to 100% distribution per district.
        height: Chart height in pixels.

    Returns:
        Altair Chart object.
    """
    col_band = "provisional_priority_band" if "provisional_priority_band" in df.columns else "priority_band"
    if df.empty or col_band not in df.columns or "district_name" not in df.columns:
        return alt.Chart(pd.DataFrame({"x": ["No Data"], "y": [0]})).mark_bar().encode(x="x:N", y="y:Q")

    agg_df = df.groupby(["district_name", col_band]).size().reset_index(name="count")
    totals = agg_df.groupby("district_name")["count"].transform("sum")
    agg_df["percentage"] = (agg_df["count"] / totals * 100).round(1)

    domain = [
        "BAND_1_CRITICAL",
        "BAND_2_SPECIALIZED",
        "BAND_3_MODERATE",
        "BAND_4_ADEQUATE",
    ]
    # Warm editorial priority palette: Terracotta, Ochre, Taupe, Olive
    colors = ["#C95D4A", "#D48B47", "#A88E75", "#7A8B63"]

    y_scale = alt.Scale(domain=[0, 1]) if normalized else alt.Scale()
    y_axis = (
        alt.Axis(format=".0%", title="Proportion of District Units")
        if normalized
        else alt.Axis(title="Number of Administrative Units")
    )
    y_encoding = alt.Y("count:Q", stack="normalize" if normalized else "zero", axis=y_axis, scale=y_scale)

    return (
        alt.Chart(agg_df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("district_name:N", title="District", axis=alt.Axis(labelAngle=0)),
            y=y_encoding,
            color=alt.Color(
                f"{col_band}:N",
                scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(
                    title="Priority Band",
                    orient="top",
                    direction="horizontal",
                    labelLimit=250,
                ),
            ),
            tooltip=[
                alt.Tooltip("district_name:N", title="District"),
                alt.Tooltip(f"{col_band}:N", title="Priority Band"),
                alt.Tooltip("count:Q", title="Units"),
                alt.Tooltip("percentage:Q", title="District Share (%)", format=".1f"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#514033",
            domainColor="#514033",
            tickColor="#514033",
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
        .configure_legend(
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
    )


def generate_tier_distribution_chart(
    df: pd.DataFrame,
    by_district: bool = False,
    height: int = 300,
) -> alt.Chart:
    """Generate Altair bar chart displaying AHP preparedness tier distribution.

    Args:
        df: DataFrame containing 'preparedness_tier' (and 'district_name' if by_district=True).
        by_district: If True, clusters bars by district.
        height: Chart height in pixels.

    Returns:
        Altair Chart object.
    """
    if df.empty or "preparedness_tier" not in df.columns:
        return alt.Chart(pd.DataFrame({"x": ["No Data"], "y": [0]})).mark_bar().encode(x="x:N", y="y:Q")

    domain = ["Low", "Medium", "High"]
    colors = ["#C95D4A", "#D48B47", "#7A8B63"]

    if by_district and "district_name" in df.columns:
        agg_df = (
            df.groupby(["district_name", "preparedness_tier"])
            .size()
            .reset_index(name="count")
        )
        totals = agg_df.groupby("district_name")["count"].transform("sum")
        agg_df["percentage"] = (agg_df["count"] / totals * 100).round(1)

        chart = (
            alt.Chart(agg_df)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("district_name:N", title="District", axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("preparedness_tier:N", sort=domain),
                y=alt.Y("count:Q", title="Administrative Units", axis=alt.Axis(grid=True)),
                color=alt.Color(
                    "preparedness_tier:N",
                    scale=alt.Scale(domain=domain, range=colors),
                    sort=domain,
                    legend=alt.Legend(title="AHP Preparedness Tier", orient="top", direction="horizontal"),
                ),
                tooltip=[
                    alt.Tooltip("district_name:N", title="District"),
                    alt.Tooltip("preparedness_tier:N", title="Tier"),
                    alt.Tooltip("count:Q", title="Units"),
                    alt.Tooltip("percentage:Q", title="District Share (%)", format=".1f"),
                ],
            )
            .properties(height=height)
        )
    else:
        agg_df = df["preparedness_tier"].value_counts().reset_index()
        agg_df.columns = ["preparedness_tier", "count"]
        total = agg_df["count"].sum()
        agg_df["percentage"] = (agg_df["count"] / total * 100).round(1)

        chart = (
            alt.Chart(agg_df)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("preparedness_tier:N", title="AHP Preparedness Tier", sort=domain, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count:Q", title="Administrative Units", axis=alt.Axis(grid=True)),
                color=alt.Color(
                    "preparedness_tier:N",
                    scale=alt.Scale(domain=domain, range=colors),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("preparedness_tier:N", title="Tier"),
                    alt.Tooltip("count:Q", title="Units"),
                    alt.Tooltip("percentage:Q", title="Regional Share (%)", format=".1f"),
                ],
            )
            .properties(height=height)
        )

    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#514033",
            domainColor="#514033",
            tickColor="#514033",
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
        .configure_legend(
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
    )


def generate_archetype_distribution_chart(
    df: pd.DataFrame,
    height: int = 300,
) -> alt.Chart:
    """Generate Altair horizontal bar chart showing K-Means vulnerability archetypes.

    Distinguishes the 4 assigned archetypes (DK/Udupi) from unassigned units (Uttara Kannada).

    Args:
        df: DataFrame containing 'vulnerability_archetype_label' and 'district_name'.
        height: Chart height in pixels.

    Returns:
        Altair Chart object.
    """
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": ["No Data"], "y": [0]})).mark_bar().encode(x="x:N", y="y:Q")

    labels = []
    statuses = []
    for _, row in df.iterrows():
        lbl = row.get("vulnerability_archetype_label")
        dist = str(row.get("district_name", ""))
        if pd.isna(lbl) or not str(lbl).strip() or dist.lower() == "uttara kannada":
            labels.append("Unassigned (Uttara Kannada Sector Fallback)")
            statuses.append("Sector Rule Fallback")
        else:
            labels.append(str(lbl).strip())
            statuses.append("Assigned Cluster")

    temp_df = pd.DataFrame({"archetype_display": labels, "assignment_status": statuses})
    agg_df = temp_df.groupby(["archetype_display", "assignment_status"]).size().reset_index(name="count")
    total = agg_df["count"].sum()
    agg_df["percentage"] = (agg_df["count"] / total * 100).round(1)

    agg_df = agg_df.sort_values("count", ascending=False)

    color_scale = alt.Scale(
        domain=["Assigned Cluster", "Sector Rule Fallback"],
        range=["#C18B62", "#918170"],
    )

    return (
        alt.Chart(agg_df)
        .mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
        .encode(
            y=alt.Y("archetype_display:N", title="Vulnerability Archetype", sort="-x", axis=alt.Axis(labelLimit=320)),
            x=alt.X("count:Q", title="Administrative Units", axis=alt.Axis(grid=True)),
            color=alt.Color(
                "assignment_status:N",
                scale=color_scale,
                legend=alt.Legend(title="Clustering Status", orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("archetype_display:N", title="Archetype"),
                alt.Tooltip("assignment_status:N", title="Status"),
                alt.Tooltip("count:Q", title="Units"),
                alt.Tooltip("percentage:Q", title="Share (%)", format=".1f"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#514033",
            domainColor="#514033",
            tickColor="#514033",
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
        .configure_legend(
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
    )


def generate_radar_chart(
    unit_scores: Dict[str, float],
    district_medians: Dict[str, float],
    unit_name: str,
    district_name: str,
) -> plt.Figure:
    """Generate an editorial polar radar chart comparing unit dimensions against district medians.

    Args:
        unit_scores: Dict of dimension scores for the selected unit.
        district_medians: Dict of median scores for the district.
        unit_name: Name of the unit.
        district_name: District name.

    Returns:
        matplotlib.figure.Figure instance.
    """
    categories = [
        "Healthcare\nDeficit",
        "Demographic\nVulnerability",
        "Infrastructure\nDeficit",
        "Coastal\nProximity",
    ]
    keys = [
        "healthcare_deficit",
        "demographic_vulnerability",
        "infrastructure_deficit",
        "coastal_exposure",
    ]

    unit_vals = [float(unit_scores.get(k, 0.0)) for k in keys]
    dist_vals = [float(district_medians.get(k, 0.0)) for k in keys]

    # Close the polygon loop
    unit_vals += unit_vals[:1]
    dist_vals += dist_vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#2B211B")
    ax.set_facecolor("#342820")

    # Plot District Median Reference (Sage/Khaki)
    ax.plot(angles, dist_vals, color="#8D9A78", linewidth=1.5, linestyle="--", label=f"{district_name} Median")
    ax.fill(angles, dist_vals, color="#8D9A78", alpha=0.12)

    # Plot Community Profile (Warm Ochre)
    ax.plot(angles, unit_vals, color="#C18B62", linewidth=2.0, label=f"{unit_name}")
    ax.fill(angles, unit_vals, color="#C18B62", alpha=0.25)

    # Aesthetics and limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#918170", size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#F3E9DC", size=8.8, fontweight="500")

    ax.grid(color="#514033", linestyle=":", linewidth=0.8)
    ax.spines["polar"].set_color("#514033")

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.15),
        frameon=True,
        facecolor="#2B211B",
        edgecolor="#514033",
        labelcolor="#F3E9DC",
        fontsize=8.5,
    )
    plt.tight_layout()
    return fig


def generate_rank_scatter_chart(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    label_x: str,
    label_y: str,
) -> alt.Chart:
    """Generate an Altair scatter plot comparing ranks between two scenarios.

    Args:
        df: DataFrame containing units with rank columns and archetype labels.
        col_x: Column name for X-axis (Rank in Scenario A).
        col_y: Column name for Y-axis (Rank in Scenario B).
        label_x: Display title for X-axis.
        label_y: Display title for Y-axis.

    Returns:
        alt.Chart object.
    """
    chart_df = df.copy()

    # Scatter points in warm tones
    points = (
        alt.Chart(chart_df)
        .mark_circle(size=50, opacity=0.8)
        .encode(
            x=alt.X(f"{col_x}:Q", title=label_x, scale=alt.Scale(domain=[1, 622])),
            y=alt.Y(f"{col_y}:Q", title=label_y, scale=alt.Scale(domain=[1, 622])),
            color=alt.Color(
                "vulnerability_archetype_label:N",
                title="Vulnerability Archetype",
                scale=alt.Scale(
                    domain=[
                        "Urban & Peri-Urban Well-Resourced Corridor",
                        "Remote Foothill & Healthcare-Isolated Communities",
                        "Socioeconomically Vulnerable Rural Interior",
                        "High-Exposure Coastal Rural Belt",
                    ],
                    range=["#7A8B63", "#D48B47", "#A76F4E", "#C95D4A"],
                ),
            ),
            tooltip=[
                alt.Tooltip("unit_name:N", title="Unit Name"),
                alt.Tooltip("district_name:N", title="District"),
                alt.Tooltip(f"{col_x}:Q", title=label_x),
                alt.Tooltip(f"{col_y}:Q", title=label_y),
                alt.Tooltip("vulnerability_archetype_label:N", title="Archetype"),
            ],
        )
    )

    # 45-degree diagonal parity line (where Rank A == Rank B)
    diag_df = pd.DataFrame({"x": [1, 622], "y": [1, 622]})
    line = (
        alt.Chart(diag_df)
        .mark_line(color="#918170", strokeDash=[4, 4], opacity=0.7)
        .encode(x="x:Q", y="y:Q")
    )

    return (
        (points + line)
        .properties(height=420)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#514033",
            domainColor="#514033",
            tickColor="#514033",
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
        .configure_legend(
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
            orient="bottom",
        )
    )


def generate_archetype_feature_chart(dimension_means_by_cluster: Dict[int, Dict[str, float]]) -> alt.Chart:
    """Generate a comparative grouped bar chart showing dimensional deficits across clusters."""
    rows = []
    dim_display = {
        "dim_healthcare_deficit": "Healthcare Access Deficit",
        "dim_demographic_vulnerability": "Demographic Vulnerability",
        "dim_infrastructure_deficit": "Infrastructure Deficit",
        "dim_coastal_exposure": "Coastal Exposure",
    }
    cluster_names = {
        0: "Cluster 0 (Urban Corridor)",
        1: "Cluster 1 (Remote Foothill)",
        2: "Cluster 2 (Rural Interior)",
        3: "Cluster 3 (Coastal Belt)",
    }

    for c_id, dims in dimension_means_by_cluster.items():
        c_label = cluster_names.get(c_id, f"Cluster {c_id}")
        for d_key, d_val in dims.items():
            rows.append({
                "Cluster": c_label,
                "Dimension": dim_display.get(d_key, d_key),
                "Mean Deficit (0-1)": round(d_val, 3),
            })

    chart_df = pd.DataFrame(rows)
    domain = list(cluster_names.values())
    colors = ["#7A8B63", "#D48B47", "#A76F4E", "#C95D4A"]

    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Dimension:N", title="Vulnerability Dimension", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Cluster:N", sort=domain),
            y=alt.Y("Mean Deficit (0-1):Q", title="Mean Dimension Deficit Score (0 to 1)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Cluster:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient="top", direction="horizontal")),
            tooltip=["Cluster", "Dimension", "Mean Deficit (0-1)"],
        )
        .properties(height=320)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#514033",
            domainColor="#514033",
            tickColor="#514033",
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
        .configure_legend(
            labelColor="#BDAE9C",
            titleColor="#F3E9DC",
            labelFont="DM Sans, sans-serif",
            titleFont="DM Sans, sans-serif",
        )
    )
