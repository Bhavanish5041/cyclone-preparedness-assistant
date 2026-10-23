"""Reusable UI components, geospatial map builders, and visual styling for Streamlit.

Provides:
- Custom CSS design system for dark/light harmony, glassmorphism cards, and badge pills.
- Folium choropleth map generator with coastline and healthcare facility overlays.
- Radar chart generator comparing administrative unit dimension scores to district medians.
- Altair scatter chart for scenario rank divergence and mover analysis.
- Archetype color palettes and operational styling metadata.
"""

from typing import Any, Dict, List, Optional, Tuple
import branca.colormap as cm
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

# Curated palette for discovered vulnerability archetypes
ARCHETYPE_COLORS = {
    0: {"name": "Urban & Peri-Urban Corridor", "hex": "#10b981", "badge": "success"},       # Emerald
    1: {"name": "Remote Foothill Communities", "hex": "#f59e0b", "badge": "warning"},       # Amber
    2: {"name": "Vulnerable Rural Interior", "hex": "#8b5cf6", "badge": "secondary"},      # Violet
    3: {"name": "High-Exposure Coastal Belt", "hex": "#ef4444", "badge": "danger"},         # Rose Red
}

METRIC_LABELS = {
    "priority_score": "Composite Priority Score",
    "dim_healthcare_deficit": "Healthcare Access Deficit",
    "dim_demographic_vulnerability": "Demographic Vulnerability",
    "dim_infrastructure_deficit": "Infrastructure Deficit",
    "dim_coastal_exposure": "Coastal Hazard Proximity",
    "vulnerability_archetype_id": "Vulnerability Archetype",
}


def get_custom_css() -> str:
    """Return polished custom CSS styles for the Streamlit dashboard."""
    return """
    <style>
    /* Global Typography & Font Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 95rem;
    }

    /* Header Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        color: #ffffff;
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        backdrop-filter: blur(10px);
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-caption {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Archetype Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-arch-0 { background-color: rgba(16, 185, 129, 0.18); color: #10b981; border: 1px solid #10b981; }
    .badge-arch-1 { background-color: rgba(245, 158, 11, 0.18); color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-arch-2 { background-color: rgba(139, 92, 246, 0.18); color: #8b5cf6; border: 1px solid #8b5cf6; }
    .badge-arch-3 { background-color: rgba(239, 68, 68, 0.18); color: #ef4444; border: 1px solid #ef4444; }

    /* Operational Checklist Item */
    .checklist-item {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #38bdf8;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }

    /* Disclaimer Callout Box */
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #d97706;
        line-height: 1.45;
    }

    /* Streamlit Sidebar Cleanliness */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """


def build_folium_choropleth_map(
    gdf: gpd.GeoDataFrame,
    metric_col: str = "priority_score",
    show_coastline: bool = True,
    show_healthcare: bool = False,
    coastline_path: str = "data/interim/coastal_karnataka_ne_10m_coastline.geojson",
    healthcare_path: str = "data/interim/coastal_healthcare_facilities.parquet",
    selected_unit_shrid: Optional[str] = None,
) -> folium.Map:
    """Build an interactive Folium map with choropleth polygons and optional overlays.

    Args:
        gdf: GeoDataFrame containing 622 units with geometry and metric columns.
        metric_col: Name of column to shade polygons by.
        show_coastline: If True, adds the Natural Earth 10m coastline overlay.
        show_healthcare: If True, adds the MoHFW healthcare facilities marker cluster.
        coastline_path: Path to coastline GeoJSON.
        healthcare_path: Path to healthcare parquet dataset.
        selected_unit_shrid: Optional shrid2 to highlight with a golden halo.

    Returns:
        folium.Map instance.
    """
    # Center on coastal Karnataka (between DK and Udupi)
    center_lat, center_lon = 13.35, 74.88
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Prepare simplified geometries for fast, responsive browser rendering
    display_gdf = gdf.copy()
    display_gdf["geom_simple"] = display_gdf.geometry.simplify(0.001)

    # Determine color mapping based on metric
    is_archetype = metric_col == "vulnerability_archetype_id"
    if not is_archetype:
        min_val = float(display_gdf[metric_col].min())
        max_val = float(display_gdf[metric_col].max())
        if max_val == min_val:
            max_val = min_val + 1.0
        # Smooth YlOrRd gradient for risk/deficit
        colormap = cm.LinearColormap(
            colors=["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"],
            vmin=min_val,
            vmax=max_val,
            caption=METRIC_LABELS.get(metric_col, metric_col),
        )
        colormap.add_to(fmap)

    def style_function(feature):
        props = feature["properties"]
        val = props.get(metric_col)
        shrid = props.get("shrid2")

        # Base highlight for selected unit
        if selected_unit_shrid and shrid == selected_unit_shrid:
            return {
                "fillColor": "#38bdf8",
                "color": "#0284c7",
                "weight": 3.5,
                "fillOpacity": 0.85,
            }

        if is_archetype:
            c_id = int(val) if val is not None else 0
            color = ARCHETYPE_COLORS.get(c_id, {}).get("hex", "#94a3b8")
            fill_opacity = 0.65
        else:
            color = colormap(val) if val is not None else "#cccccc"
            fill_opacity = 0.68

        return {
            "fillColor": color,
            "color": "#334155",
            "weight": 0.8,
            "fillOpacity": fill_opacity,
        }

    def highlight_function(feature):
        return {
            "weight": 2.5,
            "color": "#0ea5e9",
            "fillOpacity": 0.9,
        }

    # GeoJSON Choropleth Layer
    fields_to_include = [
        "shrid2",
        "pc11_id",
        "unit_name",
        "unit_type",
        "district_name",
        "priority_score",
        "priority_rank",
        "vulnerability_archetype_id",
        "vulnerability_archetype_label",
        "dim_healthcare_deficit",
        "dim_demographic_vulnerability",
        "dim_infrastructure_deficit",
        "dim_coastal_exposure",
        "geom_simple",
    ]
    # Keep only available fields
    valid_fields = [f for f in fields_to_include if f in display_gdf.columns]
    poly_gdf = display_gdf[valid_fields].set_geometry("geom_simple")

    tooltip = folium.GeoJsonTooltip(
        fields=["unit_name", "district_name", "unit_type", "priority_rank", "priority_score", "vulnerability_archetype_label"],
        aliases=["Unit:", "District:", "Type:", "Priority Rank:", "Priority Score:", "Archetype:"],
        localize=True,
        sticky=False,
        style="font-family: Inter, sans-serif; font-size: 12px; background: #0f172a; color: #f8fafc; padding: 8px; border-radius: 6px;",
    )

    geojson_layer = folium.GeoJson(
        poly_gdf,
        name="Administrative Units",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
    )
    geojson_layer.add_to(fmap)

    # Optional Coastline Overlay
    if show_coastline:
        try:
            coast_gdf = gpd.read_file(coastline_path)
            coast_layer = folium.GeoJson(
                coast_gdf,
                name="Arabian Sea Coastline (10m)",
                style_function=lambda x: {
                    "color": "#0284c7",
                    "weight": 3.0,
                    "opacity": 0.9,
                },
                tooltip=folium.Tooltip("Arabian Sea Coastline (Natural Earth 10m)"),
            )
            coast_layer.add_to(fmap)
        except Exception:
            pass

    # Optional Healthcare Facilities Overlay
    if show_healthcare:
        try:
            hosp_df = pd.read_parquet(healthcare_path)
            # Filter valid coordinates
            valid_hosp = hosp_df[hosp_df["latitude_num"].notna() & hosp_df["longitude_num"].notna()]
            cluster = MarkerCluster(name="Healthcare Facilities", show=True)

            tier_colors = {
                "HOSPITAL": "#ef4444",   # Red
                "PHC_CHC": "#f59e0b",    # Amber
                "SUBCENTRE": "#3b82f6",  # Blue
            }

            for _, row in valid_hosp.iterrows():
                tier = str(row.get("service_tier", "SUBCENTRE"))
                f_name = str(row.get("facility_name_clean", "Health Facility"))
                f_type = str(row.get("facility_type_label", tier))
                f_color = tier_colors.get(tier, "#64748b")

                popup_html = f"<b>{f_name}</b><br>Type: {f_type}<br>Tier: {tier}"
                folium.CircleMarker(
                    location=[row["latitude_num"], row["longitude_num"]],
                    radius=4,
                    color=f_color,
                    fill=True,
                    fill_color=f_color,
                    fill_opacity=0.8,
                    popup=folium.Popup(popup_html, max_width=250),
                ).add_to(cluster)

            cluster.add_to(fmap)
        except Exception:
            pass

    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
    return fmap


def generate_radar_chart(
    unit_scores: Dict[str, float],
    district_medians: Dict[str, float],
    unit_name: str,
    district_name: str,
) -> plt.Figure:
    """Generate a clean polar radar chart comparing unit dimensions against district medians.

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

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Plot District Median Reference
    ax.plot(angles, dist_vals, color="#94a3b8", linewidth=1.5, linestyle="--", label=f"{district_name} Median")
    ax.fill(angles, dist_vals, color="#94a3b8", alpha=0.12)

    # Plot Community Profile
    ax.plot(angles, unit_vals, color="#38bdf8", linewidth=2.4, label=f"{unit_name}")
    ax.fill(angles, unit_vals, color="#0284c7", alpha=0.35)

    # Aesthetics and limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#64748b", size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#e2e8f0", size=9.5, fontweight="600")

    ax.grid(color="#334155", linestyle=":", linewidth=0.8)
    ax.spines["polar"].set_color("#475569")

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.15),
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
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

    # Scatter points
    points = (
        alt.Chart(chart_df)
        .mark_circle(size=55, opacity=0.75)
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
                    range=["#10b981", "#f59e0b", "#8b5cf6", "#ef4444"],
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
        .mark_line(color="#64748b", strokeDash=[4, 4], opacity=0.8)
        .encode(x="x:Q", y="y:Q")
    )

    return (
        (points + line)
        .properties(height=420)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#334155",
            labelColor="#94a3b8",
            titleColor="#f8fafc",
        )
        .configure_legend(
            labelColor="#cbd5e1",
            titleColor="#f8fafc",
            orient="bottom",
        )
    )
