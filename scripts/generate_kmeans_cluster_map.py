#!/usr/bin/env python3
"""Generate a standalone interactive K-Means vulnerability cluster map.

Reads the existing processed clustering output
(data/processed/coastal_vulnerability_archetypes.parquet) and produces
a single self-contained HTML file at reports/kmeans_cluster_map.html.

The map colors every polygon by its K-Means cluster assignment (k=4)
using four categorical colors — no continuous gradient, no composite
priority score coloring.

Usage:
    python scripts/generate_kmeans_cluster_map.py

Output:
    reports/kmeans_cluster_map.html
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import folium
from folium.features import GeoJsonPopup, GeoJsonTooltip

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = ROOT / "data" / "processed" / "coastal_vulnerability_archetypes.parquet"
COASTLINE_PATH = ROOT / "data" / "interim" / "coastal_karnataka_ne_10m_coastline.geojson"
OUTPUT_PATH = ROOT / "reports" / "kmeans_cluster_map.html"

# Four clearly distinguishable categorical colors for K-Means clusters
CLUSTER_COLORS: dict[int, str] = {
    0: "#10b981",  # Emerald  — Urban & Peri-Urban Well-Resourced Corridor
    1: "#f59e0b",  # Amber    — Remote Foothill & Healthcare-Isolated Communities
    2: "#8b5cf6",  # Violet   — Socioeconomically Vulnerable Rural Interior
    3: "#ef4444",  # Rose-Red — High-Exposure Coastal Rural Belt
}

CLUSTER_LABELS: dict[int, str] = {
    0: "Urban & Peri-Urban Well-Resourced Corridor",
    1: "Remote Foothill & Healthcare-Isolated Communities",
    2: "Socioeconomically Vulnerable Rural Interior",
    3: "High-Exposure Coastal Rural Belt",
}

# Risk Assessment metadata by cluster
CLUSTER_RISK_PROFILES: dict[int, dict[str, str]] = {
    0: {
        "risk_tier": "Tier 4 · Moderate Direct / High Asset Risk",
        "primary_hazard": "High winds, intense cloudbursts, urban drainage overflow, electrical grid disruption",
        "primary_action": "Clear storm drain outfalls, secure hospital diesel power, prune hazard trees",
    },
    1: {
        "risk_tier": "Tier 2 · High Secondary & Isolation Risk",
        "primary_hazard": "Ghat landslides, flash riverine floods, single-road washouts, severe medical isolation",
        "primary_action": "Preposition JCB earthmovers, satellite phones, 14-day PHC medical buffer stocks",
    },
    2: {
        "risk_tier": "Tier 3 · Moderate-High Socioeconomic Risk",
        "primary_hazard": "Inland gale winds, open well contamination, kutcha roof damage, waterborne disease",
        "primary_action": "Equitable cyclone shelter access, well chlorination, dry food ration reserves",
    },
    3: {
        "risk_tier": "Tier 1 · Critical Direct Marine Surge Risk",
        "primary_hazard": "Direct cyclonic eyewall winds, 2-4m storm surge, wave wash, coastal well salinization",
        "primary_action": "Pre-emptive evacuation to MPCS shelters, protect fishing crafts, mobile desalination",
    },
}


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def _build_legend_html() -> str:
    """Build a standalone HTML/CSS legend overlay for the cluster map."""
    items = ""
    for cid in sorted(CLUSTER_LABELS):
        color = CLUSTER_COLORS[cid]
        label = CLUSTER_LABELS[cid]
        items += (
            f'<li style="margin:4px 0;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};border-radius:3px;margin-right:7px;'
            f'vertical-align:middle;border:1px solid rgba(0,0,0,0.15);"></span>'
            f'<span style="vertical-align:middle;font-size:12px;color:#1e293b;">'
            f'Cluster {cid}: {label}</span></li>'
        )

    return f"""
    <div id="cluster-legend" style="
        position: fixed;
        bottom: 30px;
        left: 14px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.18);
        border: 1px solid #e2e8f0;
        max-width: 380px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    ">
        <div style="font-weight:700;font-size:13px;margin-bottom:6px;color:#0f172a;">
            🗂️ Vulnerability Archetypes (K-Means, k=4)
        </div>
        <ul style="list-style:none;padding:0;margin:0;">
            {items}
        </ul>
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid #e2e8f0;">
            <div style="font-size:11px;font-weight:700;color:#334155;margin-bottom:4px;">
                ⚠️ Cyclone Risk Assessment Highlights
            </div>
            <div style="font-size:10.5px;color:#475569;line-height:1.45;">
                <span style="color:#ef4444;font-weight:600;">● Cl. 3 (Coastal)</span>: Tier 1 Critical · Surge, Gale &amp; Saline Intrusion<br/>
                <span style="color:#f59e0b;font-weight:600;">● Cl. 1 (Foothills)</span>: Tier 2 High · Ghat Cutoffs &amp; Hospital Isolation<br/>
                <span style="color:#8b5cf6;font-weight:600;">● Cl. 2 (Interior)</span>: Tier 3 Mod-High · Demographic &amp; Well Contamination<br/>
                <span style="color:#10b981;font-weight:600;">● Cl. 0 (Urban)</span>: Tier 4 Asset Risk · Cloudburst Waterlogging &amp; Grid Disruption
            </div>
        </div>
        <div style="font-size:10px;color:#94a3b8;margin-top:8px;">
            622 revenue villages &amp; census towns · Dakshina Kannada &amp; Udupi<br/>
            Click any polygon for localized risk assessment &amp; priority rank
        </div>
    </div>
    """


def _build_popup_html(props: dict) -> str:
    """Generate a styled HTML popup for a polygon feature."""
    name = props.get("unit_name", "N/A")
    district = props.get("district_name", "N/A")
    unit_type = props.get("unit_type", "N/A")
    cid = int(props.get("vulnerability_archetype_id", 0))
    clabel = CLUSTER_LABELS.get(cid, f"Cluster {cid}")
    color = CLUSTER_COLORS.get(cid, "#94a3b8")
    d_health = float(props.get("dim_healthcare_deficit", 0))
    d_demo = float(props.get("dim_demographic_vulnerability", 0))
    d_infra = float(props.get("dim_infrastructure_deficit", 0))
    d_coast = float(props.get("dim_coastal_exposure", 0))

    return f"""
    <div style="font-family:'Inter','Segoe UI',sans-serif;min-width:240px;">
        <div style="font-weight:700;font-size:14px;color:#0f172a;margin-bottom:4px;">
            {name}
        </div>
        <div style="font-size:11px;color:#64748b;margin-bottom:8px;">
            {unit_type} · {district} District
        </div>
        <div style="background:{color}22;border:1px solid {color};border-radius:6px;
                    padding:4px 8px;margin-bottom:8px;font-size:11px;font-weight:600;
                    color:{color};display:inline-block;">
            Cluster {cid}: {clabel}
        </div>
        <table style="width:100%;font-size:11px;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:3px 0;color:#64748b;">D<sub>health</sub></td>
                <td style="text-align:right;font-weight:600;color:#0f172a;">{d_health:.4f}</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:3px 0;color:#64748b;">D<sub>demo</sub></td>
                <td style="text-align:right;font-weight:600;color:#0f172a;">{d_demo:.4f}</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:3px 0;color:#64748b;">D<sub>infra</sub></td>
                <td style="text-align:right;font-weight:600;color:#0f172a;">{d_infra:.4f}</td>
            </tr>
            <tr>
                <td style="padding:3px 0;color:#64748b;">D<sub>coast</sub></td>
                <td style="text-align:right;font-weight:600;color:#0f172a;">{d_coast:.4f}</td>
            </tr>
        </table>
    </div>
    """


def generate_cluster_map() -> folium.Map:
    """Build and return the Folium cluster map."""
    # ------------------------------------------------------------------
    # 1. Load clustering data
    # ------------------------------------------------------------------
    gdf = gpd.read_parquet(DATA_PATH)
    assert len(gdf) == 622, f"Expected 622 units, got {len(gdf)}"
    assert gdf["vulnerability_archetype_id"].notna().all(), "Missing cluster IDs"
    assert set(gdf["vulnerability_archetype_id"].unique()) == {0, 1, 2, 3}

    # Verify actual label-to-id mapping matches our constants
    for cid, expected_label in CLUSTER_LABELS.items():
        actual = gdf.loc[
            gdf["vulnerability_archetype_id"] == cid,
            "vulnerability_archetype_label",
        ].iloc[0]
        assert actual == expected_label, (
            f"Cluster {cid}: expected '{expected_label}', got '{actual}'"
        )

    # Simplify geometries for responsive browser rendering
    gdf["geom_display"] = gdf.geometry.simplify(tolerance=0.001)

    # Attach risk assessment and prioritization attributes
    gdf["risk_tier"] = gdf["vulnerability_archetype_id"].map(
        lambda c: CLUSTER_RISK_PROFILES[c]["risk_tier"]
    )
    gdf["primary_hazard"] = gdf["vulnerability_archetype_id"].map(
        lambda c: CLUSTER_RISK_PROFILES[c]["primary_hazard"]
    )
    gdf["priority_rank_str"] = gdf["priority_rank"].astype(str) + " of 622"
    gdf["priority_action"] = gdf["vulnerability_archetype_id"].map(
        lambda c: CLUSTER_RISK_PROFILES[c]["primary_action"]
    )

    # Build a lightweight GeoDataFrame for the map
    display_cols = [
        "shrid2",
        "unit_name",
        "unit_type",
        "district_name",
        "vulnerability_archetype_id",
        "vulnerability_archetype_label",
        "risk_tier",
        "priority_rank_str",
        "primary_hazard",
        "priority_action",
        "dim_healthcare_deficit",
        "dim_demographic_vulnerability",
        "dim_infrastructure_deficit",
        "dim_coastal_exposure",
        "geom_display",
    ]
    map_gdf = gdf[display_cols].copy().set_geometry("geom_display")

    # ------------------------------------------------------------------
    # 2. Create the base Folium Map
    # ------------------------------------------------------------------
    bounds = map_gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
        control_scale=True,
        prefer_canvas=True,
    )

    # Fit map to polygon bounds
    fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # ------------------------------------------------------------------
    # 3. Add cluster-colored polygon layer
    # ------------------------------------------------------------------
    def style_function(feature):
        cid = feature["properties"].get("vulnerability_archetype_id")
        cid = int(cid) if cid is not None else 0
        return {
            "fillColor": CLUSTER_COLORS.get(cid, "#94a3b8"),
            "color": "#334155",
            "weight": 0.6,
            "fillOpacity": 0.68,
        }

    def highlight_function(feature):
        return {
            "weight": 2.5,
            "color": "#0ea5e9",
            "fillOpacity": 0.88,
        }

    tooltip = GeoJsonTooltip(
        fields=[
            "unit_name",
            "district_name",
            "vulnerability_archetype_label",
            "risk_tier",
        ],
        aliases=["Area:", "District:", "Archetype:", "Risk Tier:"],
        localize=True,
        sticky=False,
        style=(
            "font-family: Inter, 'Segoe UI', sans-serif;"
            "font-size: 12px;"
            "background: #0f172a;"
            "color: #f8fafc;"
            "padding: 8px 10px;"
            "border-radius: 6px;"
            "border: 1px solid #334155;"
        ),
    )

    popup = GeoJsonPopup(
        fields=[
            "unit_name",
            "district_name",
            "unit_type",
            "vulnerability_archetype_id",
            "vulnerability_archetype_label",
            "risk_tier",
            "priority_rank_str",
            "primary_hazard",
            "priority_action",
            "dim_healthcare_deficit",
            "dim_demographic_vulnerability",
            "dim_infrastructure_deficit",
            "dim_coastal_exposure",
        ],
        aliases=[
            "Area Name",
            "District",
            "Unit Type",
            "Cluster ID",
            "Archetype",
            "Risk Tier",
            "Priority Rank",
            "Primary Hazard Threat",
            "Recommended Action",
            "D_health",
            "D_demo",
            "D_infra",
            "D_coast",
        ],
        localize=True,
        style=(
            "font-family: Inter, 'Segoe UI', sans-serif;"
            "font-size: 12px;"
            "color: #1e293b;"
        ),
    )

    geojson_layer = folium.GeoJson(
        map_gdf,
        name="Vulnerability Clusters (K-Means, k=4)",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
        popup=popup,
    )
    geojson_layer.add_to(fmap)

    # ------------------------------------------------------------------
    # 4. Add coastline overlay if available
    # ------------------------------------------------------------------
    if COASTLINE_PATH.exists():
        coast_gdf = gpd.read_file(COASTLINE_PATH)
        folium.GeoJson(
            coast_gdf,
            name="Arabian Sea Coastline (10m)",
            style_function=lambda x: {
                "color": "#0284c7",
                "weight": 3.0,
                "opacity": 0.85,
            },
            tooltip=folium.Tooltip(
                "Arabian Sea Coastline (Natural Earth 10m)",
                style="font-size: 11px;",
            ),
        ).add_to(fmap)

    # ------------------------------------------------------------------
    # 5. Add layer control and legend
    # ------------------------------------------------------------------
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    # Inject custom HTML legend
    legend_html = _build_legend_html()
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # ------------------------------------------------------------------
    # 6. Add title bar overlay
    # ------------------------------------------------------------------
    title_html = """
    <div style="
        position: fixed;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: rgba(15, 23, 42, 0.92);
        color: #f8fafc;
        padding: 10px 24px;
        border-radius: 10px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: -0.015em;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.1);
        pointer-events: none;
    ">
        🌀 Cyclone Preparedness — Vulnerability Archetypes (K-Means, k=4)
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    return fmap


def main():
    """Entry point: generate and save the cluster map."""
    print("Loading clustering data from:", DATA_PATH)
    fmap = generate_cluster_map()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(OUTPUT_PATH))
    print(f"✅ Cluster map saved to: {OUTPUT_PATH}")
    print(f"   Open in browser: file://{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
