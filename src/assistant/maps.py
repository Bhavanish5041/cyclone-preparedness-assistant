"""Geospatial Folium map builders and layer overlay helpers in warm chocolate palette.

Provides:
- build_folium_choropleth_map: Backward-compatible choropleth map generator with metric shading.
- build_regional_gis_map: High-performance geospatial map supporting both choropleth polygons
  and administrative unit circle markers across all 1,879 coastal Karnataka units.
- Coordinated warm tones: #C95D4A, #D48B47, #A88E75, #7A8B63.
- Strictly zero emojis and zero em dashes in tooltips, popups, and layer controls.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import branca.colormap as cm
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import pandas as pd

from src.assistant.theme import ARCHETYPE_COLORS, BAND_COLORS, METRIC_LABELS, TIER_COLORS

COASTLINE_DEFAULT_PATH = "data/interim/coastal_karnataka_ne_10m_coastline.geojson"
HEALTHCARE_DEFAULT_PATH = "data/interim/coastal_healthcare_facilities.parquet"


def build_folium_choropleth_map(
    gdf: gpd.GeoDataFrame,
    metric_col: str = "priority_score",
    show_coastline: bool = True,
    show_healthcare: bool = False,
    coastline_path: str = COASTLINE_DEFAULT_PATH,
    healthcare_path: str = HEALTHCARE_DEFAULT_PATH,
    selected_unit_shrid: Optional[str] = None,
) -> folium.Map:
    """Build an interactive Folium map with choropleth polygons and optional overlays.

    Args:
        gdf: GeoDataFrame containing units with geometry and metric columns.
        metric_col: Name of column to shade polygons by.
        show_coastline: If True, adds the Natural Earth 10m coastline overlay.
        show_healthcare: If True, adds the MoHFW healthcare facilities marker cluster.
        coastline_path: Path to coastline GeoJSON.
        healthcare_path: Path to healthcare parquet dataset.
        selected_unit_shrid: Optional shrid2 to highlight with a vibrant border.

    Returns:
        folium.Map instance.
    """
    center_lat, center_lon = 13.35, 74.88
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if gdf is None or gdf.empty:
        return fmap

    display_gdf = gdf.copy()
    if "geometry" in display_gdf.columns and not display_gdf.geometry.is_empty.all():
        display_gdf["geom_simple"] = display_gdf.geometry.simplify(0.001)
    else:
        return fmap

    is_archetype = metric_col == "vulnerability_archetype_id"
    colormap = None
    if not is_archetype and metric_col in display_gdf.columns:
        min_val = float(display_gdf[metric_col].min())
        max_val = float(display_gdf[metric_col].max())
        if max_val == min_val:
            max_val = min_val + 1.0
        # Warm coordinated earthen colormap: Sand -> Taupe -> Ochre -> Terracotta
        colormap = cm.LinearColormap(
            colors=["#E0CBB3", "#D6B28C", "#C18B62", "#A76F4E", "#C95D4A"],
            vmin=min_val,
            vmax=max_val,
            caption=METRIC_LABELS.get(metric_col, metric_col),
        )
        colormap.add_to(fmap)

    def style_function(feature):
        props = feature["properties"]
        val = props.get(metric_col)
        shrid = props.get("shrid2")

        if selected_unit_shrid and shrid == selected_unit_shrid:
            return {
                "fillColor": "#C18B62",
                "color": "#F3E9DC",
                "weight": 3.0,
                "fillOpacity": 0.85,
            }

        if is_archetype:
            c_id = int(val) if val is not None else 0
            color = ARCHETYPE_COLORS.get(c_id, {}).get("hex", "#918170")
            fill_opacity = 0.65
        else:
            color = colormap(val) if (colormap and val is not None) else "#403127"
            fill_opacity = 0.68

        return {
            "fillColor": color,
            "color": "#514033",
            "weight": 0.7,
            "fillOpacity": fill_opacity,
        }

    def highlight_function(feature):
        return {
            "weight": 2.2,
            "color": "#D6B28C",
            "fillOpacity": 0.88,
        }

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
    valid_fields = [f for f in fields_to_include if f in display_gdf.columns]
    poly_gdf = display_gdf[valid_fields].set_geometry("geom_simple")

    tooltip_cols = [c for c in ["unit_name", "district_name", "unit_type", "priority_rank", "priority_score", "vulnerability_archetype_label"] if c in poly_gdf.columns]
    alias_dict = {
        "unit_name": "Unit:",
        "district_name": "District:",
        "unit_type": "Type:",
        "priority_rank": "Priority Rank:",
        "priority_score": "Priority Score:",
        "vulnerability_archetype_label": "Archetype:",
    }
    tooltip_aliases = [alias_dict.get(c, c) for c in tooltip_cols]

    tooltip = folium.GeoJsonTooltip(
        fields=tooltip_cols,
        aliases=tooltip_aliases,
        localize=True,
        sticky=False,
        style="font-family: 'DM Sans', sans-serif; font-size: 12px; background: #2B211B; color: #F3E9DC; padding: 7px 11px; border-radius: 3px; border: 1px solid #514033;",
    )

    geojson_layer = folium.GeoJson(
        poly_gdf,
        name="Administrative Units (Choropleth)",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
    )
    geojson_layer.add_to(fmap)

    # Optional Coastline Overlay
    if show_coastline and Path(coastline_path).exists():
        try:
            coast_gdf = gpd.read_file(coastline_path)
            coast_layer = folium.GeoJson(
                coast_gdf,
                name="Arabian Sea Coastline (10m)",
                style_function=lambda x: {
                    "color": "#8D9A78",
                    "weight": 2.2,
                    "opacity": 0.85,
                },
                tooltip=folium.Tooltip("Arabian Sea Coastline (Natural Earth 10m)"),
            )
            coast_layer.add_to(fmap)
        except Exception:
            pass

    # Optional Healthcare Facilities Overlay
    if show_healthcare and Path(healthcare_path).exists():
        try:
            hosp_df = pd.read_parquet(healthcare_path)
            valid_hosp = hosp_df[hosp_df["latitude_num"].notna() & hosp_df["longitude_num"].notna()]
            cluster = MarkerCluster(name="Healthcare Facilities", show=True)

            tier_colors = {
                "HOSPITAL": "#C95D4A",
                "PHC_CHC": "#D48B47",
                "SUBCENTRE": "#A88E75",
            }

            for _, row in valid_hosp.iterrows():
                tier = str(row.get("service_tier", "SUBCENTRE"))
                f_name = str(row.get("facility_name_clean", "Health Facility"))
                f_type = str(row.get("facility_type_label", tier))
                f_color = tier_colors.get(tier, "#918170")

                popup_html = f"<div style=\"font-family: 'DM Sans', sans-serif; background:#2B211B; color:#F3E9DC; padding:4px;\"><b>{f_name}</b><br>Type: {f_type}<br>Tier: {tier}</div>"
                folium.CircleMarker(
                    location=[row["latitude_num"], row["longitude_num"]],
                    radius=4,
                    color=f_color,
                    fill=True,
                    fill_color=f_color,
                    fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=250),
                ).add_to(cluster)

            cluster.add_to(fmap)
        except Exception:
            pass

    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
    return fmap


def build_regional_gis_map(
    unit_df: pd.DataFrame,
    polygon_gdf: Optional[gpd.GeoDataFrame] = None,
    color_by: str = "priority_band",
    show_polygons: bool = False,
    show_coastline: bool = True,
    show_healthcare: bool = False,
    selected_pc11_id: Optional[str] = None,
    max_markers: int = 1879,
    coastline_path: str = COASTLINE_DEFAULT_PATH,
    healthcare_path: str = HEALTHCARE_DEFAULT_PATH,
) -> folium.Map:
    """Build an interactive regional GIS map supporting point markers for all 1,879 units
    and optional choropleth polygons in warm chocolate styling.

    Args:
        unit_df: DataFrame with unit locations, bands, tiers, and scores.
        polygon_gdf: Optional GeoDataFrame with unit polygon boundaries (622 units).
        color_by: 'priority_band' or 'preparedness_tier'.
        show_polygons: If True, adds polygon choropleth layer where geometry is available.
        show_coastline: If True, overlays Arabian Sea coastline.
        show_healthcare: If True, overlays health facility marker clusters.
        selected_pc11_id: Optional PC11 ID of a selected unit to highlight.
        max_markers: Maximum point markers to render (prevents browser lockups).
        coastline_path: Path to coastline GeoJSON.
        healthcare_path: Path to healthcare parquet dataset.

    Returns:
        folium.Map instance.
    """
    if not unit_df.empty and "lat" in unit_df.columns and "lon" in unit_df.columns:
        valid_coords = unit_df[unit_df["lat"].notna() & unit_df["lon"].notna()]
        if not valid_coords.empty:
            center_lat = float(valid_coords["lat"].median())
            center_lon = float(valid_coords["lon"].median())
        else:
            center_lat, center_lon = 13.80, 74.85
    else:
        center_lat, center_lon = 13.80, 74.85

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Optional Polygon Layer (DK & Udupi 622 units)
    if show_polygons and polygon_gdf is not None and not polygon_gdf.empty:
        try:
            poly_display = polygon_gdf.copy()
            poly_display["geom_simple"] = poly_display.geometry.simplify(0.001)

            def poly_style(feature):
                return {
                    "fillColor": "#A76F4E",
                    "color": "#514033",
                    "weight": 0.8,
                    "fillOpacity": 0.32,
                }

            folium.GeoJson(
                poly_display[["unit_name", "district_name", "geom_simple"]].set_geometry("geom_simple"),
                name="Administrative Boundaries (622 Units)",
                style_function=poly_style,
                tooltip=folium.GeoJsonTooltip(fields=["unit_name", "district_name"], aliases=["Unit:", "District:"]),
            ).add_to(fmap)
        except Exception:
            pass

    # 2. Administrative Unit Markers (CircleMarkers)
    display_units = unit_df.head(max_markers)
    band_col = "provisional_priority_band" if "provisional_priority_band" in display_units.columns else "priority_band"
    name_col = "village_name" if "village_name" in display_units.columns else "unit_name"

    marker_group = folium.FeatureGroup(name=f"Administrative Units ({len(display_units):,} markers)")

    for _, row in display_units.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue

        pc11 = str(row.get("pc11_id", ""))
        u_name = str(row.get(name_col, "Administrative Unit"))
        d_name = str(row.get("district_name", ""))
        b_val = str(row.get(band_col, "BAND_4_ADEQUATE"))
        t_val = str(row.get("preparedness_tier", "Medium"))
        score = float(row.get("preparedness_score", 0.0))
        p_rank = int(row.get("overall_priority_rank", 0))

        # Color mapping in warm coordinated tones
        if color_by == "preparedness_tier":
            c_meta = TIER_COLORS.get(t_val, {"hex": "#D48B47"})
            color = c_meta["hex"]
        else:
            b_meta = BAND_COLORS.get(b_val, {"hex": "#D48B47", "short_label": "Band"})
            color = b_meta["hex"]

        is_selected = selected_pc11_id and str(selected_pc11_id) == pc11

        radius = 6.5 if is_selected else (4.5 if b_val == "BAND_1_CRITICAL" else 3.8)
        weight = 2.5 if is_selected else 1.0
        border_color = "#F3E9DC" if is_selected else "#211914"

        clean_band_label = b_val.replace("_", " ").title()

        popup_html = f"""
        <div style="font-family: 'DM Sans', sans-serif; font-size: 12px; color: #F3E9DC; background: #2B211B; padding: 6px 10px; border-radius: 3px; border: 1px solid #514033; min-width: 175px;">
            <b style="font-size: 13px; color: #F3E9DC;">{u_name}</b><br>
            <span style="color: #918170;">District:</span> {d_name}<br>
            <span style="color: #918170;">AHP Score:</span> <b>{score:.1f}</b> ({t_val})<br>
            <span style="color: #918170;">Priority:</span> <b style="color: {color};">{clean_band_label}</b><br>
            <span style="color: #918170;">Rank:</span> #{p_rank}<br>
            <span style="color: #918170;">PC11 ID:</span> <code>{pc11}</code>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=border_color,
            weight=weight,
            fill=True,
            fill_color=color,
            fill_opacity=0.92 if is_selected else 0.78,
            tooltip=f"{u_name} ({d_name}) - {clean_band_label}",
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(marker_group)

    marker_group.add_to(fmap)

    # 3. Optional Coastline Overlay
    if show_coastline and Path(coastline_path).exists():
        try:
            coast_gdf = gpd.read_file(coastline_path)
            coast_layer = folium.GeoJson(
                coast_gdf,
                name="Arabian Sea Coastline (10m)",
                style_function=lambda x: {
                    "color": "#8D9A78",
                    "weight": 2.2,
                    "opacity": 0.85,
                },
                tooltip=folium.Tooltip("Arabian Sea Coastline"),
            )
            coast_layer.add_to(fmap)
        except Exception:
            pass

    # 4. Optional Healthcare Facilities Overlay
    if show_healthcare and Path(healthcare_path).exists():
        try:
            hosp_df = pd.read_parquet(healthcare_path)
            valid_hosp = hosp_df[hosp_df["latitude_num"].notna() & hosp_df["longitude_num"].notna()]
            cluster = MarkerCluster(name="Healthcare Facilities", show=True)

            tier_colors = {
                "HOSPITAL": "#C95D4A",
                "PHC_CHC": "#D48B47",
                "SUBCENTRE": "#A88E75",
            }

            for _, h_row in valid_hosp.iterrows():
                tier = str(h_row.get("service_tier", "SUBCENTRE"))
                f_name = str(h_row.get("facility_name_clean", "Health Facility"))
                f_type = str(h_row.get("facility_type_label", tier))
                f_color = tier_colors.get(tier, "#918170")

                popup_html = f"<div style=\"font-family: 'DM Sans', sans-serif; background:#2B211B; color:#F3E9DC; padding:4px;\"><b>{f_name}</b><br>Type: {f_type}<br>Tier: {tier}</div>"
                folium.CircleMarker(
                    location=[h_row["latitude_num"], h_row["longitude_num"]],
                    radius=3.5,
                    color=f_color,
                    fill=True,
                    fill_color=f_color,
                    fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=240),
                ).add_to(cluster)

            cluster.add_to(fmap)
        except Exception:
            pass

    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
    return fmap
