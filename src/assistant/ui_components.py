"""Reusable UI components, geospatial map builders, and visual styling for Streamlit.

Unified facade re-exporting components from:
- src.assistant.theme: Design tokens, color constants, CSS stylesheets, HTML component helpers.
- src.assistant.charts: Altair and Matplotlib chart generators.
- src.assistant.maps: Folium geospatial map builders.
"""

from src.assistant.theme import (
    ARCHETYPE_COLORS,
    BAND_COLORS,
    METRIC_LABELS,
    TIER_COLORS,
    THEME_COLORS,
    get_custom_css,
    render_disclaimer,
    render_empty_state,
    render_filter_banner,
    render_header,
    render_integrity_alert,
    render_kpi_card,
)
from src.assistant.charts import (
    generate_archetype_distribution_chart,
    generate_archetype_feature_chart,
    generate_district_band_chart,
    generate_radar_chart,
    generate_rank_scatter_chart,
    generate_tier_distribution_chart,
)
from src.assistant.maps import (
    build_folium_choropleth_map,
    build_regional_gis_map,
)

__all__ = [
    "ARCHETYPE_COLORS",
    "BAND_COLORS",
    "METRIC_LABELS",
    "TIER_COLORS",
    "THEME_COLORS",
    "get_custom_css",
    "render_disclaimer",
    "render_empty_state",
    "render_filter_banner",
    "render_header",
    "render_integrity_alert",
    "render_kpi_card",
    "generate_archetype_distribution_chart",
    "generate_archetype_feature_chart",
    "generate_district_band_chart",
    "generate_radar_chart",
    "generate_rank_scatter_chart",
    "generate_tier_distribution_chart",
    "build_folium_choropleth_map",
    "build_regional_gis_map",
]
