"""Comprehensive test suite for Phase 3.1 UI Redesign and Information Architecture.

Verifies:
- Reusable UI component renderers and CSS tokens in Warm Chocolate aesthetic.
- Information architecture with all seven navigation views.
- Strict scientific preservation: 1,879 regional units, 622 archetypes, 1,257 Uttara Kannada fallback units.
- Authoritative priority band counts (155, 829, 299, 596).
- Evaluation of communities across all three coastal districts (DK, Udupi, and Uttara Kannada).
- Regional GIS Map generation for administrative unit markers and choropleth overlays.
- Report dossier generation and CSV data exports.
- Multi-page navigation lifecycle using Streamlit AppTest without emojis or em dashes.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.assistant.api_client import DecisionSupportApiClient, get_api_client
from src.assistant.charts import (
    generate_archetype_distribution_chart,
    generate_archetype_feature_chart,
    generate_district_band_chart,
    generate_radar_chart,
    generate_rank_scatter_chart,
    generate_tier_distribution_chart,
)
from src.assistant.maps import build_folium_choropleth_map, build_regional_gis_map
from src.assistant.theme import (
    ARCHETYPE_COLORS,
    BAND_COLORS,
    METRIC_LABELS,
    THEME_COLORS,
    TIER_COLORS,
    get_custom_css,
    render_disclaimer,
    render_empty_state,
    render_filter_banner,
    render_header,
    render_integrity_alert,
    render_kpi_card,
)
from src.assistant.views import (
    render_ai_readiness_page,
    render_archetypes_page,
    render_community_profile_page,
    render_gis_explorer_page,
    render_overview_page,
    render_reports_page,
    render_scenario_analysis_page,
)


@pytest.fixture(scope="module")
def client() -> DecisionSupportApiClient:
    """Fixture providing initialized direct-mode DecisionSupportApiClient."""
    return get_api_client(mode="direct")


def test_theme_tokens_and_html_builders():
    """Verify theme tokens and HTML helper generators."""
    assert len(THEME_COLORS) >= 10
    assert THEME_COLORS["background_base"] == "#211914"
    assert THEME_COLORS["accent_primary"] == "#C18B62"
    assert THEME_COLORS["surface_card"] == "#342820"

    # Header renderer
    hdr = render_header("Test Title", "Test Subtitle", "Badge 1")
    assert '<div class="hero-banner">' in hdr
    assert "Test Title" in hdr
    assert "Test Subtitle" in hdr
    assert "Badge 1" in hdr

    # Disclaimer renderer
    disc = render_disclaimer("Disclaimed text", "Test Title")
    assert '<div class="disclaimer-box">' in disc
    assert "<b>Test Title:</b> Disclaimed text" in disc

    # Integrity alert renderer
    alert = render_integrity_alert("Data mismatch alert")
    assert '<div class="data-integrity-warning">' in alert
    assert "Data mismatch alert" in alert

    # Filter banner renderer
    filt = render_filter_banner("Active filter scope")
    assert '<div class="filter-banner">' in filt
    assert "Active filter scope" in filt

    # Empty state renderer
    empty = render_empty_state("No Records", "Adjust filter criteria")
    assert '<div class="empty-state">' in empty
    assert "No Records" in empty
    assert "Adjust filter criteria" in empty


def test_scientific_integrity_record_reconciliation(client: DecisionSupportApiClient):
    """Verify exact count preservation across all 1,879 study area records."""
    summary = client.get_regional_summary()
    assert summary.total_units == 1879

    # Four priority bands
    b1 = summary.band_counts["BAND_1_CRITICAL"]
    b2 = summary.band_counts["BAND_2_SPECIALIZED"]
    b3 = summary.band_counts["BAND_3_MODERATE"]
    b4 = summary.band_counts["BAND_4_ADEQUATE"]

    assert b1 == 155
    assert b2 == 829
    assert b3 == 299
    assert b4 == 596
    assert b1 + b2 + b3 + b4 == 1879

    # District unit counts
    df_index = client.get_unit_index()
    assert len(df_index) == 1879

    dist_counts = df_index["district_name"].value_counts()
    assert dist_counts["Dakshina Kannada"] == 369
    assert dist_counts["Udupi"] == 253
    assert dist_counts["Uttara Kannada"] == 1257
    assert dist_counts.sum() == 1879

    # Archetype assignments
    assigned_mask = df_index["vulnerability_archetype_label"].notna()
    assert int(assigned_mask.sum()) == 622
    assert int((~assigned_mask).sum()) == 1257


def test_community_evaluation_all_districts(client: DecisionSupportApiClient):
    """Verify community evaluation functions across Dakshina Kannada, Udupi, and Uttara Kannada."""
    # 1. Dakshina Kannada unit
    dk_unit = client.get_unit("Shimanthur")
    assert dk_unit.district_name == "Dakshina Kannada"
    assert dk_unit.preparedness_score > 0
    assert dk_unit.provisional_priority_band in {"BAND_1_CRITICAL", "BAND_2_SPECIALIZED", "BAND_3_MODERATE", "BAND_4_ADEQUATE"}

    # 2. Udupi unit
    udupi_unit = client.get_unit("Tallur")
    assert udupi_unit.district_name == "Udupi"
    assert udupi_unit.preparedness_score > 0
    assert udupi_unit.provisional_priority_band == "BAND_4_ADEQUATE"

    # 3. Uttara Kannada unit (formerly unsupported in old community view)
    uk_unit = client.get_unit("Mudbhatkal")
    assert uk_unit.district_name == "Uttara Kannada"
    assert uk_unit.preparedness_score > 0
    assert uk_unit.provisional_priority_band in {"BAND_1_CRITICAL", "BAND_2_SPECIALIZED", "BAND_3_MODERATE", "BAND_4_ADEQUATE"}


def test_regional_gis_map_generation(client: DecisionSupportApiClient):
    """Verify regional GIS map generation with markers across all 1,879 units."""
    df_index = client.get_unit_index()

    # Map with priority band coloring
    fmap_band = build_regional_gis_map(
        unit_df=df_index,
        polygon_gdf=client.service.df,
        color_by="priority_band",
        show_polygons=False,
        show_coastline=False,
        show_healthcare=False,
        selected_pc11_id="608691",
        max_markers=200,
    )
    import folium
    assert isinstance(fmap_band, folium.Map)
    assert len(fmap_band._children) > 0

    # Map with AHP tier coloring
    fmap_tier = build_regional_gis_map(
        unit_df=df_index,
        polygon_gdf=client.service.df,
        color_by="preparedness_tier",
        show_polygons=True,
        show_coastline=False,
        show_healthcare=False,
        max_markers=100,
    )
    assert isinstance(fmap_tier, folium.Map)


def test_archetype_feature_chart_generation(client: DecisionSupportApiClient):
    """Verify generation of comparative archetype feature profile chart."""
    catalog = client.service.get_archetype_catalog()
    dim_means = {arch.archetype_id: arch.dimension_means for arch in catalog.archetypes}
    chart = generate_archetype_feature_chart(dim_means)
    import altair as alt
    assert isinstance(chart, alt.Chart)
    spec = chart.to_dict()
    assert "encoding" in spec


def test_reports_data_exports(client: DecisionSupportApiClient):
    """Verify data exports generated for reports page."""
    df_index = client.get_unit_index()

    # Priority band CSV export
    b1_units = df_index[df_index["provisional_priority_band"] == "BAND_1_CRITICAL"]
    assert len(b1_units) == 155
    csv_b1 = b1_units.to_csv(index=False)
    assert "pc11_id" in csv_b1
    assert "BAND_1_CRITICAL" in csv_b1

    # Single community incident brief verification
    eval_tallur = client.get_unit("Tallur")
    assert eval_tallur.village_name == "Tallur"
    assert eval_tallur.preparedness_score == 74.34
    assert len(eval_tallur.recommended_advisories) > 0


def test_apptest_navigation_all_seven_sections():
    """Verify multi-page navigation across all seven sections using Streamlit AppTest."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "src" / "assistant" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=20)
    assert len(at.exception) == 0

    # Test Section 1: Overview (default)
    assert at.sidebar.radio(key="platform_nav_radio").value == "Overview"

    # Test Section 2: Regional GIS Explorer
    at.sidebar.radio(key="platform_nav_radio").set_value("Regional GIS Explorer")
    at.run(timeout=20)
    assert len(at.exception) == 0

    # Test Section 3: Community Profiles
    at.sidebar.radio(key="platform_nav_radio").set_value("Community Profiles")
    at.run(timeout=20)
    assert len(at.exception) == 0

    # Test Section 4: Scenario Analysis
    at.sidebar.radio(key="platform_nav_radio").set_value("Scenario Analysis")
    at.run(timeout=20)
    assert len(at.exception) == 0

    # Test Section 5: Vulnerability Archetypes
    at.sidebar.radio(key="platform_nav_radio").set_value("Vulnerability Archetypes")
    at.run(timeout=20)
    assert len(at.exception) == 0

    # Test Section 6: Reports and Export
    at.sidebar.radio(key="platform_nav_radio").set_value("Reports and Export")
    at.run(timeout=20)
    assert len(at.exception) == 0

    # Test Section 7: AI Readiness
    at.sidebar.radio(key="platform_nav_radio").set_value("AI Readiness (Preview)")
    at.run(timeout=20)
    assert len(at.exception) == 0
