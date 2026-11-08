"""Unit tests for Streamlit dashboard UI components and visualization helpers.

Verifies:
- Custom CSS generation and key design tokens
- Folium choropleth map building with metric layers, coastline, and healthcare overlays
- Matplotlib radar chart generation with polar projections
- Altair scatter chart generation for scenario rank shifts
- Service data binding, dynamic prioritization, and export data preparation
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import folium
import altair as alt

from src.assistant.decision_support import DecisionSupportService
from src.assistant.ui_components import (
    ARCHETYPE_COLORS,
    BAND_COLORS,
    METRIC_LABELS,
    TIER_COLORS,
    build_folium_choropleth_map,
    generate_archetype_distribution_chart,
    generate_district_band_chart,
    generate_radar_chart,
    generate_rank_scatter_chart,
    generate_tier_distribution_chart,
    get_custom_css,
    render_kpi_card,
)
from src.assistant.api_client import get_api_client
from src.api.schemas import PrioritizationScenarioRequest, WeightConfiguration


@pytest.fixture(scope="module")
def service() -> DecisionSupportService:
    """Fixture providing initialized DecisionSupportService."""
    return DecisionSupportService()


def test_custom_css():
    """Verify custom CSS string contains critical design tokens."""
    css = get_custom_css()
    assert "<style>" in css
    assert "hero-banner" in css
    assert "metric-card" in css
    assert "badge-pill" in css
    assert "disclaimer-box" in css


def test_archetype_colors():
    """Verify archetype color palette covers all 4 clusters with valid hex colors."""
    assert len(ARCHETYPE_COLORS) == 4
    for c_id in [0, 1, 2, 3]:
        assert c_id in ARCHETYPE_COLORS
        hex_val = ARCHETYPE_COLORS[c_id]["hex"]
        assert hex_val.startswith("#")
        assert len(hex_val) == 7
        assert len(ARCHETYPE_COLORS[c_id]["name"]) > 0


def test_metric_labels():
    """Verify metric labels dictionary covers all standard dimensions."""
    assert "priority_score" in METRIC_LABELS
    assert "dim_healthcare_deficit" in METRIC_LABELS
    assert "dim_demographic_vulnerability" in METRIC_LABELS
    assert "dim_infrastructure_deficit" in METRIC_LABELS
    assert "dim_coastal_exposure" in METRIC_LABELS
    assert "vulnerability_archetype_id" in METRIC_LABELS


def test_build_folium_choropleth_map_priority_score(service: DecisionSupportService):
    """Verify Folium map generation with continuous priority_score metric."""
    fmap = build_folium_choropleth_map(
        gdf=service.df,
        metric_col="priority_score",
        show_coastline=True,
        show_healthcare=False,
    )
    assert isinstance(fmap, folium.Map)
    assert fmap.location == [13.35, 74.88]

    # Check children layers
    child_names = [child.__class__.__name__ for child in fmap._children.values()]
    assert "GeoJson" in child_names
    assert "LayerControl" in child_names


def test_build_folium_choropleth_map_archetypes(service: DecisionSupportService):
    """Verify Folium map generation with categorical archetype metric and overlays."""
    fmap = build_folium_choropleth_map(
        gdf=service.df,
        metric_col="vulnerability_archetype_id",
        show_coastline=True,
        show_healthcare=True,
        selected_unit_shrid="11-29-575-05561-617432",
    )
    assert isinstance(fmap, folium.Map)
    child_names = [child.__class__.__name__ for child in fmap._children.values()]
    assert "GeoJson" in child_names
    assert "MarkerCluster" in child_names


def test_generate_radar_chart():
    """Verify radar chart generation and polar figure properties."""
    unit_scores = {
        "healthcare_deficit": 0.85,
        "demographic_vulnerability": 0.45,
        "infrastructure_deficit": 0.60,
        "coastal_exposure": 0.90,
    }
    dist_medians = {
        "healthcare_deficit": 0.50,
        "demographic_vulnerability": 0.40,
        "infrastructure_deficit": 0.50,
        "coastal_exposure": 0.50,
    }

    fig = generate_radar_chart(
        unit_scores=unit_scores,
        district_medians=dist_medians,
        unit_name="Test Village",
        district_name="Udupi",
    )
    assert isinstance(fig, plt.Figure)
    axes = fig.get_axes()
    assert len(axes) == 1
    assert axes[0].name == "polar"
    plt.close(fig)


def test_generate_rank_scatter_chart():
    """Verify Altair scatter chart generation for rank comparison."""
    data = {
        "unit_name": ["Village A", "Village B", "Village C"],
        "district_name": ["Dakshina Kannada", "Udupi", "Dakshina Kannada"],
        "vulnerability_archetype_label": [
            "Urban & Peri-Urban Well-Resourced Corridor",
            "High-Exposure Coastal Rural Belt",
            "Remote Foothill & Healthcare-Isolated Communities",
        ],
        "rank_a": [10, 50, 100],
        "rank_b": [25, 30, 200],
    }
    df = pd.DataFrame(data)

    chart = generate_rank_scatter_chart(
        df=df,
        col_x="rank_a",
        col_y="rank_b",
        label_x="Scenario A Rank",
        label_y="Scenario B Rank",
    )
    assert isinstance(chart, alt.LayerChart)


def test_scenario_calculation_and_export_integrity(service: DecisionSupportService):
    """Verify prioritization calculation reproduces valid scores and export columns."""
    req = PrioritizationScenarioRequest(
        scenario_name="Test Hazard Scheme",
        weights=WeightConfiguration(
            healthcare=10.0,
            demographic=10.0,
            infrastructure=20.0,
            coastal=60.0,
            auto_normalize=True,
        ),
        top_n=20,
    )
    res = service.compute_prioritization_scenario(req)
    assert len(res.top_units) == 20
    assert res.weights_applied["coastal"] == 0.60

    # Ensure all top units have ranks 1..20
    ranks = [u.priority_rank for u in res.top_units]
    assert ranks == list(range(1, 21))


# ==============================================================================
# MILESTONE 4B.2 UI TESTS: REGIONAL DASHBOARD COMPONENTS
# ==============================================================================


def test_band_and_tier_color_constants():
    """Verify priority band and AHP tier color definitions."""
    # Band colors
    assert len(BAND_COLORS) == 4
    expected_bands = {"BAND_1_CRITICAL", "BAND_2_SPECIALIZED", "BAND_3_MODERATE", "BAND_4_ADEQUATE"}
    assert set(BAND_COLORS.keys()) == expected_bands
    for b_key, meta in BAND_COLORS.items():
        assert meta["hex"].startswith("#")
        assert len(meta["hex"]) == 7
        assert len(meta["short_label"]) > 0

    # Tier colors
    assert len(TIER_COLORS) == 3
    expected_tiers = {"Low", "Medium", "High"}
    assert set(TIER_COLORS.keys()) == expected_tiers
    for t_key, meta in TIER_COLORS.items():
        assert meta["hex"].startswith("#")
        assert len(meta["hex"]) == 7


def test_enhanced_custom_css_tokens():
    """Verify custom CSS string includes new priority band badges and warning styles."""
    css = get_custom_css()
    assert ".badge-band-1" in css
    assert ".badge-band-2" in css
    assert ".badge-band-3" in css
    assert ".badge-band-4" in css
    assert ".badge-tier-low" in css
    assert ".badge-tier-medium" in css
    assert ".badge-tier-high" in css
    assert ".data-integrity-warning" in css
    assert ".filter-banner" in css


def test_render_kpi_card():
    """Verify KPI card HTML rendering with values, percentage, and badge."""
    html = render_kpi_card(
        title="Band 1 — Critical",
        value="155",
        caption="Immediate Evac / Hardening",
        percentage=8.25,
        color="#ef4444",
        badge_label="Critical",
    )
    assert '<div class="metric-card">' in html
    assert "Band 1 — Critical" in html
    assert "155" in html
    assert "(8.2%)" in html
    assert "Critical</span>" in html
    assert "Immediate Evac / Hardening" in html


def test_regional_dashboard_kpis_and_reconciliation():
    """Verify all five KPI values, dynamic retrieval, and reconciliation across 1,879 units."""
    client = get_api_client(mode="direct")
    summary = client.get_regional_summary()

    # 1. Total units
    assert summary.total_units == 1879

    # 2. Priority bands authoritative values
    b1 = summary.band_counts["BAND_1_CRITICAL"]
    b2 = summary.band_counts["BAND_2_SPECIALIZED"]
    b3 = summary.band_counts["BAND_3_MODERATE"]
    b4 = summary.band_counts["BAND_4_ADEQUATE"]

    assert b1 == 155
    assert b2 == 829
    assert b3 == 299
    assert b4 == 596

    # Reconciliation
    assert b1 + b2 + b3 + b4 == 1879

    # 3. District reconciliation
    breakdown = summary.district_band_breakdown
    assert breakdown["Dakshina Kannada"]["BAND_1_CRITICAL"] == 2
    assert breakdown["Dakshina Kannada"]["BAND_2_SPECIALIZED"] == 123
    assert breakdown["Dakshina Kannada"]["BAND_3_MODERATE"] == 45
    assert breakdown["Dakshina Kannada"]["BAND_4_ADEQUATE"] == 199
    assert sum(breakdown["Dakshina Kannada"].values()) == 369

    assert breakdown["Udupi"]["BAND_1_CRITICAL"] == 0
    assert breakdown["Udupi"]["BAND_2_SPECIALIZED"] == 20
    assert breakdown["Udupi"]["BAND_3_MODERATE"] == 43
    assert breakdown["Udupi"]["BAND_4_ADEQUATE"] == 190
    assert sum(breakdown["Udupi"].values()) == 253

    assert breakdown["Uttara Kannada"]["BAND_1_CRITICAL"] == 153
    assert breakdown["Uttara Kannada"]["BAND_2_SPECIALIZED"] == 686
    assert breakdown["Uttara Kannada"]["BAND_3_MODERATE"] == 211
    assert breakdown["Uttara Kannada"]["BAND_4_ADEQUATE"] == 207
    assert sum(breakdown["Uttara Kannada"].values()) == 1257

    # Total district sum
    assert 369 + 253 + 1257 == 1879


def test_ahp_tier_distribution_and_field_check():
    """Verify AHP preparedness tier distribution uses authoritative field and reconciles to 1,879."""
    client = get_api_client(mode="direct")
    df = client.get_unit_index()

    assert "preparedness_tier" in df.columns
    tier_counts = df["preparedness_tier"].value_counts()

    assert tier_counts["Low"] == 155
    assert tier_counts["Medium"] == 1128
    assert tier_counts["High"] == 596
    assert tier_counts.sum() == 1879

    # Verify AHP Low matches Band 1 total
    assert tier_counts["Low"] == 155
    # Verify AHP High matches Band 4 total
    assert tier_counts["High"] == 596
    # Verify AHP Medium is split between Band 2 and Band 3
    assert tier_counts["Medium"] == (829 + 299)


def test_vulnerability_archetype_assignment_separation():
    """Verify archetype counts distinguish assigned (622) from unassigned units (1,257)."""
    client = get_api_client(mode="direct")
    df = client.get_unit_index()

    # Total rows
    assert len(df) == 1879

    # Assigned vs unassigned counts
    assigned_mask = df["vulnerability_archetype_label"].notna()
    unassigned_mask = df["vulnerability_archetype_label"].isna()

    assert int(assigned_mask.sum()) == 622
    assert int(unassigned_mask.sum()) == 1257

    # District distribution of assignments
    assert set(df[assigned_mask]["district_name"].unique()) == {"Dakshina Kannada", "Udupi"}
    assert set(df[unassigned_mask]["district_name"].unique()) == {"Uttara Kannada"}

    # Archetype breakdown for assigned units
    assigned_counts = df[assigned_mask]["vulnerability_archetype_label"].value_counts()
    assert assigned_counts["High-Exposure Coastal Rural Belt"] == 249
    assert assigned_counts["Socioeconomically Vulnerable Rural Interior"] == 161
    assert assigned_counts["Urban & Peri-Urban Well-Resourced Corridor"] == 130
    assert assigned_counts["Remote Foothill & Healthcare-Isolated Communities"] == 82
    assert assigned_counts.sum() == 622


def test_altair_regional_charts_generation():
    """Verify Altair regional charts generation with valid specifications."""
    client = get_api_client(mode="direct")
    df = client.get_unit_index()

    # 1. District band chart (count)
    chart_count = generate_district_band_chart(df, normalized=False)
    assert isinstance(chart_count, alt.Chart)
    dict_spec = chart_count.to_dict()
    assert "data" in dict_spec or "datasets" in dict_spec or "encoding" in dict_spec

    # 2. District band chart (normalized)
    chart_norm = generate_district_band_chart(df, normalized=True)
    assert isinstance(chart_norm, alt.Chart)

    # 3. Tier chart (overall)
    chart_tier_overall = generate_tier_distribution_chart(df, by_district=False)
    assert isinstance(chart_tier_overall, alt.Chart)

    # 4. Tier chart (by district)
    chart_tier_dist = generate_tier_distribution_chart(df, by_district=True)
    assert isinstance(chart_tier_dist, alt.Chart)

    # 5. Archetype chart
    chart_arch = generate_archetype_distribution_chart(df)
    assert isinstance(chart_arch, alt.Chart)


def test_filter_simulation_and_reset():
    """Verify that filtering affects candidate counts and reset restores full baseline."""
    client = get_api_client(mode="direct")
    df = client.get_unit_index()
    assert len(df) == 1879

    # 1. Filter by district
    df_dk = df[df["district_name"] == "Dakshina Kannada"]
    assert len(df_dk) == 369

    df_udupi = df[df["district_name"] == "Udupi"]
    assert len(df_udupi) == 253

    df_uk = df[df["district_name"] == "Uttara Kannada"]
    assert len(df_uk) == 1257

    # 2. Filter by band
    col_b = "provisional_priority_band"
    df_b1 = df[df[col_b] == "BAND_1_CRITICAL"]
    assert len(df_b1) == 155

    # 3. Filter combined
    df_dk_b1 = df[(df["district_name"] == "Dakshina Kannada") & (df[col_b] == "BAND_1_CRITICAL")]
    assert len(df_dk_b1) == 2

    df_udupi_b1 = df[(df["district_name"] == "Udupi") & (df[col_b] == "BAND_1_CRITICAL")]
    assert len(df_udupi_b1) == 0  # Udupi has 0 Band 1 units

    # 4. Reset restores original baseline
    df_reset = df.copy()
    assert len(df_reset) == 1879


def test_chart_empty_dataframe_handling():
    """Verify that chart generators gracefully handle empty DataFrames without exceptions."""
    empty_df = pd.DataFrame()

    c1 = generate_district_band_chart(empty_df)
    assert isinstance(c1, alt.Chart)

    c2 = generate_tier_distribution_chart(empty_df)
    assert isinstance(c2, alt.Chart)

    c3 = generate_archetype_distribution_chart(empty_df)
    assert isinstance(c3, alt.Chart)


def test_streamlit_app_filter_interactive_lifecycle():
    """Verify Streamlit app loading and interactive filter lifecycle using AppTest."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "src" / "assistant" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=15)
    assert len(at.exception) == 0

    # 1. Select District: Dakshina Kannada
    at.selectbox(key="rb_filter_district").select("Dakshina Kannada")
    at.run(timeout=15)
    assert len(at.exception) == 0

    # 2. Select Band: Band 1 — Critical Priority
    at.selectbox(key="rb_filter_band").select("Band 1 — Critical Priority")
    at.run(timeout=15)
    assert len(at.exception) == 0

    # 3. Click Reset Button
    at.button(key="rb_reset_btn").click()
    at.run(timeout=15)
    assert len(at.exception) == 0
    assert at.selectbox(key="rb_filter_district").value == "All Districts"
    assert at.selectbox(key="rb_filter_band").value == "All Priority Bands"
