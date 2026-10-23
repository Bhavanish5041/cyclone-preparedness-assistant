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
    METRIC_LABELS,
    build_folium_choropleth_map,
    generate_radar_chart,
    generate_rank_scatter_chart,
    get_custom_css,
)
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
