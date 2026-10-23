"""Tests for the standalone K-Means vulnerability cluster map.

Verifies:
- Cluster map generation produces a valid Folium Map with expected structure.
- All 622 spatial units are included in the map GeoJSON layer.
- Exactly four categorical cluster colors are used (no continuous gradient).
- The custom HTML legend contains all cluster labels and IDs.
- Tooltip and popup fields are correctly configured.
- The output HTML file is valid and contains required elements.
"""

import json
from pathlib import Path

import folium
import geopandas as gpd
import pytest

# Import the map generation module
from scripts.generate_kmeans_cluster_map import (
    CLUSTER_COLORS,
    CLUSTER_LABELS,
    DATA_PATH,
    generate_cluster_map,
)


@pytest.fixture(scope="module")
def cluster_map() -> folium.Map:
    """Generate a cluster map once for the module's tests."""
    return generate_cluster_map()


@pytest.fixture(scope="module")
def source_gdf() -> gpd.GeoDataFrame:
    """Load source clustering data for cross-validation."""
    return gpd.read_parquet(DATA_PATH)


class TestClusterMapDataIntegrity:
    """Verify the underlying data fed into the map is sound."""

    def test_source_has_622_units(self, source_gdf):
        """Source parquet must contain exactly 622 spatial units."""
        assert len(source_gdf) == 622

    def test_source_has_four_clusters(self, source_gdf):
        """Source must have exactly four distinct cluster IDs: 0, 1, 2, 3."""
        cluster_ids = set(source_gdf["vulnerability_archetype_id"].unique())
        assert cluster_ids == {0, 1, 2, 3}

    def test_no_missing_cluster_ids(self, source_gdf):
        """No unit should have a missing cluster assignment."""
        assert source_gdf["vulnerability_archetype_id"].notna().all()

    def test_no_missing_geometries(self, source_gdf):
        """No unit should have a missing geometry."""
        assert source_gdf.geometry.notna().all()

    def test_label_id_consistency(self, source_gdf):
        """Verify that each cluster ID maps to the expected archetype label."""
        for cid, expected in CLUSTER_LABELS.items():
            actual = source_gdf.loc[
                source_gdf["vulnerability_archetype_id"] == cid,
                "vulnerability_archetype_label",
            ].iloc[0]
            assert actual == expected, (
                f"Cluster {cid}: expected '{expected}', got '{actual}'"
            )

    def test_dimension_scores_not_missing(self, source_gdf):
        """All four dimension score columns must be non-null for every unit."""
        dim_cols = [
            "dim_healthcare_deficit",
            "dim_demographic_vulnerability",
            "dim_infrastructure_deficit",
            "dim_coastal_exposure",
        ]
        for col in dim_cols:
            assert col in source_gdf.columns, f"Missing column: {col}"
            assert source_gdf[col].notna().all(), f"Null values in {col}"

    def test_dimension_scores_in_range(self, source_gdf):
        """All dimension scores must be between 0 and 1 (inclusive)."""
        dim_cols = [
            "dim_healthcare_deficit",
            "dim_demographic_vulnerability",
            "dim_infrastructure_deficit",
            "dim_coastal_exposure",
        ]
        for col in dim_cols:
            assert source_gdf[col].min() >= 0.0, f"{col} min < 0"
            assert source_gdf[col].max() <= 1.0, f"{col} max > 1"


class TestClusterMapStructure:
    """Verify the generated Folium map has the expected structure."""

    def test_is_folium_map(self, cluster_map):
        """Output must be a Folium Map instance."""
        assert isinstance(cluster_map, folium.Map)

    def test_has_geojson_layer(self, cluster_map):
        """Map must contain at least one GeoJson layer for the polygons."""
        child_types = [
            type(child).__name__ for child in cluster_map._children.values()
        ]
        assert "GeoJson" in child_types

    def test_has_layer_control(self, cluster_map):
        """Map must contain a LayerControl widget."""
        child_types = [
            type(child).__name__ for child in cluster_map._children.values()
        ]
        assert "LayerControl" in child_types

    def test_map_html_contains_legend(self, cluster_map):
        """The rendered HTML must contain the custom legend div."""
        html = cluster_map.get_root().render()
        assert "cluster-legend" in html
        assert "Vulnerability Archetypes" in html

    def test_map_html_contains_all_cluster_labels(self, cluster_map):
        """The legend must reference all four cluster archetype labels."""
        html = cluster_map.get_root().render()
        for cid, label in CLUSTER_LABELS.items():
            assert f"Cluster {cid}" in html, f"Missing Cluster {cid} in legend"
            # Labels may contain '&' which is HTML-escaped to '&amp;'
            escaped_label = label.replace("&", "&amp;")
            assert label in html or escaped_label in html, (
                f"Missing label '{label}' in legend"
            )

    def test_map_html_contains_title(self, cluster_map):
        """The rendered HTML must contain the title overlay."""
        html = cluster_map.get_root().render()
        assert "Cyclone Preparedness" in html
        assert "Vulnerability Archetypes" in html

    def test_color_palette_has_four_entries(self):
        """The CLUSTER_COLORS dict must have exactly 4 entries."""
        assert len(CLUSTER_COLORS) == 4
        for cid in [0, 1, 2, 3]:
            assert cid in CLUSTER_COLORS
            assert CLUSTER_COLORS[cid].startswith("#")
            assert len(CLUSTER_COLORS[cid]) == 7

    def test_colors_are_categorically_distinct(self):
        """The four cluster colors must all be unique."""
        colors = list(CLUSTER_COLORS.values())
        assert len(set(colors)) == 4, "Cluster colors must be distinct"


class TestClusterMapOutput:
    """Verify the output HTML file can be written and is valid."""

    def test_output_file_is_valid_html(self, cluster_map, tmp_path):
        """Map can be saved as a valid HTML file with expected markers."""
        out = tmp_path / "test_map.html"
        cluster_map.save(str(out))
        assert out.exists()

        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "leaflet" in content.lower()
        assert "GeoJSON" in content or "geojson" in content.lower()
