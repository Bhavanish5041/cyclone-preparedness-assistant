"""Unit tests for census processing, key standardization, and spatial join validation."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.data.census import (
    CENSUS_VINTAGE_NOTE,
    validate_spatial_boundary_join,
)


def test_validate_spatial_boundary_join():
    """Verify join validation correctly counts exact matches, unmatched keys, and duplicate keys."""
    # Mock census DataFrame
    df_census = pd.DataFrame(
        {
            "pc11_id": ["608660", "608661", "999999"],  # 999999 is in census only
            "unit_name": ["Village A", "Village B", "Village C"],
            "total_population": [1000, 2000, 3000],
        }
    )

    # Mock boundary GeoDataFrame
    poly = Polygon([(74.5, 13.0), (74.6, 13.0), (74.6, 13.1), (74.5, 13.1)])
    gdf_boundaries = gpd.GeoDataFrame(
        {
            "pc11_id": ["608660", "608661", "888888"],  # 888888 is in boundaries only
            "district_name": ["Udupi", "Udupi", "Udupi"],
        },
        geometry=[poly, poly, poly],
        crs="EPSG:4326",
    )

    metrics = validate_spatial_boundary_join(df_census, gdf_boundaries)

    assert metrics["total_census_units"] == 3
    assert metrics["total_boundary_polygons"] == 3
    assert metrics["exact_pc11_matches"] == 2
    assert metrics["match_percentage_of_polygons"] == round(2 / 3 * 100, 2)
    assert metrics["unmatched_census_units_count"] == 1  # 999999
    assert metrics["unmatched_polygon_keys_count"] == 1  # 888888
    assert metrics["census_duplicate_key_count"] == 0


def test_duplicate_key_detection_in_census():
    """Verify detection of duplicate primary keys in census data."""
    df_census = pd.DataFrame(
        {
            "pc11_id": ["608660", "608660", "608661"],  # 608660 duplicated
            "total_population": [1000, 1050, 2000],
        }
    )
    poly = Polygon([(74.5, 13.0), (74.6, 13.0), (74.6, 13.1), (74.5, 13.1)])
    gdf_boundaries = gpd.GeoDataFrame(
        {"pc11_id": ["608660", "608661"]},
        geometry=[poly, poly],
        crs="EPSG:4326",
    )

    metrics = validate_spatial_boundary_join(df_census, gdf_boundaries)
    assert metrics["census_duplicate_key_count"] == 2  # Both duplicate rows flagged


def test_census_vintage_label_is_explicit():
    """Verify that the vintage disclaimer contains historical warning."""
    assert "2011" in CENSUS_VINTAGE_NOTE
    assert "historical" in CENSUS_VINTAGE_NOTE.lower()
    assert "not" in CENSUS_VINTAGE_NOTE.lower()
