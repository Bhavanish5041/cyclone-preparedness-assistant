"""Unit tests for coastal proximity feature engineering and geodesic validation."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from src.features.coastal import (
    COASTLINE_PROVENANCE_METADATA,
    calculate_coastal_proximity_utm,
    load_and_validate_coastline,
    verify_coastal_distance_geodesic,
)


def test_coastline_provenance_metadata():
    """Verify coastline provenance dictionary contains all required auditing metadata."""
    required_keys = [
        "source_dataset",
        "source_organization",
        "source_url",
        "source_version",
        "license",
        "retrieval_date",
        "native_crs",
        "analysis_crs",
        "scale_resolution",
        "geographic_scope",
        "known_limitations",
        "intended_use",
    ]
    for key in required_keys:
        assert key in COASTLINE_PROVENANCE_METADATA, f"Missing provenance key: {key}"
        assert COASTLINE_PROVENANCE_METADATA[key] is not None
        assert len(str(COASTLINE_PROVENANCE_METADATA[key])) > 0


def test_calculate_coastal_proximity_synthetic_geometry():
    """Verify perpendicular distance calculation on a controlled synthetic line and points."""
    # Coastline: vertical meridian line at 74.0°E from 12.0°N to 14.0°N with intermediate vertices
    coast_line = LineString([(74.0, 12.0), (74.0, 13.0), (74.0, 14.0)])
    coast_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[coast_line], crs="EPSG:4326")

    # Point 1: exactly on the line at (74.0, 13.0)
    pt_on_line = Point(74.0, 13.0)
    # Point 2: slightly east (~0.05 deg east ~ 5.4 km)
    pt_east = Point(74.05, 13.0)

    dists_km, dists_m = calculate_coastal_proximity_utm([pt_on_line, pt_east], coast_gdf)

    assert len(dists_km) == 2
    # Point on line must have distance near zero (< 1 metre)
    assert dists_km[0] < 0.001
    assert dists_m[0] < 1.0

    # Point 2 must be approximately 5.4 km east
    assert 5.0 < dists_km[1] < 6.0


def test_calculate_coastal_proximity_empty_points():
    """Verify empty input points list returns empty numpy arrays without error."""
    coast_line = LineString([(74.0, 12.0), (74.0, 14.0)])
    coast_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[coast_line], crs="EPSG:4326")

    dists_km, dists_m = calculate_coastal_proximity_utm([], coast_gdf)
    assert len(dists_km) == 0
    assert len(dists_m) == 0


def test_load_and_validate_coastline_real_file():
    """Verify loading and clipping of the downloaded Natural Earth 10m coastline."""
    coastline_path = Path("data/interim/coastal_karnataka_ne_10m_coastline.geojson")
    if not coastline_path.exists():
        pytest.skip("Local clipped coastline file not present; skipping.")

    gdf = load_and_validate_coastline(coastline_path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0
    assert gdf.crs.to_epsg() == 4326


def test_geodesic_verification_tolerance():
    """Verify UTM 43N projected distance vs WGS84 Geod ellipsoidal distance discrepancy is < 50m."""
    coastline_path = Path("data/interim/coastal_karnataka_ne_10m_coastline.geojson")
    if not coastline_path.exists():
        pytest.skip("Local clipped coastline file not present; skipping.")

    coast_gdf = load_and_validate_coastline(coastline_path)

    # Test coordinate near Mangalore coast (12.87°N, 74.84°E)
    coastal_pt = Point(74.84, 12.87)
    d_utm_m, d_geod_m, diff_m = verify_coastal_distance_geodesic(coastal_pt, coast_gdf)

    # Both should be relatively close to the coast (< 5 km)
    assert d_utm_m < 5000.0
    assert d_geod_m < 5000.0
    # Discrepancy between UTM and ellipsoidal geodesic must be under 10 metres
    assert diff_m < 10.0

    # Test coordinate far inland in Western Ghats foothills (12.67°N, 75.60°E)
    inland_pt = Point(75.60, 12.67)
    d_utm_inland, d_geod_inland, diff_inland = verify_coastal_distance_geodesic(inland_pt, coast_gdf)

    assert d_utm_inland > 50000.0  # > 50 km
    # Discrepancy for 60 km distance must be well under 50 metres (< 0.05%)
    assert diff_inland < 50.0
