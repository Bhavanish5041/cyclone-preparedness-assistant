"""Unit tests for spatial accessibility feature engineering."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from src.features.accessibility import (
    count_facilities_within_boundaries,
    find_nearest_facilities,
    get_polygon_representative_point,
    haversine_distance_matrix,
)


def test_haversine_distance_known_distance():
    """Verify haversine distance calculation between known landmarks."""
    # Mangalore (12.9141, 74.8560) and Udupi (13.3408, 74.7421)
    # Expected straight-line distance is ~49 km
    lats1 = np.array([12.9141])
    lons1 = np.array([74.8560])
    lats2 = np.array([13.3408])
    lons2 = np.array([74.7421])

    dist = haversine_distance_matrix(lats1, lons1, lats2, lons2)
    assert dist.shape == (1, 1)
    assert 48.5 < dist[0, 0] < 49.5

    # Self-distance must be zero
    self_dist = haversine_distance_matrix(lats1, lons1, lats1, lons1)
    assert np.isclose(self_dist[0, 0], 0.0)


def test_representative_point_guaranteed_inside_nonconvex_polygon():
    """Verify that representative_point falls strictly inside a U-shaped non-convex polygon.

    For a U-shaped polygon, the geometric centroid falls outside the polygon (in the open gap),
    while representative_point() is guaranteed to be contained within the polygon interior.
    """
    # U-shaped polygon in lat/lon coordinates
    u_poly = Polygon(
        [
            (74.0, 13.0),
            (74.0, 13.3),
            (74.1, 13.3),
            (74.1, 13.1),
            (74.2, 13.1),
            (74.2, 13.3),
            (74.3, 13.3),
            (74.3, 13.0),
            (74.0, 13.0),
        ]
    )

    centroid = u_poly.centroid
    assert not u_poly.contains(centroid), "Centroid should fall outside this non-convex polygon gap"

    rep_pt, lat, lon, is_inside = get_polygon_representative_point(u_poly)
    assert is_inside is True
    assert u_poly.contains(rep_pt) or u_poly.touches(rep_pt)
    assert np.isclose(rep_pt.x, lon)
    assert np.isclose(rep_pt.y, lat)


def test_find_nearest_facilities_empty_set():
    """Verify that empty facility datasets return NaNs and None without raising exceptions."""
    origin_lats = np.array([13.0, 13.5])
    origin_lons = np.array([74.8, 74.9])

    empty_df = pd.DataFrame(columns=["latitude_num", "longitude_num", "facility_name_clean", "facility_type_code"])
    res = find_nearest_facilities(origin_lats, origin_lons, empty_df)

    assert len(res["distance_km"]) == 2
    assert np.all(np.isnan(res["distance_km"]))
    assert res["nearest_name"] == [None, None]
    assert res["nearest_type"] == [None, None]


def test_find_nearest_facilities_with_missing_coords():
    """Verify that facilities with missing or NaN coordinates are omitted from distance search."""
    origin_lats = np.array([13.0])
    origin_lons = np.array([75.0])

    fac_df = pd.DataFrame(
        {
            "latitude_num": [np.nan, 13.01],
            "longitude_num": [75.0, 75.01],
            "facility_name_clean": ["BadCoordsPHC", "ValidPHC"],
            "facility_type_code": ["phc", "phc"],
        }
    )

    res = find_nearest_facilities(origin_lats, origin_lons, fac_df)
    assert res["nearest_name"] == ["ValidPHC"]
    assert res["distance_km"][0] < 2.0  # Approx 1.5 km


def test_count_facilities_within_boundaries():
    """Verify point-in-polygon counting accurately tallies facilities within polygons."""
    poly1 = Polygon([(74.0, 13.0), (74.1, 13.0), (74.1, 13.1), (74.0, 13.1)])
    poly2 = Polygon([(74.2, 13.0), (74.3, 13.0), (74.3, 13.1), (74.2, 13.1)])

    boundaries_gdf = gpd.GeoDataFrame(
        {"shrid2": ["poly1", "poly2"]},
        geometry=[poly1, poly2],
        crs="EPSG:4326",
    )

    # 2 facilities in poly1 (1 PHC, 1 Sub-centre), 0 in poly2
    facilities_gdf = gpd.GeoDataFrame(
        {
            "facility_name_clean": ["PHC 1", "SubCen 1"],
            "facility_type_code": ["phc", "sub_cen"],
        },
        geometry=[Point(74.05, 13.05), Point(74.06, 13.06)],
        crs="EPSG:4326",
    )

    counts = count_facilities_within_boundaries(boundaries_gdf, facilities_gdf)

    assert counts.loc["poly1", "facilities_in_unit_total"] == 2
    assert counts.loc["poly1", "phc_chc_in_unit_count"] == 1
    assert counts.loc["poly1", "subcentres_in_unit_count"] == 1
    assert counts.loc["poly1", "hospitals_in_unit_count"] == 0

    assert counts.loc["poly2", "facilities_in_unit_total"] == 0
    assert counts.loc["poly2", "phc_chc_in_unit_count"] == 0


def test_haversine_benchmark_against_wgs84_geodesic():
    """Benchmark haversine distance matrix against pyproj.Geod WGS84 ellipsoidal distance.

    Verifies that the spherical haversine approximation maintains < 0.5% error relative
    to Karney's ellipsoidal geodesic across multiple coastal Karnataka coordinate pairs.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    # Known landmark coordinates in coastal Karnataka
    landmarks = [
        ("Mangalore to Udupi", 12.9141, 74.8560, 13.3408, 74.7421),
        ("Kundapura to Sullia", 13.6268, 74.6917, 12.5645, 75.3892),
        ("Bantwal to Belthangady", 12.8943, 75.0347, 12.9961, 75.2572),
        ("Karkala to Puttur", 13.2133, 74.9984, 12.7663, 75.2045),
        ("Byndoor to Someshwara", 13.8732, 74.6300, 12.7915, 74.8567),
    ]

    for label, lat1, lon1, lat2, lon2 in landmarks:
        h_dist_km = haversine_distance_matrix(np.array([lat1]), np.array([lon1]), np.array([lat2]), np.array([lon2]))[0, 0]
        _, _, wgs_m = geod.inv(lon1, lat1, lon2, lat2)
        wgs_km = wgs_m / 1000.0

        diff_km = abs(h_dist_km - wgs_km)
        pct_error = (diff_km / wgs_km) * 100.0

        # Assert discrepancy is strictly under 0.5% across all regional pairs
        assert pct_error < 0.5, f"Discrepancy for {label} exceeded 0.5%: {pct_error:.4f}%"
        # Absolute difference on ~50 km hops must be under 300 metres
        if wgs_km < 60.0:
            assert diff_km < 0.300, f"Absolute error on {label} exceeded 300m: {diff_km:.4f} km"

