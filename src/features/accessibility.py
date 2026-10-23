"""Spatial accessibility feature engineering for cyclone preparedness.

Computes straight-line geodesic distance proxies from polygon representative points
(guaranteed to fall inside administrative boundary polygons) to nearest registered
healthcare facilities (PHCs/CHCs and hospitals). Incorporates in-unit facility counts,
spatial join integrity audits, and rigorous disclaimers regarding operational readiness.
"""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

# Mean Earth radius in kilometres (WGS84 spherical approximation)
WGS84_EARTH_RADIUS_KM = 6371.0088

ACCESSIBILITY_DISCLAIMER = (
    "DISCLAIMER: Straight-line / geodesic distance proxy calculated from polygon representative "
    "points to registered facility coordinates in EPSG:4326. This metric does NOT represent road "
    "network driving distances, road topography, ferry crossings, flood impasses, or the current "
    "operational readiness, doctor presence, or bed availability of the referenced facility."
)


def haversine_distance_matrix(
    lats1: np.ndarray,
    lons1: np.ndarray,
    lats2: np.ndarray,
    lons2: np.ndarray,
    radius_km: float = WGS84_EARTH_RADIUS_KM,
) -> np.ndarray:
    """Compute pairwise great-circle geodesic distances between two sets of coordinates in kilometres.

    Args:
        lats1, lons1: 1D array of origin latitudes and longitudes (in degrees).
        lats2, lons2: 1D array of destination latitudes and longitudes (in degrees).
        radius_km: Spherical Earth radius in kilometres.

    Returns:
        2D numpy array of shape (len(lats1), len(lats2)) with distances in km.
    """
    lats1 = np.asarray(lats1, dtype=float)
    lons1 = np.asarray(lons1, dtype=float)
    lats2 = np.asarray(lats2, dtype=float)
    lons2 = np.asarray(lons2, dtype=float)

    if lats1.ndim != 1 or lons1.ndim != 1 or lats2.ndim != 1 or lons2.ndim != 1:
        raise ValueError("Coordinate arrays must be 1-dimensional.")

    phi1 = np.radians(lats1[:, np.newaxis])
    phi2 = np.radians(lats2[np.newaxis, :])
    dphi = np.radians(lats2[np.newaxis, :] - lats1[:, np.newaxis])
    dlam = np.radians(lons2[np.newaxis, :] - lons1[:, np.newaxis])

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def get_polygon_representative_point(geometry: Any) -> Tuple[Point, float, float, bool]:
    """Calculate a representative point guaranteed to be inside the polygon interior.

    Unlike centroids (which may fall outside concave, crescent, or coastal inlet polygons),
    Shapely's representative_point() is mathematically guaranteed to fall within the geometry.

    Args:
        geometry: Shapely Polygon or MultiPolygon.

    Returns:
        Tuple of (Point, latitude, longitude, is_strictly_inside_or_touches).
    """
    if geometry is None or geometry.is_empty:
        raise ValueError("Cannot compute representative point for empty or None geometry.")

    if not geometry.is_valid:
        geometry = make_valid(geometry)

    rep_pt = geometry.representative_point()
    lat = float(rep_pt.y)
    lon = float(rep_pt.x)

    # Verify interior containment
    is_inside = geometry.contains(rep_pt) or geometry.touches(rep_pt)
    return rep_pt, lat, lon, is_inside


def find_nearest_facilities(
    origin_lats: np.ndarray,
    origin_lons: np.ndarray,
    facilities_df: pd.DataFrame,
    lat_col: str = "latitude_num",
    lon_col: str = "longitude_num",
    name_col: str = "facility_name_clean",
    type_col: str = "facility_type_code",
) -> Dict[str, Any]:
    """Find the nearest facility and geodesic distance for each origin coordinate.

    Handles empty facility sets gracefully by returning arrays of NaNs and None.

    Args:
        origin_lats, origin_lons: Origin latitudes and longitudes in degrees.
        facilities_df: DataFrame of target facilities with valid numeric coordinates.
        lat_col, lon_col, name_col, type_col: Relevant column names.

    Returns:
        Dictionary containing distance_km, nearest_name, nearest_type, nearest_index.
    """
    n_origins = len(origin_lats)

    if facilities_df.empty or len(facilities_df) == 0:
        return {
            "distance_km": np.full(n_origins, np.nan),
            "nearest_name": [None] * n_origins,
            "nearest_type": [None] * n_origins,
            "nearest_index": [None] * n_origins,
        }

    valid_fac = facilities_df.dropna(subset=[lat_col, lon_col]).copy()
    if valid_fac.empty:
        return {
            "distance_km": np.full(n_origins, np.nan),
            "nearest_name": [None] * n_origins,
            "nearest_type": [None] * n_origins,
            "nearest_index": [None] * n_origins,
        }

    fac_lats = valid_fac[lat_col].to_numpy(dtype=float)
    fac_lons = valid_fac[lon_col].to_numpy(dtype=float)
    fac_names = valid_fac[name_col].tolist()
    fac_types = valid_fac[type_col].tolist()

    # Calculate distance matrix (n_origins x n_facilities)
    dist_matrix = haversine_distance_matrix(origin_lats, origin_lons, fac_lats, fac_lons)

    min_indices = np.argmin(dist_matrix, axis=1)
    min_distances = np.min(dist_matrix, axis=1)

    nearest_names = [fac_names[i] for i in min_indices]
    nearest_types = [fac_types[i] for i in min_indices]

    return {
        "distance_km": np.round(min_distances, 3),
        "nearest_name": nearest_names,
        "nearest_type": nearest_types,
        "nearest_index": min_indices.tolist(),
    }


def count_facilities_within_boundaries(
    boundaries_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    type_col: str = "facility_type_code",
) -> pd.DataFrame:
    """Count how many facilities of each type fall strictly inside each administrative boundary polygon.

    Uses a spatial join (predicate='within') in EPSG:4326.
    """
    if boundaries_gdf.crs != facilities_gdf.crs:
        facilities_gdf = facilities_gdf.to_crs(boundaries_gdf.crs)

    # Perform point-in-polygon spatial join
    joined = gpd.sjoin(
        facilities_gdf,
        boundaries_gdf[["shrid2", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Aggregations by polygon shrid2
    counts_df = pd.DataFrame(index=boundaries_gdf["shrid2"])

    # Total facilities inside polygon
    total_counts = joined.groupby("shrid2").size()
    counts_df["facilities_in_unit_total"] = total_counts

    # Primary / Secondary (PHC + CHC) inside polygon
    phc_chc_mask = joined[type_col].isin(["phc", "chc"])
    phc_chc_counts = joined[phc_chc_mask].groupby("shrid2").size()
    counts_df["phc_chc_in_unit_count"] = phc_chc_counts

    # Hospitals (District Hospital + Sub-Taluk Hospital) inside polygon
    hosp_mask = joined[type_col].isin(["dis_h", "s_t_h"])
    hosp_counts = joined[hosp_mask].groupby("shrid2").size()
    counts_df["hospitals_in_unit_count"] = hosp_counts

    # Sub-centres inside polygon
    subcen_mask = joined[type_col] == "sub_cen"
    subcen_counts = joined[subcen_mask].groupby("shrid2").size()
    counts_df["subcentres_in_unit_count"] = subcen_counts

    counts_df = counts_df.fillna(0).astype(int)
    return counts_df


def engineer_spatial_accessibility_features(
    boundaries_gdf: gpd.GeoDataFrame,
    healthcare_gdf: gpd.GeoDataFrame,
    census_df: pd.DataFrame,
    output_parquet: Path,
    output_csv: Optional[Path] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Engineer comprehensive straight-line accessibility proxies and merge census context.

    Calculates:
    - Representative point for each polygon guaranteed to lie in the interior.
    - Geodesic distance in km to the nearest registered PHC/CHC.
    - Geodesic distance in km to the nearest registered Hospital.
    - Geodesic distance in km to the nearest registered facility of any type.
    - Point-in-polygon counts of facilities located directly within the village/town boundary.
    - Population and household indicators from Census 2011.

    Args:
        boundaries_gdf: Coastal boundary polygons (EPSG:4326).
        healthcare_gdf: Clean healthcare facilities with coordinates (EPSG:4326).
        census_df: Harmonized coastal census dataset.
        output_parquet: Output path for GeoParquet features.
        output_csv: Optional output path for tabular CSV.

    Returns:
        Tuple of (features_GeoDataFrame, audit_metadata_dict).
    """
    logger.info("Starting spatial accessibility feature engineering...")

    # 1. Verify CRS
    if boundaries_gdf.crs is None or boundaries_gdf.crs.to_epsg() != 4326:
        boundaries_gdf = boundaries_gdf.to_crs("EPSG:4326")
    if healthcare_gdf.crs is None or healthcare_gdf.crs.to_epsg() != 4326:
        healthcare_gdf = healthcare_gdf.to_crs("EPSG:4326")

    features_gdf = boundaries_gdf.copy()
    n_units = len(features_gdf)

    # 2. Calculate representative points guaranteed inside polygon interior
    rep_points = []
    rep_lats = []
    rep_lons = []
    interior_checks = []

    for geom in features_gdf.geometry:
        pt, lat, lon, is_in = get_polygon_representative_point(geom)
        rep_points.append(pt)
        rep_lats.append(lat)
        rep_lons.append(lon)
        interior_checks.append(is_in)

    features_gdf["rep_point_latitude"] = rep_lats
    features_gdf["rep_point_longitude"] = rep_lons
    features_gdf["rep_point_strictly_inside"] = interior_checks

    origin_lats = np.array(rep_lats, dtype=float)
    origin_lons = np.array(rep_lons, dtype=float)

    # 3. Filter facility subsets by verified administrative codes
    phc_chc_df = healthcare_gdf[healthcare_gdf["facility_type_code"].isin(["phc", "chc"])].copy()
    hospital_df = healthcare_gdf[healthcare_gdf["facility_type_code"].isin(["dis_h", "s_t_h"])].copy()
    any_fac_df = healthcare_gdf.copy()

    logger.info(f"Target facilities: PHC/CHC={len(phc_chc_df)}, Hospitals={len(hospital_df)}, Total={len(any_fac_df)}")

    # 4. Geodesic distance to nearest registered PHC/CHC
    res_phc_chc = find_nearest_facilities(origin_lats, origin_lons, phc_chc_df)
    features_gdf["dist_to_nearest_phc_chc_km"] = res_phc_chc["distance_km"]
    features_gdf["nearest_phc_chc_name"] = res_phc_chc["nearest_name"]
    features_gdf["nearest_phc_chc_type"] = res_phc_chc["nearest_type"]

    # 5. Geodesic distance to nearest registered Hospital (Sub-Taluk or District Hospital)
    res_hosp = find_nearest_facilities(origin_lats, origin_lons, hospital_df)
    features_gdf["dist_to_nearest_hospital_km"] = res_hosp["distance_km"]
    features_gdf["nearest_hospital_name"] = res_hosp["nearest_name"]
    features_gdf["nearest_hospital_type"] = res_hosp["nearest_type"]

    # 6. Geodesic distance to nearest registered health facility of any type (including Sub-Centres)
    res_any = find_nearest_facilities(origin_lats, origin_lons, any_fac_df)
    features_gdf["dist_to_nearest_facility_any_km"] = res_any["distance_km"]
    features_gdf["nearest_facility_any_name"] = res_any["nearest_name"]
    features_gdf["nearest_facility_any_type"] = res_any["nearest_type"]

    # 7. Point-in-polygon facility counts per administrative unit
    in_unit_counts = count_facilities_within_boundaries(features_gdf, healthcare_gdf)
    features_gdf["facilities_in_unit_total"] = in_unit_counts["facilities_in_unit_total"].values
    features_gdf["phc_chc_in_unit_count"] = in_unit_counts["phc_chc_in_unit_count"].values
    features_gdf["hospitals_in_unit_count"] = in_unit_counts["hospitals_in_unit_count"].values
    features_gdf["subcentres_in_unit_count"] = in_unit_counts["subcentres_in_unit_count"].values

    # 8. Merge Census 2011 demographic and infrastructure baseline
    census_subset_cols = [
        "pc11_id",
        "unit_type",
        "unit_name",
        "total_households",
        "total_population",
        "male_population",
        "female_population",
        "sc_population",
        "st_population",
        "sc_st_ratio",
        "sex_ratio_female_per_1000_male",
        "has_treated_tap_water",
        "has_all_weather_road",
        "has_pucca_road",
        "has_domestic_power",
        "drainage_system",
    ]
    # Include literacy rate where present
    if "literacy_rate" in census_df.columns:
        census_subset_cols.append("literacy_rate")

    census_clean = census_df[census_subset_cols].drop_duplicates(subset=["pc11_id"])
    features_gdf = features_gdf.merge(census_clean, on="pc11_id", how="left")

    # 9. Add metadata & disclaimers
    features_gdf["proxy_type"] = "straight_line_geodesic_haversine"
    features_gdf["distance_unit"] = "kilometres"
    features_gdf["census_vintage"] = 2011
    features_gdf["data_disclaimer"] = ACCESSIBILITY_DISCLAIMER

    # 10. Audit metrics
    phc_dists = features_gdf["dist_to_nearest_phc_chc_km"].dropna()
    hosp_dists = features_gdf["dist_to_nearest_hospital_km"].dropna()

    audit_metrics = {
        "total_administrative_polygons": n_units,
        "representative_points_interior_percentage": round(sum(interior_checks) / n_units * 100, 2),
        "target_facility_counts": {
            "phc_chc_count": len(phc_chc_df),
            "hospital_count": len(hospital_df),
            "total_facility_count": len(any_fac_df),
        },
        "distance_to_phc_chc_km": {
            "min": round(float(phc_dists.min()), 2),
            "mean": round(float(phc_dists.mean()), 2),
            "median": round(float(phc_dists.median()), 2),
            "p75": round(float(phc_dists.quantile(0.75)), 2),
            "p90": round(float(phc_dists.quantile(0.90)), 2),
            "max": round(float(phc_dists.max()), 2),
        },
        "distance_to_hospital_km": {
            "min": round(float(hosp_dists.min()), 2),
            "mean": round(float(hosp_dists.mean()), 2),
            "median": round(float(hosp_dists.median()), 2),
            "p75": round(float(hosp_dists.quantile(0.75)), 2),
            "p90": round(float(hosp_dists.quantile(0.90)), 2),
            "max": round(float(hosp_dists.max()), 2),
        },
        "units_with_phc_chc_in_boundary": int((features_gdf["phc_chc_in_unit_count"] > 0).sum()),
        "units_with_hospital_in_boundary": int((features_gdf["hospitals_in_unit_count"] > 0).sum()),
        "units_with_any_facility_in_boundary": int((features_gdf["facilities_in_unit_total"] > 0).sum()),
        "unmatched_census_units_count": int(features_gdf["total_population"].isna().sum()),
        "crs": "EPSG:4326",
        "output_file": str(output_parquet),
    }

    # 11. Save derived feature dataset
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    features_gdf.to_parquet(output_parquet)
    logger.info(f"Saved {len(features_gdf)} features to {output_parquet}")

    if output_csv is not None:
        df_csv = pd.DataFrame(features_gdf.drop(columns=["geometry"]))
        df_csv.to_csv(output_csv, index=False)
        logger.info(f"Saved tabular CSV copy to {output_csv}")

    return features_gdf, audit_metrics
