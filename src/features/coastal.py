"""Coastal proximity feature engineering and coastline dataset validation.

Computes straight-line / geodesic proximity proxies from administrative polygon
representative points to the validated Arabian Sea coastline in EPSG:32643 (UTM Zone 43N)
with WGS84 ellipsoidal geodesic verification.

DISCLAIMER: This metric represents straight-line proximity to the generalized continental
coastline. It does NOT represent storm-surge probability, flood inundation risk,
cyclone landfall likelihood, or local hydrodynamic bathymetry.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
import shapely.geometry
import shapely.ops

logger = logging.getLogger(__name__)

# Coordinate bounds for Coastal Karnataka study area (Dakshina Kannada + Udupi)
STUDY_AREA_BBOX = (74.3, 12.3, 75.8, 14.2)  # minx, miny, maxx, maxy

# UTM Zone 43N is the optimal projected CRS for Karnataka (72°E - 78°E)
# Central meridian 75°E minimizes scale distortion (< 0.04% across study area)
ANALYSIS_PROJECTED_CRS = "EPSG:32643"
GEOGRAPHIC_CRS = "EPSG:4326"

COASTAL_PROXIMITY_DISCLAIMER = (
    "DISCLAIMER: Straight-line proximity proxy calculated from polygon representative points "
    "to the validated Natural Earth 10m continental coastline in EPSG:32643 (UTM Zone 43N). "
    "This metric does NOT represent storm-surge inundation depth, storm-surge probability, "
    "cyclone track exposure, elevation, or local hydrodynamic bathymetry."
)

COASTLINE_PROVENANCE_METADATA: Dict[str, Any] = {
    "source_dataset": "Natural Earth 10m Coastline (ne_10m_coastline)",
    "source_organization": "Natural Earth / North American Cartographic Information Society (NACIS)",
    "source_url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson",
    "source_version": "5.1.1",
    "license": "Public Domain (CC0 equivalent)",
    "retrieval_date": "2026-10-22",
    "native_crs": GEOGRAPHIC_CRS,
    "analysis_crs": ANALYSIS_PROJECTED_CRS,
    "scale_resolution": "1:10,000,000 (10m medium-scale cartographic vector)",
    "geographic_scope": "Arabian Sea coast of Karnataka (Dakshina Kannada and Udupi)",
    "known_limitations": (
        "Global 1:10,000,000 scale means spatial generalization is substantial (~1.7 km between vertices). "
        "Does NOT resolve fine-scale estuaries (Netravati-Gurupura at Mangalore, Swarna-Sita at Malpe, "
        "Panchagangavalli at Kundapura), tidal creeks, sand spit dynamics, or offshore islands. "
        "Must be interpreted strictly as continental coastline proximity, not storm surge hazard."
    ),
    "intended_use": "Straight-line geographic proximity proxy to the continental coastline for cyclone preparedness vulnerability assessment.",
}


def load_and_validate_coastline(
    coastline_path: Union[str, Path],
    clip_bbox: Tuple[float, float, float, float] = STUDY_AREA_BBOX,
) -> gpd.GeoDataFrame:
    """Load, validate, and clip the coastline vector dataset.

    Args:
        coastline_path: Path to coastline GeoJSON or Shapefile.
        clip_bbox: Tuple of (minx, miny, maxx, maxy) in EPSG:4326 to clip.

    Returns:
        GeoDataFrame of clipped coastline lines in EPSG:4326.

    Raises:
        FileNotFoundError: If coastline file does not exist.
        ValueError: If file contains no geometries or CRS cannot be established.
    """
    path = Path(coastline_path)
    if not path.exists():
        raise FileNotFoundError(f"Coastline file not found: {path}")

    logger.info(f"Loading coastline from {path}...")
    gdf = gpd.read_file(path)

    if gdf.empty or len(gdf) == 0:
        raise ValueError(f"Coastline file {path} is empty.")

    # Ensure EPSG:4326
    if gdf.crs is None:
        logger.warning("Coastline CRS is missing; assuming EPSG:4326.")
        gdf = gdf.set_crs(GEOGRAPHIC_CRS)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(GEOGRAPHIC_CRS)

    # Clip to study area bounding box
    minx, miny, maxx, maxy = clip_bbox
    clip_geom = shapely.geometry.box(minx, miny, maxx, maxy)
    clipped = gdf.clip(clip_geom)

    if clipped.empty or len(clipped) == 0:
        raise ValueError(f"No coastline geometries found within bounding box {clip_bbox}.")

    logger.info(f"Validated coastline with {len(clipped)} features in study area bounding box.")
    return clipped


def calculate_coastal_proximity_utm(
    points_wgs84: List[shapely.geometry.Point],
    coastline_gdf_wgs84: gpd.GeoDataFrame,
    projected_crs: str = ANALYSIS_PROJECTED_CRS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate perpendicular Euclidean distance from points to coastline in projected metric CRS.

    Args:
        points_wgs84: List of Shapely Point objects in EPSG:4326.
        coastline_gdf_wgs84: Coastline GeoDataFrame in EPSG:4326.
        projected_crs: Planar projected CRS in metres (default: EPSG:32643 UTM 43N).

    Returns:
        Tuple of:
        - dists_km: 1D numpy array of distances in kilometres (rounded to 3 decimals).
        - dists_m: 1D numpy array of exact distances in metres.
    """
    if len(points_wgs84) == 0:
        return np.array([]), np.array([])

    # Create GeoSeries for points and reproject to UTM
    pts_gs = gpd.GeoSeries(points_wgs84, crs=GEOGRAPHIC_CRS).to_crs(projected_crs)

    # Reproject coastline to UTM and create unified geometry
    coast_proj = coastline_gdf_wgs84.to_crs(projected_crs)
    coast_geom = coast_proj.union_all()

    # Calculate distances in metres
    dists_m = np.array([coast_geom.distance(pt) for pt in pts_gs], dtype=float)
    dists_km = np.round(dists_m / 1000.0, 3)

    return dists_km, dists_m


def verify_coastal_distance_geodesic(
    point_wgs84: shapely.geometry.Point,
    coastline_gdf_wgs84: gpd.GeoDataFrame,
    projected_crs: str = ANALYSIS_PROJECTED_CRS,
) -> Tuple[float, float, float]:
    """Verify UTM projected distance against ellipsoidal WGS84 geodesic distance for a single point.

    Finds the nearest point on the coastline in UTM 43N, unprojects that point back to EPSG:4326,
    and computes the geodesic distance via pyproj.Geod on the WGS84 ellipsoid.

    Args:
        point_wgs84: Point in EPSG:4326.
        coastline_gdf_wgs84: Coastline in EPSG:4326.
        projected_crs: Projected metric CRS.

    Returns:
        Tuple of (dist_utm_m, dist_geodesic_m, diff_m).
    """
    pt_proj = gpd.GeoSeries([point_wgs84], crs=GEOGRAPHIC_CRS).to_crs(projected_crs).iloc[0]
    coast_proj = coastline_gdf_wgs84.to_crs(projected_crs).union_all()

    # Distance in UTM
    dist_utm_m = float(coast_proj.distance(pt_proj))

    # Nearest point on coastline
    nearest_pt_utm = shapely.ops.nearest_points(coast_proj, pt_proj)[0]
    nearest_pt_wgs = gpd.GeoSeries([nearest_pt_utm], crs=projected_crs).to_crs(GEOGRAPHIC_CRS).iloc[0]

    # Geodesic on WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    _, _, dist_geod_m = geod.inv(point_wgs84.x, point_wgs84.y, nearest_pt_wgs.x, nearest_pt_wgs.y)

    diff_m = abs(dist_utm_m - dist_geod_m)
    return dist_utm_m, dist_geod_m, diff_m


def engineer_coastal_exposure_features(
    input_features_path: Union[str, Path],
    coastline_path: Union[str, Path],
    output_parquet: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Engineer coastal proximity features and merge with existing accessibility features.

    Args:
        input_features_path: Path to coastal_spatial_accessibility_features.parquet.
        coastline_path: Path to coastline GeoJSON or Shapefile.
        output_parquet: Destination path for combined GeoParquet dataset.
        output_csv: Optional destination path for CSV copy.

    Returns:
        Tuple of (updated_GeoDataFrame, audit_metadata_dict).
    """
    input_path = Path(input_features_path)
    output_pq = Path(output_parquet)

    logger.info(f"Loading input accessibility features from {input_path}...")
    gdf = gpd.read_parquet(input_path)

    # Load and validate coastline
    coastline_gdf = load_and_validate_coastline(coastline_path)

    # Reconstruct Point objects from verified rep_point columns
    rep_points = [
        shapely.geometry.Point(lon, lat)
        for lon, lat in zip(gdf["rep_point_longitude"], gdf["rep_point_latitude"])
    ]

    # Calculate coastal proximity
    dists_km, dists_m = calculate_coastal_proximity_utm(rep_points, coastline_gdf)

    gdf["dist_to_coastline_km"] = dists_km
    gdf["coastal_proximity_proxy_type"] = "straight_line_utm43n_perpendicular"
    gdf["coastal_proximity_disclaimer"] = COASTAL_PROXIMITY_DISCLAIMER
    gdf["coastline_source"] = COASTLINE_PROVENANCE_METADATA["source_dataset"]
    gdf["coastline_vintage"] = COASTLINE_PROVENANCE_METADATA["source_version"]

    # Benchmark geodesic accuracy on sample points (min, median, max)
    sample_indices = [
        int(np.argmin(dists_km)),
        int(np.argsort(dists_km)[len(dists_km) // 2]),
        int(np.argmax(dists_km)),
    ]

    geodesic_benchmarks = []
    for idx in sample_indices:
        pt = rep_points[idx]
        d_utm, d_geod, diff = verify_coastal_distance_geodesic(pt, coastline_gdf)
        geodesic_benchmarks.append(
            {
                "unit_name": gdf.iloc[idx].get("unit_name", f"Unit {idx}"),
                "shrid2": gdf.iloc[idx].get("shrid2", "unknown"),
                "dist_utm_km": round(d_utm / 1000.0, 3),
                "dist_geodesic_km": round(d_geod / 1000.0, 3),
                "discrepancy_meters": round(diff, 2),
                "discrepancy_percent": round((diff / max(d_geod, 1.0)) * 100, 4),
            }
        )

    # Audit statistics
    dist_series = pd.Series(dists_km)
    audit_metrics = {
        "total_administrative_units": len(gdf),
        "coastline_provenance": COASTLINE_PROVENANCE_METADATA,
        "coastal_proximity_km": {
            "min": round(float(dist_series.min()), 3),
            "mean": round(float(dist_series.mean()), 3),
            "median": round(float(dist_series.median()), 3),
            "p25": round(float(dist_series.quantile(0.25)), 3),
            "p75": round(float(dist_series.quantile(0.75)), 3),
            "p90": round(float(dist_series.quantile(0.90)), 3),
            "max": round(float(dist_series.max()), 3),
        },
        "units_within_1km_coast": int((dist_series < 1.0).sum()),
        "units_within_5km_coast": int((dist_series < 5.0).sum()),
        "units_within_10km_coast": int((dist_series < 10.0).sum()),
        "geodesic_accuracy_benchmarks": geodesic_benchmarks,
        "output_parquet": str(output_pq),
    }

    # Save to output
    output_pq.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_pq)
    logger.info(f"Saved {len(gdf)} coastal exposure features to {output_pq}")

    if output_csv is not None:
        output_csv_path = Path(output_csv)
        df_csv = pd.DataFrame(gdf.drop(columns=["geometry"]))
        df_csv.to_csv(output_csv_path, index=False)
        logger.info(f"Saved tabular CSV copy to {output_csv_path}")

    return gdf, audit_metrics
