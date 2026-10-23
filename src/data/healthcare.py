"""Healthcare facility extraction, coordinate validation, and standardization.

Processes the canonical national geocode_health_centre.csv dataset, filters to
target coastal districts (Dakshina Kannada and Udupi), validates geographic coordinates,
removes derived duplicates, applies verified facility tier classifications, and exports
intermediate GeoParquet and CSV datasets.
"""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Validated Bounding Box for Coastal Karnataka (Dakshina Kannada & Udupi)
# Latitude: 12.0°N to 14.5°N | Longitude: 74.0°E to 76.0°E
LAT_MIN, LAT_MAX = 12.0, 14.5
LON_MIN, LON_MAX = 74.0, 76.0

# Verified facility type mapping based on Ministry of Health / NHM conventions
FACILITY_TYPE_LABELS: Dict[str, str] = {
    "sub_cen": "Health Sub-Centre",
    "phc": "Primary Health Centre (PHC)",
    "chc": "Community Health Centre (CHC)",
    "s_t_h": "Sub-Taluk Hospital",
    "dis_h": "District Hospital",
}

# Defensible service-tier grouping for emergency response capability
FACILITY_TIER_MAPPING: Dict[str, str] = {
    "sub_cen": "Tier 1: Primary Outpost (Basic first aid & outreach)",
    "phc": "Tier 2: Primary Care (Outpatient, basic maternity, essential triage)",
    "chc": "Tier 2: Secondary / CHC (Inpatient beds, emergency care, surgical/specialist)",
    "s_t_h": "Tier 3: Referral / Taluk Hospital (Secondary inpatient & surgical capacity)",
    "dis_h": "Tier 3: Apex / District Hospital (Full trauma, ICU, inpatient & emergency capacity)",
}

DATA_DISCLAIMER = (
    "DISCLAIMER: Facility presence in this directory records administrative registration "
    "and geocoded coordinates. It does NOT establish current operational status, active staffing, "
    "bed availability, oxygen supply, generator power, or cyclone emergency readiness."
)


def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Validate numeric latitude/longitude values and flag invalid or out-of-bounds coordinates.

    Args:
        df: DataFrame containing raw coordinate columns.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        lat_min, lat_max: Valid latitude range.
        lon_min, lon_max: Valid longitude range.

    Returns:
        Tuple of (clean_df_with_numeric_lat_lon, quality_metrics_dict).
    """
    df = df.copy()

    # Convert to numeric, coercing unparseable strings to NaN
    df["latitude_num"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude_num"] = pd.to_numeric(df[lon_col], errors="coerce")

    # Metrics
    total = len(df)
    missing_lat = int(df["latitude_num"].isna().sum())
    missing_lon = int(df["longitude_num"].isna().sum())
    zero_coords = int(((df["latitude_num"] == 0) | (df["longitude_num"] == 0)).sum())

    # Only evaluate geographic bounds on non-null, non-zero coordinates
    has_coords = (
        df["latitude_num"].notna()
        & df["longitude_num"].notna()
        & (df["latitude_num"] != 0)
        & (df["longitude_num"] != 0)
    )
    out_of_bounds = int(
        (
            has_coords
            & (
                (df["latitude_num"] < lat_min)
                | (df["latitude_num"] > lat_max)
                | (df["longitude_num"] < lon_min)
                | (df["longitude_num"] > lon_max)
            )
        ).sum()
    )

    valid_mask = (
        has_coords
        & (df["latitude_num"] >= lat_min)
        & (df["latitude_num"] <= lat_max)
        & (df["longitude_num"] >= lon_min)
        & (df["longitude_num"] <= lon_max)
    )

    valid_df = df[valid_mask].copy()

    metrics = {
        "total_records": total,
        "missing_latitude": missing_lat,
        "missing_longitude": missing_lon,
        "zero_coordinates": zero_coords,
        "out_of_bounds_coordinates": out_of_bounds,
        "valid_coordinate_count": int(valid_mask.sum()),
        "invalid_coordinate_count": total - int(valid_mask.sum()),
    }

    return valid_df, metrics


def process_coastal_healthcare(
    csv_path: Path,
    output_parquet: Path,
    output_csv: Optional[Path] = None,
    target_districts: Optional[List[str]] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Load canonical health facilities CSV, filter to coastal districts, validate, and export.

    Args:
        csv_path: Path to raw geocode_health_centre.csv.
        output_parquet: Destination for clean GeoParquet file.
        output_csv: Optional destination for clean CSV file.
        target_districts: List of district names (default: ['Dakshina Kannada', 'Udupi']).

    Returns:
        Tuple of (clean_GeoDataFrame, audit_metadata_dict).
    """
    if target_districts is None:
        target_districts = ["Dakshina Kannada", "Udupi"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    # Compute source SHA-256
    h = hashlib.sha256()
    with csv_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    source_sha256 = h.hexdigest()

    # Load canonical CSV
    logger.info(f"Loading {csv_path}...")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    raw_total_records = len(df_raw)

    # Filter to Karnataka and target districts
    is_karnataka = df_raw["State Name"].astype(str).str.strip().str.upper() == "KARNATAKA"
    clean_target_districts = [d.strip().upper() for d in target_districts]
    is_target_district = (
        df_raw["District Name"].astype(str).str.strip().str.upper().isin(clean_target_districts)
    )

    df_filtered = df_raw[is_karnataka & is_target_district].copy()
    filtered_count = len(df_filtered)
    logger.info(f"Filtered {filtered_count} facilities in {target_districts} (from {raw_total_records} national rows)")

    # Standardize column naming and strip whitespace
    df_filtered["district_name_standardized"] = (
        df_filtered["District Name"].astype(str).str.strip().str.title()
    )
    df_filtered["subdistrict_name_standardized"] = (
        df_filtered["Subdistrict Name"].astype(str).str.strip().str.title()
    )
    df_filtered["facility_name_clean"] = df_filtered["Facility Name"].astype(str).str.strip()
    raw_facility_type = df_filtered["Facility Type"].astype(str).str.strip().str.lower()
    df_filtered["facility_type_code"] = raw_facility_type
    df_filtered["facility_type_label"] = raw_facility_type.map(FACILITY_TYPE_LABELS).fillna("Other")
    df_filtered["service_tier"] = raw_facility_type.map(FACILITY_TIER_MAPPING).fillna("Unclassified")

    # Validate coordinates
    df_valid, coord_metrics = validate_coordinates(df_filtered)

    # Check and remove duplicates in derived dataset
    initial_valid = len(df_valid)
    dedup_subset = ["facility_name_clean", "district_name_standardized", "facility_type_code", "latitude_num", "longitude_num"]
    df_dedup = df_valid.drop_duplicates(subset=dedup_subset).copy()
    duplicates_removed = initial_valid - len(df_dedup)
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate facility records.")

    # Assign persistent data disclaimer
    df_dedup["data_disclaimer"] = DATA_DISCLAIMER
    df_dedup["source_vintage"] = "approx. 2015-2019"

    # Create GeoDataFrame with WGS84 CRS (EPSG:4326)
    geometry = [Point(xy) for xy in zip(df_dedup["longitude_num"], df_dedup["latitude_num"])]
    gdf = gpd.GeoDataFrame(df_dedup, geometry=geometry, crs="EPSG:4326")

    # Sort deterministically
    gdf = gdf.sort_values(["district_name_standardized", "facility_type_code", "facility_name_clean"]).reset_index(drop=True)

    # Export to GeoParquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_parquet)
    logger.info(f"Saved {len(gdf)} facilities to {output_parquet}")

    if output_csv is not None:
        # Save CSV copy without geometry object for lightweight tabular consumption
        df_export = pd.DataFrame(gdf.drop(columns=["geometry"]))
        df_export.to_csv(output_csv, index=False)
        logger.info(f"Saved CSV copy to {output_csv}")

    type_counts = gdf.groupby(["district_name_standardized", "facility_type_label"]).size().unstack(fill_value=0)

    audit_metadata = {
        "source_file": str(csv_path),
        "source_sha256": source_sha256,
        "raw_national_records": raw_total_records,
        "filtered_coastal_records": filtered_count,
        "coordinate_validation": coord_metrics,
        "duplicates_removed_in_derived": duplicates_removed,
        "final_clean_facility_count": len(gdf),
        "breakdown_by_district": gdf["district_name_standardized"].value_counts().to_dict(),
        "breakdown_by_facility_type": gdf["facility_type_label"].value_counts().to_dict(),
        "breakdown_by_tier": gdf["service_tier"].value_counts().to_dict(),
        "coordinate_bounds": [round(float(b), 6) for b in gdf.total_bounds],
        "crs": "EPSG:4326",
        "output_parquet": str(output_parquet),
        "disclaimer": DATA_DISCLAIMER,
    }

    return gdf, audit_metadata
