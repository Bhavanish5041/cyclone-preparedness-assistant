"""Geographic boundary extraction, CRS validation, and spatial processing.

Provides memory-safe extraction of administrative polygons from the all-India
SHRUG shapefile using pyogrio SQL/where filtering, avoiding loading 615 MB into RAM.
Extracts and structures coastal Karnataka boundaries (Dakshina Kannada and Udupi)
into compact GeoParquet format.
"""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import pyogrio

logger = logging.getLogger(__name__)

# Validated Census 2011 District Codes for Coastal Karnataka
COASTAL_DISTRICT_CODES: Dict[int, str] = {
    569: "Udupi",
    575: "Dakshina Kannada",
}


def build_shrid2_where_clause(district_codes: List[int], state_code: int = 29) -> str:
    """Build a SQL WHERE clause matching SHRUG shrid2 identifiers for target districts.

    In SHRUG v2.1, shrid2 follows the schema: '11-{state_code:02d}-{district_code:03d}-...'.
    Example for Karnataka (29) Udupi (569) and Dakshina Kannada (575):
    "shrid2 LIKE '11-29-569-%' OR shrid2 LIKE '11-29-575-%'"

    Args:
        district_codes: List of 3-digit Census 2011 district codes.
        state_code: 2-digit Census 2011 state code (default 29 for Karnataka).

    Returns:
        SQL WHERE clause string for pyogrio.
    """
    clauses = [
        f"shrid2 LIKE '11-{state_code:02d}-{code:03d}-%'"
        for code in sorted(district_codes)
    ]
    return " OR ".join(clauses)


def parse_shrid2_components(shrid2: str) -> Dict[str, Any]:
    """Parse Census 2011 hierarchy components from a SHRUG shrid2 string.

    Format: '11-SS-DDD-TTTTT-VVVVVV'
    - 11: Census round (2011)
    - SS: State code (e.g. 29)
    - DDD: District code (e.g. 569, 575)
    - TTTTT: Subdistrict / Taluk code
    - VVVVVV: Village / Town code (pc11_id)
    """
    parts = shrid2.split("-")
    if len(parts) >= 5:
        dist_code = int(parts[2]) if parts[2].isdigit() else None
        return {
            "census_year": 2011,
            "state_code": int(parts[1]) if parts[1].isdigit() else 29,
            "district_code": dist_code,
            "district_name": COASTAL_DISTRICT_CODES.get(dist_code, "Other"),
            "subdistrict_code": parts[3],
            "pc11_id": parts[4],
        }
    return {
        "census_year": 2011,
        "state_code": 29,
        "district_code": None,
        "district_name": "Unknown",
        "subdistrict_code": None,
        "pc11_id": parts[-1] if parts else None,
    }


def extract_coastal_boundaries(
    shp_path: Path,
    output_path: Path,
    district_codes: Optional[List[int]] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Extract coastal boundary polygons from SHRUG shapefile and save to GeoParquet.

    Uses pyogrio with attribute filtering to ensure only relevant features are read,
    minimizing memory consumption.

    Args:
        shp_path: Path to raw shrid2_open.shp file.
        output_path: Destination path for clean GeoParquet file.
        district_codes: List of district codes (defaults to 569 and 575).

    Returns:
        Tuple of (GeoDataFrame, metadata_dict).
    """
    if district_codes is None:
        district_codes = list(COASTAL_DISTRICT_CODES.keys())

    if not shp_path.exists():
        raise FileNotFoundError(f"Source shapefile does not exist: {shp_path}")

    # Compute source file hash
    h = hashlib.sha256()
    with shp_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    source_sha256 = h.hexdigest()

    where_clause = build_shrid2_where_clause(district_codes)
    logger.info(f"Reading SHRUG shapefile with filter: {where_clause}")

    gdf = pyogrio.read_dataframe(str(shp_path), where=where_clause)
    logger.info(f"Extracted {len(gdf)} coastal boundary features.")

    if gdf.empty:
        raise ValueError(f"No features matched filter: {where_clause}")

    # Validate CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Parse and enrich administrative hierarchy from shrid2
    parsed_meta = gdf["shrid2"].apply(parse_shrid2_components)
    meta_df = pd.DataFrame(parsed_meta.tolist(), index=gdf.index)

    for col in ["state_code", "district_code", "district_name", "subdistrict_code", "pc11_id"]:
        gdf[col] = meta_df[col]

    # Ensure pc11_id is standard 6-character zero-padded string
    gdf["pc11_id"] = gdf["pc11_id"].astype(str).str.zfill(6)

    # Sort deterministically by district and pc11_id
    gdf = gdf.sort_values(["district_code", "pc11_id"]).reset_index(drop=True)

    # Save to GeoParquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    output_size_kb = round(output_path.stat().st_size / 1024, 2)
    logger.info(f"Saved {len(gdf)} features to {output_path} ({output_size_kb} KB)")

    bounds = [round(float(b), 6) for b in gdf.total_bounds]

    metadata = {
        "source_file": str(shp_path),
        "source_sha256": source_sha256,
        "selection_criteria": where_clause,
        "target_districts": {code: COASTAL_DISTRICT_CODES[code] for code in district_codes},
        "feature_count": len(gdf),
        "feature_breakdown": gdf["district_name"].value_counts().to_dict(),
        "crs": "EPSG:4326",
        "total_bounds": bounds,
        "output_file": str(output_path),
        "output_size_kb": output_size_kb,
        "geometry_types": gdf.geometry.geom_type.value_counts().to_dict(),
    }

    return gdf, metadata
