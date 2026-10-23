#!/usr/bin/env python3
"""CLI script to engineer coastal proximity features and perform geodesic validation.

Reads:
- data/processed/coastal_spatial_accessibility_features.parquet
- data/interim/coastal_karnataka_ne_10m_coastline.geojson

Outputs:
- data/processed/coastal_exposure_features.parquet
- data/processed/coastal_exposure_features.csv

Prints:
- Validation benchmarks and distance distribution summaries.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from src.features.coastal import (
    COASTLINE_PROVENANCE_METADATA,
    engineer_coastal_exposure_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("engineer_coastal_features")


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineer coastal exposure features.")
    parser.add_argument(
        "--input-features",
        type=Path,
        default=Path("data/processed/coastal_spatial_accessibility_features.parquet"),
        help="Path to input accessibility features parquet.",
    )
    parser.add_argument(
        "--coastline",
        type=Path,
        default=Path("data/interim/coastal_karnataka_ne_10m_coastline.geojson"),
        help="Path to clipped coastline GeoJSON.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/processed/coastal_exposure_features.parquet"),
        help="Output GeoParquet path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/coastal_exposure_features.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    logger.info("Executing Coastal Proximity Feature Engineering...")
    logger.info(f"Input features: {args.input_features}")
    logger.info(f"Coastline: {args.coastline}")

    gdf, audit = engineer_coastal_exposure_features(
        input_features_path=args.input_features,
        coastline_path=args.coastline,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
    )

    print("\n" + "=" * 80)
    print("COASTAL PROXIMITY FEATURE ENGINEERING — AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Administrative Units Evaluated: {audit['total_administrative_units']}")
    print(f"Coastline Source: {audit['coastline_provenance']['source_dataset']} (v{audit['coastline_provenance']['source_version']})")
    print(f"Native CRS: {audit['coastline_provenance']['native_crs']} | Analysis Projected CRS: {audit['coastline_provenance']['analysis_crs']}")
    print("-" * 80)
    print("Distance to Coastline Distribution (km):")
    for stat, val in audit["coastal_proximity_km"].items():
        print(f"  {stat.upper():<8}: {val:>8.3f} km")
    print("-" * 80)
    print("Exposure Band Counts:")
    print(f"  Within 1 km of Coastline:  {audit['units_within_1km_coast']:>4} units ({audit['units_within_1km_coast']/len(gdf)*100:.1f}%)")
    print(f"  Within 5 km of Coastline:  {audit['units_within_5km_coast']:>4} units ({audit['units_within_5km_coast']/len(gdf)*100:.1f}%)")
    print(f"  Within 10 km of Coastline: {audit['units_within_10km_coast']:>4} units ({audit['units_within_10km_coast']/len(gdf)*100:.1f}%)")
    print("-" * 80)
    print("Geodesic Accuracy Benchmark (UTM 43N vs WGS84 Geod):")
    for bm in audit["geodesic_accuracy_benchmarks"]:
        print(
            f"  {bm['unit_name']:<22} | UTM: {bm['dist_utm_km']:>7.3f} km | "
            f"Geod: {bm['dist_geodesic_km']:>7.3f} km | Diff: {bm['discrepancy_meters']:>6.2f} m ({bm['discrepancy_percent']:.4f}%)"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
