"""Milestone 1: Relational reconciliation and archetype integration module.

Integrates the AHP/MCDA village preparedness assessment (from teammate's repo)
with the K-Means unsupervised vulnerability archetypes (from clustering repo)
into a unified, verified village-level dataset.

Preserves all 1,879 preparedness records, enforces canonical join on shrid2 == area_id,
restores canonical village and town names, and assigns explicit coverage status flags.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("join_preparedness_archetypes")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CYCLONE_DATA_DIR", PROJECT_ROOT / "data" / "processed"))

DEFAULT_PREPAREDNESS_PATH = DATA_DIR / "master_village_preparedness.parquet"
DEFAULT_ARCHETYPES_PATH = DATA_DIR / "coastal_vulnerability_archetypes.parquet"
DEFAULT_OUTPUT_PATH = DATA_DIR / "unified_village_preparedness.parquet"

INTEGRATION_DISCLAIMER = (
    "DISCLAIMER: This dataset integrates the AHP/MCDA preparedness index with descriptive "
    "K-Means vulnerability archetypes. Vulnerability archetypes represent empirical spatial "
    "similarity groupings derived from four normalized indicators (healthcare deficit, "
    "demographic vulnerability, infrastructure deficit, coastal exposure) and are NOT hazard "
    "predictions or casualty forecasts. Preparedness scores represent structural baseline "
    "capacity derived from 2011 Census amenities, institutional registries, and physical proxies."
)


def validate_source_datasets(
    df_prep: pd.DataFrame,
    df_clust: Union[pd.DataFrame, gpd.GeoDataFrame],
) -> Tuple[int, int, int]:
    """Validate primary key uniqueness, nullability, and intersection counts.

    Args:
        df_prep: Preparedness master DataFrame containing area_id.
        df_clust: Clustering archetypes GeoDataFrame containing shrid2.

    Returns:
        Tuple of (n_prep, n_clust, n_intersection).

    Raises:
        ValueError: If duplicate keys, null keys, or unexpected intersection counts are detected.
    """
    logger.info("Validating source dataset identifiers and key integrity...")

    # Preparedness checks
    if "area_id" not in df_prep.columns:
        raise ValueError("Preparedness master dataset is missing primary key column 'area_id'")
    n_prep = len(df_prep)
    if df_prep["area_id"].isna().sum() > 0:
        raise ValueError(f"Found {df_prep['area_id'].isna().sum()} null area_id keys in preparedness dataset")
    if df_prep["area_id"].nunique() != n_prep:
        raise ValueError(f"Duplicate area_id detected in preparedness dataset: {df_prep['area_id'].nunique()} unique vs {n_prep} total")

    # Clustering checks
    if "shrid2" not in df_clust.columns:
        raise ValueError("Clustering archetypes dataset is missing primary key column 'shrid2'")
    n_clust = len(df_clust)
    if df_clust["shrid2"].isna().sum() > 0:
        raise ValueError(f"Found {df_clust['shrid2'].isna().sum()} null shrid2 keys in clustering dataset")
    if df_clust["shrid2"].nunique() != n_clust:
        raise ValueError(f"Duplicate shrid2 detected in clustering dataset: {df_clust['shrid2'].nunique()} unique vs {n_clust} total")

    # Intersection check
    prep_keys = set(df_prep["area_id"])
    clust_keys = set(df_clust["shrid2"])
    intersection = prep_keys.intersection(clust_keys)
    n_inter = len(intersection)

    logger.info(
        f"Validation passed: Preparedness={n_prep} units, Clustering={n_clust} units, "
        f"Exact Intersection={n_inter} units."
    )

    if n_inter != n_clust:
        unmatched_clust = clust_keys - prep_keys
        raise ValueError(f"Expected all {n_clust} clustering records to match, but {len(unmatched_clust)} were unmatched")

    return n_prep, n_clust, n_inter


def integrate_preparedness_and_archetypes(
    preparedness_path: Union[str, Path] = DEFAULT_PREPAREDNESS_PATH,
    archetypes_path: Union[str, Path] = DEFAULT_ARCHETYPES_PATH,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Execute the deterministic integration of preparedness and vulnerability archetypes.

    Performs a validated 1:1 left join, restores verified town and village names,
    reconciles identifiers, stamps explicit coverage status, and writes the unified Parquet.

    Args:
        preparedness_path: Path to master_village_preparedness.parquet.
        archetypes_path: Path to coastal_vulnerability_archetypes.parquet.
        output_path: Path where unified_village_preparedness.parquet will be written.
        save_csv: Whether to also write a companion CSV copy.

    Returns:
        The resulting unified DataFrame.
    """
    prep_p = Path(preparedness_path)
    clust_p = Path(archetypes_path)
    out_p = Path(output_path)

    logger.info(f"Loading preparedness master dataset from {prep_p}...")
    df_prep = pd.read_parquet(prep_p)

    logger.info(f"Loading clustering archetypes dataset from {clust_p}...")
    df_clust = gpd.read_parquet(clust_p) if clust_p.suffix in [".parquet", ".pq"] else pd.read_parquet(clust_p)

    # 1. Validate datasets
    n_prep, n_clust, n_inter = validate_source_datasets(df_prep, df_clust)

    # 2. Extract verified archetype and administrative columns from clustering repo
    clust_subset = df_clust[
        [
            "shrid2",
            "pc11_id",
            "unit_name",
            "unit_type",
            "vulnerability_archetype_id",
            "vulnerability_archetype_label",
            "archetype_intervention_need",
        ]
    ].copy()

    # 3. Perform validated Left Join (preserves all 1,879 preparedness records)
    logger.info("Executing 1:1 relational left join on area_id == shrid2...")
    merged = df_prep.merge(
        clust_subset,
        left_on="area_id",
        right_on="shrid2",
        how="left",
        suffixes=("", "_clust"),
    )

    if len(merged) != n_prep:
        raise RuntimeError(
            f"Row count mismatch after join! Expected {n_prep} rows, got {len(merged)} rows. Possible duplicate keys."
        )

    # 4. Canonical Key and Identifier Reconciliation
    # shrid2 is the canonical SHRUG identifier across the entire dataset
    merged["shrid2"] = merged["area_id"]

    # pc11_id: Clean Census 2011 code.
    # For the 622 matched units, clustering pc11_id is already clean.
    # For unmatched units, parse the 5th token of shrid2 (11-SS-DDD-TTTTT-VVVVVV),
    # which eliminates the malformed '00None' bug found in village_code.
    shrid_tokens = merged["shrid2"].str.split("-", expand=True)
    clean_pc11_from_shrid = shrid_tokens[4]
    merged["pc11_id"] = merged["pc11_id"].fillna(clean_pc11_from_shrid)

    # Also fix village_code so both identifier fields are clean
    merged["village_code"] = merged["pc11_id"]

    # 5. Entity Name Hygiene
    # For matched units: restore clean unit_name from clustering (restores 59 town names and strips whitespace)
    # For unmatched units: keep preparedness village_name after stripping trailing whitespace
    cleaned_clust_name = merged["unit_name"].astype(str).str.strip()
    cleaned_prep_name = merged["village_name"].astype(str).str.strip()

    merged["village_name"] = np.where(
        merged["unit_name"].notna(),
        cleaned_clust_name,
        cleaned_prep_name,
    )

    # 6. Explicit Cluster Assignment Status Tracking
    merged["cluster_assignment_status"] = np.where(
        merged["vulnerability_archetype_id"].notna(),
        "ASSIGNED",
        "UNASSIGNED_OUT_OF_STUDY_AREA",
    )

    # 7. Cast Archetype ID to Nullable Integer (Int16) to preserve integer semantics with nulls
    merged["vulnerability_archetype_id"] = merged["vulnerability_archetype_id"].astype("Int16")

    # 8. Attach Methodological Disclaimer
    merged["data_disclaimer"] = INTEGRATION_DISCLAIMER

    # 9. Structure and Order the Unified Columns Logically
    # Canonical primary identifiers & administrative fields
    admin_cols = [
        "shrid2",
        "area_id",
        "pc11_id",
        "village_code",
        "village_name",
        "district_name",
        "district_code",
        "subdistrict_name",
        "unit_type",
        "lat",
        "lon",
        "village_area_sqkm",
    ]

    # Preparedness domain & composite scores
    preparedness_cols = [
        "preparedness_score",
        "preparedness_tier",
        "score_medical",
        "score_shelters",
        "score_infrastructure",
        "score_early_warning",
        "score_socioeconomic",
        "vulnerability_tier",
        "cvi_physical",
        "cvi_social",
        "cvi_integrated",
    ]

    # Vulnerability archetype fields
    archetype_cols = [
        "vulnerability_archetype_id",
        "vulnerability_archetype_label",
        "archetype_intervention_need",
        "cluster_assignment_status",
    ]

    # Contextual indicators and raw proxy amenities
    context_cols = [
        "population",
        "households",
        "pop_density_sqkm",
        "avg_household_size",
        "sc_share",
        "st_share",
        "vulnerable_pop_share",
        "elevation_mean",
        "dist_coast_km",
        "dist_nearest_shelter_km",
        "n_shelters_10km",
        "shelter_capacity_nearby",
        "shelter_capacity_per_capita",
        "shelter_road_access",
        "dist_nearest_hc_km",
        "dist_nearest_phc_km",
        "dist_nearest_hospital_km",
        "n_hc_10km",
        "n_phc_chc_10km",
        "has_phc_or_chc",
        "dist_nearest_gauge_km",
        "n_gauges_20km",
        "nearest_gauge_name",
        "road_connectivity_score",
        "drinking_water_score",
        "has_pucca_road",
        "has_major_district_road",
        "bus_connectivity",
        "tauktae_affected",
        "data_disclaimer",
    ]

    # Construct final column selection
    all_ordered_cols = admin_cols + preparedness_cols + archetype_cols + context_cols
    # Ensure any additional columns in merged are not lost
    remaining_cols = [c for c in merged.columns if c not in all_ordered_cols and c != "unit_name"]
    final_cols = all_ordered_cols + remaining_cols

    unified_df = merged[final_cols].copy()

    # 10. Persist Output Parquet (and optional CSV)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing unified dataset ({len(unified_df)} rows, {len(unified_df.columns)} cols) to {out_p}...")
    unified_df.to_parquet(out_p, index=False)

    if save_csv:
        csv_p = out_p.with_suffix(".csv")
        logger.info(f"Writing companion CSV to {csv_p}...")
        unified_df.to_csv(csv_p, index=False)

    logger.info("Milestone 1 integration ETL successfully completed.")
    return unified_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate Preparedness Master with Vulnerability Archetypes.")
    parser.add_argument(
        "--preparedness-path",
        type=Path,
        default=DEFAULT_PREPAREDNESS_PATH,
        help="Path to master_village_preparedness.parquet",
    )
    parser.add_argument(
        "--archetypes-path",
        type=Path,
        default=DEFAULT_ARCHETYPES_PATH,
        help="Path to coastal_vulnerability_archetypes.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output unified_village_preparedness.parquet",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing companion CSV copy.",
    )
    args = parser.parse_args()

    integrate_preparedness_and_archetypes(
        preparedness_path=args.preparedness_path,
        archetypes_path=args.archetypes_path,
        output_path=args.output_path,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
