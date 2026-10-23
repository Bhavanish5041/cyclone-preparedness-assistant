"""Census 2011 extraction, harmonization, and spatial indicator alignment.

Processes District Census Handbook (DCHB) Village and Town releases for Karnataka
(focusing on coastal districts: Dakshina Kannada 575 and Udupi 569), extracts demographic
and infrastructure indicators, checks join integrity with SHRUG spatial boundaries,
and exports clean intermediate datasets stamped with explicit 2011 vintage disclaimers.
"""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import geopandas as gpd
import openpyxl
import pandas as pd

logger = logging.getLogger(__name__)

CENSUS_VINTAGE_NOTE = (
    "HISTORICAL BASELINE: All demographic, household, water access, road connectivity, "
    "and institutional counts are derived from the 2011 Census of India. These figures "
    "represent a structural baseline and must NOT be interpreted as 2026 ground truth."
)

TARGET_DISTRICTS = {569: "Udupi", 575: "Dakshina Kannada"}


def extract_coastal_villages_dchb(
    village_xlsx_path: Path,
    target_district_codes: Set[int],
) -> pd.DataFrame:
    """Stream and extract coastal villages from the 55 MB DCHB Village Release workbook.

    Uses openpyxl read_only mode to filter by district code efficiently without loading
    all 29,340 statewide records into memory at once.
    """
    logger.info(f"Streaming coastal villages from {village_xlsx_path}...")
    wb = openpyxl.load_workbook(village_xlsx_path, read_only=True)
    ws = wb["Village_Data_2900"]

    # Header mapping
    raw_headers = [str(c.value) if c.value is not None else "" for c in next(ws.iter_rows(max_row=1))]

    def get_idx(name: str) -> int:
        target_norm = " ".join(name.strip().lower().split())
        for idx, h in enumerate(raw_headers):
            curr_norm = " ".join(h.strip().lower().split())
            if curr_norm == target_norm:
                return idx
        raise KeyError(f"Column '{name}' not found in Village_Data_2900 headers")

    # Map column indices using exact headers
    dist_idx = get_idx("District Code")
    v_code_idx = get_idx("Village Code")
    v_name_idx = get_idx("Village Name")
    subdist_code_idx = get_idx("Sub District Code")
    subdist_name_idx = get_idx("Sub District Name")
    hh_idx = get_idx("Total   Households ")
    pop_idx = get_idx("Total Population of Village")
    pop_m_idx = get_idx("Total Male Population of Village")
    pop_f_idx = get_idx("Total Female Population of Village")
    sc_idx = get_idx("Total Scheduled Castes Population of Village")
    st_idx = get_idx("Total Scheduled Tribes Population of Village")

    # Amenities
    chc_idx = get_idx("Community Health Centre (Numbers)")
    phc_idx = get_idx("Primary Health Centre (Numbers)")
    sub_cen_idx = get_idx("Primary Heallth Sub Centre (Numbers)")  # Note source typo in official file
    hosp_idx = get_idx("Hospital Allopathic (Numbers)")
    disp_idx = get_idx("Dispensary (Numbers)")

    tap_water_idx = get_idx("Tap Water-Treated (Status A(1)/NA(2))")
    all_weather_road_idx = get_idx("All Weather Road (Status A(1)/NA(2))")
    pucca_road_idx = get_idx("Black Topped (pucca) Road (Status A(1)/NA(2))")
    bus_idx = get_idx("Public Bus Service (Status A(1)/NA(2))")
    power_idx = get_idx("Power Supply For Domestic Use  (Status A(1)/NA(2))")
    drain_closed_idx = get_idx("Closed Drainage (Status A(1)/NA(2))")
    drain_open_idx = get_idx(" Open Drainage (Status A(1)/NA(2))")
    drain_none_idx = get_idx("No  Drainage (Status A(1)/NA(2))")

    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        dcode = row[dist_idx]
        if dcode in target_district_codes:
            # Parse status codes (Status 1 = Available / Yes; 2 = Not Available / No)
            tap_water_avail = 1 if row[tap_water_idx] == 1 else 0
            all_weather_avail = 1 if row[all_weather_road_idx] == 1 else 0
            pucca_avail = 1 if row[pucca_road_idx] == 1 else 0
            bus_avail = 1 if row[bus_idx] == 1 else 0
            power_avail = 1 if row[power_idx] == 1 else 0

            drain_str = "None"
            if row[drain_closed_idx] == 1:
                drain_str = "Closed"
            elif row[drain_open_idx] == 1:
                drain_str = "Open"

            rows.append(
                {
                    "unit_type": "VILLAGE",
                    "pc11_id": str(row[v_code_idx]).strip().zfill(6),
                    "state_code": 29,
                    "district_code": int(dcode),
                    "district_name": TARGET_DISTRICTS.get(int(dcode), "Other"),
                    "subdistrict_code": str(row[subdist_code_idx]),
                    "subdistrict_name": str(row[subdist_name_idx]),
                    "unit_name": str(row[v_name_idx]).strip(),
                    "total_households": pd.to_numeric(row[hh_idx], errors="coerce"),
                    "total_population": pd.to_numeric(row[pop_idx], errors="coerce"),
                    "male_population": pd.to_numeric(row[pop_m_idx], errors="coerce"),
                    "female_population": pd.to_numeric(row[pop_f_idx], errors="coerce"),
                    "sc_population": pd.to_numeric(row[sc_idx], errors="coerce"),
                    "st_population": pd.to_numeric(row[st_idx], errors="coerce"),
                    "phc_count": pd.to_numeric(row[phc_idx], errors="coerce"),
                    "chc_count": pd.to_numeric(row[chc_idx], errors="coerce"),
                    "subcentre_count": pd.to_numeric(row[sub_cen_idx], errors="coerce"),
                    "allopathic_hospital_count": pd.to_numeric(row[hosp_idx], errors="coerce"),
                    "dispensary_count": pd.to_numeric(row[disp_idx], errors="coerce"),
                    "has_treated_tap_water": tap_water_avail,
                    "has_all_weather_road": all_weather_avail,
                    "has_pucca_road": pucca_avail,
                    "has_public_bus": bus_avail,
                    "has_domestic_power": power_avail,
                    "drainage_system": drain_str,
                }
            )

    wb.close()
    df_villages = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df_villages)} coastal villages.")
    return df_villages


def extract_coastal_towns_dchb(
    town_xlsx_path: Path,
    target_district_codes: Set[int],
) -> pd.DataFrame:
    """Extract coastal statutory and census towns from the DCHB Town Release workbook."""
    logger.info(f"Extracting coastal towns from {town_xlsx_path}...")
    df_town_raw = pd.read_excel(town_xlsx_path, sheet_name="Town_2900")

    is_coastal = df_town_raw["District Code"].isin(target_district_codes)
    df_town = df_town_raw[is_coastal].copy()

    # Drainage mapping: Open-1, Closed-2, Both-3, No-4
    drainage_map = {1: "Open", 2: "Closed", 3: "Both", 4: "None"}

    rows = []
    for _, r in df_town.iterrows():
        dcode = int(r["District Code"])
        drain_val = drainage_map.get(r.get("Drainage System (Open-1/Closed-2/Both-3/No-4 ?)"), "Unknown")
        fire_status = 1 if r.get("Fire Fighting Service (Status A(1)/NA(2))") == 1 else 0

        rows.append(
            {
                "unit_type": "TOWN",
                "pc11_id": str(r["Town Code"]).strip().zfill(6),
                "state_code": 29,
                "district_code": dcode,
                "district_name": TARGET_DISTRICTS.get(dcode, "Other"),
                "subdistrict_code": str(r.get("Sub District Code", "")),
                "subdistrict_name": str(r.get("Sub District Name", "")),
                "unit_name": str(r["Town Name"]).strip(),
                "total_households": pd.to_numeric(r.get("Total   Households "), errors="coerce"),
                "total_population": pd.to_numeric(r.get("Total Population of Town"), errors="coerce"),
                "male_population": pd.to_numeric(r.get("Total Male Population of Town"), errors="coerce"),
                "female_population": pd.to_numeric(r.get("Total Female Population of Town"), errors="coerce"),
                "sc_population": pd.to_numeric(r.get("Total Scheduled Castes Population of Town"), errors="coerce"),
                "st_population": pd.to_numeric(r.get("Total Scheduled Tribes Population of Town"), errors="coerce"),
                "phc_count": pd.to_numeric(r.get("Dispensary/Health Centre (Numbers)"), errors="coerce"),  # Proxy in town dir
                "chc_count": pd.to_numeric(r.get("Family Welfare Centre (Numbers)"), errors="coerce"),
                "subcentre_count": pd.to_numeric(r.get("Maternity and Child Welfare Centre (Numbers)"), errors="coerce"),
                "allopathic_hospital_count": pd.to_numeric(r.get("Hospital Allopathic (Numbers)"), errors="coerce"),
                "dispensary_count": pd.to_numeric(r.get("Dispensary/Health Centre (Numbers)"), errors="coerce"),
                "has_treated_tap_water": 1,  # Statutory/census towns have municipal water connections
                "has_all_weather_road": 1,
                "has_pucca_road": 1,
                "has_public_bus": 1,
                "has_domestic_power": 1,
                "drainage_system": drain_val,
                "has_fire_fighting_service": fire_status,
            }
        )

    df_towns = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df_towns)} coastal towns.")
    return df_towns


def enrich_with_udupi_pca(
    df_units: pd.DataFrame,
    pca_xls_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enrich Udupi administrative units with literacy and worker counts from PCA_CDB_2915_F_Census.xls."""
    logger.info(f"Loading Udupi PCA from {pca_xls_path}...")
    df_pca = pd.read_excel(pca_xls_path)

    # Filter to village and town records
    pca_units = df_pca[df_pca["Level"].isin(["VILLAGE", "TOWN", "OG"])].copy()
    pca_units["pc11_id"] = pca_units["Town/Village_Code"].astype(str).str.strip().str.zfill(6)

    # Key columns from PCA
    pca_cols = {
        "pc11_id": "pc11_id",
        "Literates Population Person": "literate_population",
        "Total Worker Population Person": "total_workers",
        "Population in the age group 0-6 Person": "child_0_6_population",
    }
    pca_subset = pca_units[list(pca_cols.keys())].rename(columns=pca_cols).drop_duplicates(subset=["pc11_id"])

    # Merge onto coastal units
    initial_len = len(df_units)
    enriched = pd.merge(df_units, pca_subset, on="pc11_id", how="left")
    assert len(enriched) == initial_len, "PCA merge altered row count"

    # Compute literacy rate where available
    enriched["literacy_rate"] = (
        enriched["literate_population"] / enriched["total_population"] * 100.0
    ).round(2)
    enriched["child_ratio"] = (
        enriched["child_0_6_population"] / enriched["total_population"] * 100.0
    ).round(2)

    matched_pca_count = int(enriched["literate_population"].notna().sum())
    logger.info(f"Enriched {matched_pca_count} Udupi units with literacy/worker indicators.")

    return enriched, {"pca_source_records": len(pca_units), "matched_records": matched_pca_count}


def validate_spatial_boundary_join(
    df_census: pd.DataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """Validate cross-dataset join between census indicators and spatial boundary polygons."""
    census_keys = set(df_census["pc11_id"])
    boundary_keys = set(boundaries_gdf["pc11_id"])

    matched_keys = census_keys.intersection(boundary_keys)
    unmatched_in_boundaries = census_keys - boundary_keys
    unmatched_in_census = boundary_keys - census_keys

    # Check for duplicate keys in census dataset
    census_duplicates = df_census[df_census.duplicated(subset=["pc11_id"], keep=False)]

    join_metrics = {
        "total_census_units": len(df_census),
        "total_boundary_polygons": len(boundaries_gdf),
        "exact_pc11_matches": len(matched_keys),
        "match_percentage_of_polygons": round(len(matched_keys) / len(boundaries_gdf) * 100, 2),
        "unmatched_census_units_count": len(unmatched_in_boundaries),
        "unmatched_polygon_keys_count": len(unmatched_in_census),
        "census_duplicate_key_count": len(census_duplicates),
    }

    return join_metrics


def process_coastal_census(
    village_xlsx_path: Path,
    town_xlsx_path: Path,
    pca_xls_path: Path,
    boundaries_gdf: gpd.GeoDataFrame,
    output_parquet: Path,
    output_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute end-to-end extraction, harmonization, PCA enrichment, and join validation."""
    target_district_codes = set(TARGET_DISTRICTS.keys())

    # 1. Extract villages and towns
    df_villages = extract_coastal_villages_dchb(village_xlsx_path, target_district_codes)
    df_towns = extract_coastal_towns_dchb(town_xlsx_path, target_district_codes)

    # 2. Combine into unified administrative units table
    df_combined = pd.concat([df_villages, df_towns], ignore_index=True)

    # Deduplicate in case a census town was listed in both releases
    df_dedup = df_combined.drop_duplicates(subset=["pc11_id"]).copy()

    # 3. Derive vulnerability indicators
    df_dedup["sc_st_population"] = (
        df_dedup["sc_population"].fillna(0) + df_dedup["st_population"].fillna(0)
    )
    df_dedup["sc_st_ratio"] = (
        df_dedup["sc_st_population"] / df_dedup["total_population"].replace(0, pd.NA)
    ).round(4)

    df_dedup["sex_ratio_female_per_1000_male"] = (
        df_dedup["female_population"] / df_dedup["male_population"].replace(0, pd.NA) * 1000
    ).round(1)

    # 4. Enrich with Udupi PCA
    df_enriched, pca_stats = enrich_with_udupi_pca(df_dedup, pca_xls_path)

    # 5. Stamp historical vintage disclaimer
    df_enriched["census_vintage"] = 2011
    df_enriched["data_limitation_note"] = CENSUS_VINTAGE_NOTE

    # 6. Validate join against spatial boundaries
    join_metrics = validate_spatial_boundary_join(df_enriched, boundaries_gdf)

    # 7. Export intermediate datasets
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(output_parquet, index=False)
    logger.info(f"Saved {len(df_enriched)} census records to {output_parquet}")

    if output_csv is not None:
        df_enriched.to_csv(output_csv, index=False)
        logger.info(f"Saved CSV copy to {output_csv}")

    audit_metadata = {
        "target_districts": TARGET_DISTRICTS,
        "total_extracted_units": len(df_enriched),
        "villages_count": int((df_enriched["unit_type"] == "VILLAGE").sum()),
        "towns_count": int((df_enriched["unit_type"] == "TOWN").sum()),
        "units_by_district": df_enriched["district_name"].value_counts().to_dict(),
        "pca_enrichment": pca_stats,
        "boundary_join_validation": join_metrics,
        "census_vintage": 2011,
        "output_file": str(output_parquet),
    }

    return df_enriched, audit_metadata
