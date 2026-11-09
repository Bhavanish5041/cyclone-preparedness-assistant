"""Automated validation test suite for Milestone 1: Unified Village Preparedness Dataset.

Verifies:
1. Output contains exactly 1,879 rows.
2. shrid2 is unique and has no null values.
3. Exactly 622 records are ASSIGNED.
4. Exactly 1,257 records are UNASSIGNED_OUT_OF_STUDY_AREA.
5. All 622 original archetype IDs match the clustering source exactly.
6. All 622 original archetype labels match exactly.
7. All 622 original intervention descriptions match exactly.
8. All 1,879 original preparedness scores and tiers are preserved.
9. No Cartesian join expansion or accidental row loss occurs.
10. All matched town records retain their correct names and unit types.
11. Unmatched Uttara Kannada records have null archetype fields.
12. The output has no duplicate primary keys.
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_PATH = PROJECT_ROOT / "data" / "processed" / "unified_village_preparedness.parquet"
PREPAREDNESS_PATH = PROJECT_ROOT / "data" / "processed" / "master_village_preparedness.parquet"
CLUSTERING_PATH = PROJECT_ROOT / "data" / "processed" / "coastal_vulnerability_archetypes.parquet"


@pytest.fixture(scope="module")
def df_unified() -> pd.DataFrame:
    """Load the generated unified village preparedness dataset."""
    assert UNIFIED_PATH.exists(), f"Unified dataset not found at {UNIFIED_PATH}"
    df = pd.read_parquet(UNIFIED_PATH)
    assert len(df) > 0, "Unified dataset is empty"
    return df


@pytest.fixture(scope="module")
def df_prep() -> pd.DataFrame:
    """Load the source preparedness master dataset."""
    assert PREPAREDNESS_PATH.exists(), f"Preparedness dataset not found at {PREPAREDNESS_PATH}"
    return pd.read_parquet(PREPAREDNESS_PATH)


@pytest.fixture(scope="module")
def df_clust() -> pd.DataFrame:
    """Load the source clustering archetypes dataset."""
    assert CLUSTERING_PATH.exists(), f"Clustering dataset not found at {CLUSTERING_PATH}"
    return gpd.read_parquet(CLUSTERING_PATH) if CLUSTERING_PATH.suffix == ".parquet" else pd.read_parquet(CLUSTERING_PATH)


def test_1_row_count(df_unified):
    """Test 1: Output contains exactly 1,879 rows."""
    assert len(df_unified) == 1879, f"Expected 1,879 rows, got {len(df_unified)}"


def test_2_shrid2_unique_and_non_null(df_unified):
    """Test 2: shrid2 is unique and has no null values."""
    assert df_unified["shrid2"].isna().sum() == 0, "Found null values in shrid2"
    assert df_unified["shrid2"].nunique() == 1879, "shrid2 values are not strictly unique"
    # Format pattern check: 11-SS-DDD-TTTTT-VVVVVV
    parts = df_unified["shrid2"].str.split("-")
    assert (parts.str.len() == 5).all(), "Malformed shrid2 format detected"


def test_3_assigned_count(df_unified):
    """Test 3: Exactly 622 records are ASSIGNED."""
    assigned_count = (df_unified["cluster_assignment_status"] == "ASSIGNED").sum()
    assert assigned_count == 622, f"Expected exactly 622 ASSIGNED records, got {assigned_count}"


def test_4_unassigned_count(df_unified):
    """Test 4: Exactly 1,257 records are UNASSIGNED_OUT_OF_STUDY_AREA."""
    unassigned_count = (df_unified["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA").sum()
    assert unassigned_count == 1257, f"Expected exactly 1,257 UNASSIGNED records, got {unassigned_count}"


def test_5_archetype_ids_match_clustering_source(df_unified, df_clust):
    """Test 5: All 622 original archetype IDs match the clustering source exactly."""
    assigned = df_unified[df_unified["cluster_assignment_status"] == "ASSIGNED"]
    merged = assigned.merge(df_clust[["shrid2", "vulnerability_archetype_id"]], on="shrid2")
    assert len(merged) == 622, "Failed to match all 622 assigned records with clustering source"
    assert (merged["vulnerability_archetype_id_x"] == merged["vulnerability_archetype_id_y"]).all(), (
        "Discrepancy detected in vulnerability_archetype_id between unified and clustering source"
    )


def test_6_archetype_labels_match_clustering_source(df_unified, df_clust):
    """Test 6: All 622 original archetype labels match exactly."""
    assigned = df_unified[df_unified["cluster_assignment_status"] == "ASSIGNED"]
    merged = assigned.merge(df_clust[["shrid2", "vulnerability_archetype_label"]], on="shrid2")
    assert len(merged) == 622
    assert (merged["vulnerability_archetype_label_x"] == merged["vulnerability_archetype_label_y"]).all(), (
        "Discrepancy detected in vulnerability_archetype_label"
    )


def test_7_intervention_descriptions_match_clustering_source(df_unified, df_clust):
    """Test 7: All 622 original intervention descriptions match exactly."""
    assigned = df_unified[df_unified["cluster_assignment_status"] == "ASSIGNED"]
    merged = assigned.merge(df_clust[["shrid2", "archetype_intervention_need"]], on="shrid2")
    assert len(merged) == 622
    assert (merged["archetype_intervention_need_x"] == merged["archetype_intervention_need_y"]).all(), (
        "Discrepancy detected in archetype_intervention_need"
    )


def test_8_preparedness_scores_and_tiers_preserved(df_unified, df_prep):
    """Test 8: All 1,879 original preparedness scores and tiers are preserved."""
    merged = df_unified.merge(
        df_prep[["area_id", "preparedness_score", "preparedness_tier"]],
        left_on="shrid2",
        right_on="area_id",
        suffixes=("_uni", "_orig"),
    )
    assert len(merged) == 1879
    np.testing.assert_allclose(
        merged["preparedness_score_uni"],
        merged["preparedness_score_orig"],
        atol=1e-5,
        err_msg="Discrepancy in preparedness_score",
    )
    assert (merged["preparedness_tier_uni"] == merged["preparedness_tier_orig"]).all(), (
        "Discrepancy in preparedness_tier"
    )


def test_9_no_cartesian_join_or_row_loss(df_unified, df_prep):
    """Test 9: No Cartesian join expansion or accidental row loss occurs."""
    assert len(df_unified) == len(df_prep) == 1879
    assert set(df_unified["shrid2"]) == set(df_prep["area_id"])


def test_10_matched_town_records_names_and_unit_types(df_unified, df_clust):
    """Test 10: All matched town records retain their correct names and unit types."""
    towns_unified = df_unified[df_unified["unit_type"] == "TOWN"]
    assert len(towns_unified) == 59, f"Expected exactly 59 towns, got {len(towns_unified)}"
    
    # Ensure no town has synthetic 'Village_' name
    assert not towns_unified["village_name"].str.startswith("Village_").any(), (
        "Found synthetic 'Village_' prefix in matched town records"
    )
    
    # Match against clustering source towns
    towns_clust = df_clust[df_clust["unit_type"] == "TOWN"]
    merged_towns = towns_unified.merge(towns_clust, on="shrid2")
    assert len(merged_towns) == 59
    assert (merged_towns["village_name"] == merged_towns["unit_name"].str.strip()).all(), (
        "Town names do not match canonical clustering unit_name"
    )


def test_11_unmatched_records_have_null_archetype_fields(df_unified):
    """Test 11: Unmatched Uttara Kannada records have null archetype fields."""
    unmatched = df_unified[df_unified["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"]
    assert len(unmatched) == 1257
    assert (unmatched["district_name"] == "Uttara Kannada").all()
    assert unmatched["vulnerability_archetype_id"].isna().all(), "Unmatched unit has non-null archetype ID"
    assert unmatched["vulnerability_archetype_label"].isna().all(), "Unmatched unit has non-null archetype label"
    assert unmatched["archetype_intervention_need"].isna().all(), "Unmatched unit has non-null intervention text"


def test_12_no_duplicate_primary_keys(df_unified):
    """Test 12: The output has no duplicate primary keys."""
    assert df_unified["shrid2"].duplicated().sum() == 0, "Duplicate shrid2 keys found in unified dataset"
    assert df_unified["area_id"].duplicated().sum() == 0, "Duplicate area_id keys found in unified dataset"
    assert (df_unified["shrid2"] == df_unified["area_id"]).all(), "shrid2 does not equal area_id"


def test_13_canonical_administrative_names_and_known_60_legacy_differences(df_unified, df_prep):
    """Test 13: Canonical administrative names are preserved, documenting the 60 known legacy differences.

    In the legacy master preparedness dataset, 59 census towns used synthetic 'Village_<census_code>'
    placeholders and 1 rural village used 'Village_00None'. The unified dataset resolves all 60 to
    canonical Census administrative names (e.g. 'Kundapura (TMC)', 'Kodimbala'), while the remaining
    1,819 regular villages maintain exact string equality when trimmed.
    """
    merged = df_unified.merge(
        df_prep[["area_id", "village_name"]],
        left_on="shrid2",
        right_on="area_id",
        suffixes=("_uni", "_prep"),
    )
    assert len(merged) == 1879

    name_diff = merged["village_name_uni"].str.strip() != merged["village_name_prep"].str.strip()
    assert name_diff.sum() == 60, f"Expected exactly 60 known legacy name differences, got {name_diff.sum()}"

    # 1. Exactly 59 differences are matched census towns where synthetic names were replaced
    diff_records = merged[name_diff]
    town_diffs = diff_records[diff_records["unit_type"] == "TOWN"]
    assert len(town_diffs) == 59
    assert town_diffs["village_name_prep"].str.startswith("Village_").all(), (
        "Expected all 59 legacy town placeholders to start with 'Village_'"
    )
    assert not town_diffs["village_name_uni"].str.startswith("Village_").any(), (
        "Unified dataset towns must not contain synthetic 'Village_' prefixes"
    )

    # 2. Exactly 1 difference is a rural village (Kodimbala) where legacy placeholder was Village_00None
    village_diffs = diff_records[diff_records["unit_type"] == "VILLAGE"]
    assert len(village_diffs) == 1
    kodimbala = village_diffs.iloc[0]
    assert kodimbala["shrid2"] == "11-29-575-05564-617728"
    assert kodimbala["village_name_prep"] == "Village_00None"
    assert kodimbala["village_name_uni"] == "Kodimbala"

    # 3. All other 1,819 records have exact 1-to-1 matching village names
    same_records = merged[~name_diff]
    assert len(same_records) == 1819
    assert (same_records["village_name_uni"].str.strip() == same_records["village_name_prep"].str.strip()).all()
