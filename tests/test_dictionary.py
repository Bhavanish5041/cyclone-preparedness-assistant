"""Unit tests for candidate feature dictionary, quality registry, and collinearity auditing."""

import numpy as np
import pandas as pd
import pytest

from src.features.dictionary import (
    CANDIDATE_FEATURE_DICTIONARY,
    audit_feature_quality,
    detect_collinear_pairs,
    get_feature_dictionary_df,
)


def test_candidate_feature_dictionary_completeness():
    """Verify that all essential candidate variables are registered with complete metadata."""
    expected_features = [
        "dist_to_nearest_phc_chc_km",
        "dist_to_nearest_hospital_km",
        "dist_to_nearest_facility_any_km",
        "phc_chc_in_unit_count",
        "hospitals_in_unit_count",
        "facilities_in_unit_total",
        "total_population",
        "total_households",
        "sc_st_ratio",
        "sex_ratio_female_per_1000_male",
        "literacy_rate",
        "has_treated_tap_water",
        "has_all_weather_road",
        "has_pucca_road",
        "has_domestic_power",
        "drainage_system",
        "dist_to_coastline_km",
        "rep_point_strictly_inside",
        "census_vintage",
    ]

    valid_domains = {
        "Healthcare Accessibility",
        "Demographic Indicators",
        "Water, Road, Drainage and Power Infrastructure",
        "Coastal Proximity",
        "Data Freshness and Quality",
    }

    valid_directions = {
        "higher_indicates_lower_preparedness",
        "higher_indicates_better_preparedness",
        "contextual",
    }

    for feat in expected_features:
        assert feat in CANDIDATE_FEATURE_DICTIONARY, f"Feature {feat} missing from dictionary"
        meta = CANDIDATE_FEATURE_DICTIONARY[feat]
        assert meta.domain in valid_domains, f"Invalid domain {meta.domain} for {feat}"
        assert meta.interpretation_direction in valid_directions, f"Invalid direction for {feat}"
        assert len(meta.source_dataset) > 0
        assert len(meta.known_bias_or_limitation) > 0


def test_get_feature_dictionary_df():
    """Verify conversion of feature dictionary to DataFrame."""
    df = get_feature_dictionary_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(CANDIDATE_FEATURE_DICTIONARY)
    assert "domain" in df.columns
    assert "known_bias_or_limitation" in df.columns


def test_audit_feature_quality():
    """Verify audit_feature_quality accurately tallies missingness and distribution statistics."""
    test_df = pd.DataFrame(
        {
            "dist_to_nearest_phc_chc_km": [2.0, 4.0, 6.0, np.nan],
            "total_population": [1000, 2000, 3000, 4000],
            "dist_to_coastline_km": [1.0, 2.0, 3.0, 4.0],
        }
    )

    audit_df = audit_feature_quality(test_df)
    assert isinstance(audit_df, pd.DataFrame)

    phc_row = audit_df[audit_df["feature_name"] == "dist_to_nearest_phc_chc_km"].iloc[0]
    assert phc_row["missing_count"] == 1
    assert phc_row["missing_pct"] == 25.0
    assert phc_row["min"] == 2.0
    assert phc_row["median"] == 4.0
    assert phc_row["max"] == 6.0


def test_detect_collinear_pairs_synthetic():
    """Verify that detect_collinear_pairs correctly flags correlated variables."""
    # Synthetic dataset with perfect collinearity between total_population and total_households
    pop = np.linspace(500, 10000, 50)
    hh = pop / 4.5 + np.random.normal(0, 5, 50)  # r > 0.99
    dist_coast = np.random.uniform(1, 50, 50)

    test_df = pd.DataFrame(
        {
            "total_population": pop,
            "total_households": hh,
            "dist_to_coastline_km": dist_coast,
        }
    )

    collinear = detect_collinear_pairs(test_df, threshold=0.85, method="pearson")
    assert len(collinear) >= 1
    top_pair = collinear.iloc[0]
    pair_names = {top_pair["feature_1"], top_pair["feature_2"]}
    assert pair_names == {"total_population", "total_households"}
    assert bool(top_pair["is_severe"]) is True
    assert top_pair["correlation"] > 0.95
