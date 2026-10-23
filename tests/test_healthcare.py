"""Unit tests for healthcare facility filtering, coordinate validation, and classification."""

import numpy as np
import pandas as pd
import pytest

from src.data.healthcare import (
    FACILITY_TIER_MAPPING,
    FACILITY_TYPE_LABELS,
    validate_coordinates,
)


def test_validate_coordinates_clean():
    """Verify coordinate validation accepts strictly valid points."""
    df = pd.DataFrame(
        {
            "Facility Name": ["PHC 1", "CHC 2"],
            "Latitude": ["12.9141", 13.3408],
            "Longitude": ["74.8560", 74.7421],
        }
    )

    valid_df, metrics = validate_coordinates(df)
    assert len(valid_df) == 2
    assert metrics["valid_coordinate_count"] == 2
    assert metrics["invalid_coordinate_count"] == 0
    assert metrics["missing_latitude"] == 0
    assert metrics["zero_coordinates"] == 0
    assert metrics["out_of_bounds_coordinates"] == 0


def test_validate_coordinates_flags_anomalies():
    """Verify coordinate validation flags missing, zero, and out-of-bounds coordinates."""
    df = pd.DataFrame(
        {
            "Facility Name": ["Valid", "ZeroCoords", "MissingLat", "OutOfBoundsNorth", "Unparseable"],
            "Latitude": [13.0, 0.0, None, 28.6139, "not_a_number"],  # 28.6 is Delhi
            "Longitude": [74.9, 0.0, 75.0, 77.2090, 74.8],
        }
    )

    valid_df, metrics = validate_coordinates(df)
    assert len(valid_df) == 1
    assert valid_df.iloc[0]["Facility Name"] == "Valid"
    assert metrics["zero_coordinates"] == 1
    assert metrics["missing_latitude"] == 2  # None + unparseable coerced to NaN
    assert metrics["out_of_bounds_coordinates"] == 1  # Delhi point
    assert metrics["invalid_coordinate_count"] == 4


def test_facility_type_and_tier_mapping():
    """Verify that facility codes map correctly to human-readable names and tiers."""
    test_codes = ["sub_cen", "phc", "chc", "s_t_h", "dis_h"]
    for code in test_codes:
        assert code in FACILITY_TYPE_LABELS
        assert code in FACILITY_TIER_MAPPING

    assert "Sub-Centre" in FACILITY_TYPE_LABELS["sub_cen"]
    assert "District Hospital" in FACILITY_TYPE_LABELS["dis_h"]
    assert "Tier 3" in FACILITY_TIER_MAPPING["dis_h"]
    assert "Tier 1" in FACILITY_TIER_MAPPING["sub_cen"]
