"""Unit tests for geographic boundary extraction and shrid2 processing."""

import pytest

from src.geo.boundaries import (
    COASTAL_DISTRICT_CODES,
    build_shrid2_where_clause,
    parse_shrid2_components,
)


def test_build_shrid2_where_clause():
    """Verify that build_shrid2_where_clause constructs valid SQL LIKE queries."""
    clause = build_shrid2_where_clause([569, 575], state_code=29)
    assert clause == "shrid2 LIKE '11-29-569-%' OR shrid2 LIKE '11-29-575-%'"

    # Single district
    clause_single = build_shrid2_where_clause([569], state_code=29)
    assert clause_single == "shrid2 LIKE '11-29-569-%'"


def test_parse_shrid2_components_valid():
    """Verify parsing of standard 5-part shrid2 identifier."""
    # Udupi village
    shrid_udupi = "11-29-569-05522-608660"
    meta = parse_shrid2_components(shrid_udupi)
    assert meta["census_year"] == 2011
    assert meta["state_code"] == 29
    assert meta["district_code"] == 569
    assert meta["district_name"] == "Udupi"
    assert meta["subdistrict_code"] == "05522"
    assert meta["pc11_id"] == "608660"

    # Dakshina Kannada village
    shrid_dk = "11-29-575-05561-617425"
    meta_dk = parse_shrid2_components(shrid_dk)
    assert meta_dk["census_year"] == 2011
    assert meta_dk["state_code"] == 29
    assert meta_dk["district_code"] == 575
    assert meta_dk["district_name"] == "Dakshina Kannada"
    assert meta_dk["subdistrict_code"] == "05561"
    assert meta_dk["pc11_id"] == "617425"


def test_parse_shrid2_components_malformed():
    """Verify that incomplete or non-standard shrid2 strings degrade gracefully."""
    meta = parse_shrid2_components("invalid-string")
    assert meta["district_code"] is None
    assert meta["district_name"] == "Unknown"
