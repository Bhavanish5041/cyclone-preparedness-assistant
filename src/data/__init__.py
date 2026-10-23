"""Data management, ingestion, inventory, and processing subpackage."""

from src.data.census import (
    CENSUS_VINTAGE_NOTE,
    TARGET_DISTRICTS,
    extract_coastal_towns_dchb,
    extract_coastal_villages_dchb,
    process_coastal_census,
    validate_spatial_boundary_join,
)
from src.data.healthcare import (
    DATA_DISCLAIMER,
    FACILITY_TIER_MAPPING,
    FACILITY_TYPE_LABELS,
    process_coastal_healthcare,
    validate_coordinates,
)
from src.data.inventory import (
    DataInventory,
    compute_sha256,
    detect_duplicates,
    inspect_file,
)
from src.data.provenance import get_provenance_info

__all__ = [
    "DataInventory",
    "compute_sha256",
    "detect_duplicates",
    "inspect_file",
    "get_provenance_info",
    "process_coastal_healthcare",
    "validate_coordinates",
    "FACILITY_TYPE_LABELS",
    "FACILITY_TIER_MAPPING",
    "DATA_DISCLAIMER",
    "process_coastal_census",
    "validate_spatial_boundary_join",
    "extract_coastal_villages_dchb",
    "extract_coastal_towns_dchb",
    "TARGET_DISTRICTS",
    "CENSUS_VINTAGE_NOTE",
]
