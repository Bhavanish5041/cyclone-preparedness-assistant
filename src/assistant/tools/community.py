"""Community lookup and granular infrastructure exposure analysis tools.

Provides:
- `lookup_community`: Case-insensitive autocomplete search across all 1,879 administrative units.
- `analyze_infrastructure_and_exposure`: Granular physical indicators, healthcare distances,
  shelter access, and explicit Census 2011 baseline provenance caveats.
- `resolve_community_row`: Deterministic community resolution with ambiguity detection.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools.base import (
    AmbiguousCommunityError,
    BaseTool,
    CommunityNotFoundError,
    ToolValidationError,
)

logger = logging.getLogger(__name__)

VALID_DISTRICTS = {"dakshina kannada", "udupi", "uttara kannada"}
DISTRICT_CANONICAL = {
    "dakshina kannada": "Dakshina Kannada",
    "udupi": "Udupi",
    "uttara kannada": "Uttara Kannada",
}


def _safe_float(val: Any, decimals: int = 2) -> Optional[float]:
    """Convert value to float safely, returning None for NaNs or invalid values."""
    if val is None:
        return None
    try:
        f = float(val)
        return round(f, decimals) if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert value to int safely, returning None for NaNs or invalid values."""
    if val is None:
        return None
    try:
        f = float(val)
        return int(f) if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def resolve_community_row(df: pd.DataFrame, identifier: str) -> pd.Series:
    """Resolve an identifier against ranked_df with ambiguity and not-found handling.

    Args:
        df: ranked_df DataFrame containing all 1,879 records.
        identifier: Census PC11 ID, SHRID, or village name.

    Returns:
        pd.Series for the uniquely matching unit.

    Raises:
        ToolValidationError: If identifier is empty or whitespace.
        AmbiguousCommunityError: If village name matches multiple records.
        CommunityNotFoundError: If identifier matches zero records.
    """
    clean_id = str(identifier).strip()
    if not clean_id:
        raise ToolValidationError("Community identifier must be a non-empty string.")

    # 1. Exact match on unique canonical keys (pc11_id, shrid2, area_id)
    id_matches = df[
        (df["pc11_id"].astype(str) == clean_id)
        | (df["shrid2"].astype(str) == clean_id)
        | (df["area_id"].astype(str) == clean_id)
    ]
    if len(id_matches) == 1:
        return id_matches.iloc[0]

    # 2. Exact case-insensitive match on village_name
    name_matches = df[df["village_name"].astype(str).str.strip().str.lower() == clean_id.lower()]
    if len(name_matches) == 1:
        return name_matches.iloc[0]
    elif len(name_matches) > 1:
        candidates = []
        for _, row in name_matches.iterrows():
            candidates.append(
                {
                    "pc11_id": str(row["pc11_id"]),
                    "shrid2": str(row["shrid2"]),
                    "village_name": str(row["village_name"]),
                    "district_name": str(row["district_name"]),
                    "subdistrict_name": str(row["subdistrict_name"]),
                    "priority_band": str(row.get("priority_band", "N/A")),
                }
            )
        raise AmbiguousCommunityError(name=clean_id, candidates=candidates)

    # 3. Not found: search for partial matches to offer helpful suggestions
    partial_matches = df[df["village_name"].astype(str).str.lower().str.contains(clean_id.lower())]
    if 0 < len(partial_matches) <= 4:
        suggestions = [
            f"{r['village_name']} (PC11: {r['pc11_id']}, {r['district_name']})"
            for _, r in partial_matches.iterrows()
        ]
        sug_str = f" Did you mean one of: {', '.join(suggestions)}?"
    else:
        sug_str = ""

    raise CommunityNotFoundError(
        f"Administrative unit '{clean_id}' not found in coastal Karnataka study area (1,879 surveyed units).{sug_str}"
    )


# ==============================================================================
# TOOL 1: lookup_community
# ==============================================================================


class LookupCommunityInput(BaseModel):
    """Input parameters for lookup_community tool."""

    query: str = Field(
        ...,
        min_length=1,
        description="Search string matching village name, Census 2011 PC11 ID, or SHRID.",
        examples=["Tallur", "608720", "Edmoge"],
    )
    district: Optional[str] = Field(
        None,
        description="Optional filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada').",
        examples=["Udupi", "Dakshina Kannada"],
    )
    limit: int = Field(
        5,
        ge=1,
        le=15,
        description="Maximum number of candidate matches to return (1 to 15, default 5).",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Search query cannot be empty or solely whitespace.")
        return clean

    @field_validator("district")
    @classmethod
    def validate_district(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip().lower()
        if clean not in VALID_DISTRICTS:
            raise ValueError(
                f"Invalid district '{v}'. Allowed values: 'Dakshina Kannada', 'Udupi', 'Uttara Kannada'."
            )
        return DISTRICT_CANONICAL[clean]


class LookupCommunityTool(BaseTool):
    """Tool for community lookup and autocomplete across all 1,879 administrative units."""

    name = "lookup_community"
    description = (
        "Search and resolve administrative units across coastal Karnataka (1,879 units across Dakshina Kannada, "
        "Udupi, and Uttara Kannada) by village name, Census 2011 PC11 ID, or SHRID. Returns standard identifiers, "
        "district, priority band, and authoritative AHP preparedness score."
    )
    input_schema = LookupCommunityInput

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()

    def run(self, query: str, district: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        df = self.service.ranked_df
        clean_q = query.strip().lower()

        mask = (
            df["village_name"].astype(str).str.lower().str.contains(clean_q)
            | df["pc11_id"].astype(str).str.lower().str.contains(clean_q)
            | df["shrid2"].astype(str).str.lower().str.contains(clean_q)
        )

        if district:
            mask = mask & (df["district_name"].str.lower() == district.strip().lower())

        matches_df = df[mask]
        total_matches = len(matches_df)
        slice_df = matches_df.head(limit)

        results = []
        for _, row in slice_df.iterrows():
            results.append(
                {
                    "shrid2": str(row["shrid2"]),
                    "pc11_id": str(row["pc11_id"]),
                    "village_name": str(row["village_name"]),
                    "district_name": str(row["district_name"]),
                    "subdistrict_name": str(row["subdistrict_name"]),
                    "unit_type": str(row["unit_type"]) if pd.notna(row.get("unit_type")) else None,
                    "priority_band": str(row.get("priority_band", "BAND_3_MODERATE")),
                    "overall_priority_rank": int(row.get("overall_priority_rank", 0)),
                    "band_rank": int(row.get("band_rank", 0)),
                    "preparedness_score": _safe_float(row.get("preparedness_score"), 2),
                    "preparedness_tier": str(row.get("preparedness_tier", "Unknown")).capitalize(),
                }
            )

        return {
            "query": query,
            "district_filter": district,
            "total_matches": total_matches,
            "returned_count": len(results),
            "matches": results,
        }


# ==============================================================================
# TOOL 5: analyze_infrastructure_and_exposure
# ==============================================================================


class AnalyzeInfrastructureAndExposureInput(BaseModel):
    """Input parameters for analyze_infrastructure_and_exposure tool."""

    identifier: str = Field(
        ...,
        min_length=1,
        description="Census 2011 PC11 ID (e.g., '608720'), SHRID (e.g., '11-29-569-05522-608720'), or village name.",
        examples=["608720", "Tallur", "Edmoge"],
    )

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Identifier cannot be empty or solely whitespace.")
        return clean


class AnalyzeInfrastructureAndExposureTool(BaseTool):
    """Tool to inspect granular physical indicators, healthcare distances, and shelter connectivity."""

    name = "analyze_infrastructure_and_exposure"
    description = (
        "Inspect granular physical infrastructure and exposure indicators for a specific community: geodesic "
        "distance to coast, apex hospital and PHC proximities, local cyclone shelters within 10 km, all-weather "
        "pucca road connectivity, drinking water score, and demographic baselines with explicit Census 2011 "
        "provenance and operational caveats."
    )
    input_schema = AnalyzeInfrastructureAndExposureInput

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()

    def run(self, identifier: str) -> Dict[str, Any]:
        row = resolve_community_row(self.service.ranked_df, identifier)

        # 1. Physical Exposure
        dist_coast = _safe_float(row.get("dist_coast_km"), 2)
        elevation = _safe_float(row.get("elevation_mean"), 2)
        tauktae = bool(row.get("tauktae_affected", 0) == 1)

        # 2. Healthcare Proximity
        dist_hosp = _safe_float(row.get("dist_nearest_hospital_km"), 2)
        dist_phc = _safe_float(row.get("dist_nearest_phc_km"), 2)
        dist_local_hc = _safe_float(row.get("dist_nearest_hc_km"), 2)
        n_hc = _safe_int(row.get("n_hc_10km"))
        n_phc_chc = _safe_int(row.get("n_phc_chc_10km"))

        # 3. Shelter and Transit
        dist_shelter = _safe_float(row.get("dist_nearest_shelter_km"), 2)
        n_shelters = _safe_int(row.get("n_shelters_10km"))
        shelter_cap = _safe_int(row.get("shelter_capacity_nearby"))
        has_pucca = bool(row.get("has_pucca_road", 0) == 1)
        has_major_road = bool(row.get("has_major_district_road", 0) == 1)
        bus_conn = bool(row.get("bus_connectivity", 0) == 1)
        road_score = _safe_float(row.get("road_connectivity_score"), 2)

        # 4. Amenities and Demographics
        pop = _safe_int(row.get("population"))
        hh = _safe_int(row.get("households"))
        density = _safe_float(row.get("pop_density_sqkm"), 1)
        sc_share = _safe_float(row.get("sc_share"), 4)
        st_share = _safe_float(row.get("st_share"), 4)
        vul_share = _safe_float(row.get("vulnerable_pop_share"), 4)
        water_score = _safe_float(row.get("drinking_water_score"), 2)

        return {
            "community_name": str(row["village_name"]),
            "pc11_id": str(row["pc11_id"]),
            "shrid2": str(row["shrid2"]),
            "district_name": str(row["district_name"]),
            "subdistrict_name": str(row["subdistrict_name"]),
            "physical_exposure": {
                "dist_coast_km": dist_coast,
                "elevation_mean_m": elevation,
                "tauktae_affected_2021": tauktae,
            },
            "healthcare_access": {
                "dist_nearest_hospital_km": dist_hosp,
                "dist_nearest_phc_km": dist_phc,
                "dist_nearest_local_facility_km": dist_local_hc,
                "hospitals_within_10km": n_hc,
                "phc_chc_within_10km": n_phc_chc,
                "facility_disambiguation": {
                    "primary_health_centre_phc_km": dist_phc,
                    "local_health_subcentre_km": dist_local_hc,
                    "apex_hospital_km": dist_hosp,
                    "definition_note": (
                        "Primary Health Centre (PHC) measures geodesic distance to the nearest dedicated 24/7 PHC. "
                        "Local health subcentre (dist_nearest_hc_km) measures distance to any local clinic or "
                        "sub-centre (0.0 km indicates facility located within village boundary). Apex hospital "
                        "measures distance to tertiary referral hospital."
                    ),
                },
            },
            "shelter_and_transit": {
                "dist_nearest_shelter_km": dist_shelter,
                "shelters_within_10km": n_shelters,
                "shelter_capacity_nearby": shelter_cap,
                "has_pucca_road": has_pucca,
                "has_major_district_road": has_major_road,
                "bus_connectivity": bus_conn,
                "road_connectivity_score": road_score,
            },
            "amenities_and_demographics": {
                "population": pop,
                "households": hh,
                "pop_density_sqkm": density,
                "sc_share": sc_share,
                "st_share": st_share,
                "vulnerable_pop_share": vul_share,
                "drinking_water_score": water_score,
            },
            "operational_limitations": {
                "census_vintage": 2011,
                "healthcare_registry_source": "MoHFW/NHM (2015-2019)",
                "coastline_source": "Natural Earth 10m (v5.1.1, 2022) in UTM Zone 43N",
                "operational_caveat": (
                    "Physical facility coordinates represent administrative registration, NOT confirmed active "
                    "doctor staffing, functioning ICU beds, or backup generator diesel status during cyclonic "
                    "power failure. Demographic and amenity counts reflect the 2011 Census historical baseline."
                ),
            },
        }
