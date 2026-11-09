"""Regional comparison and deterministic community ranking tools.

Provides:
- `compare_districts`: Comparative cross-tabulation and metric distributions between coastal districts.
- `filter_and_rank_communities`: Bounded, deterministically sorted community rosters filtered
  by priority band or district (strictly capped at 20 results to avoid context bloat).
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools.base import BaseTool, ToolValidationError
from src.assistant.tools.community import DISTRICT_CANONICAL, VALID_DISTRICTS, _safe_float

logger = logging.getLogger(__name__)

VALID_PRIORITY_BANDS = {
    "BAND_1_CRITICAL",
    "BAND_2_SPECIALIZED",
    "BAND_3_MODERATE",
    "BAND_4_ADEQUATE",
}

VALID_SORT_COLUMNS = {
    "overall_priority_rank",
    "band_rank",
    "preparedness_score",
    "dist_coast_km",
    "score_shelters",
    "score_medical",
    "population",
}


# ==============================================================================
# TOOL 3: compare_districts
# ==============================================================================


class CompareDistrictsInput(BaseModel):
    """Input parameters for compare_districts tool."""

    district_a: str = Field(
        ...,
        description="First coastal district for comparison ('Dakshina Kannada', 'Udupi', 'Uttara Kannada').",
        examples=["Dakshina Kannada", "Udupi"],
    )
    district_b: str = Field(
        ...,
        description="Second coastal district for comparison ('Dakshina Kannada', 'Udupi', 'Uttara Kannada').",
        examples=["Uttara Kannada", "Udupi"],
    )

    @field_validator("district_a", "district_b")
    @classmethod
    def validate_district(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in VALID_DISTRICTS:
            raise ValueError(
                f"Invalid district '{v}'. Allowed values: 'Dakshina Kannada', 'Udupi', 'Uttara Kannada'."
            )
        return DISTRICT_CANONICAL[clean]


class CompareDistrictsTool(BaseTool):
    """Tool to generate comparative statistics and priority distributions between two coastal districts."""

    name = "compare_districts"
    description = (
        "Generate authoritative comparative cross-tabulations and metric distributions between two coastal "
        "districts (Dakshina Kannada, Udupi, Uttara Kannada): total units, priority band breakdowns, Band 1 "
        "Critical proportions, average AHP preparedness scores, coastal distances, and clustering coverage methodology."
    )
    input_schema = CompareDistrictsInput

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()

    def run(self, district_a: str, district_b: str) -> Dict[str, Any]:
        if district_a == district_b:
            raise ToolValidationError(
                f"district_a ('{district_a}') and district_b ('{district_b}') must be different districts."
            )

        df = self.service.ranked_df
        df_a = df[df["district_name"].str.lower() == district_a.lower()]
        df_b = df[df["district_name"].str.lower() == district_b.lower()]

        def compute_district_metrics(sub_df: pd.DataFrame, dist_name: str) -> Dict[str, Any]:
            n_total = len(sub_df)
            band_counts = sub_df["priority_band"].value_counts().to_dict()
            b1 = int(band_counts.get("BAND_1_CRITICAL", 0))
            b2 = int(band_counts.get("BAND_2_SPECIALIZED", 0))
            b3 = int(band_counts.get("BAND_3_MODERATE", 0))
            b4 = int(band_counts.get("BAND_4_ADEQUATE", 0))

            mean_score = _safe_float(sub_df["preparedness_score"].mean(), 2)
            mean_coast = _safe_float(sub_df["dist_coast_km"].dropna().mean(), 2)
            mean_hosp = _safe_float(sub_df["dist_nearest_hospital_km"].dropna().mean(), 2)

            agreement_count = int((sub_df["model_agreement_status"] == "FULL_AGREEMENT").sum())
            agreement_rate = round((agreement_count / n_total) * 100, 1) if n_total > 0 else 0.0

            clustering_status = (
                "Assigned to K-Means Vulnerability Archetypes (k=4 Study Area)"
                if dist_name in ("Dakshina Kannada", "Udupi")
                else "Unassigned (Evaluated strictly via Deterministic 3-Sector Fallback Engine)"
            )

            return {
                "total_units": n_total,
                "priority_bands": {
                    "BAND_1_CRITICAL": b1,
                    "BAND_2_SPECIALIZED": b2,
                    "BAND_3_MODERATE": b3,
                    "BAND_4_ADEQUATE": b4,
                },
                "band_1_critical_percentage": round((b1 / n_total) * 100, 1) if n_total > 0 else 0.0,
                "band_2_specialized_percentage": round((b2 / n_total) * 100, 1) if n_total > 0 else 0.0,
                "mean_preparedness_score": mean_score,
                "mean_dist_coast_km": mean_coast,
                "mean_dist_nearest_hospital_km": mean_hosp,
                "model_agreement_rate_pct": agreement_rate,
                "clustering_coverage": clustering_status,
            }

        metrics_a = compute_district_metrics(df_a, district_a)
        metrics_b = compute_district_metrics(df_b, district_b)

        # High-level analytical takeaways
        disparities = []
        if metrics_a["band_1_critical_percentage"] > metrics_b["band_1_critical_percentage"]:
            diff = round(metrics_a["band_1_critical_percentage"] - metrics_b["band_1_critical_percentage"], 1)
            disparities.append(
                f"{district_a} has a higher concentration of Critical Band 1 units ({metrics_a['band_1_critical_percentage']}% vs {metrics_b['band_1_critical_percentage']}%, +{diff}%)."
            )
        elif metrics_b["band_1_critical_percentage"] > metrics_a["band_1_critical_percentage"]:
            diff = round(metrics_b["band_1_critical_percentage"] - metrics_a["band_1_critical_percentage"], 1)
            disparities.append(
                f"{district_b} has a higher concentration of Critical Band 1 units ({metrics_b['band_1_critical_percentage']}% vs {metrics_a['band_1_critical_percentage']}%, +{diff}%)."
            )

        if metrics_a["mean_dist_coast_km"] and metrics_b["mean_dist_coast_km"]:
            if metrics_a["mean_dist_coast_km"] < metrics_b["mean_dist_coast_km"]:
                disparities.append(
                    f"{district_a} units have higher average coastal proximity (mean {metrics_a['mean_dist_coast_km']} km vs {metrics_b['mean_dist_coast_km']} km)."
                )
            else:
                disparities.append(
                    f"{district_b} units have higher average coastal proximity (mean {metrics_b['mean_dist_coast_km']} km vs {metrics_a['mean_dist_coast_km']} km)."
                )

        return {
            "districts_compared": [district_a, district_b],
            "comparison": {
                district_a: metrics_a,
                district_b: metrics_b,
            },
            "key_observations": disparities,
            "methodological_note": (
                "Dakshina Kannada (369 units) and Udupi (253 units) form the primary K-Means clustering study area. "
                "Uttara Kannada (1,257 units) is evaluated via authoritative AHP domain scores and the audited "
                "three-sector fallback engine (coastal proximity < 5 km, shelter score < 20, medical score < 35)."
            ),
        }


# ==============================================================================
# TOOL 4: filter_and_rank_communities
# ==============================================================================


class FilterAndRankCommunitiesInput(BaseModel):
    """Input parameters for filter_and_rank_communities tool."""

    priority_band: Optional[str] = Field(
        None,
        description="Filter by operational priority band: 'BAND_1_CRITICAL', 'BAND_2_SPECIALIZED', 'BAND_3_MODERATE', 'BAND_4_ADEQUATE'.",
        examples=["BAND_1_CRITICAL", "BAND_2_SPECIALIZED"],
    )
    district: Optional[str] = Field(
        None,
        description="Filter by district ('Dakshina Kannada', 'Udupi', 'Uttara Kannada').",
        examples=["Uttara Kannada", "Dakshina Kannada"],
    )
    sort_by: str = Field(
        "overall_priority_rank",
        description="Deterministic ranking attribute ('overall_priority_rank', 'band_rank', 'preparedness_score', 'dist_coast_km', 'score_shelters', 'score_medical', 'population').",
    )
    ascending: bool = Field(
        True,
        description="Sort direction (True for ascending, False for descending).",
    )
    limit: int = Field(
        10,
        ge=1,
        le=20,
        description="Maximum records to return. Strictly capped at 20 to prevent context bloat.",
    )
    offset: int = Field(
        0,
        ge=0,
        description="Pagination offset index (default 0).",
    )

    @field_validator("priority_band")
    @classmethod
    def validate_priority_band(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip().upper()
        if clean not in VALID_PRIORITY_BANDS:
            raise ValueError(
                f"Invalid priority_band '{v}'. Allowed values: {sorted(VALID_PRIORITY_BANDS)}."
            )
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

    @field_validator("sort_by")
    @classmethod
    def validate_sort_by(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in VALID_SORT_COLUMNS:
            raise ValueError(
                f"Invalid sort_by '{v}'. Allowed values: {sorted(VALID_SORT_COLUMNS)}."
            )
        return clean


class FilterAndRankCommunitiesTool(BaseTool):
    """Tool to filter and rank communities deterministically without dumping large datasets."""

    name = "filter_and_rank_communities"
    description = (
        "Filter and deterministically rank administrative units by priority band (e.g. BAND_1_CRITICAL) and/or "
        "district, sorted by priority rank, AHP preparedness score, coastal proximity, or shelter score. "
        "Enforces a strict maximum limit of 20 units per query to ensure compact, high-relevance payloads."
    )
    input_schema = FilterAndRankCommunitiesInput

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()

    def run(
        self,
        priority_band: Optional[str] = None,
        district: Optional[str] = None,
        sort_by: str = "overall_priority_rank",
        ascending: bool = True,
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        df = self.service.ranked_df
        filtered = df

        if priority_band:
            filtered = filtered[filtered["priority_band"].str.upper() == priority_band]

        if district:
            filtered = filtered[filtered["district_name"].str.lower() == district.lower()]

        total_matching = len(filtered)

        # Null-safe deterministic sorting
        sorted_df = filtered.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position="last",
        )

        slice_df = sorted_df.iloc[offset : offset + limit]

        units = []
        for _, row in slice_df.iterrows():
            units.append(
                {
                    "shrid2": str(row["shrid2"]),
                    "pc11_id": str(row["pc11_id"]),
                    "village_name": str(row["village_name"]),
                    "district_name": str(row["district_name"]),
                    "subdistrict_name": str(row["subdistrict_name"]),
                    "priority_band": str(row.get("priority_band", "BAND_3_MODERATE")),
                    "overall_priority_rank": int(row.get("overall_priority_rank", 0)),
                    "band_rank": int(row.get("band_rank", 0)),
                    "preparedness_score": _safe_float(row.get("preparedness_score"), 2),
                    "sort_metric_value": _safe_float(row.get(sort_by), 2),
                }
            )

        return {
            "total_matching": total_matching,
            "returned_count": len(units),
            "offset": offset,
            "limit": limit,
            "applied_filters": {
                "priority_band": priority_band,
                "district": district,
                "sort_by": sort_by,
                "ascending": ascending,
            },
            "units": units,
        }
