"""Vulnerability archetype explanation and spatial clustering tools.

Provides:
- `explain_vulnerability_archetype`: Descriptive profiles, statistical dimensions,
  and spatial boundaries for the 4 K-Means vulnerability archetypes, with explicit
  handling for unassigned Uttara Kannada units.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools.base import BaseTool, ToolValidationError
from src.assistant.tools.community import resolve_community_row
from src.models.clustering import ARCHETYPE_DISCLAIMER, ARCHETYPE_LABELS_K4

logger = logging.getLogger(__name__)


class ExplainVulnerabilityArchetypeInput(BaseModel):
    """Input parameters for explain_vulnerability_archetype tool."""

    archetype_id: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description="Vulnerability archetype cluster ID (0: Urban Corridor, 1: Remote Foothill, 2: Rural Interior, 3: Coastal Belt).",
        examples=[0, 1, 2, 3],
    )
    community_identifier: Optional[str] = Field(
        None,
        description="Optional Census PC11 ID, SHRID, or unique village name to inspect its archetype assignment or fallback status.",
        examples=["Edmoge", "Tallur", "Bappalgundi"],
    )

    @field_validator("community_identifier")
    @classmethod
    def validate_community_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        return clean if clean else None


class ExplainVulnerabilityArchetypeTool(BaseTool):
    """Tool to inspect discovered vulnerability archetypes and community assignment status."""

    name = "explain_vulnerability_archetype"
    description = (
        "Inspect descriptive profiles, dimension means, and operational interventions for the 4 K-Means "
        "vulnerability archetypes in coastal Karnataka (Cluster 0: Urban/Peri-Urban Corridor, Cluster 1: Remote Foothill "
        "Communities, Cluster 2: Vulnerable Rural Interior, Cluster 3: High-Exposure Coastal Belt). If a community "
        "identifier is provided, returns its cluster assignment or explains why Uttara Kannada units are unassigned."
    )
    input_schema = ExplainVulnerabilityArchetypeInput

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()

    def run(
        self,
        archetype_id: Optional[int] = None,
        community_identifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        catalog = self.service.get_archetype_catalog()

        # Scenario A: Specific community identifier provided
        if community_identifier:
            row = resolve_community_row(self.service.ranked_df, community_identifier)
            assignment_status = str(row.get("cluster_assignment_status", "UNASSIGNED_OUT_OF_STUDY_AREA")).strip()

            # Path 1: Unassigned Uttara Kannada unit
            if assignment_status != "ASSIGNED":
                dist_coast = float(row.get("dist_coast_km", 99.0))
                score_shelter = float(row.get("score_shelters", 50.0))
                score_med = float(row.get("score_medical", 50.0))
                prep_score = float(row.get("preparedness_score", 50.0))

                return {
                    "community_name": str(row["village_name"]),
                    "pc11_id": str(row["pc11_id"]),
                    "shrid2": str(row["shrid2"]),
                    "district_name": str(row["district_name"]),
                    "cluster_assignment_status": "UNASSIGNED_OUT_OF_STUDY_AREA",
                    "archetype_id": None,
                    "archetype_label": None,
                    "explanation": (
                        "This administrative unit is located in Uttara Kannada, which was outside the 622-unit "
                        "K-Means clustering study area (restricted to Dakshina Kannada and Udupi). Unassigned units "
                        "are evaluated strictly via deterministic AHP domain scores and the audited three-sector fallback "
                        "engine rather than unsupervised cluster membership."
                    ),
                    "fallback_trigger_evaluation": {
                        "coastal_proximity_trigger (<5km)": bool(dist_coast < 5.0),
                        "shelter_deficit_trigger (score <20)": bool(score_shelter < 20.0),
                        "medical_remoteness_trigger (score <35)": bool(score_med < 35.0),
                        "baseline_low_tier_trigger (score <40)": bool(prep_score < 40.0),
                    },
                    "spatial_scope_caveat": (
                        "K-Means archetype clustering is descriptive only and does not supersede AHP decision rankings."
                    ),
                    "disclaimer": ARCHETYPE_DISCLAIMER,
                }

            # Path 2: Assigned Dakshina Kannada or Udupi unit
            c_id = int(row["vulnerability_archetype_id"])
            c_meta = ARCHETYPE_LABELS_K4.get(c_id, {})
            summary_match = next((s for s in catalog.archetypes if s.archetype_id == c_id), None)

            return {
                "community_name": str(row["village_name"]),
                "pc11_id": str(row["pc11_id"]),
                "shrid2": str(row["shrid2"]),
                "district_name": str(row["district_name"]),
                "cluster_assignment_status": "ASSIGNED",
                "archetype_id": c_id,
                "short_label": c_meta.get("short_label", f"Cluster {c_id}"),
                "detailed_description": c_meta.get("detailed_description", "No description available."),
                "primary_intervention_need": c_meta.get("primary_intervention_need", "Standard response."),
                "community_dimension_scores": {
                    "healthcare_deficit": round(float(row.get("dim_healthcare_deficit", 0.0)), 4),
                    "demographic_vulnerability": round(float(row.get("dim_demographic_vulnerability", 0.0)), 4),
                    "infrastructure_deficit": round(float(row.get("dim_infrastructure_deficit", 0.0)), 4),
                    "coastal_exposure": round(float(row.get("dim_coastal_exposure", 0.0)), 4),
                },
                "cluster_profile_means": summary_match.dimension_means if summary_match else {},
                "spatial_coverage": "Dakshina Kannada and Udupi study area (N=622).",
                "disclaimer": ARCHETYPE_DISCLAIMER,
            }

        # Scenario B: Specific archetype_id requested
        if archetype_id is not None:
            c_meta = ARCHETYPE_LABELS_K4.get(archetype_id)
            if not c_meta:
                raise ToolValidationError(f"Invalid archetype_id {archetype_id}. Allowed values: 0, 1, 2, 3.")

            summary_match = next((s for s in catalog.archetypes if s.archetype_id == archetype_id), None)
            if not summary_match:
                raise ToolValidationError(f"Summary profile for archetype_id {archetype_id} not found in catalog.")

            return {
                "archetype_id": archetype_id,
                "short_label": summary_match.short_label,
                "detailed_description": summary_match.detailed_description,
                "primary_intervention_need": summary_match.primary_intervention_need,
                "unit_count": summary_match.unit_count,
                "percentage_of_total": summary_match.percentage_of_total,
                "distribution": {
                    "dakshina_kannada": summary_match.count_dakshina_kannada,
                    "udupi": summary_match.count_udupi,
                    "uttara_kannada": 0,
                },
                "dimension_means": summary_match.dimension_means,
                "raw_indicator_means": summary_match.raw_indicator_means,
                "spatial_boundary_note": (
                    "Covers 622 revenue units in Dakshina Kannada and Udupi. "
                    "Uttara Kannada (1,257 units) is unassigned and evaluated via deterministic sector triggers."
                ),
                "disclaimer": ARCHETYPE_DISCLAIMER,
            }

        # Scenario C: Overview of all 4 archetypes
        summaries = []
        for s in catalog.archetypes:
            summaries.append(
                {
                    "archetype_id": s.archetype_id,
                    "short_label": s.short_label,
                    "unit_count": s.unit_count,
                    "percentage_of_total": s.percentage_of_total,
                    "primary_intervention_need": s.primary_intervention_need,
                    "dimension_means": s.dimension_means,
                }
            )

        return {
            "total_archetypes": len(summaries),
            "study_area_units": 622,
            "clustering_methodology": (
                "K-Means (k=4) on standardized z-score transformed dimension scores "
                "(Healthcare Deficit, Demographic Vulnerability, Infrastructure Deficit, Coastal Proximity)."
            ),
            "archetypes": summaries,
            "unassigned_region_note": (
                "Uttara Kannada (1,257 revenue units) was not included in the K-Means clustering study area "
                "and is evaluated strictly via deterministic AHP domain scores and 3-sector fallback rules."
            ),
            "disclaimer": ARCHETYPE_DISCLAIMER,
        }
