"""Decision support evaluation and intervention recommendation tools.

Provides:
- `explain_preparedness_assessment`: Detailed multi-criteria AHP preparedness evaluation,
  domain sub-scores, surrogate model concordance, and transparent priority band justification.
- `get_intervention_recommendations`: Evidence-grounded operational advisories and field
  inspection checklists generated via deterministic rules.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools.base import BaseTool
from src.assistant.tools.community import resolve_community_row
from src.integration.decision_matrix import (
    DECISION_SUPPORT_DISCLAIMER,
    build_unit_evaluation,
    evaluate_unit_advisories,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_PRECEDENCE_MANDATE = (
    "AHP PRECEDENCE MANDATE: The deterministic Analytical Hierarchy Process (AHP) composite preparedness index "
    "and its 4-band priority hierarchy maintain absolute operational precedence for emergency planning and resource allocation. "
    "Logistic Regression outputs serve strictly as an empirical tier surrogate approximating AHP classification boundaries; "
    "they are uncalibrated class likelihoods and do NOT represent physical cyclone strike, storm-surge inundation, "
    "or casualty forecasts."
)


# ==============================================================================
# TOOL 2: explain_preparedness_assessment
# ==============================================================================


class ExplainPreparednessAssessmentInput(BaseModel):
    """Input parameters for explain_preparedness_assessment tool."""

    identifier: str = Field(
        ...,
        min_length=1,
        description="Census 2011 PC11 ID (e.g., '608720'), SHRID (e.g., '11-29-569-05522-608720'), or unique village name.",
        examples=["608720", "Tallur", "Edmoge", "Bappalgundi"],
    )

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Identifier cannot be empty or solely whitespace.")
        return clean


class ExplainPreparednessAssessmentTool(BaseTool):
    """Tool to retrieve comprehensive preparedness assessment, domain scores, and model stability."""

    name = "explain_preparedness_assessment"
    description = (
        "Retrieve the authoritative multi-criteria preparedness evaluation for a specific community: deterministic "
        "AHP index (0-100), 5 domain sub-scores (medical, shelters, infrastructure, early warning, socioeconomic), "
        "priority band (Band 1 Critical to Band 4 Adequate), within-band and overall priority ranks, surrogate "
        "Logistic Regression concordance and boundary margin, and plain-English factor synthesis."
    )
    input_schema = ExplainPreparednessAssessmentInput

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()

    def run(self, identifier: str) -> Dict[str, Any]:
        row = resolve_community_row(self.service.ranked_df, identifier)
        evaluation = build_unit_evaluation(row)

        return {
            "community_name": evaluation.village_name,
            "pc11_id": evaluation.pc11_id,
            "shrid2": evaluation.shrid2,
            "district_name": evaluation.district_name,
            "subdistrict_name": evaluation.subdistrict_name,
            "unit_type": evaluation.unit_type,
            "deterministic_ahp": {
                "preparedness_score": evaluation.preparedness_score,
                "preparedness_tier": evaluation.preparedness_tier,
                "domain_scores": evaluation.domain_scores,
            },
            "surrogate_ml": {
                "predicted_preparedness_tier": evaluation.predicted_preparedness_tier,
                "class_probabilities": evaluation.class_probabilities,
                "max_class_probability": evaluation.max_class_probability,
                "probability_margin": evaluation.probability_margin,
                "model_agreement_status": evaluation.model_agreement_status,
                "is_boundary_case": evaluation.is_boundary_case,
                "prediction_provenance": evaluation.prediction_provenance,
            },
            "operational_prioritization": {
                "priority_band": evaluation.provisional_priority_band,
                "band_rank": evaluation.band_rank,
                "overall_priority_rank": evaluation.overall_priority_rank,
                "priority_explanation": evaluation.priority_explanation,
            },
            "vulnerability_archetype": {
                "cluster_assignment_status": evaluation.cluster_assignment_status,
                "archetype_id": evaluation.vulnerability_archetype_id,
                "archetype_label": evaluation.vulnerability_archetype_label,
            },
            "data_integrity_warning": evaluation.has_unverified_indicators,
            "scientific_precedence_mandate": SCIENTIFIC_PRECEDENCE_MANDATE,
            "disclaimer": DECISION_SUPPORT_DISCLAIMER,
        }


# ==============================================================================
# TOOL 7: get_intervention_recommendations
# ==============================================================================


class GetInterventionRecommendationsInput(BaseModel):
    """Input parameters for get_intervention_recommendations tool."""

    identifier: str = Field(
        ...,
        min_length=1,
        description="Census 2011 PC11 ID (e.g., '608720'), SHRID (e.g., '11-29-569-05522-608720'), or unique village name.",
        examples=["608720", "Tallur", "Edmoge", "Bappalgundi"],
    )

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Identifier cannot be empty or solely whitespace.")
        return clean


class GetInterventionRecommendationsTool(BaseTool):
    """Tool to retrieve evidence-grounded operational advisories and field inspection checklists."""

    name = "get_intervention_recommendations"
    description = (
        "Retrieve verified, evidence-grounded operational advisories and field inspection checklists for an "
        "administrative unit. Generated deterministically based on empirical thresholds (shelter score < 20, "
        "hospital distance > 20 km, coastal proximity < 5 km, unpaved road access) distinguishing archetype-specific "
        "rules for Dakshina Kannada/Udupi and the independent 3-sector fallback for Uttara Kannada."
    )
    input_schema = GetInterventionRecommendationsInput

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()

    def run(self, identifier: str) -> Dict[str, Any]:
        row = resolve_community_row(self.service.ranked_df, identifier)
        advisories = evaluate_unit_advisories(row)

        has_unverified = any(a.directive_id == "ADV_DATA_INCOMPLETE" for a in advisories)

        return {
            "community_name": str(row["village_name"]),
            "pc11_id": str(row["pc11_id"]),
            "shrid2": str(row["shrid2"]),
            "district_name": str(row["district_name"]),
            "subdistrict_name": str(row["subdistrict_name"]),
            "priority_band": str(row.get("priority_band", "BAND_3_MODERATE")),
            "cluster_assignment_status": str(row.get("cluster_assignment_status", "UNASSIGNED_OUT_OF_STUDY_AREA")),
            "advisories_count": len(advisories),
            "has_unverified_indicators": has_unverified,
            "advisories": [a.model_dump() for a in advisories],
            "disclaimer": DECISION_SUPPORT_DISCLAIMER,
        }
