"""Structured argument extraction and Pydantic validation for assistant tools.

Extracts, disambiguates, and validates parameters for the 7 deterministic tools.
Independent of CycloneLM.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, ValidationError

from src.assistant.decision_support import DecisionSupportService
from src.assistant.tools.archetypes import ExplainVulnerabilityArchetypeInput
from src.assistant.tools.community import (
    AnalyzeInfrastructureAndExposureInput,
    DISTRICT_CANONICAL,
    LookupCommunityInput,
    VALID_DISTRICTS,
)
from src.assistant.tools.decision import (
    ExplainPreparednessAssessmentInput,
    GetInterventionRecommendationsInput,
)
from src.assistant.tools.regional import (
    CompareDistrictsInput,
    FilterAndRankCommunitiesInput,
    VALID_PRIORITY_BANDS,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Standardized output from argument extraction."""

    parameters: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = False
    is_ambiguous: bool = False
    ambiguous_candidates: List[Dict[str, Any]] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    validation_error: Optional[str] = None
    pydantic_instance: Optional[BaseModel] = None


class ArgumentExtractor:
    """Extracts, disambiguates, and validates parameters for the 7 deterministic tools."""

    TOOL_SCHEMA_MAP = {
        "lookup_community": LookupCommunityInput,
        "analyze_infrastructure_and_exposure": AnalyzeInfrastructureAndExposureInput,
        "explain_preparedness_assessment": ExplainPreparednessAssessmentInput,
        "get_intervention_recommendations": GetInterventionRecommendationsInput,
        "compare_districts": CompareDistrictsInput,
        "filter_and_rank_communities": FilterAndRankCommunitiesInput,
        "explain_vulnerability_archetype": ExplainVulnerabilityArchetypeInput,
    }

    PRIORITY_BAND_SYNONYMS = {
        "band 1": "BAND_1_CRITICAL",
        "band 1 critical": "BAND_1_CRITICAL",
        "critical": "BAND_1_CRITICAL",
        "band 2": "BAND_2_SPECIALIZED",
        "band 2 specialized": "BAND_2_SPECIALIZED",
        "specialized": "BAND_2_SPECIALIZED",
        "band 3": "BAND_3_MODERATE",
        "band 3 moderate": "BAND_3_MODERATE",
        "moderate": "BAND_3_MODERATE",
        "band 4": "BAND_4_ADEQUATE",
        "band 4 adequate": "BAND_4_ADEQUATE",
        "adequate": "BAND_4_ADEQUATE",
    }

    ARCHETYPE_SYNONYMS = {
        "cluster 0": 0,
        "archetype 0": 0,
        "urban corridor": 0,
        "cluster 1": 1,
        "archetype 1": 1,
        "foothill": 1,
        "cluster 2": 2,
        "archetype 2": 2,
        "rural interior": 2,
        "cluster 3": 3,
        "archetype 3": 3,
        "coastal belt": 3,
    }

    def __init__(self, service: Optional[DecisionSupportService] = None) -> None:
        self.service = service or DecisionSupportService()
        self._init_gazetteer()

    def _init_gazetteer(self) -> None:
        """Index village names and administrative codes for fast, accurate lookup."""
        df = self.service.ranked_df
        self.village_index: Dict[str, List[Dict[str, Any]]] = {}
        self.pc11_index: Dict[str, Dict[str, Any]] = {}

        def _add_alias(alias: str, entry: Dict[str, Any]):
            clean = alias.strip().lower()
            if not clean:
                return
            if clean not in self.village_index:
                self.village_index[clean] = []
            if not any(c["pc11_id"] == entry["pc11_id"] for c in self.village_index[clean]):
                self.village_index[clean].append(entry)

        for _, row in df.iterrows():
            raw_vname = str(row.get("village_name", "")).strip()
            pc11 = str(row.get("pc11_id", "")).strip()
            dist = str(row.get("district_name", "")).strip()
            subdist = str(row.get("subdistrict_name", "")).strip()

            entry = {
                "village_name": raw_vname,
                "pc11_id": pc11,
                "district_name": dist,
                "subdistrict_name": subdist,
            }

            if raw_vname:
                _add_alias(raw_vname, entry)
                # Strip administrative suffixes like (CT), (TMC), (TP)
                base_name = re.sub(r"\s*\([^)]*\)", "", raw_vname).strip()
                if base_name and base_name.lower() != raw_vname.lower():
                    _add_alias(base_name, entry)

                # Add Kannada transliteration variants (-a suffix)
                clean_base = base_name.lower()
                if clean_base.endswith("a") and len(clean_base) > 3:
                    _add_alias(clean_base[:-1], entry)
                elif not clean_base.endswith("a"):
                    _add_alias(clean_base + "a", entry)

            if pc11:
                self.pc11_index[pc11] = entry

    def extract_arguments(self, intent: str, text: str) -> ExtractionResult:
        """Extract arguments for the identified intent and validate via Pydantic."""
        if intent not in self.TOOL_SCHEMA_MAP:
            return ExtractionResult(is_valid=False, validation_error=f"Unsupported tool intent: {intent}")

        schema_cls = self.TOOL_SCHEMA_MAP[intent]

        if intent in ("analyze_infrastructure_and_exposure", "explain_preparedness_assessment", "get_intervention_recommendations"):
            return self._extract_single_identifier(schema_cls, text)
        elif intent == "lookup_community":
            return self._extract_lookup(schema_cls, text)
        elif intent == "compare_districts":
            return self._extract_compare_districts(schema_cls, text)
        elif intent == "filter_and_rank_communities":
            return self._extract_filter_rank(schema_cls, text)
        elif intent == "explain_vulnerability_archetype":
            return self._extract_archetype(schema_cls, text)

        return ExtractionResult(is_valid=False, validation_error="Unknown intent handler.")

    def _extract_district(self, text: str) -> Optional[str]:
        lower = text.lower()
        if "dakshina kannada" in lower or "dakshina" in lower:
            return "Dakshina Kannada"
        if "uttara kannada" in lower or "uttara" in lower:
            return "Uttara Kannada"
        if "udupi" in lower:
            return "Udupi"
        return None

    def _extract_all_districts(self, text: str) -> List[str]:
        lower = text.lower()
        found = []
        if "dakshina kannada" in lower or "dakshina" in lower:
            found.append("Dakshina Kannada")
        if "udupi" in lower:
            found.append("Udupi")
        if "uttara kannada" in lower or "uttara" in lower:
            found.append("Uttara Kannada")
        return found

    def _find_community_in_text(self, text: str, filter_district: Optional[str] = None) -> Tuple[Optional[str], bool, List[Dict[str, Any]]]:
        """Search text for matching village names or PC11 IDs with ambiguity detection."""
        # 1. Check for PC11 ID (numeric 6-digit)
        pc11_match = re.search(r"\b(\d{6})\b", text)
        if pc11_match:
            code = pc11_match.group(1)
            if code in self.pc11_index:
                return self.pc11_index[code]["village_name"], False, [self.pc11_index[code]]

        # 2. Check for village names from gazetteer (longest match first)
        lower_text = text.lower()
        # Sort candidate village names by character length descending to prefer specific multi-word towns
        sorted_villages = sorted(self.village_index.keys(), key=len, reverse=True)

        for vname in sorted_villages:
            # Word boundary check
            pattern = r"\b" + re.escape(vname) + r"\b"
            if re.search(pattern, lower_text):
                candidates = self.village_index[vname]
                if filter_district:
                    filtered = [c for c in candidates if c["district_name"].lower() == filter_district.lower()]
                    if filtered:
                        candidates = filtered

                if len(candidates) > 1:
                    return candidates[0]["village_name"], True, candidates
                return candidates[0]["village_name"], False, candidates

        return None, False, []

    def _extract_single_identifier(self, schema_cls: Any, text: str) -> ExtractionResult:
        district_hint = self._extract_district(text)
        comm_name, is_ambig, candidates = self._find_community_in_text(text, filter_district=district_hint)

        if not comm_name:
            return ExtractionResult(
                parameters={},
                is_valid=False,
                missing_required=["identifier"],
                validation_error="Missing required community identifier (village name or Census PC11 ID).",
            )

        if is_ambig and not district_hint:
            return ExtractionResult(
                parameters={"identifier": comm_name},
                is_valid=False,
                is_ambiguous=True,
                ambiguous_candidates=candidates,
                validation_error=(
                    f"Ambiguous community '{comm_name}' matches multiple revenue units across districts: "
                    + ", ".join(f"{c['village_name']} in {c['district_name']} ({c['subdistrict_name']})" for c in candidates)
                    + ". Please specify the district or Census PC11 ID."
                ),
            )

        raw_params = {"identifier": comm_name}
        try:
            instance = schema_cls(**raw_params)
            return ExtractionResult(
                parameters=instance.model_dump(),
                is_valid=True,
                pydantic_instance=instance,
            )
        except ValidationError as e:
            return ExtractionResult(
                parameters=raw_params,
                is_valid=False,
                validation_error=str(e),
            )

    def _extract_lookup(self, schema_cls: Any, text: str) -> ExtractionResult:
        district_hint = self._extract_district(text)
        comm_name, is_ambig, candidates = self._find_community_in_text(text, filter_district=district_hint)

        query_str = comm_name if comm_name else text.strip()
        raw_params: Dict[str, Any] = {"query": query_str}
        if district_hint:
            raw_params["district"] = district_hint

        try:
            instance = schema_cls(**raw_params)
            return ExtractionResult(
                parameters=instance.model_dump(),
                is_valid=True,
                is_ambiguous=is_ambig and (district_hint is None),
                ambiguous_candidates=candidates if is_ambig else [],
                pydantic_instance=instance,
            )
        except ValidationError as e:
            return ExtractionResult(
                parameters=raw_params,
                is_valid=False,
                validation_error=str(e),
            )

    def _extract_compare_districts(self, schema_cls: Any, text: str) -> ExtractionResult:
        districts = self._extract_all_districts(text)
        if len(districts) < 2:
            return ExtractionResult(
                parameters={"districts_found": districts},
                is_valid=False,
                missing_required=["district_a", "district_b"] if not districts else ["district_b"],
                validation_error="Comparison requires two coastal districts ('Dakshina Kannada', 'Udupi', or 'Uttara Kannada').",
            )

        raw_params = {"district_a": districts[0], "district_b": districts[1]}
        try:
            instance = schema_cls(**raw_params)
            return ExtractionResult(
                parameters=instance.model_dump(),
                is_valid=True,
                pydantic_instance=instance,
            )
        except ValidationError as e:
            return ExtractionResult(parameters=raw_params, is_valid=False, validation_error=str(e))

    def _extract_filter_rank(self, schema_cls: Any, text: str) -> ExtractionResult:
        lower = text.lower()
        raw_params: Dict[str, Any] = {}

        # 1. District
        dist = self._extract_district(text)
        if dist:
            raw_params["district"] = dist

        # 2. Priority Band
        for syn, band in self.PRIORITY_BAND_SYNONYMS.items():
            if syn in lower:
                raw_params["priority_band"] = band
                break

        # 3. Limit (e.g. "top 5", "top 10")
        limit_match = re.search(r"\btop\s+(\d+)\b", lower)
        if limit_match:
            raw_params["limit"] = min(20, max(1, int(limit_match.group(1))))

        # 4. Sort by coastal distance
        if any(w in lower for w in ["proximity to coast", "coastal distance", "closest to coast"]):
            raw_params["sort_by"] = "dist_coast_km"
            raw_params["ascending"] = True

        try:
            instance = schema_cls(**raw_params)
            return ExtractionResult(
                parameters=instance.model_dump(),
                is_valid=True,
                pydantic_instance=instance,
            )
        except ValidationError as e:
            return ExtractionResult(parameters=raw_params, is_valid=False, validation_error=str(e))

    def _extract_archetype(self, schema_cls: Any, text: str) -> ExtractionResult:
        lower = text.lower()
        raw_params: Dict[str, Any] = {}

        # 1. Archetype ID
        for syn, arch_id in self.ARCHETYPE_SYNONYMS.items():
            if syn in lower:
                raw_params["archetype_id"] = arch_id
                break

        # 2. Community identifier if checking specific village
        comm_name, _, _ = self._find_community_in_text(text)
        if comm_name:
            raw_params["community_identifier"] = comm_name

        try:
            instance = schema_cls(**raw_params)
            return ExtractionResult(
                parameters=instance.model_dump(),
                is_valid=True,
                pydantic_instance=instance,
            )
        except ValidationError as e:
            return ExtractionResult(parameters=raw_params, is_valid=False, validation_error=str(e))
