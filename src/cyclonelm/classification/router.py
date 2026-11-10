"""High-level Intent Router orchestrating refusal, classification, and validation."""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional

from src.assistant.decision_support import DecisionSupportService
from src.cyclonelm.classification.argument_extractor import ArgumentExtractor, ExtractionResult
from src.cyclonelm.classification.classifiers.base import BaseIntentClassifier, ClassificationPrediction
from src.cyclonelm.classification.classifiers.logistic_classifier import LogisticIntentClassifier
from src.cyclonelm.classification.config import (
    ABSTENTION_THRESHOLD,
    AMBIGUOUS_CATEGORY,
    DEFAULT_ACCEPTANCE_THRESHOLD,
    OUT_OF_SCOPE_CATEGORY,
    REFUSAL_CATEGORY,
    SUPPORTED_INTENTS,
)
from src.cyclonelm.classification.refusal_gate import RefusalDecision, RefusalGate

logger = logging.getLogger(__name__)


class RoutingStatus(str, Enum):
    """Categorical routing status."""

    ROUTED = "ROUTED"  # Successfully routed to deterministic tool with valid parameters
    REFUSED = "REFUSED"  # Refused: dynamic operational or unrecorded live request
    ABSTAINED = "ABSTAINED"  # Abstained: model is uncertain or below confidence threshold
    AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"  # Query is underspecified or vague
    OUT_OF_SCOPE = "OUT_OF_SCOPE"  # Query is unrelated to cyclone disaster management
    AMBIGUOUS_ARGUMENT = "AMBIGUOUS_ARGUMENT"  # Community name matches multiple units across districts
    MISSING_ARGUMENT = "MISSING_ARGUMENT"  # Intent recognized, but required parameter is absent
    INVALID_ARGUMENT = "INVALID_ARGUMENT"  # Parameter extraction failed Pydantic validation


@dataclass
class RoutingResult:
    """Standardized decision payload produced by the IntentRouter."""

    status: RoutingStatus
    query: str
    tool_name: Optional[str] = None
    validated_arguments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    refusal_reason: Optional[str] = None
    clarification_message: Optional[str] = None
    ambiguous_candidates: List[Dict[str, Any]] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)
    can_execute: bool = False


class IntentRouter:
    """Orchestrates refusal gating, intent classification, and parameter extraction.

    Integration Boundary Invariants:
    - Identifies intended tool and validates arguments via Pydantic.
    - NEVER executes tools automatically.
    - NEVER modifies AHP calculations, tiers, or priority bands.
    - NEVER invents evidence or live telemetry.
    """

    def __init__(
        self,
        classifier: Optional[BaseIntentClassifier] = None,
        refusal_gate: Optional[RefusalGate] = None,
        argument_extractor: Optional[ArgumentExtractor] = None,
        service: Optional[DecisionSupportService] = None,
        acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        abstention_threshold: float = ABSTENTION_THRESHOLD,
    ) -> None:
        self.service = service or DecisionSupportService()
        self.refusal_gate = refusal_gate or RefusalGate()
        self.argument_extractor = argument_extractor or ArgumentExtractor(service=self.service)
        self.classifier = classifier or LogisticIntentClassifier()
        self.acceptance_threshold = acceptance_threshold
        self.abstention_threshold = abstention_threshold

    def route(self, query: str) -> RoutingResult:
        """Route a natural language query through the 4-stage pipeline."""
        clean_query = query.strip()
        if not clean_query:
            return RoutingResult(
                status=RoutingStatus.AMBIGUOUS_QUERY,
                query=query,
                confidence=0.0,
                clarification_message="Empty query. Please provide a question or instruction.",
            )

        # ---------------------------------------------------------------------
        # STAGE 1: Dedicated Refusal Gate (Live Telemetry & Safety Check)
        # ---------------------------------------------------------------------
        refusal_res = self.refusal_gate.evaluate(clean_query)
        if refusal_res.is_refused:
            return RoutingResult(
                status=RoutingStatus.REFUSED,
                query=query,
                confidence=refusal_res.confidence,
                refusal_reason=refusal_res.reason,
                clarification_message=refusal_res.reason,
                can_execute=False,
            )

        # ---------------------------------------------------------------------
        # STAGE 2: Intent Classification
        # ---------------------------------------------------------------------
        pred: ClassificationPrediction = self.classifier.predict(clean_query)

        # Handle explicit refusal class from classifier
        if pred.intent == REFUSAL_CATEGORY:
            ref_msg = (
                "Refusal required: Request involves dynamic live telemetry. CYCLONE INTEL "
                "provides static baseline preparedness assessments. For live operational updates, "
                "consult IMD or district DDMA emergency bulletins."
            )
            return RoutingResult(
                status=RoutingStatus.REFUSED,
                query=query,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                refusal_reason=ref_msg,
                clarification_message=ref_msg,
                can_execute=False,
            )

        # Handle out-of-scope queries
        if pred.intent == OUT_OF_SCOPE_CATEGORY:
            return RoutingResult(
                status=RoutingStatus.OUT_OF_SCOPE,
                query=query,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                clarification_message=(
                    "Query is outside the scope of coastal Karnataka cyclone preparedness intelligence. "
                    "Supported tasks include community lookups, AHP evaluations, intervention recommendations, "
                    "district comparisons, priority band filtering, and vulnerability archetypes."
                ),
                can_execute=False,
            )

        # Handle ambiguous queries
        if pred.intent == AMBIGUOUS_CATEGORY:
            return RoutingResult(
                status=RoutingStatus.AMBIGUOUS_QUERY,
                query=query,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                clarification_message=(
                    "The query is too brief or ambiguous to determine the requested analysis. "
                    "Please specify the community name and the desired action (e.g., 'Explain preparedness "
                    "for Ullal', 'Compare Udupi and Dakshina Kannada', or 'Interventions for Malpe')."
                ),
                can_execute=False,
            )

        # Handle low-confidence abstention
        if pred.confidence < self.abstention_threshold:
            return RoutingResult(
                status=RoutingStatus.ABSTAINED,
                query=query,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                clarification_message=(
                    f"Low classification confidence ({pred.confidence:.2f} < {self.abstention_threshold:.2f}). "
                    "CycloneLM abstains from automatic routing. Candidate intents: "
                    + ", ".join(f"{c}: {s:.2f}" for c, s in pred.all_scores[:3])
                ),
                can_execute=False,
            )

        # ---------------------------------------------------------------------
        # STAGE 3: Argument Extraction and Disambiguation
        # ---------------------------------------------------------------------
        ext_res: ExtractionResult = self.argument_extractor.extract_arguments(pred.intent, clean_query)

        # Check for ambiguous community matches across districts
        if ext_res.is_ambiguous:
            return RoutingResult(
                status=RoutingStatus.AMBIGUOUS_ARGUMENT,
                query=query,
                tool_name=pred.intent,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                ambiguous_candidates=ext_res.ambiguous_candidates,
                clarification_message=ext_res.validation_error,
                can_execute=False,
            )

        # Check for missing required parameters
        if ext_res.missing_required:
            return RoutingResult(
                status=RoutingStatus.MISSING_ARGUMENT,
                query=query,
                tool_name=pred.intent,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                missing_parameters=ext_res.missing_required,
                clarification_message=ext_res.validation_error,
                can_execute=False,
            )

        # Check Pydantic validation failure
        if not ext_res.is_valid:
            return RoutingResult(
                status=RoutingStatus.INVALID_ARGUMENT,
                query=query,
                tool_name=pred.intent,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                clarification_message=ext_res.validation_error,
                can_execute=False,
            )

        # ---------------------------------------------------------------------
        # STAGE 4: Successfully Routed (Ready for Authoritative Tool Execution)
        # ---------------------------------------------------------------------
        return RoutingResult(
            status=RoutingStatus.ROUTED,
            query=query,
            tool_name=pred.intent,
            validated_arguments=ext_res.parameters,
            confidence=pred.confidence,
            probabilities=pred.probabilities,
            can_execute=True,
        )
