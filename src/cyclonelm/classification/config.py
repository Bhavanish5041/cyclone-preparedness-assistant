"""Configuration constants and thresholds for CycloneLM Intent Classification and Refusal."""

from typing import Dict, List, Set, Tuple

# Exact 7 supported tool names registered in src.assistant.tools.registry
SUPPORTED_INTENTS: Tuple[str, ...] = (
    "analyze_infrastructure_and_exposure",
    "compare_districts",
    "explain_preparedness_assessment",
    "explain_vulnerability_archetype",
    "filter_and_rank_communities",
    "get_intervention_recommendations",
    "lookup_community",
)

# Special non-execution categories
REFUSAL_CATEGORY = "refusal_live_telemetry"
OUT_OF_SCOPE_CATEGORY = "out_of_scope"
AMBIGUOUS_CATEGORY = "ambiguous"

ALL_CLASSIFIER_CLASSES: Tuple[str, ...] = SUPPORTED_INTENTS + (
    REFUSAL_CATEGORY,
    OUT_OF_SCOPE_CATEGORY,
    AMBIGUOUS_CATEGORY,
)

# Default confidence and acceptance thresholds
# Note: Raw embedding similarities must not be treated as calibrated probabilities.
DEFAULT_ACCEPTANCE_THRESHOLD = 0.50
REFUSAL_ACCEPTANCE_THRESHOLD = 0.45
ABSTENTION_THRESHOLD = 0.35

# Backbone embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Live telemetry refusal triggers: unsupported dynamic queries
REFUSAL_DOMAINS: Dict[str, List[str]] = {
    "live_weather": [
        "live wind speed",
        "current wind",
        "wind speed right now",
        "real-time wind",
        "current storm track",
        "live cyclone forecast",
        "live satellite",
        "radar telemetry",
        "current atmospheric pressure",
        "live rainfall right now",
        "storm surge right now",
        "making landfall today",
        "landfall right now",
    ],
    "live_shelters": [
        "live shelter occupancy",
        "current occupancy",
        "shelter occupancy right now",
        "how many people in shelter right now",
        "live evacuee headcount",
        "shelter capacity remaining right now",
    ],
    "live_medical": [
        "hospital bed availability right now",
        "vacant hospital beds",
        "live bed vacancy",
        "beds currently vacant",
        "available beds right now",
        "active doctors right now",
        "ambulances dispatched right now",
    ],
    "live_infrastructure": [
        "passable right now",
        "flooded right now",
        "road passable right now",
        "live road passability",
        "power grid operational right now",
        "water level right now",
    ],
    "live_staffing": [
        "active rescue personnel right now",
        "ndrf personnel on duty right now",
        "police personnel deployed right now",
        "current staffing right now",
    ],
}
