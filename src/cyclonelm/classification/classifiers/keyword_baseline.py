"""Keyword baseline heuristic classifier for comparative benchmarking."""

from typing import Dict, List, Optional
from src.cyclonelm.classification.classifiers.base import BaseIntentClassifier, ClassificationPrediction
from src.cyclonelm.classification.config import (
    AMBIGUOUS_CATEGORY,
    OUT_OF_SCOPE_CATEGORY,
    REFUSAL_CATEGORY,
    SUPPORTED_INTENTS,
)


class KeywordBaselineClassifier(BaseIntentClassifier):
    """Rule-based keyword matching baseline."""

    KEYWORD_MAP: Dict[str, List[str]] = {
        "analyze_infrastructure_and_exposure": [
            "infrastructure",
            "hospital",
            "phc",
            "buffer",
            "exposure",
            "road",
            "drinking water",
            "connectivity",
            "shelters within",
        ],
        "compare_districts": [
            "compare",
            "contrast",
            "versus",
            "vs",
            "difference between",
            "between udupi",
            "between dakshina",
            "between uttara",
        ],
        "explain_preparedness_assessment": [
            "preparedness",
            "score",
            "ahp",
            "tier",
            "assessment",
            "breakdown",
            "criteria",
            "domain score",
            "why is",
        ],
        "explain_vulnerability_archetype": [
            "archetype",
            "cluster",
            "k-means",
            "k means",
            "urban corridor",
            "foothill",
            "rural interior",
            "coastal belt",
        ],
        "filter_and_rank_communities": [
            "filter",
            "rank",
            "band 1",
            "band 2",
            "band 3",
            "band 4",
            "critical",
            "top 5",
            "top 10",
            "most vulnerable",
        ],
        "get_intervention_recommendations": [
            "intervention",
            "recommend",
            "recommendations",
            "upgrade",
            "checklist",
            "action",
            "needed for",
            "capital",
        ],
        "lookup_community": [
            "look up",
            "lookup",
            "find",
            "search",
            "retrieve",
            "community data",
            "pc11",
            "village",
        ],
        REFUSAL_CATEGORY: [
            "live",
            "right now",
            "wind speed",
            "at this moment",
            "occupancy",
            "vacant bed",
            "flooded right now",
        ],
    }

    def fit(self, texts: List[str], labels: List[str]) -> "KeywordBaselineClassifier":
        # Static rules do not require fitting
        return self

    def predict(self, text: str) -> ClassificationPrediction:
        lower = text.lower()

        scores: Dict[str, int] = {}
        for intent, kw_list in self.KEYWORD_MAP.items():
            count = sum(1 for kw in kw_list if kw in lower)
            if count > 0:
                scores[intent] = count

        if not scores:
            return ClassificationPrediction(
                intent=AMBIGUOUS_CATEGORY,
                confidence=0.1,
                is_supported_tool=False,
                probabilities={AMBIGUOUS_CATEGORY: 1.0},
            )

        # Pick class with max matches
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_count = sorted_intents[0]

        total_matches = sum(scores.values())
        conf = min(0.95, round(best_count / max(1, total_matches), 3))

        probs = {k: round(v / total_matches, 3) for k, v in scores.items()}

        return ClassificationPrediction(
            intent=best_intent,
            confidence=conf,
            is_supported_tool=(best_intent in SUPPORTED_INTENTS),
            probabilities=probs,
            all_scores=[(k, float(v)) for k, v in sorted_intents],
        )
