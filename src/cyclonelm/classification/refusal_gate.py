"""Refusal and abstention gate for dynamic, unrecorded live operational telemetry."""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.cyclonelm.classification.config import (
    REFUSAL_ACCEPTANCE_THRESHOLD,
    REFUSAL_CATEGORY,
    REFUSAL_DOMAINS,
)
from src.cyclonelm.classification.embedder import MiniLMEmbedder


@dataclass
class RefusalDecision:
    """Decision output from the RefusalGate."""

    is_refused: bool
    domain: Optional[str] = None
    reason: Optional[str] = None
    confidence: float = 0.0
    matched_pattern: Optional[str] = None


class RefusalGate:
    """Detects requests for live dynamic telemetry that CYCLONE INTEL does not support.

    CYCLONE INTEL provides static, baseline preparedness assessments (Census 2011,
    multi-criteria AHP weights, K-Means vulnerability archetypes, and GIS buffers).
    It NEVER records or invents live operational conditions:
    - Real-time wind speeds or live cyclone forecasts (IMD authority).
    - Current shelter occupancy or live evacuee numbers (DDMA authority).
    - Hospital bed vacancies or active physician staffing (Health authority).
    - Live road waterlogging or passability (Public Works authority).
    - Active rescue personnel deployment or resource allocations (NDRF/SDRF authority).
    """

    # High-precision live operational indicators
    DYNAMIC_INDICATORS = [
        r"\b(right now|at this moment|currently|live|real[- ]time|today)\b.*?\b(wind speed|wind|forecast|storm track|landfall|surge|pressure|radar)\b",
        r"\b(wind speed|forecast|track|surge)\b.*?\b(right now|at this moment|currently|live|today)\b",
        r"\b(live|current|vacant|available)\b.*?\b(hospital bed|bed|beds|doctor|doctors|staff|personnel|ambulance|ambulances)\b",
        r"\b(bed vacancy|hospital beds|vacant beds)\b",
        r"\b(live|current|real[- ]time)\b.*?\b(shelter occupancy|occupancy|evacuee|evacuees|headcount)\b",
        r"\b(shelter occupancy)\b.*?\b(right now|today|currently)\b",
        r"\b(passable|flooded|waterlogged|submerged)\b.*?\b(right now|today|at this moment|currently)\b",
        r"\b(road passability|highway passable)\b",
        r"\b(ndrf|police|fire force|rescue team)\b.*?\b(on duty|deployed|active)\b.*?\b(right now|today)\b",
    ]

    def __init__(self, embedder: Optional[MiniLMEmbedder] = None) -> None:
        self.embedder = embedder or MiniLMEmbedder()
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.DYNAMIC_INDICATORS]
        self._init_semantic_prototypes()

    def _init_semantic_prototypes(self) -> None:
        """Precompute prototype embeddings for each refusal domain."""
        self.domain_prototypes: Dict[str, np.ndarray] = {}
        all_prototype_texts = []
        domain_labels = []

        for domain, phrases in REFUSAL_DOMAINS.items():
            for p in phrases:
                all_prototype_texts.append(p)
                domain_labels.append(domain)

        all_embs = self.embedder.embed_texts(all_prototype_texts, normalize=True)
        # Compute mean centroid per domain
        for domain in REFUSAL_DOMAINS:
            indices = [i for i, d in enumerate(domain_labels) if d == domain]
            centroid = np.mean(all_embs[indices], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            self.domain_prototypes[domain] = centroid

    def evaluate(self, text: str) -> RefusalDecision:
        """Evaluate if the user query requests dynamic operational telemetry.

        Returns:
            RefusalDecision indicating whether to refuse, the refusal domain,
            confidence score, and standard governance explanation.
        """
        clean_text = text.strip()
        if not clean_text:
            return RefusalDecision(is_refused=False, confidence=0.0)

        # 1. Lexical Pattern Check (High Precision)
        for pattern in self.compiled_patterns:
            match = pattern.search(clean_text)
            if match:
                matched_str = match.group(0)
                domain = self._infer_domain(matched_str)
                return RefusalDecision(
                    is_refused=True,
                    domain=domain,
                    confidence=0.98,
                    matched_pattern=matched_str,
                    reason=(
                        f"Refusal required: Request involves dynamic live telemetry ({domain}). "
                        "CYCLONE INTEL provides static baseline preparedness assessments derived "
                        "from Census 2011 and geospatial criteria. For live weather, shelter occupancy, "
                        "hospital beds, or road conditions, please consult IMD, KSDMA, or district DDMA channels."
                    ),
                )

        # 2. Semantic Embedding Similarity Check
        query_emb = self.embedder.embed_single(clean_text, normalize=True)
        max_sim = -1.0
        best_domain = None

        for domain, centroid in self.domain_prototypes.items():
            sim = float(np.dot(query_emb, centroid))
            if sim > max_sim:
                max_sim = sim
                best_domain = domain

        # Threshold check: requires high similarity to refuse on semantics alone
        # to prevent false-refusals of legitimate historical or baseline questions
        if max_sim >= 0.78:
            return RefusalDecision(
                is_refused=True,
                domain=best_domain,
                confidence=round(max_sim, 3),
                reason=(
                    f"Refusal required: Request semantically matches live operational telemetry ({best_domain}). "
                    "CYCLONE INTEL does not record or simulate dynamic weather, hospital vacancies, "
                    "or real-time shelter statuses. Consult official state emergency operation centers."
                ),
            )

        return RefusalDecision(is_refused=False, confidence=round(max(0.0, max_sim), 3))

    def _infer_domain(self, matched_text: str) -> str:
        lower = matched_text.lower()
        if any(w in lower for w in ["wind", "forecast", "track", "landfall", "surge", "radar", "pressure"]):
            return "live_weather"
        if any(w in lower for w in ["shelter", "occupancy", "evacuee", "headcount"]):
            return "live_shelters"
        if any(w in lower for w in ["bed", "hospital", "doctor", "ambulance"]):
            return "live_medical"
        if any(w in lower for w in ["passable", "road", "flooded", "highway"]):
            return "live_infrastructure"
        if any(w in lower for w in ["ndrf", "personnel", "staff", "police", "deployed"]):
            return "live_staffing"
        return "live_operational_telemetry"
