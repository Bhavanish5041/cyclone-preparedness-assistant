"""CycloneLM evaluation package."""

from src.cyclonelm.evaluation.benchmark import CycloneLMBenchmark
from src.cyclonelm.evaluation.metrics import (
    compute_perplexity,
    evaluate_json_validity,
    evaluate_scientific_governance_rules,
    measure_inference_speed,
    verify_checkpoint_reload,
)

__all__ = [
    "CycloneLMBenchmark",
    "compute_perplexity",
    "verify_checkpoint_reload",
    "measure_inference_speed",
    "evaluate_json_validity",
    "evaluate_scientific_governance_rules",
]
