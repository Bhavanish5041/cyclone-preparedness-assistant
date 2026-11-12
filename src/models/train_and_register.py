"""Unified Training, Evaluation, and Registry Staging CLI.
======================================================

Orchestrates reproducible model generation, artifact serialization,
and local registry verification for:
  1. village-preparedness-classifier (Logistic Regression)
  2. coastal-vulnerability-kmeans (K-Means Clustering)
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path_entry in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from src.models.clustering import train_kmeans_model
from src.models.quality_checks import QualityGateRunner
from src.models.registry.local_registry import LocalModelRegistry

COASTAL_PATH = PROJECT_ROOT / "data" / "processed" / "coastal_prioritization_baseline.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_and_register")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, and verify CYCLONE INTEL models.")
    parser.add_argument("--skip-training", action="store_true", help="Skip retraining existing models.")
    parser.add_argument("--candidate-version", type=str, default=None, help="Version tag for candidate model registration (e.g. v1.1.0-candidate).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "candidates", help="Directory for candidate artifacts (preserves production models/).")
    args = parser.parse_args()

    # Verify existing production models before any operation
    registry = LocalModelRegistry()
    initial_models = {m["model_identifier"]: m["active_version"] for m in registry.list_models()}
    logger.info(f"Initial active production models: {initial_models}")

    candidate_dir = args.output_dir
    candidate_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        logger.info(f"Training canonical K-Means model (k=4, random_state=42) to staging directory: {candidate_dir}...")
        km_model, km_scaler, km_meta, km_metrics, km_labels = train_kmeans_model(
            data_or_path=COASTAL_PATH,
            n_clusters=4,
            random_state=42,
            n_init=20,
            output_dir=candidate_dir,
            save_artifacts=True,
        )

        if args.candidate_version:
            logger.info(f"Registering candidate version '{args.candidate_version}' without altering active production aliases...")
            registry.register_candidate(
                model_id="coastal-vulnerability-kmeans",
                version=args.candidate_version,
                artifact_files=[
                    {"path": str(candidate_dir / "kmeans_k4_model.joblib")},
                    {"path": str(candidate_dir / "kmeans_scaler.joblib")},
                    {"path": str(candidate_dir / "kmeans_metadata.json")},
                ],
                input_schema={"dimensions": 4, "dimension_names": km_meta["features"]},
                output_schema={"target": "vulnerability_archetype_id", "clusters": 4},
                evaluation_metrics=km_metrics,
                author="pipeline",
                note="Trained and staged via train_and_register CLI.",
            )

    logger.info("Validating local model registry manifest...")
    summary = registry.list_models()
    for m in summary:
        logger.info(f"Registered Model: {m['model_identifier']} ({m['active_version']}) — Status: {m['status']}")

    logger.info("Executing comprehensive model quality gates...")
    runner = QualityGateRunner()
    passed = runner.run_all()

    # Re-verify production aliases remain strictly unchanged
    registry_after = LocalModelRegistry()
    final_models = {m["model_identifier"]: m["active_version"] for m in registry_after.list_models()}
    assert initial_models == final_models, f"CRITICAL: Active production versions altered! Before: {initial_models}, After: {final_models}"

    if passed:
        logger.info("All model quality gates PASSED. Staged candidate models are eligible for promotion review.")
        sys.exit(0)
    else:
        logger.error("Quality gate failures detected. Promotion blocked. Production models remain untouched.")
        sys.exit(1)


if __name__ == "__main__":
    main()
