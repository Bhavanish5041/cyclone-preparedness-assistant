"""Milestone 2C: Enrich the Unified Preparedness Dataset with Logistic Regression Predictions and Probabilities.

Loads:
- Milestone 1 unified dataset: data/processed/unified_village_preparedness.parquet
- Model probability predictions: data/processed/village_preparedness_probabilities.parquet

Performs a validated 1:1 left-join on unified.shrid2 == probabilities.area_id.
Preserves all 1,879 records, retains all original AHP and K-Means columns unchanged,
and writes data/processed/unified_village_preparedness_v2.parquet (and .csv).
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("enrich_unified_v2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CYCLONE_DATA_DIR", PROJECT_ROOT / "data" / "processed"))

DEFAULT_UNIFIED_PATH = DATA_DIR / "unified_village_preparedness.parquet"
DEFAULT_PROBABILITIES_PATH = DATA_DIR / "village_preparedness_probabilities.parquet"
DEFAULT_OUTPUT_PATH = DATA_DIR / "unified_village_preparedness_v2.parquet"


def enrich_unified_dataset_v2(
    unified_path: Path = DEFAULT_UNIFIED_PATH,
    probabilities_path: Path = DEFAULT_PROBABILITIES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Enrich Milestone 1 unified dataset with model predictions and probabilities.

    Args:
        unified_path: Path to unified_village_preparedness.parquet.
        probabilities_path: Path to village_preparedness_probabilities.parquet.
        output_path: Path to output unified_village_preparedness_v2.parquet.
        save_csv: Whether to write companion CSV.

    Returns:
        Enriched DataFrame (1,879 rows).
    """
    logger.info(f"Loading Milestone 1 unified dataset from {unified_path}...")
    df_uni = pd.read_parquet(unified_path)

    logger.info(f"Loading probabilities dataset from {probabilities_path}...")
    df_prob = pd.read_parquet(probabilities_path)

    # 1. Verification of records and keys
    if len(df_uni) != 1879:
        raise ValueError(f"Expected 1,879 rows in unified dataset, found {len(df_uni)}")
    if len(df_prob) != 1879:
        raise ValueError(f"Expected 1,879 rows in probabilities dataset, found {len(df_prob)}")

    if df_uni["shrid2"].isna().any():
        raise ValueError("Null values found in unified shrid2")
    if df_prob["area_id"].isna().any():
        raise ValueError("Null values found in probabilities area_id")

    if df_uni["shrid2"].nunique() != 1879:
        raise ValueError("Duplicate shrid2 values found in unified dataset")
    if df_prob["area_id"].nunique() != 1879:
        raise ValueError("Duplicate area_id values found in probabilities dataset")

    # 2. Probability numerical validation
    prob_cols = ["prob_low", "prob_medium", "prob_high"]
    for col in prob_cols:
        if col not in df_prob.columns:
            raise KeyError(f"Expected probability column '{col}' missing from probabilities dataset")
        if not np.isfinite(df_prob[col]).all():
            raise ValueError(f"Non-finite values found in '{col}'")
        if (df_prob[col] < 0.0).any() or (df_prob[col] > 1.0).any():
            raise ValueError(f"Values outside [0, 1] found in '{col}'")

    prob_sums = df_prob["prob_low"] + df_prob["prob_medium"] + df_prob["prob_high"]
    if not np.allclose(prob_sums, 1.0, atol=1e-3):
        raise ValueError("Probability simplex violation: sums do not equal 1.0")

    # 3. Provenance validation
    prov_counts = df_prob["prediction_provenance"].value_counts().to_dict()
    logger.info(f"Prediction provenance counts: {prov_counts}")
    if prov_counts.get("TRAIN_IN_SAMPLE") != 1315 or prov_counts.get("VAL_OUT_OF_SAMPLE") != 282 or prov_counts.get("TEST_OUT_OF_SAMPLE") != 282:
        raise ValueError(f"Unexpected prediction provenance counts: {prov_counts}")

    # 4. Join prediction fields
    cols_to_add = [
        "area_id",
        "predicted_preparedness_tier",
        "prob_low",
        "prob_medium",
        "prob_high",
        "prediction_provenance",
        "probability_disclaimer",
    ]
    df_prob_sub = df_prob[cols_to_add].copy()

    logger.info("Executing 1:1 left-join on unified.shrid2 == probabilities.area_id...")
    merged = df_uni.merge(df_prob_sub, left_on="shrid2", right_on="area_id", how="left", suffixes=("", "_prob_table"))

    if len(merged) != 1879:
        raise RuntimeError(f"Row count expansion! Expected 1,879 rows, got {len(merged)}")

    # Drop redundant area_id from probability sub-table if present
    if "area_id_prob_table" in merged.columns:
        merged.drop(columns=["area_id_prob_table"], inplace=True)

    # 5. Verify preservation of original AHP and K-Means values
    np.testing.assert_allclose(merged["preparedness_score"], df_uni["preparedness_score"], atol=1e-5)
    assert (merged["preparedness_tier"] == df_uni["preparedness_tier"]).all()
    assert (merged["cluster_assignment_status"] == df_uni["cluster_assignment_status"]).all()

    assigned_mask = merged["cluster_assignment_status"] == "ASSIGNED"
    assert assigned_mask.sum() == 622
    assert (merged.loc[assigned_mask, "vulnerability_archetype_id"] == df_uni.loc[assigned_mask, "vulnerability_archetype_id"]).all()
    assert (merged.loc[assigned_mask, "vulnerability_archetype_label"] == df_uni.loc[assigned_mask, "vulnerability_archetype_label"]).all()
    assert (merged.loc[assigned_mask, "archetype_intervention_need"] == df_uni.loc[assigned_mask, "archetype_intervention_need"]).all()

    unassigned_mask = merged["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"
    assert unassigned_mask.sum() == 1257
    assert merged.loc[unassigned_mask, "vulnerability_archetype_id"].isna().all()
    assert merged.loc[unassigned_mask, "vulnerability_archetype_label"].isna().all()

    # 6. Re-order columns logically to place predictions beside original tiers
    # Place predicted tier and probabilities directly after preparedness_tier
    existing_cols = list(df_uni.columns)
    tier_idx = existing_cols.index("preparedness_tier")

    new_model_cols = [
        "predicted_preparedness_tier",
        "prob_low",
        "prob_medium",
        "prob_high",
        "prediction_provenance",
        "probability_disclaimer",
    ]

    reordered_cols = (
        existing_cols[: tier_idx + 1]
        + new_model_cols
        + [c for c in existing_cols[tier_idx + 1 :] if c not in new_model_cols]
    )

    enriched_df = merged[reordered_cols].copy()

    # 7. Persist to data/processed/unified_village_preparedness_v2.parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing enriched dataset v2 ({len(enriched_df)} rows, {len(enriched_df.columns)} cols) to {output_path}...")
    enriched_df.to_parquet(output_path, index=False)

    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        logger.info(f"Writing companion CSV to {csv_path}...")
        enriched_df.to_csv(csv_path, index=False)

    logger.info("Milestone 2C enrichment completed successfully.")
    return enriched_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Unified Preparedness Dataset with Model Probabilities.")
    parser.add_argument(
        "--unified-path",
        type=Path,
        default=DEFAULT_UNIFIED_PATH,
        help="Path to unified_village_preparedness.parquet",
    )
    parser.add_argument(
        "--probabilities-path",
        type=Path,
        default=DEFAULT_PROBABILITIES_PATH,
        help="Path to village_preparedness_probabilities.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output unified_village_preparedness_v2.parquet",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing companion CSV copy",
    )
    args = parser.parse_args()

    enrich_unified_dataset_v2(
        unified_path=args.unified_path,
        probabilities_path=args.probabilities_path,
        output_path=args.output_path,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
