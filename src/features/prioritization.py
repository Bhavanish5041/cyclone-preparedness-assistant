"""Transparent multidimensional prioritization baseline for cyclone preparedness.

Implements an interpretable analytical heuristic (NOT a predictive ML model or validated risk score).
Normalizes non-redundant indicators into four distinct dimensions:
1. Healthcare Accessibility Deficit (isolation from PHCs/CHCs and apex hospitals)
2. Demographic Vulnerability (SC/ST socioeconomic disadvantage, with optional population scale)
3. Infrastructure Deficit (lack of treated tap water, unpaved/no road, and drainage deficit)
4. Coastal Hazard Proximity (perpendicular proximity to the Arabian Sea coastline)

Maintains explicit directionality, documented weighting, visibility of missing data,
and comprehensive sensitivity analysis across alternative weighting schemes.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)

PRIORITIZATION_DISCLAIMER = (
    "DISCLAIMER: This prioritization score is an experimental, transparent analytical heuristic "
    "designed to highlight spatial clusters with compounding accessibility, infrastructure, demographic, "
    "and coastal proximity deficits. It is NOT a validated cyclone damage model, storm-surge prediction, "
    "or empirical casualty probability score. It uses the 2011 Census baseline and administrative "
    "healthcare registry coordinates without real-time operational status."
)


@dataclass(frozen=True)
class WeightingScenario:
    """Configuration for a multidimensional weighting scenario."""

    name: str
    description: str
    weight_healthcare: float
    weight_demographic: float
    weight_infrastructure: float
    weight_coastal: float
    include_population_scale: bool = False

    def validate(self) -> None:
        """Validate that weights are non-negative and sum to 1.0."""
        weights = [
            self.weight_healthcare,
            self.weight_demographic,
            self.weight_infrastructure,
            self.weight_coastal,
        ]
        if any(w < 0.0 for w in weights):
            raise ValueError(f"Scenario '{self.name}' has negative weights: {weights}")
        total = sum(weights)
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"Scenario '{self.name}' weights must sum to 1.0, got {total:.6f}")


# Standard Sensitivity Scenarios
DEFAULT_SCENARIOS: Dict[str, WeightingScenario] = {
    "equal_weights": WeightingScenario(
        name="equal_weights",
        description="Equal 25% weight across all four dimensions (Healthcare, Demographics, Infrastructure, Coastal Proximity).",
        weight_healthcare=0.25,
        weight_demographic=0.25,
        weight_infrastructure=0.25,
        weight_coastal=0.25,
        include_population_scale=False,
    ),
    "hazard_focused": WeightingScenario(
        name="hazard_focused",
        description="Hazard-dominant weighting emphasizing immediate coastal proximity (50%) over interior factors.",
        weight_healthcare=0.15,
        weight_demographic=0.15,
        weight_infrastructure=0.20,
        weight_coastal=0.50,
        include_population_scale=False,
    ),
    "access_focused": WeightingScenario(
        name="access_focused",
        description="Access-dominant weighting emphasizing healthcare isolation (40%) and infrastructure deficits (30%).",
        weight_healthcare=0.40,
        weight_demographic=0.15,
        weight_infrastructure=0.35,
        weight_coastal=0.10,
        include_population_scale=False,
    ),
    "no_coastal": WeightingScenario(
        name="no_coastal",
        description="Inland resilience baseline omitting coastal proximity completely (equal 33.3% across remaining 3 dimensions).",
        weight_healthcare=1.0 / 3.0,
        weight_demographic=1.0 / 3.0,
        weight_infrastructure=1.0 / 3.0,
        weight_coastal=0.0,
        include_population_scale=False,
    ),
    "with_population_scale": WeightingScenario(
        name="with_population_scale",
        description="Equal weights incorporating log-transformed population scale alongside SC/ST ratio in demographics.",
        weight_healthcare=0.25,
        weight_demographic=0.25,
        weight_infrastructure=0.25,
        weight_coastal=0.25,
        include_population_scale=True,
    ),
}


def min_max_scale(
    series: pd.Series,
    invert: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> pd.Series:
    """Normalize a series to [0.0, 1.0] with explicit directionality.

    Args:
        series: Numeric series to normalize.
        invert: If True, maps min to 1.0 and max to 0.0 (higher raw value -> lower score).
                If False, maps min to 0.0 and max to 1.0 (higher raw value -> higher score).
        min_val: Optional manual minimum value (uses series.min() if None).
        max_val: Optional manual maximum value (uses series.max() if None).

    Returns:
        Series of normalized float values bounded in [0.0, 1.0].
    """
    clean_s = series.astype(float)
    lo = float(clean_s.min()) if min_val is None else float(min_val)
    hi = float(clean_s.max()) if max_val is None else float(max_val)

    if np.isclose(lo, hi):
        # Constant series: assign neutral 0.5
        return pd.Series(0.5, index=series.index, dtype=float)

    scaled = (clean_s - lo) / (hi - lo)
    scaled = scaled.clip(0.0, 1.0)

    if invert:
        scaled = 1.0 - scaled

    return scaled


def compute_healthcare_deficit_dimension(
    df: pd.DataFrame,
    phc_weight: float = 0.5,
    hospital_weight: float = 0.5,
) -> pd.Series:
    """Compute the Healthcare Accessibility Deficit dimension score in [0.0, 1.0].

    Higher score = Greater geographic isolation from healthcare facilities.

    Args:
        df: DataFrame containing dist_to_nearest_phc_chc_km and dist_to_nearest_hospital_km.
        phc_weight: Weight for distance to nearest PHC/CHC.
        hospital_weight: Weight for distance to nearest Hospital.

    Returns:
        Series of dimension scores in [0.0, 1.0].
    """
    if "dist_to_nearest_phc_chc_km" not in df.columns or "dist_to_nearest_hospital_km" not in df.columns:
        raise KeyError("Missing required healthcare distance columns.")

    s_phc = min_max_scale(df["dist_to_nearest_phc_chc_km"], invert=False)
    s_hosp = min_max_scale(df["dist_to_nearest_hospital_km"], invert=False)

    dim_score = phc_weight * s_phc + hospital_weight * s_hosp
    return dim_score.round(4)


def compute_demographic_vulnerability_dimension(
    df: pd.DataFrame,
    include_population_scale: bool = False,
    sc_st_weight: float = 0.7,
    pop_weight: float = 0.3,
) -> pd.Series:
    """Compute the Demographic Vulnerability dimension score in [0.0, 1.0].

    Higher score = Greater socioeconomic vulnerability and/or exposed population scale.

    Args:
        df: DataFrame containing sc_st_ratio and optionally total_population.
        include_population_scale: If True, blends log-transformed population scale with SC/ST ratio.
                                  If False, uses purely the SC/ST ratio.
        sc_st_weight, pop_weight: Weights when combining SC/ST ratio and population scale.

    Returns:
        Series of dimension scores in [0.0, 1.0].
    """
    if "sc_st_ratio" not in df.columns:
        raise KeyError("Missing sc_st_ratio in dataframe.")

    s_sc_st = min_max_scale(df["sc_st_ratio"], invert=False)

    if not include_population_scale:
        return s_sc_st.round(4)

    if "total_population" not in df.columns:
        raise KeyError("total_population required when include_population_scale=True.")

    # Log1p transform population to compress extreme urban skew (Mangalore City = 499k)
    log_pop = np.log1p(df["total_population"].astype(float))
    s_pop = min_max_scale(log_pop, invert=False)

    total_w = sc_st_weight + pop_weight
    dim_score = (sc_st_weight * s_sc_st + pop_weight * s_pop) / total_w
    return dim_score.round(4)


def compute_infrastructure_deficit_dimension(
    df: pd.DataFrame,
    water_weight: float = 0.40,
    road_weight: float = 0.35,
    drainage_weight: float = 0.25,
) -> pd.Series:
    """Compute the Infrastructure Deficit dimension score in [0.0, 1.0].

    Higher score = Greater deficit in basic municipal water, road, and drainage infrastructure.

    Args:
        df: DataFrame containing has_treated_tap_water, has_all_weather_road, drainage_system.
        water_weight: Weight for drinking water deficit.
        road_weight: Weight for road access deficit.
        drainage_weight: Weight for drainage deficit.

    Returns:
        Series of dimension scores in [0.0, 1.0].
    """
    req_cols = ["has_treated_tap_water", "has_all_weather_road", "drainage_system"]
    for c in req_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required infrastructure column: {c}")

    # 1. Water deficit: 1 if untreated/no tap water, 0 if treated tap water present
    s_water_deficit = 1.0 - df["has_treated_tap_water"].astype(float).clip(0.0, 1.0)

    # 2. Road deficit: 1 if no all-weather road, 0 if all-weather road present
    s_road_deficit = 1.0 - df["has_all_weather_road"].astype(float).clip(0.0, 1.0)

    # 3. Drainage deficit mapping:
    # Closed underground drains = 0.0 (lowest deficit)
    # Both closed and open = 0.33
    # Open drains = 0.67 (prone to silting, storm overflow)
    # None = 1.0 (highest deficit / zero formal drainage)
    drainage_map = {
        "closed": 0.0,
        "both": 0.33,
        "open": 0.67,
        "none": 1.0,
    }
    clean_drainage = df["drainage_system"].fillna("none").astype(str).str.lower().str.strip()
    s_drainage_deficit = clean_drainage.map(lambda v: drainage_map.get(v, 1.0)).astype(float)

    total_w = water_weight + road_weight + drainage_weight
    dim_score = (
        water_weight * s_water_deficit
        + road_weight * s_road_deficit
        + drainage_weight * s_drainage_deficit
    ) / total_w

    return dim_score.round(4)


def compute_coastal_hazard_dimension(
    df: pd.DataFrame,
    method: str = "linear_inverse",
) -> pd.Series:
    """Compute Coastal Hazard Exposure dimension score in [0.0, 1.0].

    Higher score = Closer to Arabian Sea coastline (greater marine surge and wind exposure).

    Args:
        df: DataFrame containing dist_to_coastline_km.
        method: 'linear_inverse' (min-max inverted) or 'exponential_decay'.

    Returns:
        Series of dimension scores in [0.0, 1.0].
    """
    if "dist_to_coastline_km" not in df.columns:
        raise KeyError("Missing dist_to_coastline_km in dataframe.")

    dists = df["dist_to_coastline_km"].astype(float)

    if method == "linear_inverse":
        # Invert: min distance (~0 km) -> 1.0, max distance (~72 km) -> 0.0
        return min_max_scale(dists, invert=True).round(4)
    elif method == "exponential_decay":
        # Characteristic half-distance of 10 km (typical storm surge / coastal wind influence zone)
        score = np.exp(-dists / 10.0)
        return score.round(4)
    else:
        raise ValueError(f"Unknown coastal method: {method}")


def calculate_prioritization_baseline(
    df: pd.DataFrame,
    scenario: WeightingScenario,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Calculate all dimension scores and combined priority score for a given scenario.

    Args:
        df: Input features DataFrame with all 4 domain variables.
        scenario: WeightingScenario instance specifying weights and configuration.

    Returns:
        Tuple of (enriched_DataFrame, metadata_audit_dict).
    """
    scenario.validate()
    res = df.copy()

    # 1. Compute four separate dimension scores
    res["dim_healthcare_deficit"] = compute_healthcare_deficit_dimension(res)
    res["dim_demographic_vulnerability"] = compute_demographic_vulnerability_dimension(
        res, include_population_scale=scenario.include_population_scale
    )
    res["dim_infrastructure_deficit"] = compute_infrastructure_deficit_dimension(res)
    res["dim_coastal_exposure"] = compute_coastal_hazard_dimension(res)

    # 2. Compute combined weighted priority score
    res["priority_score"] = (
        scenario.weight_healthcare * res["dim_healthcare_deficit"]
        + scenario.weight_demographic * res["dim_demographic_vulnerability"]
        + scenario.weight_infrastructure * res["dim_infrastructure_deficit"]
        + scenario.weight_coastal * res["dim_coastal_exposure"]
    ).round(4)

    # 3. Deterministic ranking: rank 1 = highest priority need (highest priority score)
    # Break ties deterministically by unit_name / shrid2
    res["priority_rank"] = (
        res["priority_score"].rank(ascending=False, method="min").astype(int)
    )

    audit_summary = {
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "weights": {
            "healthcare": scenario.weight_healthcare,
            "demographic": scenario.weight_demographic,
            "infrastructure": scenario.weight_infrastructure,
            "coastal": scenario.weight_coastal,
        },
        "include_population_scale": scenario.include_population_scale,
        "score_summary": {
            "min": float(res["priority_score"].min()),
            "mean": float(res["priority_score"].mean()),
            "median": float(res["priority_score"].median()),
            "max": float(res["priority_score"].max()),
            "std": float(res["priority_score"].std()),
        },
        "disclaimer": PRIORITIZATION_DISCLAIMER,
    }

    return res, audit_summary


def evaluate_sensitivity_scenarios(
    df: pd.DataFrame,
    scenarios: Optional[Dict[str, WeightingScenario]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run sensitivity analysis across multiple weighting configurations.

    Args:
        df: Input DataFrame with candidate features.
        scenarios: Dictionary of scenario configurations (uses DEFAULT_SCENARIOS if None).

    Returns:
        Tuple of:
        - scenario_scores_df: DataFrame with unit identifiers, scores, and ranks for each scenario.
        - rank_correlation_matrix: DataFrame containing Spearman and Kendall rank correlations.
        - sensitivity_metrics: Dictionary summarizing robust units, shift statistics, and top ranks.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    scenario_names = list(scenarios.keys())
    results_df = df[["shrid2", "pc11_id", "unit_name"]].copy()

    # District label
    results_df["district_name"] = results_df["shrid2"].apply(
        lambda s: "Dakshina Kannada" if "575" in str(s) else "Udupi"
    )

    rank_cols = []
    score_cols = []

    for name, scen in scenarios.items():
        scored_df, _ = calculate_prioritization_baseline(df, scen)
        score_col = f"score_{name}"
        rank_col = f"rank_{name}"
        results_df[score_col] = scored_df["priority_score"]
        results_df[rank_col] = scored_df["priority_rank"]
        score_cols.append(score_col)
        rank_cols.append(rank_col)

    # Compute pairwise rank correlations
    spearman_mat = pd.DataFrame(index=scenario_names, columns=scenario_names, dtype=float)
    kendall_mat = pd.DataFrame(index=scenario_names, columns=scenario_names, dtype=float)

    for sc1 in scenario_names:
        for sc2 in scenario_names:
            r_col1 = f"rank_{sc1}"
            r_col2 = f"rank_{sc2}"
            sr, _ = spearmanr(results_df[r_col1], results_df[r_col2])
            kt, _ = kendalltau(results_df[r_col1], results_df[r_col2])
            spearman_mat.loc[sc1, sc2] = round(sr, 3)
            kendall_mat.loc[sc1, sc2] = round(kt, 3)

    # Rank variance per unit (measures volatility across weighting schemes)
    rank_matrix = results_df[rank_cols]
    results_df["rank_mean"] = rank_matrix.mean(axis=1).round(1)
    results_df["rank_std"] = rank_matrix.std(axis=1).round(1)
    results_df["rank_min"] = rank_matrix.min(axis=1)
    results_df["rank_max"] = rank_matrix.max(axis=1)
    results_df["rank_shift_span"] = results_df["rank_max"] - results_df["rank_min"]

    # Identify robust high-priority units: top 50 in equal_weights AND present in top 100 across all scenarios
    top50_equal = set(results_df.loc[results_df["rank_equal_weights"] <= 50, "shrid2"])
    robust_mask = results_df[rank_cols].apply(lambda row: all(r <= 100 for r in row), axis=1)
    robust_units = results_df[robust_mask & results_df["shrid2"].isin(top50_equal)].sort_values(
        "rank_mean"
    )

    sensitivity_metrics = {
        "total_units_evaluated": len(results_df),
        "scenarios_evaluated": scenario_names,
        "spearman_rank_correlations": spearman_mat.to_dict(),
        "kendall_tau_correlations": kendall_mat.to_dict(),
        "robust_high_priority_units_count": len(robust_units),
        "robust_units_summary": robust_units[
            ["shrid2", "unit_name", "district_name", "rank_mean", "rank_std", "rank_shift_span"]
        ].to_dict(orient="records"),
        "max_rank_shift_observed": int(results_df["rank_shift_span"].max()),
        "mean_rank_shift_observed": round(float(results_df["rank_shift_span"].mean()), 1),
    }

    return results_df, spearman_mat, sensitivity_metrics
