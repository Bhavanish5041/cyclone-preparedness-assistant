"""Unit tests for transparent multidimensional prioritization baseline and sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest

from src.features.prioritization import (
    DEFAULT_SCENARIOS,
    WeightingScenario,
    calculate_prioritization_baseline,
    compute_coastal_hazard_dimension,
    compute_demographic_vulnerability_dimension,
    compute_healthcare_deficit_dimension,
    compute_infrastructure_deficit_dimension,
    evaluate_sensitivity_scenarios,
    min_max_scale,
)


def test_min_max_scale_basic_and_invert():
    """Verify min-max scaling with forward and inverted directionality."""
    s = pd.Series([10.0, 20.0, 30.0, 50.0])

    # Forward direction: 10 -> 0.0, 50 -> 1.0
    fwd = min_max_scale(s, invert=False)
    assert np.isclose(fwd.iloc[0], 0.0)
    assert np.isclose(fwd.iloc[-1], 1.0)
    assert np.isclose(fwd.iloc[1], 0.25)
    assert all(0.0 <= x <= 1.0 for x in fwd)

    # Inverted direction: 10 -> 1.0, 50 -> 0.0
    inv = min_max_scale(s, invert=True)
    assert np.isclose(inv.iloc[0], 1.0)
    assert np.isclose(inv.iloc[-1], 0.0)
    assert np.isclose(inv.iloc[1], 0.75)


def test_min_max_scale_constant_series():
    """Verify that constant series gracefully returns neutral 0.5 without divide-by-zero error."""
    const_s = pd.Series([5.0, 5.0, 5.0])
    res = min_max_scale(const_s)
    assert np.allclose(res, 0.5)


def test_healthcare_deficit_dimension():
    """Verify healthcare deficit dimension score calculation and bounds."""
    df = pd.DataFrame(
        {
            "dist_to_nearest_phc_chc_km": [0.5, 3.0, 10.0],
            "dist_to_nearest_hospital_km": [1.0, 15.0, 35.0],
        }
    )
    scores = compute_healthcare_deficit_dimension(df)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Unit 0 is closest to both facilities -> lowest deficit
    assert scores.iloc[0] == 0.0
    # Unit 2 is farthest from both facilities -> highest deficit
    assert scores.iloc[2] == 1.0
    # Monotonicity check
    assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]


def test_demographic_vulnerability_dimension():
    """Verify demographic dimension with and without population scale."""
    df = pd.DataFrame(
        {
            "sc_st_ratio": [0.05, 0.20, 0.40],
            "total_population": [1000, 5000, 50000],
        }
    )

    # Without population scale: purely normalized SC/ST ratio
    pure_sc_st = compute_demographic_vulnerability_dimension(df, include_population_scale=False)
    assert np.isclose(pure_sc_st.iloc[0], 0.0)
    assert np.isclose(pure_sc_st.iloc[-1], 1.0)

    # With population scale
    scaled = compute_demographic_vulnerability_dimension(df, include_population_scale=True)
    assert all(0.0 <= s <= 1.0 for s in scaled)
    assert scaled.iloc[0] < scaled.iloc[-1]


def test_infrastructure_deficit_dimension():
    """Verify infrastructure deficit scoring and categorical drainage mapping."""
    df = pd.DataFrame(
        {
            "has_treated_tap_water": [1, 0, 0],
            "has_all_weather_road": [1, 1, 0],
            "drainage_system": ["Closed", "Open", "None"],
        }
    )
    scores = compute_infrastructure_deficit_dimension(df)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Row 0 has treated tap water (deficit 0), all-weather road (deficit 0), closed drainage (deficit 0) -> 0.0
    assert np.isclose(scores.iloc[0], 0.0)

    # Row 2 lacks treated water (deficit 1), lacks road (deficit 1), has no drainage (deficit 1) -> 1.0
    assert np.isclose(scores.iloc[2], 1.0)

    # Row 1 has intermediate deficit
    assert 0.0 < scores.iloc[1] < 1.0


def test_coastal_hazard_dimension():
    """Verify coastal hazard proximity inversion."""
    df = pd.DataFrame(
        {
            "dist_to_coastline_km": [0.02, 5.0, 70.0],
        }
    )
    scores = compute_coastal_hazard_dimension(df, method="linear_inverse")
    assert len(scores) == 3
    # Nearest unit (0.02 km) must have highest exposure score
    assert scores.iloc[0] > 0.99
    # Farthest inland unit (70 km) must have lowest exposure score
    assert scores.iloc[2] < 0.01
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]


def test_weighting_scenario_validation():
    """Verify that WeightingScenario enforces non-negative weights summing to 1.0."""
    valid_scen = WeightingScenario(
        name="valid",
        description="Valid test scenario",
        weight_healthcare=0.25,
        weight_demographic=0.25,
        weight_infrastructure=0.25,
        weight_coastal=0.25,
    )
    valid_scen.validate()  # Must not raise

    # Negative weight
    with pytest.raises(ValueError, match="negative weights"):
        bad_neg = WeightingScenario("neg", "test", -0.1, 0.4, 0.3, 0.4)
        bad_neg.validate()

    # Sum does not equal 1.0
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        bad_sum = WeightingScenario("bad_sum", "test", 0.5, 0.5, 0.5, 0.5)
        bad_sum.validate()


def test_calculate_prioritization_baseline():
    """Verify full baseline scoring and deterministic rank assignment."""
    df = pd.DataFrame(
        {
            "shrid2": ["id1", "id2", "id3"],
            "pc11_id": ["p1", "p2", "p3"],
            "unit_name": ["Hamlet A", "Hamlet B", "Hamlet C"],
            "dist_to_nearest_phc_chc_km": [1.0, 5.0, 10.0],
            "dist_to_nearest_hospital_km": [5.0, 15.0, 30.0],
            "sc_st_ratio": [0.05, 0.20, 0.35],
            "has_treated_tap_water": [1, 0, 0],
            "has_all_weather_road": [1, 1, 0],
            "drainage_system": ["Closed", "Open", "None"],
            "dist_to_coastline_km": [50.0, 10.0, 0.5],
        }
    )

    scen = DEFAULT_SCENARIOS["equal_weights"]
    res, audit = calculate_prioritization_baseline(df, scen)

    assert "priority_score" in res.columns
    assert "priority_rank" in res.columns
    assert len(res) == 3
    # Check bounds
    assert all(0.0 <= s <= 1.0 for s in res["priority_score"])
    # Rank 1 must correspond to max score
    max_idx = res["priority_score"].idxmax()
    assert res.loc[max_idx, "priority_rank"] == 1
    # All ranks must be distinct integers 1, 2, 3
    assert set(res["priority_rank"]) == {1, 2, 3}


def test_evaluate_sensitivity_scenarios():
    """Verify multi-scenario sensitivity evaluation and rank correlations."""
    df = pd.DataFrame(
        {
            "shrid2": [f"11-29-575-05561-617{i:03d}" for i in range(10)],
            "pc11_id": [f"617{i:03d}" for i in range(10)],
            "unit_name": [f"Village {i}" for i in range(10)],
            "dist_to_nearest_phc_chc_km": np.linspace(1.0, 12.0, 10),
            "dist_to_nearest_hospital_km": np.linspace(5.0, 35.0, 10),
            "sc_st_ratio": np.linspace(0.05, 0.45, 10),
            "has_treated_tap_water": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "has_all_weather_road": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            "drainage_system": ["Closed", "Both", "Open", "Open", "None", "None", "None", "None", "None", "None"],
            "dist_to_coastline_km": np.linspace(0.1, 70.0, 10),
            "total_population": np.linspace(1000, 20000, 10),
        }
    )

    results_df, spearman_mat, sens_metrics = evaluate_sensitivity_scenarios(df, DEFAULT_SCENARIOS)

    assert len(results_df) == 10
    # Diagonal of correlation matrix must be exactly 1.0
    for name in DEFAULT_SCENARIOS.keys():
        assert np.isclose(spearman_mat.loc[name, name], 1.0)

    # Rank shift span must be >= 0
    assert all(results_df["rank_shift_span"] >= 0)
    assert sens_metrics["total_units_evaluated"] == 10
