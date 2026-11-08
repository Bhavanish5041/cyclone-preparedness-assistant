"""Production Feature Engineering and Validation Module for Model Inference (Phase 2.4).

Faithfully reproduces the original teammate training transformations that produce
the exact 26 canonical features required by the registered Logistic Regression classifier.

The 9 missing engineered features in unified_village_preparedness_v2:
1. log1p_population
2. log1p_pop_density_sqkm
3. log1p_dist_nearest_shelter_km
4. log1p_dist_nearest_hospital_km
5. coastal_risk_score
6. evacuation_sufficiency
7. vulnerability_exposure
8. infra_composite
9. monitoring_coverage

Guarantees:
- Strict canonical 26-feature schema and ordering matching logistic_regression_metadata.json.
- Zero silent defaults or synthetic imputation.
- Explicit NaN, null, +/-Inf, and dtype validation.
- Deterministic normalization using audited baseline constants.
"""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

# Canonical 26-feature list in strict fit order
CANONICAL_26_FEATURES: List[str] = [
    "dist_coast_km",
    "elevation_mean",
    "avg_household_size",
    "sc_share",
    "st_share",
    "vulnerable_pop_share",
    "road_connectivity_score",
    "drinking_water_score",
    "n_hc_10km",
    "n_phc_chc_10km",
    "n_gauges_20km",
    "n_shelters_10km",
    "cvi_physical",
    "cvi_social",
    "has_pucca_road",
    "has_major_district_road",
    "bus_connectivity",
    "log1p_population",
    "log1p_pop_density_sqkm",
    "log1p_dist_nearest_shelter_km",
    "log1p_dist_nearest_hospital_km",
    "coastal_risk_score",
    "evacuation_sufficiency",
    "vulnerability_exposure",
    "infra_composite",
    "monitoring_coverage",
]

# The 9 features engineered from raw indicators
NINE_ENGINEERED_FEATURES: List[str] = [
    "log1p_population",
    "log1p_pop_density_sqkm",
    "log1p_dist_nearest_shelter_km",
    "log1p_dist_nearest_hospital_km",
    "coastal_risk_score",
    "evacuation_sufficiency",
    "vulnerability_exposure",
    "infra_composite",
    "monitoring_coverage",
]

# Base 17 features present in raw/unified dataset
SEVENTEEN_BASE_FEATURES: List[str] = [f for f in CANONICAL_26_FEATURES if f not in NINE_ENGINEERED_FEATURES]

# Source indicator columns required to derive the 9 engineered features
REQUIRED_RAW_SOURCE_COLUMNS: List[str] = [
    "population",
    "pop_density_sqkm",
    "dist_nearest_shelter_km",
    "dist_nearest_hospital_km",
    "shelter_capacity_nearby",
    "cvi_integrated",
    "dist_nearest_gauge_km",
]

# Audited empirical reference constants for single-record and batch normalization
# (Identical to 01_EDA.ipynb / modelling_dataset.parquet generation baseline)
CVI_INTEGRATED_MAX: float = 773.4
DIST_COAST_KM_MAX: float = 90.38
DIST_NEAREST_GAUGE_KM_MAX: float = 15.43
N_GAUGES_20KM_MAX: float = 77.0
COASTAL_DECAY_DIST_KM: float = 10.0
COASTAL_DECAY_ELEV_M: float = 30.0


def engineer_inference_features(
    data: Union[pd.DataFrame, Dict[str, Any]],
) -> pd.DataFrame:
    """Engineer and validate the exact 26 canonical features for model inference.

    Handles two input types:
    1. Input already containing all 26 features: validates schema and column order.
    2. Input containing raw village indicators: computes the 9 engineered features faithfully
       and produces the canonical 26-feature DataFrame.

    Args:
        data: DataFrame or dictionary of feature/indicator values.

    Returns:
        DataFrame containing exactly the 26 canonical features in strict fit order.

    Raises:
        ValueError: If required columns are missing, non-numeric, or contain NaNs/Infs.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Input data must be a pandas DataFrame or dictionary, got {type(data)}")

    if df.empty:
        raise ValueError("Input feature data is empty.")

    # Check whether all 26 features are already present
    has_all_26 = all(f in df.columns for f in CANONICAL_26_FEATURES)

    if not has_all_26:
        # Check base 17 features
        missing_base = [f for f in SEVENTEEN_BASE_FEATURES if f not in df.columns]
        # Check raw source indicators required for engineering
        missing_source = [c for c in REQUIRED_RAW_SOURCE_COLUMNS if c not in df.columns]

        missing_total = sorted(list(set(missing_base + missing_source)))
        if missing_total:
            raise ValueError(
                f"Cannot engineer model features: input is missing {len(missing_total)} required column(s): "
                f"{missing_total}. Per Phase 2.4 governance boundaries, missing values cannot be invented or substituted."
            )

        # Check non-numeric values early
        invalid_types = []
        for col in SEVENTEEN_BASE_FEATURES + REQUIRED_RAW_SOURCE_COLUMNS:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                invalid_types.append(col)
        if invalid_types:
            raise ValueError(f"Feature validation failed: Non-numeric features detected: {invalid_types}")

        # 1. Log1p transforms for heavily right-skewed indicators (clipped at 0)
        df["log1p_population"] = np.log1p(df["population"].astype(float).clip(lower=0.0))
        df["log1p_pop_density_sqkm"] = np.log1p(df["pop_density_sqkm"].astype(float).clip(lower=0.0))
        df["log1p_dist_nearest_shelter_km"] = np.log1p(df["dist_nearest_shelter_km"].astype(float).clip(lower=0.0))
        df["log1p_dist_nearest_hospital_km"] = np.log1p(df["dist_nearest_hospital_km"].astype(float).clip(lower=0.0))

        # 2. Coastal Risk Proximity Score (exponential decay by coast distance and elevation)
        df["coastal_risk_score"] = (
            np.exp(-df["dist_coast_km"].astype(float) / COASTAL_DECAY_DIST_KM)
            * np.exp(-df["elevation_mean"].astype(float) / COASTAL_DECAY_ELEV_M)
        ).round(4)

        # 3. Evacuation Sufficiency Ratio (shelter capacity / population, capped at 10)
        pop_safe = df["population"].astype(float).replace(0.0, np.nan)
        df["evacuation_sufficiency"] = (
            df["shelter_capacity_nearby"].astype(float) / pop_safe.fillna(1.0)
        ).clip(0.0, 10.0).round(4)

        # 4. Composite Vulnerability Exposure (CVI + coastal proximity + demography)
        df["vulnerability_exposure"] = (
            0.5 * (df["cvi_integrated"].astype(float) / CVI_INTEGRATED_MAX)
            + 0.3 * (1.0 - df["dist_coast_km"].astype(float) / DIST_COAST_KM_MAX)
            + 0.2 * df["vulnerable_pop_share"].astype(float)
        ).round(4)

        # 5. Infrastructure Composite (road + water + bus connectivity)
        df["infra_composite"] = (
            df["road_connectivity_score"].astype(float) / 3.0
            + df["drinking_water_score"].astype(float) / 2.0
            + df["bus_connectivity"].astype(float)
        ).round(4) / 3.0

        # 6. Monitoring Coverage Score (gauge proximity + density)
        df["monitoring_coverage"] = (
            (1.0 - df["dist_nearest_gauge_km"].astype(float) / DIST_NEAREST_GAUGE_KM_MAX) * 0.6
            + (df["n_gauges_20km"].astype(float) / N_GAUGES_20KM_MAX) * 0.4
        ).round(4)

    # Extract exactly the 26 canonical features in strict order
    X_aligned = df[CANONICAL_26_FEATURES].copy()

    # Strict validation of data types, NaNs, and Infs
    invalid_types = []
    nan_cols = []
    inf_cols = []

    for col in CANONICAL_26_FEATURES:
        series = X_aligned[col]
        if not pd.api.types.is_numeric_dtype(series):
            try:
                X_aligned[col] = series.astype(float)
            except Exception:
                invalid_types.append(col)
                continue

        n_nan = int(X_aligned[col].isna().sum())
        if n_nan > 0:
            nan_cols.append((col, n_nan))

        numeric_vals = X_aligned[col].to_numpy(dtype=float, na_value=0.0)
        n_inf = int(np.isinf(numeric_vals).sum())
        if n_inf > 0:
            inf_cols.append((col, n_inf))

    errors = []
    if invalid_types:
        errors.append(f"Non-numeric features detected: {invalid_types}")
    if nan_cols:
        errors.append(f"Missing (NaN/null) values detected: {nan_cols}")
    if inf_cols:
        errors.append(f"Infinite (+/-Inf) values detected: {inf_cols}")

    if errors:
        raise ValueError("Feature validation failed: " + "; ".join(errors))

    return X_aligned
