"""Candidate feature dictionary and quality registry for cyclone preparedness.

Maintains comprehensive metadata for candidate variables across 5 distinct domains:
1. Healthcare Accessibility Proxies
2. Demographic Vulnerability Indicators
3. Water, Road, Drainage, and Power Infrastructure
4. Coastal Proximity Proxies
5. Data Freshness and Spatial Quality Flags

Defines explicit interpretation directions, missingness rates, known biases,
and provides utilities for correlation analysis, skewness, and multicollinearity auditing.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata specification for a single candidate feature."""

    name: str
    domain: str
    source_dataset: str
    source_organization: str
    vintage: str
    unit: str
    interpretation_direction: str  # "higher_indicates_lower_preparedness", "higher_indicates_better_preparedness", "contextual"
    description: str
    known_bias_or_limitation: str


CANDIDATE_FEATURE_DICTIONARY: Dict[str, FeatureMetadata] = {
    # ---------------------------------------------------------
    # Domain 1: Healthcare Accessibility
    # ---------------------------------------------------------
    "dist_to_nearest_phc_chc_km": FeatureMetadata(
        name="dist_to_nearest_phc_chc_km",
        domain="Healthcare Accessibility",
        source_dataset="geocode_health_centre.csv (MoHFW)",
        source_organization="Ministry of Health and Family Welfare (MoHFW), Govt. of India",
        vintage="2015-2019",
        unit="kilometres",
        interpretation_direction="higher_indicates_lower_preparedness",
        description="Straight-line geodesic distance from polygon representative point to nearest registered PHC or CHC.",
        known_bias_or_limitation=(
            "Geodesic proxy does NOT account for road routing, topography, ferry crossings, or flood impassability. "
            "Registration does not guarantee active doctor staffing, functioning power, or cyclone emergency readiness."
        ),
    ),
    "dist_to_nearest_hospital_km": FeatureMetadata(
        name="dist_to_nearest_hospital_km",
        domain="Healthcare Accessibility",
        source_dataset="geocode_health_centre.csv (MoHFW)",
        source_organization="Ministry of Health and Family Welfare (MoHFW), Govt. of India",
        vintage="2015-2019",
        unit="kilometres",
        interpretation_direction="higher_indicates_lower_preparedness",
        description="Straight-line geodesic distance from polygon representative point to nearest registered Hospital (STH or DH).",
        known_bias_or_limitation=(
            "Apex facilities are scarce (only 9 in study area); distances reach up to 36.4 km in Western Ghats foothills. "
            "Does not confirm operational trauma, surgical, or ICU surge capacity."
        ),
    ),
    "dist_to_nearest_facility_any_km": FeatureMetadata(
        name="dist_to_nearest_facility_any_km",
        domain="Healthcare Accessibility",
        source_dataset="geocode_health_centre.csv (MoHFW)",
        source_organization="Ministry of Health and Family Welfare (MoHFW), Govt. of India",
        vintage="2015-2019",
        unit="kilometres",
        interpretation_direction="higher_indicates_lower_preparedness",
        description="Straight-line geodesic distance to nearest registered health facility of any type (including Sub-Centres).",
        known_bias_or_limitation=(
            "Over 82% of these facilities are Sub-Centres with zero inpatient beds, single ANMs, and no power backup."
        ),
    ),
    "phc_chc_in_unit_count": FeatureMetadata(
        name="phc_chc_in_unit_count",
        domain="Healthcare Accessibility",
        source_dataset="geocode_health_centre.csv (MoHFW)",
        source_organization="Ministry of Health and Family Welfare (MoHFW), Govt. of India",
        vintage="2015-2019",
        unit="facility count",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Number of registered PHCs and CHCs located strictly within the village/town administrative boundary.",
        known_bias_or_limitation=(
            "Heavily zero-inflated (78.8% of units have zero in-unit PHCs/CHCs). Subject to boundary edge slivers."
        ),
    ),
    "hospitals_in_unit_count": FeatureMetadata(
        name="hospitals_in_unit_count",
        domain="Healthcare Accessibility",
        source_dataset="geocode_health_centre.csv (MoHFW)",
        source_organization="Ministry of Health and Family Welfare (MoHFW), Govt. of India",
        vintage="2015-2019",
        unit="facility count",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Number of registered District or Sub-Taluk Hospitals located strictly within the unit boundary.",
        known_bias_or_limitation="Severely zero-inflated (98.7% of units have zero hospitals).",
    ),
    "facilities_in_unit_total": FeatureMetadata(
        name="facilities_in_unit_total",
        domain="Healthcare Accessibility",
        source_dataset="geocode_health_centre.csv (MoHFW)",
        source_organization="Ministry of Health and Family Welfare (MoHFW), Govt. of India",
        vintage="2015-2019",
        unit="facility count",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Total count of registered health facilities of all types inside the administrative unit boundary.",
        known_bias_or_limitation="Reflects administrative boundary size and sub-centre density rather than tertiary capacity.",
    ),
    # ---------------------------------------------------------
    # Domain 2: Demographic Indicators
    # ---------------------------------------------------------
    "total_population": FeatureMetadata(
        name="total_population",
        domain="Demographic Indicators",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="persons",
        interpretation_direction="higher_indicates_lower_preparedness",
        description="Total resident population in the administrative unit as of Census 2011.",
        known_bias_or_limitation=(
            "Census vintage is 2011 (over 15 years old); does not capture subsequent urban growth, peri-urbanization, "
            "or seasonal tourism in coastal hubs like Mangalore, Udupi, and Malpe."
        ),
    ),
    "total_households": FeatureMetadata(
        name="total_households",
        domain="Demographic Indicators",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="households",
        interpretation_direction="higher_indicates_lower_preparedness",
        description="Total number of occupied residential households in the unit.",
        known_bias_or_limitation="Extremely high collinearity with total_population (r > 0.98); prone to double-counting.",
    ),
    "sc_st_ratio": FeatureMetadata(
        name="sc_st_ratio",
        domain="Demographic Indicators",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="ratio (proportion of total pop)",
        interpretation_direction="higher_indicates_lower_preparedness",
        description="Proportion of unit population belonging to Scheduled Castes (SC) or Scheduled Tribes (ST).",
        known_bias_or_limitation=(
            "Proxy for historical socio-economic disadvantage; does not capture household income or individual assets."
        ),
    ),
    "sex_ratio_female_per_1000_male": FeatureMetadata(
        name="sex_ratio_female_per_1000_male",
        domain="Demographic Indicators",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="females per 1000 males",
        interpretation_direction="contextual",
        description="Number of females per 1,000 males.",
        known_bias_or_limitation="Higher female ratios in coastal Karnataka often reflect male out-migration for Gulf/urban employment.",
    ),
    "literacy_rate": FeatureMetadata(
        name="literacy_rate",
        domain="Demographic Indicators",
        source_dataset="PCA_CDB_2915_F_Census.xls (Census 2011 PCA)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="percentage",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Percentage of literate population in unit.",
        known_bias_or_limitation=(
            "CRITICAL MISSINGNESS: Only available for Udupi district (250 units); missing for 100% of Dakshina Kannada (372 units) "
            "because DCHB village release lacked PCA education columns. Cannot be used directly in joint cross-district modeling."
        ),
    ),
    # ---------------------------------------------------------
    # Domain 3: Infrastructure Indicators
    # ---------------------------------------------------------
    "has_treated_tap_water": FeatureMetadata(
        name="has_treated_tap_water",
        domain="Water, Road, Drainage and Power Infrastructure",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="boolean (0 or 1)",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Indicator whether treated municipal/piped tap water infrastructure exists within the unit.",
        known_bias_or_limitation="Does not capture water salinity intrusion post-cyclone surge or pipe breakage vulnerabilities.",
    ),
    "has_all_weather_road": FeatureMetadata(
        name="has_all_weather_road",
        domain="Water, Road, Drainage and Power Infrastructure",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="boolean (0 or 1)",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Presence of an all-weather motorable approach road to the village or town.",
        known_bias_or_limitation=(
            "Binary indicator only; nearly 98% saturation in coastal Karnataka. Highly collinear with has_pucca_road."
        ),
    ),
    "has_pucca_road": FeatureMetadata(
        name="has_pucca_road",
        domain="Water, Road, Drainage and Power Infrastructure",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="boolean (0 or 1)",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Presence of paved/pucca internal roads.",
        known_bias_or_limitation="High collinearity with has_all_weather_road (r > 0.88); does not guarantee storm flood elevation.",
    ),
    "has_domestic_power": FeatureMetadata(
        name="has_domestic_power",
        domain="Water, Road, Drainage and Power Infrastructure",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="boolean (0 or 1)",
        interpretation_direction="higher_indicates_better_preparedness",
        description="Indicator whether domestic grid electricity supply is connected to the unit.",
        known_bias_or_limitation="Nearly 100% saturation (low variance); does not measure grid blackout duration during high winds.",
    ),
    "drainage_system": FeatureMetadata(
        name="drainage_system",
        domain="Water, Road, Drainage and Power Infrastructure",
        source_dataset="Census of India 2011 (DCHB Village & Town Release)",
        source_organization="Office of the Registrar General & Census Commissioner, India",
        vintage="2011",
        unit="categorical string (closed / open / none)",
        interpretation_direction="contextual",
        description="Type of community wastewater/stormwater drainage system present in the unit.",
        known_bias_or_limitation="Vast majority are open drains (prone to storm silting and overflow) or have no formal system.",
    ),
    # ---------------------------------------------------------
    # Domain 4: Coastal Proximity
    # ---------------------------------------------------------
    "dist_to_coastline_km": FeatureMetadata(
        name="dist_to_coastline_km",
        domain="Coastal Proximity",
        source_dataset="ne_10m_coastline (Natural Earth)",
        source_organization="Natural Earth / NACIS",
        vintage="v5.1.1 (2022)",
        unit="kilometres",
        interpretation_direction="higher_indicates_better_preparedness",  # Closer to coast = higher storm surge exposure
        description="Straight-line perpendicular distance from representative point to the validated Arabian Sea coastline in UTM 43N.",
        known_bias_or_limitation=(
            "Measures continental shoreline proximity only. Does NOT account for elevation, storm surge propagation models, "
            "bathymetry, or estuarine tidal flooding. Must NOT be reported as inundation probability."
        ),
    ),
    # ---------------------------------------------------------
    # Domain 5: Data Freshness & Quality Flags
    # ---------------------------------------------------------
    "rep_point_strictly_inside": FeatureMetadata(
        name="rep_point_strictly_inside",
        domain="Data Freshness and Quality",
        source_dataset="Derived via Shapely representative_point()",
        source_organization="Antigravity GIS pipeline",
        vintage="2026",
        unit="boolean",
        interpretation_direction="contextual",
        description="Quality assurance flag verifying representative point lies strictly inside polygon boundary interior.",
        known_bias_or_limitation="Currently 100% verified True across all 622 units.",
    ),
    "census_vintage": FeatureMetadata(
        name="census_vintage",
        domain="Data Freshness and Quality",
        source_dataset="Census 2011",
        source_organization="Office of the Registrar General, India",
        vintage="2011",
        unit="year",
        interpretation_direction="contextual",
        description="Baseline vintage year for demographic and infrastructure indicators.",
        known_bias_or_limitation="Fixed constant (2011). Signals over 15-year temporal gap requiring explicit caveat in index formulation.",
    ),
}


def get_feature_dictionary_df() -> pd.DataFrame:
    """Return the candidate feature dictionary as a pandas DataFrame."""
    records = [asdict(meta) for meta in CANDIDATE_FEATURE_DICTIONARY.values()]
    return pd.DataFrame(records)


def audit_feature_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Audit missingness, numeric distribution stats, and skewness for candidate features.

    Args:
        df: DataFrame containing candidate feature columns.

    Returns:
        DataFrame summarizing audit metrics for each candidate feature.
    """
    audit_rows = []
    n_rows = len(df)

    for col_name, meta in CANDIDATE_FEATURE_DICTIONARY.items():
        if col_name not in df.columns:
            audit_rows.append(
                {
                    "feature_name": col_name,
                    "domain": meta.domain,
                    "present_in_dataset": False,
                    "missing_count": n_rows,
                    "missing_pct": 100.0,
                    "dtype": "missing",
                    "min": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "max": np.nan,
                    "skewness": np.nan,
                }
            )
            continue

        s = df[col_name]
        n_missing = int(s.isna().sum())
        pct_missing = round(n_missing / n_rows * 100, 2)

        # If numeric
        if pd.api.types.is_numeric_dtype(s):
            clean_s = s.dropna().astype(float)
            if len(clean_s) > 0:
                audit_rows.append(
                    {
                        "feature_name": col_name,
                        "domain": meta.domain,
                        "present_in_dataset": True,
                        "missing_count": n_missing,
                        "missing_pct": pct_missing,
                        "dtype": str(s.dtype),
                        "min": round(float(clean_s.min()), 3),
                        "mean": round(float(clean_s.mean()), 3),
                        "median": round(float(clean_s.median()), 3),
                        "p75": round(float(clean_s.quantile(0.75)), 3),
                        "max": round(float(clean_s.max()), 3),
                        "skewness": round(float(clean_s.skew()), 3),
                    }
                )
            else:
                audit_rows.append(
                    {
                        "feature_name": col_name,
                        "domain": meta.domain,
                        "present_in_dataset": True,
                        "missing_count": n_missing,
                        "missing_pct": 100.0,
                        "dtype": str(s.dtype),
                        "min": np.nan,
                        "mean": np.nan,
                        "median": np.nan,
                        "p75": np.nan,
                        "max": np.nan,
                        "skewness": np.nan,
                    }
                )
        else:
            audit_rows.append(
                {
                    "feature_name": col_name,
                    "domain": meta.domain,
                    "present_in_dataset": True,
                    "missing_count": n_missing,
                    "missing_pct": pct_missing,
                    "dtype": str(s.dtype),
                    "min": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "max": np.nan,
                    "skewness": np.nan,
                }
            )

    return pd.DataFrame(audit_rows)


def detect_collinear_pairs(
    df: pd.DataFrame,
    threshold: float = 0.70,
    method: str = "spearman",
) -> pd.DataFrame:
    """Identify pairs of numeric candidate features with absolute correlation above threshold.

    Flags potential information leakage, direct redundancy, or multicollinearity.

    Args:
        df: DataFrame containing features.
        threshold: Absolute correlation threshold to flag (default 0.70).
        method: Correlation method ('pearson' or 'spearman').

    Returns:
        DataFrame of collinear pairs with correlation coefficient and interpretation notes.
    """
    numeric_cols = [
        col
        for col in CANDIDATE_FEATURE_DICTIONARY.keys()
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    corr_matrix = df[numeric_cols].corr(method=method)

    pairs = []
    seen = set()

    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i >= j:
                continue
            r = corr_matrix.loc[col1, col2]
            if not np.isnan(r) and abs(r) >= threshold:
                pairs.append(
                    {
                        "feature_1": col1,
                        "feature_2": col2,
                        "domain_1": CANDIDATE_FEATURE_DICTIONARY[col1].domain,
                        "domain_2": CANDIDATE_FEATURE_DICTIONARY[col2].domain,
                        "correlation": round(float(r), 3),
                        "correlation_type": method,
                        "is_severe": bool(abs(r) >= 0.85),
                        "leakage_risk_note": _interpret_collinear_pair(col1, col2, r),
                    }
                )

    return pd.DataFrame(pairs).sort_values("correlation", ascending=False)


def _interpret_collinear_pair(col1: str, col2: str, r: float) -> str:
    """Provide domain explanation for collinear feature pair."""
    pair_set = {col1, col2}
    if pair_set == {"total_population", "total_households"}:
        return "Direct demographic redundancy: population scale and household count move in lockstep; use only one."
    if pair_set == {"has_all_weather_road", "has_pucca_road"}:
        return "Infrastructure redundancy: pucca roads almost always provide all-weather access; combine or drop one."
    if "dist_to_nearest_phc_chc_km" in pair_set and "dist_to_nearest_facility_any_km" in pair_set:
        return "Structural hierarchy correlation: units close to Sub-Centres often cluster near PHCs; distinct service tiers."
    if "dist_to_nearest_phc_chc_km" in pair_set and "dist_to_nearest_hospital_km" in pair_set:
        return "Spatial gradient correlation: distance from coastal urban centres expands both primary and tertiary distances."
    return f"High pairwise correlation (|r| = {abs(r):.2f}); requires dimensionality reduction or single variable selection."
