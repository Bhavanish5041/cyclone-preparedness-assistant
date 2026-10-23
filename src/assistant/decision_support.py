"""Decision support service layer for cyclone preparedness planning.

Implements core analytical workflows:
- Geographic unit lookup (by name, PC11 ID, or SHRID)
- Granular indicator inspection (dimensions, raw proxies, provenance, limitations)
- Dynamic re-weighting and prioritization recomputation
- Scenario comparison and rank divergence analysis
- Discovered archetype catalog inspection and operational intervention guidance
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src.api.schemas import (
    ArchetypeCatalogResponse,
    ArchetypeSummary,
    HealthCheckResponse,
    PrioritizationScenarioRequest,
    PrioritizationScenarioResponse,
    RankDivergenceItem,
    ScenarioComparisonRequest,
    ScenarioComparisonResponse,
    UnitDetailResponse,
    UnitSummary,
    WeightConfiguration,
)
from src.features.prioritization import (
    PRIORITIZATION_DISCLAIMER,
    compute_coastal_hazard_dimension,
    compute_demographic_vulnerability_dimension,
    compute_healthcare_deficit_dimension,
    compute_infrastructure_deficit_dimension,
)
from src.models.clustering import (
    ARCHETYPE_DISCLAIMER,
    ARCHETYPE_LABELS_K4,
    compute_archetype_profiles,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/processed/coastal_vulnerability_archetypes.parquet")


class DecisionSupportService:
    """Singleton/service class encapsulating decision support operations on processed data."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self._df: Optional[gpd.GeoDataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load and cache the archetypes GeoDataFrame."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Underlying dataset not found at {self.data_path}")

        logger.info(f"Loading decision support dataset from {self.data_path}...")
        self._df = gpd.read_parquet(self.data_path)
        logger.info(f"Successfully loaded {len(self._df)} units into DecisionSupportService.")

    @property
    def df(self) -> gpd.GeoDataFrame:
        if self._df is None:
            self._load_data()
        return self._df  # type: ignore

    def get_health_status(self) -> HealthCheckResponse:
        """Return system health, record counts, and dataset metadata."""
        n_units = len(self.df)
        n_villages = int((self.df["unit_type"] == "VILLAGE").sum())
        n_towns = int((self.df["unit_type"] == "TOWN").sum())
        districts = sorted(self.df["district_name"].unique().tolist())

        return HealthCheckResponse(
            status="healthy",
            dataset_loaded=True,
            total_units=n_units,
            total_villages=n_villages,
            total_census_towns=n_towns,
            districts=districts,
            archetypes_count=4,
            version="1.0.0",
        )

    def search_units(
        self,
        query: str,
        district_filter: Optional[str] = None,
        limit: int = 15,
    ) -> List[UnitSummary]:
        """Search administrative units by name, PC11 ID, or SHRID.

        Args:
            query: Search string (e.g., 'Tallur', '608720', '11-29-569').
            district_filter: Optional filter ('Dakshina Kannada' or 'Udupi').
            limit: Maximum results to return.

        Returns:
            List of UnitSummary models matching query.
        """
        clean_q = query.strip().lower()
        if not clean_q:
            return []

        mask = (
            self.df["unit_name"].astype(str).str.lower().str.contains(clean_q)
            | self.df["pc11_id"].astype(str).str.lower().str.contains(clean_q)
            | self.df["shrid2"].astype(str).str.lower().str.contains(clean_q)
        )

        if district_filter:
            mask = mask & (self.df["district_name"].str.lower() == district_filter.strip().lower())

        matches = self.df[mask].head(limit)
        results = []
        for _, row in matches.iterrows():
            results.append(
                UnitSummary(
                    shrid2=str(row["shrid2"]),
                    pc11_id=str(row["pc11_id"]),
                    unit_name=str(row["unit_name"]),
                    unit_type=str(row["unit_type"]),
                    district_name=str(row["district_name"]),
                    priority_score=round(float(row["priority_score"]), 4),
                    priority_rank=int(row["priority_rank"]),
                    vulnerability_archetype_id=int(row["vulnerability_archetype_id"]),
                    vulnerability_archetype_label=str(row["vulnerability_archetype_label"]),
                )
            )
        return results

    def get_unit_detail(self, identifier: str) -> UnitDetailResponse:
        """Retrieve complete indicators, dimension scores, and archetype guidance for a unit.

        Args:
            identifier: Exact match for pc11_id, shrid2, or unit_name.

        Returns:
            UnitDetailResponse model.

        Raises:
            KeyError: If unit is not found.
        """
        clean_id = identifier.strip()
        match = self.df[
            (self.df["pc11_id"] == clean_id)
            | (self.df["shrid2"] == clean_id)
            | (self.df["unit_name"].str.lower() == clean_id.lower())
        ]

        if match.empty:
            raise KeyError(f"Administrative unit '{identifier}' not found in study area dataset.")

        row = match.iloc[0]
        c_id = int(row["vulnerability_archetype_id"])
        archetype_meta = ARCHETYPE_LABELS_K4.get(c_id, {})

        dim_scores = {
            "healthcare_deficit": round(float(row["dim_healthcare_deficit"]), 4),
            "demographic_vulnerability": round(float(row["dim_demographic_vulnerability"]), 4),
            "infrastructure_deficit": round(float(row["dim_infrastructure_deficit"]), 4),
            "coastal_exposure": round(float(row["dim_coastal_exposure"]), 4),
        }

        raw_indicators = {
            "dist_to_coastline_km": round(float(row["dist_to_coastline_km"]), 2),
            "dist_to_nearest_hospital_km": round(float(row["dist_to_nearest_hospital_km"]), 2),
            "nearest_hospital_name": str(row.get("nearest_hospital_name", "N/A")),
            "nearest_hospital_type": str(row.get("nearest_hospital_type", "N/A")),
            "dist_to_nearest_phc_chc_km": round(float(row["dist_to_nearest_phc_chc_km"]), 2),
            "nearest_phc_chc_name": str(row.get("nearest_phc_chc_name", "N/A")),
            "total_population": int(row["total_population"]),
            "total_households": int(row["total_households"]),
            "sc_st_ratio": round(float(row["sc_st_ratio"]), 4),
            "sex_ratio_female_per_1000_male": round(float(row["sex_ratio_female_per_1000_male"]), 1),
            "has_treated_tap_water": bool(row["has_treated_tap_water"]),
            "has_all_weather_road": bool(row["has_all_weather_road"]),
            "drainage_system": str(row["drainage_system"]),
        }

        archetype_dict = {
            "archetype_id": c_id,
            "label": archetype_meta.get("short_label", f"Cluster {c_id}"),
            "description": archetype_meta.get("detailed_description", "No description available."),
            "recommended_interventions": archetype_meta.get("primary_intervention_need", "Standard disaster response protocol."),
            "disclaimer": ARCHETYPE_DISCLAIMER,
        }

        provenance = {
            "census_vintage": int(row.get("census_vintage", 2011)),
            "healthcare_directory_source": "MoHFW/NHM (2015-19)",
            "coastline_source": "Natural Earth 10m (v5.1.1, 2022) in UTM Zone 43N",
            "representative_point_interior": bool(row.get("rep_point_strictly_inside", True)),
            "operational_limitations": (
                "Facility coordinates represent administrative registration, NOT confirmed active doctor staffing, "
                "functioning ICU beds, or backup generator diesel status during cyclonic power failure. "
                "Census statistics represent the 2011 decennial baseline (over 15 years old)."
            ),
        }

        return UnitDetailResponse(
            shrid2=str(row["shrid2"]),
            pc11_id=str(row["pc11_id"]),
            unit_name=str(row["unit_name"]),
            unit_type=str(row["unit_type"]),
            district_name=str(row["district_name"]),
            latitude=round(float(row["rep_point_latitude"]), 6),
            longitude=round(float(row["rep_point_longitude"]), 6),
            dimension_scores=dim_scores,
            raw_indicators=raw_indicators,
            priority_score=round(float(row["priority_score"]), 4),
            priority_rank=int(row["priority_rank"]),
            archetype=archetype_dict,
            provenance_and_limitations=provenance,
        )

    def compute_prioritization_scenario(
        self,
        request: PrioritizationScenarioRequest,
    ) -> PrioritizationScenarioResponse:
        """Dynamically compute priority scores and ranks under custom weights.

        Args:
            request: PrioritizationScenarioRequest payload.

        Returns:
            PrioritizationScenarioResponse with top ranked units and score distributions.
        """
        weights = request.weights.get_normalized_weights()
        temp_df = self.df.copy()

        # If population scale is requested, recompute demographic dimension
        if request.weights.include_population_scale:
            dim_demo = compute_demographic_vulnerability_dimension(temp_df, include_population_scale=True)
        else:
            dim_demo = temp_df["dim_demographic_vulnerability"]

        # Calculate combined score
        custom_score = (
            weights["healthcare"] * temp_df["dim_healthcare_deficit"]
            + weights["demographic"] * dim_demo
            + weights["infrastructure"] * temp_df["dim_infrastructure_deficit"]
            + weights["coastal"] * temp_df["dim_coastal_exposure"]
        ).round(4)

        temp_df["custom_score"] = custom_score
        temp_df["custom_rank"] = custom_score.rank(ascending=False, method="min").astype(int)

        # Optional district filter
        if request.district_filter:
            temp_df = temp_df[temp_df["district_name"].str.lower() == request.district_filter.strip().lower()]

        score_stats = {
            "min": round(float(custom_score.min()), 4),
            "median": round(float(custom_score.median()), 4),
            "mean": round(float(custom_score.mean()), 4),
            "max": round(float(custom_score.max()), 4),
            "std": round(float(custom_score.std()), 4),
        }

        top_df = temp_df.sort_values("custom_score", ascending=False).head(request.top_n)
        top_units = []
        for _, row in top_df.iterrows():
            top_units.append(
                UnitSummary(
                    shrid2=str(row["shrid2"]),
                    pc11_id=str(row["pc11_id"]),
                    unit_name=str(row["unit_name"]),
                    unit_type=str(row["unit_type"]),
                    district_name=str(row["district_name"]),
                    priority_score=round(float(row["custom_score"]), 4),
                    priority_rank=int(row["custom_rank"]),
                    vulnerability_archetype_id=int(row["vulnerability_archetype_id"]),
                    vulnerability_archetype_label=str(row["vulnerability_archetype_label"]),
                )
            )

        return PrioritizationScenarioResponse(
            scenario_name=request.scenario_name,
            weights_applied=weights,
            include_population_scale=request.weights.include_population_scale,
            total_units_evaluated=len(self.df),
            score_statistics=score_stats,
            top_units=top_units,
            disclaimer=PRIORITIZATION_DISCLAIMER,
        )

    def compare_scenarios(
        self,
        request: ScenarioComparisonRequest,
    ) -> ScenarioComparisonResponse:
        """Compare rankings between two weighting scenarios and identify major divergence."""
        w_a = request.scenario_a_weights.get_normalized_weights()
        w_b = request.scenario_b_weights.get_normalized_weights()

        score_a = (
            w_a["healthcare"] * self.df["dim_healthcare_deficit"]
            + w_a["demographic"] * self.df["dim_demographic_vulnerability"]
            + w_a["infrastructure"] * self.df["dim_infrastructure_deficit"]
            + w_a["coastal"] * self.df["dim_coastal_exposure"]
        )
        rank_a = score_a.rank(ascending=False, method="min").astype(int)

        score_b = (
            w_b["healthcare"] * self.df["dim_healthcare_deficit"]
            + w_b["demographic"] * self.df["dim_demographic_vulnerability"]
            + w_b["infrastructure"] * self.df["dim_infrastructure_deficit"]
            + w_b["coastal"] * self.df["dim_coastal_exposure"]
        )
        rank_b = score_b.rank(ascending=False, method="min").astype(int)

        sr, _ = spearmanr(rank_a, rank_b)
        kt, _ = kendalltau(rank_a, rank_b)

        # Rank shift: rank_a - rank_b (positive means unit ranked higher/closer to 1 in B than A)
        shift = rank_a - rank_b
        abs_shift = np.abs(shift)

        comp_df = pd.DataFrame(
            {
                "shrid2": self.df["shrid2"],
                "unit_name": self.df["unit_name"],
                "district_name": self.df["district_name"],
                "vulnerability_archetype_label": self.df["vulnerability_archetype_label"],
                "rank_a": rank_a,
                "rank_b": rank_b,
                "rank_shift": shift,
                "abs_shift": abs_shift,
            }
        )

        top_shifts = comp_df.sort_values("abs_shift", ascending=False).head(request.top_divergent_count)
        divergent_items = []
        for _, row in top_shifts.iterrows():
            sh = int(row["rank_shift"])
            dir_str = "Surged in Priority (in B)" if sh > 0 else "Dropped in Priority (in B)"
            divergent_items.append(
                RankDivergenceItem(
                    shrid2=str(row["shrid2"]),
                    unit_name=str(row["unit_name"]),
                    district_name=str(row["district_name"]),
                    vulnerability_archetype_label=str(row["vulnerability_archetype_label"]),
                    rank_scenario_a=int(row["rank_a"]),
                    rank_scenario_b=int(row["rank_b"]),
                    rank_shift=sh,
                    direction=dir_str,
                )
            )

        note = (
            f"Comparing '{request.scenario_a_name}' vs '{request.scenario_b_name}'. "
            f"Spearman rank correlation is {sr:.3f}. A positive rank shift indicates that a community "
            f"gains higher priority under Scenario B."
        )

        return ScenarioComparisonResponse(
            scenario_a_name=request.scenario_a_name,
            scenario_b_name=request.scenario_b_name,
            spearman_rank_correlation=round(float(sr), 4),
            kendall_tau_correlation=round(float(kt), 4),
            mean_absolute_rank_shift=round(float(abs_shift.mean()), 1),
            max_rank_shift=int(abs_shift.max()),
            top_divergent_units=divergent_items,
            methodological_note=note,
        )

    def get_archetype_catalog(self) -> ArchetypeCatalogResponse:
        """Return complete descriptive catalog and summary statistics of all archetypes."""
        profiles = compute_archetype_profiles(self.df, self.df["vulnerability_archetype_id"].values)
        summaries = []

        for c_id, row in profiles.iterrows():
            c_int = int(c_id)
            meta = ARCHETYPE_LABELS_K4.get(c_int, {})
            summaries.append(
                ArchetypeSummary(
                    archetype_id=c_int,
                    short_label=meta.get("short_label", f"Cluster {c_int}"),
                    detailed_description=meta.get("detailed_description", "No description available."),
                    primary_intervention_need=meta.get("primary_intervention_need", "Standard response."),
                    unit_count=int(row["unit_count"]),
                    percentage_of_total=round(float(row["pct_of_total"]), 1),
                    count_dakshina_kannada=int(row["count_dk"]),
                    count_udupi=int(row["count_udupi"]),
                    dimension_means={
                        "healthcare_deficit": round(float(row["dim_healthcare_deficit_mean"]), 4),
                        "demographic_vulnerability": round(float(row["dim_demographic_vulnerability_mean"]), 4),
                        "infrastructure_deficit": round(float(row["dim_infrastructure_deficit_mean"]), 4),
                        "coastal_exposure": round(float(row["dim_coastal_exposure_mean"]), 4),
                    },
                    raw_indicator_means={
                        "dist_to_coastline_km": round(float(row.get("dist_to_coastline_km_mean", 0)), 1),
                        "dist_to_hospital_km": round(float(row.get("dist_to_hospital_km_mean", 0)), 1),
                        "dist_to_phc_km": round(float(row.get("dist_to_phc_chc_km_mean", 0)), 1),
                        "sc_st_ratio": round(float(row.get("sc_st_ratio_mean", 0)), 3),
                    },
                )
            )

        return ArchetypeCatalogResponse(
            total_archetypes=len(summaries),
            archetypes=summaries,
            clustering_methodology=(
                "Partioning via K-Means (k=4) on standardized z-score transformed dimension scores "
                "(Healthcare Deficit, Demographic Vulnerability, Infrastructure Deficit, Coastal Proximity)."
            ),
            clustering_disclaimer=ARCHETYPE_DISCLAIMER,
        )
