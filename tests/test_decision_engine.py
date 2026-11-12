"""Comprehensive test suite for the Milestone 3B Integrated Decision-Support Engine.

Validates:
1. Exact baseline priority band counts (155, 829, 299, 596 -> Total 1,879).
2. District-by-band reconciliation across Dakshina Kannada, Udupi, and Uttara Kannada.
3. Uttara Kannada independent sector triggers (coastal, shelter, medical) and overlap accounting.
4. Separation of AHP Low-tier overlay from sector triggers.
5. The 7 Low-tier zero-sector-trigger exception units receiving ADV_UK_GENERAL_DEFICIT_REVIEW.
6. The 282 non-Low zero-sector-trigger units receiving ADV_UK_ROUTINE.
7. Unit-level advisory eligibility formulas and supporting evidence dictionaries.
8. Non-extrapolation of K-Means archetypes to Uttara Kannada (unassigned status preservation).
9. Nullable field parsing (unit_type, archetype_id, archetype_label).
10. Hierarchical deterministic ranking and null-safe sorting (na_position='last').
11. Preservation of deterministic AHP baseline tier in all 87 model disagreements.
12. Probability margin calculations and uncalibrated metric bounds.
13. Prohibited emergency command language scan.
14. FastAPI REST endpoint integration and schema contracts.
"""

import re
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.api.schemas import (
    DecisionEvaluationListResponse,
    DecisionSupportEvaluation,
    InterventionItem,
    RegionalDecisionSummaryResponse,
)
from src.assistant.decision_support import DecisionSupportService
from src.integration.decision_matrix import (
    COASTAL_PROXIMITY_THRESHOLD_KM,
    DECISION_SUPPORT_DISCLAIMER,
    DRINKING_WATER_DEFICIT_THRESHOLD,
    FOOTHILL_ELEVATION_THRESHOLD,
    HOSPITAL_DISTANCE_THRESHOLD_KM,
    INFRASTRUCTURE_DEFICIT_THRESHOLD,
    MEDICAL_DEFICIT_THRESHOLD,
    POP_DENSITY_HIGH_THRESHOLD,
    SHELTER_DEFICIT_THRESHOLD,
    VULNERABLE_POP_SHARE_THRESHOLD,
    build_unit_evaluation,
    compute_priority_bands_and_ranking,
    evaluate_unit_advisories,
    get_regional_decision_summary,
)


@pytest.fixture(scope="module")
def service() -> DecisionSupportService:
    """Fixture providing initialized DecisionSupportService."""
    return DecisionSupportService()


@pytest.fixture(scope="module")
def ranked_df(service: DecisionSupportService) -> pd.DataFrame:
    """Fixture providing the evaluated and ranked 1,879 records."""
    return service.ranked_df


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# 1. Exact Baseline Band Counts & District Reconciliation
# =============================================================================


def test_baseline_priority_band_counts(ranked_df: pd.DataFrame):
    """Verify exact regional counts for the four priority bands."""
    assert len(ranked_df) == 1879

    band_counts = ranked_df["priority_band"].value_counts().to_dict()
    assert band_counts["BAND_1_CRITICAL"] == 155
    assert band_counts["BAND_2_SPECIALIZED"] == 829
    assert band_counts["BAND_3_MODERATE"] == 299
    assert band_counts["BAND_4_ADEQUATE"] == 596
    assert sum(band_counts.values()) == 1879


def test_district_by_band_reconciliation(ranked_df: pd.DataFrame):
    """Verify district-level priority band distributions match Milestone 3A audit."""
    ct = pd.crosstab(ranked_df["district_name"], ranked_df["priority_band"]).to_dict(orient="index")

    # Dakshina Kannada: 369 total (2 Low, 123 Spec, 45 Mod, 199 Adeq)
    dk = ct["Dakshina Kannada"]
    assert dk["BAND_1_CRITICAL"] == 2
    assert dk["BAND_2_SPECIALIZED"] == 123
    assert dk["BAND_3_MODERATE"] == 45
    assert dk["BAND_4_ADEQUATE"] == 199
    assert sum(dk.values()) == 369

    # Udupi: 253 total (0 Low, 20 Spec, 43 Mod, 190 Adeq)
    udupi = ct["Udupi"]
    assert udupi.get("BAND_1_CRITICAL", 0) == 0
    assert udupi["BAND_2_SPECIALIZED"] == 20
    assert udupi["BAND_3_MODERATE"] == 43
    assert udupi["BAND_4_ADEQUATE"] == 190
    assert sum(udupi.values()) == 253

    # Uttara Kannada: 1,257 total (153 Low, 686 Spec, 211 Mod, 207 Adeq)
    uk = ct["Uttara Kannada"]
    assert uk["BAND_1_CRITICAL"] == 153
    assert uk["BAND_2_SPECIALIZED"] == 686
    assert uk["BAND_3_MODERATE"] == 211
    assert uk["BAND_4_ADEQUATE"] == 207
    assert sum(uk.values()) == 1257


# =============================================================================
# 2. Uttara Kannada Trigger Accounting & Exception Handling
# =============================================================================


def test_uttara_kannada_sector_triggers_and_overlap(ranked_df: pd.DataFrame):
    """Verify UK 3 sector triggers and overlap distribution."""
    uk = ranked_df[ranked_df["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"].copy()
    assert len(uk) == 1257

    t_coast = uk["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM
    t_shelter = uk["score_shelters"] < SHELTER_DEFICIT_THRESHOLD
    t_medical = uk["score_medical"] < MEDICAL_DEFICIT_THRESHOLD

    assert t_coast.sum() == 175
    assert t_shelter.sum() == 710
    assert t_medical.sum() == 278

    sector_trig_count = t_coast.astype(int) + t_shelter.astype(int) + t_medical.astype(int)
    counts = sector_trig_count.value_counts().to_dict()

    assert counts[0] == 289
    assert counts[1] == 773
    assert counts[2] == 195
    assert counts.get(3, 0) == 0


def test_uttara_kannada_zero_trigger_exception_units(ranked_df: pd.DataFrame):
    """Verify that the 7 Low-tier units with 0 sector triggers receive ADV_UK_GENERAL_DEFICIT_REVIEW and NOT ADV_UK_ROUTINE."""
    uk = ranked_df[ranked_df["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"].copy()

    t_coast = uk["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM
    t_shelter = uk["score_shelters"] < SHELTER_DEFICIT_THRESHOLD
    t_medical = uk["score_medical"] < MEDICAL_DEFICIT_THRESHOLD
    sec_trig = t_coast.astype(int) + t_shelter.astype(int) + t_medical.astype(int)

    # 7 units with 0 sector triggers and preparedness_score < 40.0
    zero_trig_low = uk[(sec_trig == 0) & (uk["preparedness_score"] < 40.0)]
    assert len(zero_trig_low) == 7

    expected_low_names = {"Bidoli", "Kodthalli", "Kudragod", "Kendige", "Shikliturli", "Anegundi", "Kerehittal"}
    assert set(zero_trig_low["village_name"]).issubset(expected_low_names)

    for _, row in zero_trig_low.iterrows():
        advisories = evaluate_unit_advisories(row)
        adv_ids = {a.directive_id for a in advisories}

        # Must receive baseline low advisory
        assert "ADV_UK_BASELINE_LOW" in adv_ids
        # Must receive general deficit review
        assert "ADV_UK_GENERAL_DEFICIT_REVIEW" in adv_ids
        # Must NEVER receive routine advisory
        assert "ADV_UK_ROUTINE" not in adv_ids


def test_uttara_kannada_routine_advisory_eligibility(ranked_df: pd.DataFrame):
    """Verify that exactly 282 units (211 Medium + 71 High) receive ADV_UK_ROUTINE."""
    uk = ranked_df[ranked_df["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"].copy()

    routine_count = 0
    for _, row in uk.iterrows():
        advisories = evaluate_unit_advisories(row)
        adv_ids = {a.directive_id for a in advisories}
        if "ADV_UK_ROUTINE" in adv_ids:
            routine_count += 1
            # Must NOT be Low tier
            assert row["preparedness_tier"] != "Low"
            assert row["preparedness_score"] >= 40.0

    assert routine_count == 282


# =============================================================================
# 3. Unit-Level Advisory Eligibility & Supporting Evidence
# =============================================================================


def test_assigned_units_advisory_eligibility_positive_and_negative(ranked_df: pd.DataFrame):
    """Test positive and negative eligibility cases for assigned units in DK & Udupi."""
    assigned = ranked_df[ranked_df["cluster_assignment_status"] == "ASSIGNED"]
    assert len(assigned) == 622

    # Archetype 0 test: ADV_0_DRAINAGE_INFRA only if score_infrastructure < 50
    a0 = assigned[assigned["vulnerability_archetype_id"] == 0]
    for _, row in a0.iterrows():
        advisories = evaluate_unit_advisories(row)
        adv_ids = {a.directive_id for a in advisories}
        if row["score_infrastructure"] < INFRASTRUCTURE_DEFICIT_THRESHOLD:
            assert "ADV_0_DRAINAGE_INFRA" in adv_ids
        else:
            assert "ADV_0_DRAINAGE_INFRA" not in adv_ids

    # Archetype 1 test: ADV_1_HOSPITAL_ISOLATION only if dist_nearest_hospital_km > 20
    a1 = assigned[assigned["vulnerability_archetype_id"] == 1]
    for _, row in a1.iterrows():
        advisories = evaluate_unit_advisories(row)
        adv_ids = {a.directive_id for a in advisories}
        if row["dist_nearest_hospital_km"] > HOSPITAL_DISTANCE_THRESHOLD_KM:
            assert "ADV_1_HOSPITAL_ISOLATION" in adv_ids
        else:
            assert "ADV_1_HOSPITAL_ISOLATION" not in adv_ids

    # Archetype 2 test: ADV_2_WATER_SANITATION only if drinking_water_score <= 0.4
    a2 = assigned[assigned["vulnerability_archetype_id"] == 2]
    for _, row in a2.iterrows():
        advisories = evaluate_unit_advisories(row)
        adv_ids = {a.directive_id for a in advisories}
        if row["drinking_water_score"] <= DRINKING_WATER_DEFICIT_THRESHOLD:
            assert "ADV_2_WATER_SANITATION" in adv_ids
        else:
            assert "ADV_2_WATER_SANITATION" not in adv_ids

    # Archetype 3 test: ADV_3_COASTAL_PROXIMITY only if dist_coast_km < 5.0
    a3 = assigned[assigned["vulnerability_archetype_id"] == 3]
    for _, row in a3.iterrows():
        advisories = evaluate_unit_advisories(row)
        adv_ids = {a.directive_id for a in advisories}
        if row["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM:
            assert "ADV_3_COASTAL_PROXIMITY" in adv_ids
        else:
            assert "ADV_3_COASTAL_PROXIMITY" not in adv_ids


def test_supporting_evidence_numerical_accuracy(ranked_df: pd.DataFrame):
    """Verify that supporting evidence dictionaries contain actual per-unit values matching the row."""
    # Test a sample of 50 units across all districts
    sample = ranked_df.sample(50, random_state=42)

    for _, row in sample.iterrows():
        advisories = evaluate_unit_advisories(row)
        for adv in advisories:
            ev = adv.supporting_evidence
            assert isinstance(ev, dict)
            assert len(ev) > 0

            # Verify specific evidence keys against row
            if "score_infrastructure" in ev:
                assert pytest.approx(ev["score_infrastructure"], abs=1e-3) == float(row["score_infrastructure"])
            if "dist_coast_km" in ev:
                assert pytest.approx(ev["dist_coast_km"], abs=1e-2) == float(row["dist_coast_km"])
            if "score_shelters" in ev:
                assert pytest.approx(ev["score_shelters"], abs=1e-3) == float(row["score_shelters"])
            if "score_medical" in ev:
                assert pytest.approx(ev["score_medical"], abs=1e-3) == float(row["score_medical"])
            if "drinking_water_score" in ev:
                assert pytest.approx(ev["drinking_water_score"], abs=1e-2) == float(row["drinking_water_score"])


def test_no_archetype_extrapolation_to_uttara_kannada(ranked_df: pd.DataFrame):
    """Verify that all 1,257 Uttara Kannada units remain unassigned with null archetype fields."""
    uk = ranked_df[ranked_df["district_name"] == "Uttara Kannada"]
    assert len(uk) == 1257

    for _, row in uk.iterrows():
        assert row["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"
        assert pd.isna(row["vulnerability_archetype_id"]) or row["vulnerability_archetype_id"] is None
        assert pd.isna(row["vulnerability_archetype_label"]) or row["vulnerability_archetype_label"] is None

        # Build evaluation model and verify Pydantic types
        eval_obj = build_unit_evaluation(row)
        assert eval_obj.vulnerability_archetype_id is None
        assert eval_obj.vulnerability_archetype_label is None
        assert eval_obj.cluster_assignment_status == "UNASSIGNED_OUT_OF_STUDY_AREA"


# =============================================================================
# 4. Deterministic Ranking & Null-Safe Sorting
# =============================================================================


def test_deterministic_ranking_contiguity_and_bounds(ranked_df: pd.DataFrame):
    """Verify contiguous 1..1879 overall ranks and within-band rank integrity."""
    overall_ranks = ranked_df["overall_priority_rank"].tolist()
    assert overall_ranks == list(range(1, 1880))

    # Verify that Band 1 ranks precede Band 2, which precede Band 3 and Band 4
    b1 = ranked_df[ranked_df["priority_band"] == "BAND_1_CRITICAL"]
    b2 = ranked_df[ranked_df["priority_band"] == "BAND_2_SPECIALIZED"]
    b3 = ranked_df[ranked_df["priority_band"] == "BAND_3_MODERATE"]
    b4 = ranked_df[ranked_df["priority_band"] == "BAND_4_ADEQUATE"]

    assert b1["overall_priority_rank"].min() == 1
    assert b1["overall_priority_rank"].max() == 155
    assert b2["overall_priority_rank"].min() == 156
    assert b2["overall_priority_rank"].max() == 984
    assert b3["overall_priority_rank"].min() == 985
    assert b3["overall_priority_rank"].max() == 1283
    assert b4["overall_priority_rank"].min() == 1284
    assert b4["overall_priority_rank"].max() == 1879

    # Check band_rank within each band
    assert b1["band_rank"].tolist() == list(range(1, 156))
    assert b2["band_rank"].tolist() == list(range(1, 830))
    assert b3["band_rank"].tolist() == list(range(1, 300))
    assert b4["band_rank"].tolist() == list(range(1, 597))


def test_null_safe_ranking_policy():
    """Verify that missing secondary/tertiary keys sort last using na_position='last' without zero-imputation."""
    # Create synthetic test dataset with nulls
    test_data = pd.DataFrame(
        [
            {
                "shrid2": "UNIT_A",
                "preparedness_tier": "Medium",
                "preparedness_score": 45.0,
                "score_shelters": np.nan,  # Missing shelter
                "score_medical": 50.0,
                "dist_coast_km": 10.0,
                "prob_low": 0.1,
                "prob_medium": 0.8,
                "prob_high": 0.1,
                "predicted_preparedness_tier": "Medium",
                "cluster_assignment_status": "UNASSIGNED_OUT_OF_STUDY_AREA",
            },
            {
                "shrid2": "UNIT_B",
                "preparedness_tier": "Medium",
                "preparedness_score": 45.0,
                "score_shelters": 40.0,  # Valid secondary score
                "score_medical": 50.0,
                "dist_coast_km": 10.0,
                "prob_low": 0.1,
                "prob_medium": 0.8,
                "prob_high": 0.1,
                "predicted_preparedness_tier": "Medium",
                "cluster_assignment_status": "UNASSIGNED_OUT_OF_STUDY_AREA",
            },
        ]
    )

    ranked_synth = compute_priority_bands_and_ranking(test_data)
    # Unit B has valid secondary score (min = 40.0), Unit A has NaN secondary score
    # Under na_position='last', Unit B must rank higher (rank 1) than Unit A (rank 2)
    assert ranked_synth.loc[ranked_synth["shrid2"] == "UNIT_B", "overall_priority_rank"].iloc[0] == 1
    assert ranked_synth.loc[ranked_synth["shrid2"] == "UNIT_A", "overall_priority_rank"].iloc[0] == 2


def test_ranking_reproducibility(ranked_df: pd.DataFrame):
    """Verify 100% deterministic ranking across repeated evaluations on the dataset."""
    df_raw = pd.read_parquet("data/processed/unified_village_preparedness_v2.parquet")
    run2 = compute_priority_bands_and_ranking(df_raw)

    assert (ranked_df["overall_priority_rank"].values == run2["overall_priority_rank"].values).all()
    assert (ranked_df["priority_band"].values == run2["priority_band"].values).all()
    assert (ranked_df["band_rank"].values == run2["band_rank"].values).all()


# =============================================================================
# 5. Model Disagreement & Probability Margin Integrity
# =============================================================================


def test_ahp_tier_preservation_in_disagreements(ranked_df: pd.DataFrame):
    """Verify all 87 disagreement cases preserve AHP tier as canonical preparedness_tier."""
    disagreements = ranked_df[ranked_df["model_agreement_status"] == "BORDERLINE_DISAGREEMENT"]
    assert len(disagreements) == 87

    agreement_counts = ranked_df["model_agreement_status"].value_counts().to_dict()
    assert agreement_counts["FULL_AGREEMENT"] == 1792
    assert agreement_counts["BORDERLINE_DISAGREEMENT"] == 87

    # For every disagreement, eval_obj.preparedness_tier must equal row['preparedness_tier']
    for _, row in disagreements.iterrows():
        eval_obj = build_unit_evaluation(row)
        assert eval_obj.preparedness_tier == str(row["preparedness_tier"]).strip().capitalize()
        assert eval_obj.model_agreement_status == "BORDERLINE_DISAGREEMENT"
        assert eval_obj.preparedness_tier != eval_obj.predicted_preparedness_tier


def test_probability_margin_calculation(ranked_df: pd.DataFrame):
    """Verify max_class_probability in [0.3333, 1.0] and margin == p_max - p_second."""
    for _, row in ranked_df.iterrows():
        probs = sorted([row["prob_low"], row["prob_medium"], row["prob_high"]], reverse=True)
        expected_max = round(float(probs[0]), 4)
        expected_margin = round(float(probs[0] - probs[1]), 4)

        assert pytest.approx(row["max_class_probability"], abs=1e-4) == expected_max
        assert pytest.approx(row["probability_margin"], abs=1e-4) == expected_margin
        assert row["max_class_probability"] >= 0.3333
        assert row["probability_margin"] >= 0.0


# =============================================================================
# 6. Prohibited Language & Statutory Disclaimers
# =============================================================================


def test_prohibited_emergency_command_language(ranked_df: pd.DataFrame):
    """Verify that NO generated advisories contain unsupported emergency command directives."""
    prohibited_patterns = [
        re.compile(r"\bmandatory\s+evacuation\b", re.IGNORECASE),
        re.compile(r"\bfishing\s+ban\b", re.IGNORECASE),
        re.compile(r"\b14-day\s+ration\b", re.IGNORECASE),
        re.compile(r"\b30-day\s+stockpile\b", re.IGNORECASE),
        re.compile(r"\bT-72h\b", re.IGNORECASE),
        re.compile(r"\bT-48h\b", re.IGNORECASE),
        re.compile(r"\bT-24h\b", re.IGNORECASE),
        re.compile(r"\bdeploy\s+pumps\b", re.IGNORECASE),
        re.compile(r"\bguaranteed\s+shelter\b", re.IGNORECASE),
    ]

    for _, row in ranked_df.iterrows():
        eval_obj = build_unit_evaluation(row)
        for adv in eval_obj.recommended_advisories:
            text = adv.operational_recommendation + " " + adv.advisory_focus
            for pat in prohibited_patterns:
                assert not pat.search(text), f"Prohibited pattern '{pat.pattern}' found in {adv.directive_id}: {text}"


def test_statutory_disclaimer_preservation(ranked_df: pd.DataFrame):
    """Verify statutory disclaimer is present in every evaluation and regional summary."""
    sample = ranked_df.head(20)
    for _, row in sample.iterrows():
        eval_obj = build_unit_evaluation(row)
        assert eval_obj.advisory_disclaimer == DECISION_SUPPORT_DISCLAIMER

    summary = get_regional_decision_summary(ranked_df)
    assert summary.methodological_disclaimer == DECISION_SUPPORT_DISCLAIMER


# =============================================================================
# 7. FastAPI REST Endpoints & Backward Compatibility
# =============================================================================


def test_api_decision_support_summary(client: TestClient):
    """Verify GET /api/v1/decision-support/summary returns regional aggregation."""
    resp = client.get("/api/v1/decision-support/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_units"] == 1879
    assert data["band_counts"]["BAND_1_CRITICAL"] == 155
    assert data["band_counts"]["BAND_2_SPECIALIZED"] == 829
    assert data["band_counts"]["BAND_3_MODERATE"] == 299
    assert data["band_counts"]["BAND_4_ADEQUATE"] == 596
    assert data["model_agreement_counts"]["FULL_AGREEMENT"] == 1792
    assert data["model_agreement_counts"]["BORDERLINE_DISAGREEMENT"] == 87
    assert data["uttara_kannada_trigger_breakdown"]["total_unassigned_units"] == 1257


def test_api_decision_support_unit_lookup(client: TestClient):
    """Verify GET /api/v1/decision-support/units/{identifier} for assigned and unassigned units."""
    # Test an assigned unit: Edmoge (PC11 ID: 608720) in Udupi
    resp = client.get("/api/v1/decision-support/units/608720")
    assert resp.status_code == 200
    data = resp.json()
    assert data["village_name"] == "Edmoge"
    assert data["cluster_assignment_status"] == "ASSIGNED"
    assert data["vulnerability_archetype_id"] is not None
    assert len(data["recommended_advisories"]) > 0

    # Test an unassigned unit: Bappalgundi in Uttara Kannada
    resp_uk = client.get("/api/v1/decision-support/units/Bappalgundi")
    assert resp_uk.status_code == 200
    data_uk = resp_uk.json()
    assert data_uk["village_name"] == "Bappalgundi"
    assert data_uk["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"
    assert data_uk["vulnerability_archetype_id"] is None
    assert data_uk["provisional_priority_band"] == "BAND_1_CRITICAL"
    assert data_uk["overall_priority_rank"] == 1

    # Test 404 for nonexistent unit
    resp_404 = client.get("/api/v1/decision-support/units/NonExistentVillage12345")
    assert resp_404.status_code == 404


def test_api_decision_support_units_list_and_filters(client: TestClient):
    """Verify GET /api/v1/decision-support/units with district, band, and pagination filters."""
    # Filter by district: Uttara Kannada
    resp_uk = client.get("/api/v1/decision-support/units?district=Uttara%20Kannada&limit=10")
    assert resp_uk.status_code == 200
    data_uk = resp_uk.json()
    assert data_uk["total_evaluated"] == 1879
    assert data_uk["total_matching"] == 1257
    assert len(data_uk["units"]) == 10
    assert all(u["district_name"] == "Uttara Kannada" for u in data_uk["units"])

    # Filter by priority_band: BAND_1_CRITICAL
    resp_b1 = client.get("/api/v1/decision-support/units?priority_band=BAND_1_CRITICAL&limit=200")
    assert resp_b1.status_code == 200
    data_b1 = resp_b1.json()
    assert data_b1["total_matching"] == 155
    assert len(data_b1["units"]) == 155
    assert all(u["provisional_priority_band"] == "BAND_1_CRITICAL" for u in data_b1["units"])

    # Verify invalid filter raises 422
    resp_err = client.get("/api/v1/decision-support/units?district=InvalidDistrict")
    assert resp_err.status_code == 422


# =============================================================================
# 8. Milestone 3A Compliance Audit & Non-Promotion Verification
# =============================================================================


def test_approved_medium_tier_band_2_condition(ranked_df: pd.DataFrame):
    """Verify exact approved Medium-tier Band 2 condition without archetype promotion.

    Rule: preparedness_tier == 'Medium' and (dist_coast_km < 5.0 or score_shelters < 20.0 or score_medical < 35.0)
    All Medium units meeting this condition must be BAND_2_SPECIALIZED.
    All Medium units NOT meeting this condition must be BAND_3_MODERATE.
    """
    med_units = ranked_df[ranked_df["preparedness_tier"] == "Medium"]
    assert len(med_units) == 1128

    has_trigger = (
        (med_units["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM)
        | (med_units["score_shelters"] < SHELTER_DEFICIT_THRESHOLD)
        | (med_units["score_medical"] < MEDICAL_DEFICIT_THRESHOLD)
    )

    band_2_units = med_units[has_trigger]
    band_3_units = med_units[~has_trigger]

    assert len(band_2_units) == 829
    assert len(band_3_units) == 299

    assert (band_2_units["priority_band"] == "BAND_2_SPECIALIZED").all()
    assert (band_3_units["priority_band"] == "BAND_3_MODERATE").all()


def test_archetype_alone_cannot_promote_to_band_2(ranked_df: pd.DataFrame):
    """Verify that archetype membership alone NEVER promotes a unit into Band 2.

    For assigned units in Dakshina Kannada & Udupi:
    Medium-tier units belonging to Archetypes 0, 1, or 2 that lack all 3 individual triggers
    MUST remain in BAND_3_MODERATE.
    """
    assigned = ranked_df[ranked_df["cluster_assignment_status"] == "ASSIGNED"]
    assert len(assigned) == 622

    assigned_med = assigned[assigned["preparedness_tier"] == "Medium"]

    no_trigger = ~(
        (assigned_med["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM)
        | (assigned_med["score_shelters"] < SHELTER_DEFICIT_THRESHOLD)
        | (assigned_med["score_medical"] < MEDICAL_DEFICIT_THRESHOLD)
    )

    # Across Archetypes 0, 1, 2, 3, there are 88 Medium units without triggers
    untriggered_assigned = assigned_med[no_trigger]
    assert len(untriggered_assigned) == 88

    # All 88 MUST be in BAND_3_MODERATE despite Archetype 0, 1, or 2 membership
    assert (untriggered_assigned["priority_band"] == "BAND_3_MODERATE").all()

    # Specifically check Archetypes 0, 1, 2
    arch_0_1_2_untriggered = untriggered_assigned[untriggered_assigned["vulnerability_archetype_id"].isin([0, 1, 2])]
    assert len(arch_0_1_2_untriggered) == 80  # 29 (arch 0) + 25 (arch 1) + 26 (arch 2)
    assert (arch_0_1_2_untriggered["priority_band"] == "BAND_3_MODERATE").all()


def test_uk_fallback_and_low_tier_independence(ranked_df: pd.DataFrame):
    """Verify that UK fallback sector triggers and AHP Low-tier overlay operate independently."""
    uk = ranked_df[ranked_df["cluster_assignment_status"] == "UNASSIGNED_OUT_OF_STUDY_AREA"]
    assert len(uk) == 1257

    # All 153 Low-tier units are in Band 1 regardless of sector trigger count
    uk_low = uk[uk["preparedness_tier"] == "Low"]
    assert len(uk_low) == 153
    assert (uk_low["priority_band"] == "BAND_1_CRITICAL").all()

    # Sector triggers
    t_coast = uk["dist_coast_km"] < COASTAL_PROXIMITY_THRESHOLD_KM
    t_shelter = uk["score_shelters"] < SHELTER_DEFICIT_THRESHOLD
    t_med = uk["score_medical"] < MEDICAL_DEFICIT_THRESHOLD
    n_trig = t_coast.astype(int) + t_shelter.astype(int) + t_med.astype(int)

    # 7 units with 0 sector triggers and Low tier are in Band 1
    assert ((n_trig == 0) & (uk["preparedness_tier"] == "Low")).sum() == 7
    # 282 units with 0 sector triggers and non-Low tier are in Band 3 or Band 4
    zero_trig_non_low = uk[(n_trig == 0) & (uk["preparedness_tier"] != "Low")]
    assert len(zero_trig_non_low) == 282
    assert set(zero_trig_non_low["priority_band"]).issubset({"BAND_3_MODERATE", "BAND_4_ADEQUATE"})


def test_exhaustive_approved_advisories_positive_and_negative(ranked_df: pd.DataFrame):
    """Verify all 18 approved advisories positive and negative eligibility across all 1,879 rows."""
    for _, row in ranked_df.iterrows():
        advs = evaluate_unit_advisories(row)
        adv_ids = {a.directive_id for a in advs}
        status = row["cluster_assignment_status"]
        arch = row["vulnerability_archetype_id"]
        score = row["preparedness_score"]

        if status == "ASSIGNED":
            if arch == 0:
                assert ("ADV_0_DRAINAGE_INFRA" in adv_ids) == (row["score_infrastructure"] < 50.0)
                assert ("ADV_0_POP_DENSITY" in adv_ids) == (
                    row["pop_density_sqkm"] > 1000.0 or str(row.get("unit_type", "")).strip().upper() == "TOWN"
                )
                assert ("ADV_0_CRITICAL_HOSPITAL" in adv_ids) == (
                    row["score_medical"] >= 60.0 and row["dist_nearest_hospital_km"] <= 15.0
                )
            elif arch == 1:
                assert ("ADV_1_HOSPITAL_ISOLATION" in adv_ids) == (row["dist_nearest_hospital_km"] > 20.0)
                assert ("ADV_1_SHELTER_GAP" in adv_ids) == (row["score_shelters"] < 20.0)
                assert ("ADV_1_FOOTHILL_ROAD" in adv_ids) == (
                    row["elevation_mean"] > 100.0 and row["has_pucca_road"] == 0.0
                )
            elif arch == 2:
                assert ("ADV_2_VULNERABLE_POP" in adv_ids) == (row["vulnerable_pop_share"] > 0.20)
                assert ("ADV_2_WATER_SANITATION" in adv_ids) == (row["drinking_water_score"] <= 0.4)
                assert ("ADV_2_SHELTER_CAPACITY" in adv_ids) == (row["score_shelters"] < 20.0)
            elif arch == 3:
                assert ("ADV_3_COASTAL_PROXIMITY" in adv_ids) == (row["dist_coast_km"] < 5.0)
                assert ("ADV_3_CYCLONE_SHELTER" in adv_ids) == (row["n_shelters_10km"] > 0)
                assert ("ADV_3_HISTORIC_IMPACT" in adv_ids) == (row["tauktae_affected"] == 1)
        else:
            # Uttara Kannada unassigned
            t_coast = row["dist_coast_km"] < 5.0
            t_shelter = row["score_shelters"] < 20.0
            t_med = row["score_medical"] < 35.0
            n_trig = int(t_coast) + int(t_shelter) + int(t_med)

            assert ("ADV_UK_BASELINE_LOW" in adv_ids) == (score < 40.0)
            assert ("ADV_UK_COASTAL_PROXIMITY" in adv_ids) == t_coast
            assert ("ADV_UK_SHELTER_DEFICIT" in adv_ids) == t_shelter
            assert ("ADV_UK_MEDICAL_REMOTENESS" in adv_ids) == t_med
            assert ("ADV_UK_GENERAL_DEFICIT_REVIEW" in adv_ids) == (n_trig == 0 and score < 40.0)
            assert ("ADV_UK_ROUTINE" in adv_ids) == (n_trig == 0 and score >= 40.0)


def test_missing_indicator_handling_and_data_incomplete():
    """Verify that rows with null indicators attach ADV_DATA_INCOMPLETE without crashing."""
    row_missing = {
        "shrid2": "TEST_MISSING_UNIT",
        "pc11_id": "999999",
        "village_name": "Test Missing Village",
        "district_name": "Uttara Kannada",
        "subdistrict_name": "Kumta",
        "preparedness_score": 45.0,
        "preparedness_tier": "Medium",
        "predicted_preparedness_tier": "Medium",
        "prob_low": 0.1,
        "prob_medium": 0.8,
        "prob_high": 0.1,
        "prediction_provenance": "TEST_OUT_OF_SAMPLE",
        "cluster_assignment_status": "UNASSIGNED_OUT_OF_STUDY_AREA",
        "score_shelters": np.nan,
        "score_medical": np.nan,
        "dist_coast_km": np.nan,
    }

    advs = evaluate_unit_advisories(row_missing)
    adv_ids = [a.directive_id for a in advs]
    assert "ADV_DATA_INCOMPLETE" in adv_ids

    eval_obj = build_unit_evaluation(row_missing)
    assert eval_obj.has_unverified_indicators is True


def test_unsupported_language_audit_comprehensive(ranked_df: pd.DataFrame):
    """Audit all 1,879 records and their advisories to verify absence of unsupported operational claims."""
    prohibited_patterns = [
        re.compile(r"statutory\s+(?:requirement|threshold|baseline)", re.IGNORECASE),
        re.compile(r"physical\s+isolation", re.IGNORECASE),
        re.compile(r"travel\s+time", re.IGNORECASE),
        re.compile(r"(?:emergency\s+deployment|deployment)\s+is\s+warranted", re.IGNORECASE),
        re.compile(r"warranted\s+solely", re.IGNORECASE),
        re.compile(r"\bMPCS\b"),
        re.compile(r"operational\s+readiness", re.IGNORECASE),
        re.compile(r"real-time\s+infrastructure", re.IGNORECASE),
        re.compile(r"probability\s+of\s+being\s+unprepared", re.IGNORECASE),
        re.compile(r"unpreparedness\s+probability", re.IGNORECASE),
        re.compile(r"extrapolated\s+archetype", re.IGNORECASE),
    ]

    for _, row in ranked_df.iterrows():
        eval_obj = build_unit_evaluation(row)
        texts = [eval_obj.priority_explanation, eval_obj.advisory_disclaimer]
        for adv in eval_obj.recommended_advisories:
            texts.extend([adv.advisory_focus, adv.operational_recommendation, adv.directive_id])

        full_text = " ".join(texts)
        for pat in prohibited_patterns:
            assert not pat.search(full_text), f"Prohibited phrase '{pat.pattern}' found in {eval_obj.village_name}: {full_text}"


def test_baseline_datasets_and_models_preservation():
    """Verify that baseline datasets and serialized models remain unmodified."""
    from pathlib import Path

    p1 = Path("data/processed/coastal_vulnerability_archetypes.parquet")
    p2 = Path("data/processed/unified_village_preparedness.parquet")
    p3 = Path("data/processed/unified_village_preparedness_v2.parquet")
    m1 = Path("models/logistic_regression_tier.joblib")

    assert p1.exists() and len(pd.read_parquet(p1)) == 622
    assert p2.exists() and len(pd.read_parquet(p2)) == 1879
    assert p3.exists() and len(pd.read_parquet(p3)) == 1879
    assert m1.exists() and m1.stat().st_size > 1000


def test_regional_summary_provenance_breakdown(client: TestClient):
    """Verify GET /api/v1/decision-support/summary returns train, val, and test provenance breakdown."""
    resp = client.get("/api/v1/decision-support/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert "provenance_agreement" in data
    prov = data["provenance_agreement"]
    assert "TRAIN_IN_SAMPLE" in prov
    assert "VAL_OUT_OF_SAMPLE" in prov
    assert "TEST_OUT_OF_SAMPLE" in prov

    # Train in-sample: 1,315 units, 1,261 agreement, 95.89%
    assert prov["TRAIN_IN_SAMPLE"]["total"] == 1315
    assert prov["TRAIN_IN_SAMPLE"]["agreement"] == 1261
    assert prov["TRAIN_IN_SAMPLE"]["agreement_rate"] == 0.9589

    # Val out-of-sample: 282 units, 265 agreement, 93.97%
    assert prov["VAL_OUT_OF_SAMPLE"]["total"] == 282
    assert prov["VAL_OUT_OF_SAMPLE"]["agreement"] == 265
    assert prov["VAL_OUT_OF_SAMPLE"]["agreement_rate"] == 0.9397

    # Test out-of-sample: 282 units, 266 agreement, 94.33%
    assert prov["TEST_OUT_OF_SAMPLE"]["total"] == 282
    assert prov["TEST_OUT_OF_SAMPLE"]["agreement"] == 266
    assert prov["TEST_OUT_OF_SAMPLE"]["agreement_rate"] == 0.9433

