"""Unit tests for ArgumentExtractor and Pydantic validation."""

import pytest
from src.cyclonelm.classification.argument_extractor import ArgumentExtractor


@pytest.fixture(scope="module")
def extractor():
    return ArgumentExtractor()


class TestArgumentExtractor:
    def test_extract_lookup_community_single(self, extractor):
        res = extractor.extract_arguments("lookup_community", "Check community data for Bengre")
        assert res.is_valid is True
        assert res.parameters["query"] == "Bengre"

    def test_extract_lookup_with_district(self, extractor):
        res = extractor.extract_arguments("lookup_community", "Look up Someshwara in Dakshina Kannada")
        assert res.is_valid is True
        assert "someshwar" in res.parameters["query"].lower()
        assert res.parameters["district"] == "Dakshina Kannada"

    def test_extract_infrastructure_pc11_code(self, extractor):
        res = extractor.extract_arguments("analyze_infrastructure_and_exposure", "Analyze infrastructure for 608720")
        assert res.is_valid is True
        assert res.parameters["identifier"] == "Edmoge"

    def test_extract_intervention_recommendations(self, extractor):
        res = extractor.extract_arguments("get_intervention_recommendations", "What interventions are needed for Tallur?")
        assert res.is_valid is True
        assert res.parameters["identifier"] == "Tallur"

    def test_extract_intervention_recommendations_with_district(self, extractor):
        res = extractor.extract_arguments("get_intervention_recommendations", "What interventions are needed for Ullal in Dakshina Kannada?")
        assert res.is_valid is True
        assert "ullal" in res.parameters["identifier"].lower()

    def test_extract_compare_districts(self, extractor):
        res = extractor.extract_arguments("compare_districts", "Compare Dakshina Kannada with Udupi")
        assert res.is_valid is True
        assert res.parameters["district_a"] == "Dakshina Kannada"
        assert res.parameters["district_b"] == "Udupi"

    def test_extract_compare_districts_missing_second(self, extractor):
        res = extractor.extract_arguments("compare_districts", "Compare Dakshina Kannada with Mumbai")
        assert res.is_valid is False
        assert "district_b" in res.missing_required

    def test_extract_filter_and_rank(self, extractor):
        res = extractor.extract_arguments("filter_and_rank_communities", "Filter Band 1 Critical communities in Dakshina Kannada")
        assert res.is_valid is True
        assert res.parameters["priority_band"] == "BAND_1_CRITICAL"
        assert res.parameters["district"] == "Dakshina Kannada"

    def test_extract_archetype(self, extractor):
        res = extractor.extract_arguments("explain_vulnerability_archetype", "Explain vulnerability archetype Cluster 3")
        assert res.is_valid is True
        assert res.parameters["archetype_id"] == 3

    def test_missing_identifier(self, extractor):
        res = extractor.extract_arguments("explain_preparedness_assessment", "Explain preparedness scores for")
        assert res.is_valid is False
        assert "identifier" in res.missing_required

    def test_ambiguous_community_cross_districts(self, extractor):
        # "siddapur" is present in multiple taluks / districts in Karnataka
        res = extractor.extract_arguments("explain_preparedness_assessment", "Explain preparedness scores for Siddapur")
        # Without specifying district, it must be flagged as ambiguous
        assert res.is_ambiguous is True
        assert len(res.ambiguous_candidates) > 1

    def test_disambiguated_community_with_district(self, extractor):
        res = extractor.extract_arguments("explain_preparedness_assessment", "Explain preparedness scores for Siddapur in Uttara Kannada")
        # With district specified, it resolves cleanly
        assert res.is_valid is True
        assert res.is_ambiguous is False
        assert res.parameters["identifier"].lower() == "siddapur"
