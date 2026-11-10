"""Unit tests for RefusalGate."""

import pytest
from src.cyclonelm.classification.config import REFUSAL_CATEGORY
from src.cyclonelm.classification.refusal_gate import RefusalDecision, RefusalGate


@pytest.fixture(scope="module")
def refusal_gate():
    return RefusalGate()


class TestRefusalGate:
    def test_live_wind_speed_refusal(self, refusal_gate):
        res = refusal_gate.evaluate("What is the current live wind speed right now in Mangalore?")
        assert res.is_refused is True
        assert res.confidence >= 0.75
        assert "dynamic live telemetry" in res.reason.lower() or "imd" in res.reason.lower()

    def test_live_shelter_occupancy_refusal(self, refusal_gate):
        res = refusal_gate.evaluate("What is the live shelter occupancy in Malpe at this moment?")
        assert res.is_refused is True
        assert "shelter" in res.domain

    def test_live_hospital_beds_refusal(self, refusal_gate):
        res = refusal_gate.evaluate("How many vacant hospital beds are currently available in Udupi right now?")
        assert res.is_refused is True
        assert "medical" in res.domain or "hospital" in res.domain

    def test_live_road_passability_refusal(self, refusal_gate):
        res = refusal_gate.evaluate("Is the coastal highway passable right now or flooded?")
        assert res.is_refused is True

    def test_live_rescue_staffing_refusal(self, refusal_gate):
        res = refusal_gate.evaluate("How many active NDRF personnel are deployed right now in Karwar?")
        assert res.is_refused is True

    def test_static_baseline_query_not_refused(self, refusal_gate):
        # Legitimate static preparedness requests must NOT be refused
        res1 = refusal_gate.evaluate("Explain preparedness scores for Ullal")
        assert res1.is_refused is False

        res2 = refusal_gate.evaluate("Compare Dakshina Kannada with Udupi")
        assert res2.is_refused is False

        res3 = refusal_gate.evaluate("What is the distance to nearest apex hospital for Bengre?")
        assert res3.is_refused is False

        res4 = refusal_gate.evaluate("Explain vulnerability archetype Cluster 3")
        assert res4.is_refused is False

    def test_empty_string(self, refusal_gate):
        res = refusal_gate.evaluate("")
        assert res.is_refused is False
