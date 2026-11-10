"""Unit tests for IntentRouter and integration boundaries."""

import json
from pathlib import Path
import pytest

from src.assistant.tools.registry import get_default_registry
from src.cyclonelm.classification.classifiers import LogisticIntentClassifier
from src.cyclonelm.classification.config import SUPPORTED_INTENTS
from src.cyclonelm.classification.router import IntentRouter, RoutingStatus


@pytest.fixture(scope="module")
def trained_router():
    # Load training dataset
    train_path = Path("data/cyclonelm/v0.8/intent/train.jsonl")
    texts = []
    labels = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                texts.append(item["text"])
                labels.append(item["intent"])

    clf = LogisticIntentClassifier(c_param=2.0).fit(texts, labels)
    return IntentRouter(classifier=clf)


class TestIntentRouter:
    def test_route_successful_tool_execution(self, trained_router):
        res = trained_router.route("What interventions are needed for Tallur?")
        assert res.status == RoutingStatus.ROUTED
        assert res.tool_name == "get_intervention_recommendations"
        assert res.validated_arguments["identifier"] == "Tallur"
        assert res.can_execute is True

    def test_route_successful_tool_execution_with_district_qualification(self, trained_router):
        res = trained_router.route("What interventions are needed for Ullal in Dakshina Kannada?")
        assert res.status == RoutingStatus.ROUTED
        assert res.tool_name == "get_intervention_recommendations"
        assert "ullal" in res.validated_arguments["identifier"].lower()
        assert res.can_execute is True

    def test_route_live_refusal(self, trained_router):
        res = trained_router.route("What is the current live wind speed right now in Mangalore?")
        assert res.status == RoutingStatus.REFUSED
        assert res.can_execute is False
        assert "dynamic" in res.refusal_reason.lower() or "imd" in res.refusal_reason.lower()

    def test_route_out_of_scope(self, trained_router):
        res = trained_router.route("Write a Python function to sort a list using quicksort.")
        assert res.status == RoutingStatus.OUT_OF_SCOPE
        assert res.can_execute is False

    def test_route_ambiguous_query(self, trained_router):
        res = trained_router.route("Cyclone")
        assert res.status in (RoutingStatus.AMBIGUOUS_QUERY, RoutingStatus.ABSTAINED)
        assert res.can_execute is False

    def test_route_missing_required_argument(self, trained_router):
        # User indicates intent but omits the community identifier
        res = trained_router.route("What interventions are needed?")
        assert res.status == RoutingStatus.MISSING_ARGUMENT
        assert "identifier" in res.missing_parameters
        assert res.can_execute is False

    def test_route_ambiguous_argument(self, trained_router):
        # Village name that exists in multiple districts without qualification
        res = trained_router.route("Explain preparedness scores for Siddapur")
        assert res.status == RoutingStatus.AMBIGUOUS_ARGUMENT
        assert len(res.ambiguous_candidates) > 1
        assert res.can_execute is False

    def test_deterministic_tools_remain_unchanged(self):
        # Verify the production tool registry has all 7 tools registered and untouched
        registry = get_default_registry()
        registered = registry.list_tools()
        for expected in SUPPORTED_INTENTS:
            assert expected in registered
            tool = registry.get_tool(expected)
            assert tool is not None
            assert hasattr(tool, "execute")
            assert hasattr(tool, "input_schema")
