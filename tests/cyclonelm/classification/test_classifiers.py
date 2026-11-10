"""Unit tests for candidate classifiers."""

import pytest
from src.cyclonelm.classification.classifiers import (
    EmbeddingSimilarityClassifier,
    KeywordBaselineClassifier,
    LogisticIntentClassifier,
    SupervisedHeadClassifier,
)
from src.cyclonelm.classification.config import SUPPORTED_INTENTS
from src.cyclonelm.classification.embedder import MiniLMEmbedder


@pytest.fixture(scope="module")
def train_data():
    texts = [
        "Analyze shelter and healthcare buffers for Ullal",
        "Compare Dakshina Kannada with Udupi",
        "Explain preparedness scores for Someshwara",
        "Explain vulnerability archetype Cluster 3",
        "Filter Band 1 Critical communities in Dakshina Kannada",
        "What interventions are needed for Bengre?",
        "Check community data for Tallur",
        "What is the live wind speed right now in Mangalore?",
        "Write a Python script to sort a list",
        "Ullal",
    ]
    labels = [
        "analyze_infrastructure_and_exposure",
        "compare_districts",
        "explain_preparedness_assessment",
        "explain_vulnerability_archetype",
        "filter_and_rank_communities",
        "get_intervention_recommendations",
        "lookup_community",
        "refusal_live_telemetry",
        "out_of_scope",
        "ambiguous",
    ]
    return texts, labels


@pytest.fixture(scope="module")
def shared_embedder():
    return MiniLMEmbedder()


class TestClassifiers:
    def test_keyword_baseline(self, train_data):
        texts, labels = train_data
        clf = KeywordBaselineClassifier().fit(texts, labels)
        pred = clf.predict("Compare Dakshina Kannada with Udupi")
        assert pred.intent == "compare_districts"
        assert pred.is_supported_tool is True

    def test_embedding_similarity_classifier(self, train_data, shared_embedder):
        texts, labels = train_data
        clf = EmbeddingSimilarityClassifier(embedder=shared_embedder).fit(texts, labels)
        pred = clf.predict("Show infrastructure and PHC distance for Malpe")
        assert pred.intent == "analyze_infrastructure_and_exposure"
        assert 0.0 <= pred.confidence <= 1.0

    def test_logistic_classifier(self, train_data, shared_embedder):
        texts, labels = train_data
        clf = LogisticIntentClassifier(embedder=shared_embedder, c_param=2.0).fit(texts, labels)
        pred = clf.predict("What risk reduction upgrades are recommended for Edmoge?")
        assert pred.intent == "get_intervention_recommendations"
        assert 0.0 <= pred.confidence <= 1.0
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-3

    def test_supervised_head_classifier(self, train_data, shared_embedder):
        texts, labels = train_data
        clf = SupervisedHeadClassifier(embedder=shared_embedder, epochs=15).fit(texts, labels)
        pred = clf.predict("Explain vulnerability cluster 0")
        assert pred.intent == "explain_vulnerability_archetype"
        assert 0.0 <= pred.confidence <= 1.0

    def test_all_seven_intents_represented(self, shared_embedder):
        # Verify that all 7 supported tools are valid tool names in config
        assert len(SUPPORTED_INTENTS) == 7
        for t in SUPPORTED_INTENTS:
            assert isinstance(t, str)
            assert len(t) > 3
