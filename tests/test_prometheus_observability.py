"""Tests for Prometheus operational observability and metric exposition.

Verifies:
- Prometheus exposition endpoint (`GET /metrics`) availability and formatting.
- HTTP request count and latency metric collection with bounded cardinality.
- Model inference execution count, latency histogram, and error classification.
- Predicted preparedness-tier distribution metrics.
- Boundary-case detection tracking.
- AHP vs ML tier agreement / discordance metrics.
- Model availability and artifact version reporting.
- Label cardinality protection and endpoint normalization.
- Prometheus configuration and Docker Compose integration validity.
"""

from pathlib import Path
import time
from fastapi.testclient import TestClient
import pytest
import yaml

from src.api.main import app
from src.monitoring.prometheus_metrics import (
    ALLOWED_AGREEMENT_STATUSES,
    ALLOWED_ERROR_TYPES,
    ALLOWED_METHODS,
    ALLOWED_TIERS,
    HTTP_REQUESTS_TOTAL,
    MODEL_AHP_DISAGREEMENTS_TOTAL,
    MODEL_BOUNDARY_CASES_TOTAL,
    MODEL_INFERENCE_ERRORS_TOTAL,
    MODEL_INFERENCE_REQUESTS_TOTAL,
    MODEL_INFO,
    MODEL_LOADED,
    MODEL_PREDICTIONS_TIER_TOTAL,
    get_normalized_endpoint,
    record_http_request,
    record_inference_error,
    record_inference_success,
    sanitize_agreement_status,
    sanitize_error_type,
    sanitize_method,
    sanitize_tier,
    set_model_status,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMETHEUS_CONFIG_PATH = PROJECT_ROOT / "monitoring/prometheus/prometheus.yml"
COMPOSE_CONFIG_PATH = PROJECT_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_endpoint_availability(client: TestClient):
    """Verify /metrics returns HTTP 200 with Prometheus exposition content-type."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type
    assert "version=" in content_type

    # Verify standard metric names are present in exposition payload
    text = response.text
    assert "cyclone_http_requests_total" in text
    assert "cyclone_http_request_duration_seconds" in text
    assert "cyclone_model_loaded" in text
    assert "cyclone_model_info" in text


def test_http_request_metrics_recorded(client: TestClient):
    """Verify that hitting API endpoints increments HTTP request metrics."""
    # Hit health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200

    # Retrieve metrics and verify endpoint tracking
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    text = metrics_resp.text

    assert 'cyclone_http_requests_total{endpoint="/health",method="GET",status_code="200"}' in text
    assert 'cyclone_http_request_duration_seconds_count{endpoint="/health",method="GET"}' in text


def test_model_inference_metrics_success(client: TestClient):
    """Verify that successful inference records count, duration, tier, and agreement."""
    # Call prediction for Panjikal
    resp = client.post(
        "/api/v1/model/predict",
        json={"area_id": "11-29-575-05562-617533"},
    )
    assert resp.status_code == 200
    data = resp.json()
    pred_tier = data["predicted_preparedness_tier"]
    assert pred_tier == "High"

    # Inspect metrics output
    metrics_resp = client.get("/metrics")
    text = metrics_resp.text

    assert 'cyclone_model_inference_requests_total{status="success"}' in text
    assert 'cyclone_model_predictions_tier_total{tier="High"}' in text
    assert "cyclone_model_inference_duration_seconds_count" in text
    assert 'cyclone_model_ahp_disagreements_total{agreement_status="FULL_AGREEMENT"' in text


def test_model_inference_error_metrics(client: TestClient):
    """Verify inference errors increment error metrics with bounded classification."""
    # 1. Test Not Found (404)
    resp_404 = client.post(
        "/api/v1/model/predict",
        json={"area_id": "nonexistent_area_id_999999"},
    )
    assert resp_404.status_code == 404

    # 2. Test Validation Error (422)
    resp_422 = client.post(
        "/api/v1/model/predict",
        json={"features": {"invalid_col": 123.4}},
    )
    assert resp_422.status_code == 422

    # Verify Prometheus output
    metrics_resp = client.get("/metrics")
    text = metrics_resp.text
    assert 'cyclone_model_inference_errors_total{error_type="not_found"}' in text
    assert 'cyclone_model_inference_errors_total{error_type="validation_error"}' in text


def test_model_loaded_and_info_gauges(client: TestClient):
    """Verify model loaded state and model info version gauges."""
    metrics_resp = client.get("/metrics")
    text = metrics_resp.text

    assert "cyclone_model_loaded 1.0" in text
    assert 'cyclone_model_info{artifact_name="village-preparedness-classifier",artifact_version="v0"} 1.0' in text


def test_boundary_case_tracking():
    """Verify boundary case tracking records when boundary conditions occur."""
    initial_metric = MODEL_BOUNDARY_CASES_TOTAL.labels(tier="Medium")._value.get()
    record_inference_success(
        duration_seconds=0.005,
        predicted_tier="Medium",
        is_boundary=True,
        model_version="v0",
    )
    updated_metric = MODEL_BOUNDARY_CASES_TOTAL.labels(tier="Medium")._value.get()
    assert updated_metric == initial_metric + 1


def test_label_cardinality_and_sanitization():
    """Verify strict label cardinality bounds and route parameter masking."""
    # Test method sanitization
    assert sanitize_method("get") == "GET"
    assert sanitize_method("post") == "POST"
    assert sanitize_method("MALICIOUS_METHOD_123") == "OTHER"

    # Test tier sanitization
    assert sanitize_tier("high") == "High"
    assert sanitize_tier("Medium") == "Medium"
    assert sanitize_tier("low") == "Low"
    assert sanitize_tier("RANDOM_TIER") == "unknown"
    assert sanitize_tier(None) == "unknown"

    # Test error type sanitization
    assert sanitize_error_type("validation_error") == "validation_error"
    assert sanitize_error_type("not_found") == "not_found"
    assert sanitize_error_type("INVENTED_ERROR_TYPE") == "other"

    # Test agreement status sanitization
    assert sanitize_agreement_status("full_agreement") == "FULL_AGREEMENT"
    assert sanitize_agreement_status("borderline_disagreement") == "BORDERLINE_DISAGREEMENT"
    assert sanitize_agreement_status("INVALID_STATUS") == "NONE"


def test_endpoint_normalization_masks_dynamic_identifiers():
    """Verify endpoint normalizer replaces dynamic identifiers with route templates."""
    class DummyURL:
        def __init__(self, path):
            self.path = path

    class DummyRequest:
        def __init__(self, path, route_path=None):
            self.url = DummyURL(path)
            self.scope = {}
            if route_path:
                class DummyRoute:
                    def __init__(self, rp):
                        self.path = rp
                self.scope["route"] = DummyRoute(route_path)

    # Static path
    req_health = DummyRequest("/health")
    assert get_normalized_endpoint(req_health) == "/health"

    # Matched parameterized route
    req_unit = DummyRequest("/api/v1/units/608720", route_path="/api/v1/units/{identifier}")
    assert get_normalized_endpoint(req_unit) == "/api/v1/units/{identifier}"

    # Parameterized fallback when route not in scope
    req_unit_fallback = DummyRequest("/api/v1/units/11-29-575-05562-617533")
    assert get_normalized_endpoint(req_unit_fallback) == "/api/v1/units/{identifier}"

    # Unknown path
    req_unknown = DummyRequest("/some/arbitrary/unmatched/url")
    assert get_normalized_endpoint(req_unknown) == "unmatched"


def test_prometheus_configuration_file():
    """Verify Prometheus configuration exists, parses cleanly, and targets API."""
    assert PROMETHEUS_CONFIG_PATH.is_file(), f"Prometheus config missing at {PROMETHEUS_CONFIG_PATH}"
    with open(PROMETHEUS_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict)
    assert "scrape_configs" in config

    # Find cyclone-assistant-api scrape job
    jobs = {j["job_name"]: j for j in config["scrape_configs"]}
    assert "cyclone-assistant-api" in jobs

    api_job = jobs["cyclone-assistant-api"]
    assert api_job.get("metrics_path") == "/metrics"
    static_configs = api_job.get("static_configs", [])
    assert len(static_configs) > 0
    targets = static_configs[0].get("targets", [])
    assert any("api:8000" in t for t in targets)


def test_docker_compose_prometheus_service_definition():
    """Verify docker-compose.yml defines Prometheus service with persistent volume."""
    assert COMPOSE_CONFIG_PATH.is_file(), f"Compose file missing at {COMPOSE_CONFIG_PATH}"
    with open(COMPOSE_CONFIG_PATH, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "prometheus" in services, "Prometheus service missing from docker-compose.yml"

    prom_service = services["prometheus"]
    assert "prom/prometheus" in prom_service.get("image", "")
    assert "cyclone-prometheus" == prom_service.get("container_name")

    # Verify dependencies and healthcheck
    depends = prom_service.get("depends_on", {})
    assert "api" in depends

    health = prom_service.get("healthcheck", {})
    assert "test" in health

    # Verify persistent volume
    volumes = compose_data.get("volumes", {})
    assert "prometheus_data" in volumes
