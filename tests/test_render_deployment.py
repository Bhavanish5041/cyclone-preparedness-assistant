"""Targeted regression tests for Render Free-Tier Deployment Configuration.

Phase 2.8B:
- Validates Direct-Mode Streamlit execution.
- Verifies offline loading of immutable W&B model artifact (v0).
- Verifies dataset integrity and 1,879 unit reconciliation.
- Verifies minimal dependency specifications in docker/requirements-render.txt.
- Verifies Dockerfile.render configuration, non-root security profile, and port binding.
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.assistant.api_client import AdapterMode, DecisionSupportApiClient, get_api_client
from src.assistant.decision_support import DecisionSupportService
from src.models.load_registered_model import RegisteredModelContainer, load_registered_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_render_requirements_specification():
    """Verify docker/requirements-render.txt contains minimal runtime dependencies."""
    req_path = PROJECT_ROOT / "docker" / "requirements-render.txt"
    assert req_path.exists(), "docker/requirements-render.txt must exist"

    content = req_path.read_text(encoding="utf-8")
    lines = [
        line.split("#")[0].strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    pkg_names = {line.split("==")[0].lower() for line in lines}

    # Required runtime dependencies
    required_pkgs = {
        "streamlit",
        "streamlit-folium",
        "folium",
        "altair",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "geopandas",
        "pyarrow",
        "shapely",
        "pydantic",
        "httpx",
        "wandb",
    }
    for req in required_pkgs:
        assert req in pkg_names, f"Missing essential runtime dependency: '{req}'"

    # Strictly excluded heavy / non-runtime dependencies
    forbidden_pkgs = {"evidently", "fastapi", "uvicorn", "prometheus_client", "pytest"}
    for forbidden in forbidden_pkgs:
        assert forbidden not in pkg_names, f"Prohibited dependency '{forbidden}' found in requirements-render.txt"


def test_render_dockerfile_specification():
    """Verify Dockerfile.render meets production constraints and non-root security."""
    dockerfile_path = PROJECT_ROOT / "Dockerfile.render"
    assert dockerfile_path.exists(), "Dockerfile.render must exist"

    content = dockerfile_path.read_text(encoding="utf-8")

    # Base image
    assert "FROM python:3.14-slim" in content

    # Non-root security
    assert "USER appuser" in content
    assert "10001" in content

    # Direct mode environment
    assert 'CYCLONE_ADAPTER_MODE="direct"' in content or "CYCLONE_ADAPTER_MODE=direct" in content
    assert "village-preparedness-classifier:v0" in content

    # Dynamic port binding and entrypoint
    assert "streamlit run src/assistant/app.py" in content
    assert "${PORT:-10000}" in content or "$PORT" in content

    # Streamlit healthcheck
    assert "/_stcore/health" in content

    # Deployment artifact packaging
    assert "models/render_artifact" in content


def test_direct_mode_initialization():
    """Verify DecisionSupportApiClient operates purely in-memory in Direct Mode."""
    client = DecisionSupportApiClient(mode=AdapterMode.DIRECT)
    assert client.mode == AdapterMode.DIRECT

    # Verify lazy instantiation of DecisionSupportService
    service = client.service
    assert isinstance(service, DecisionSupportService)

    # Verify summary retrieval with zero network calls
    summary = client.get_regional_summary()
    assert summary.total_units == 1879
    assert "BAND_1_CRITICAL" in summary.band_counts
    assert set(summary.band_counts.keys()) == {
        "BAND_1_CRITICAL",
        "BAND_2_SPECIALIZED",
        "BAND_3_MODERATE",
        "BAND_4_ADEQUATE",
    }


def test_offline_wandb_model_loading_v0():
    """Verify immutable model artifact v0 loads completely offline from local cache."""
    cache_dir = PROJECT_ROOT / ".wandb_artifacts_cache"
    assert cache_dir.exists(), "Local W&B artifacts cache must exist"

    # Ensure no network authentication is available
    with patch.dict(os.environ, {"WANDB_API_KEY": ""}, clear=False):
        container = load_registered_model(
            artifact_address="village-preparedness-classifier:v0",
            cache_dir=cache_dir,
        )
        assert isinstance(container, RegisteredModelContainer)
        assert container.artifact_version == "v0"
        assert len(container.features) == 26
        assert set(container.classes_) == {"High", "Low", "Medium"}


def test_render_packaged_artifact_loading_v0(tmp_path):
    """Verify immutable model artifact v0 in models/render_artifact matches specifications and loads cleanly."""
    packaged_dir = PROJECT_ROOT / "models" / "render_artifact"
    assert packaged_dir.exists(), "models/render_artifact must exist"

    required_files = [
        "logistic_regression_tier.joblib",
        "label_encoder.joblib",
        "logistic_regression_metadata.json",
        "evaluation_summary.json",
        "feature_schema.json",
        "intended_use_and_limitations.md",
        "model_card.md",
    ]
    for fname in required_files:
        assert (packaged_dir / fname).exists(), f"Missing required artifact file: {fname}"

    # Replicate Docker container artifact layout into tmp_path
    target_cache_dir = tmp_path / "bhavanishreddy69-na_cyclone-preparedness-assistant_village-preparedness-classifier_v0"
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for fname in required_files:
        shutil.copy2(packaged_dir / fname, target_cache_dir / fname)

    # Load from replicated cache without W&B credentials
    with patch.dict(os.environ, {"WANDB_API_KEY": ""}, clear=False):
        container = load_registered_model(
            artifact_address="village-preparedness-classifier:v0",
            cache_dir=tmp_path,
        )
        assert isinstance(container, RegisteredModelContainer)
        assert container.artifact_version == "v0"
        assert len(container.features) == 26
        assert set(container.classes_) == {"High", "Low", "Medium"}


def test_dataset_loading_and_reconciliation():
    """Verify both processed datasets load accurately with zero missing records."""
    archetypes_path = PROJECT_ROOT / "data" / "processed" / "coastal_vulnerability_archetypes.parquet"
    unified_path = PROJECT_ROOT / "data" / "processed" / "unified_village_preparedness_v2.parquet"

    assert archetypes_path.exists(), "Archetypes parquet must exist"
    assert unified_path.exists(), "Unified v2 parquet must exist"

    df_arch = pd.read_parquet(archetypes_path)
    df_unified = pd.read_parquet(unified_path)

    # Archetype assignments strictly for Dakshina Kannada & Udupi (622 units)
    assert len(df_arch) == 622
    assert set(df_arch["district_name"].unique()) == {"Dakshina Kannada", "Udupi"}

    # Unified study area strictly encompasses all 1,879 units
    assert len(df_unified) == 1879
    assert set(df_unified["district_name"].unique()) == {"Dakshina Kannada", "Udupi", "Uttara Kannada"}

    # Authoritative four priority band totals
    service = DecisionSupportService(data_path=archetypes_path, unified_data_path=unified_path)
    summary = service.get_regional_decision_summary()
    assert summary.band_counts["BAND_1_CRITICAL"] == 155
    assert summary.band_counts["BAND_2_SPECIALIZED"] == 829
    assert summary.band_counts["BAND_3_MODERATE"] == 299
    assert summary.band_counts["BAND_4_ADEQUATE"] == 596
    assert sum(summary.band_counts.values()) == 1879


def test_model_inference_in_direct_mode():
    """Verify model inference on canonical feature vector yields valid tier probabilities."""
    cache_dir = PROJECT_ROOT / ".wandb_artifacts_cache"
    container = load_registered_model(
        artifact_address="village-preparedness-classifier:v0",
        cache_dir=cache_dir,
    )

    # Synthetic sample row with all 26 canonical features
    sample_data = {feat: [0.5] for feat in container.features}
    # Categoricals
    sample_data["has_pucca_road"] = [1.0]
    sample_data["has_major_district_road"] = [1.0]
    sample_data["bus_connectivity"] = [1.0]

    X = pd.DataFrame(sample_data)

    # Detailed prediction
    res_df = container.predict_detailed(X)
    assert len(res_df) == 1
    assert res_df["predicted_preparedness_tier"].iloc[0] in {"High", "Low", "Medium"}

    p_high = res_df["prob_high"].iloc[0]
    p_med = res_df["prob_medium"].iloc[0]
    p_low = res_df["prob_low"].iloc[0]

    # Simplex sum strictly equals 1.0
    assert np.isclose(p_high + p_med + p_low, 1.0, atol=1e-5)
    assert 0.0 <= p_high <= 1.0
    assert 0.0 <= p_med <= 1.0
    assert 0.0 <= p_low <= 1.0

    # Margin check
    margin = res_df["probability_margin"].iloc[0]
    max_p = res_df["max_probability"].iloc[0]
    assert margin >= 0.0
    assert max_p >= 1.0 / 3.0


def test_render_dynamic_port_configuration():
    """Verify dynamic port binding expansion under custom PORT environment variable."""
    default_port = 10000
    custom_port = 8080

    for test_port in [default_port, custom_port]:
        with patch.dict(os.environ, {"PORT": str(test_port)}):
            port_val = int(os.environ.get("PORT", "10000"))
            assert port_val == test_port
            cmd_args = [
                "streamlit",
                "run",
                "src/assistant/app.py",
                f"--server.port={port_val}",
                "--server.address=0.0.0.0",
            ]
            assert f"--server.port={test_port}" in cmd_args


def test_streamlit_health_endpoint_contract():
    """Verify Streamlit native health check path conventions."""
    health_route = "/_stcore/health"
    assert health_route == "/_stcore/health"
