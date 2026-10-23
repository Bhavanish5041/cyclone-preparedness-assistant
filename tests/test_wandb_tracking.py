"""Unit tests for Weights & Biases experiment tracking module using mocked calls.

Ensures:
- Safe authentication detection and mode resolution
- Comprehensive logging of clustering models, benchmark tables, and seed stability
- Comprehensive logging of prioritization sensitivity scenarios and correlation tables
- Zero online network calls or authentication requirements in test environments
- Full backward compatibility of existing helper functions
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.features.prioritization import DEFAULT_SCENARIOS
from src.models.clustering import log_clustering_experiments_wandb
from src.tracking.wandb_logger import (
    is_wandb_authenticated,
    log_clustering_experiments,
    log_prioritization_experiments,
    resolve_wandb_mode,
)


def test_is_wandb_authenticated():
    """Verify authentication detection under simulated environment variables."""
    # When WANDB_API_KEY is present
    with patch.dict("os.environ", {"WANDB_API_KEY": "fake_test_key_12345"}):
        assert is_wandb_authenticated() is True

    # When no key is set
    with patch.dict("os.environ", {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            with patch("wandb.setup") as mock_setup:
                mock_setup.return_value.settings.api_key = None
                with patch("wandb.Api") as mock_api:
                    mock_api.return_value.api_key = None
                    mock_api.return_value.default_entity = None
                    assert is_wandb_authenticated() is False


def test_resolve_wandb_mode():
    """Verify mode resolution priority and fallback logic."""
    # Explicit CLI mode: disabled
    assert resolve_wandb_mode("disabled") == "disabled"

    # Explicit CLI mode: offline
    assert resolve_wandb_mode("offline") == "offline"

    # Explicit CLI mode: online with no auth -> falls back to offline
    with patch("src.tracking.wandb_logger.is_wandb_authenticated", return_value=False):
        assert resolve_wandb_mode("online") == "offline"

    # Explicit CLI mode: online with auth -> resolves to online
    with patch("src.tracking.wandb_logger.is_wandb_authenticated", return_value=True):
        assert resolve_wandb_mode("online") == "online"

    # Invalid mode -> defaults to offline
    assert resolve_wandb_mode("invalid_mode_name") == "offline"


def test_log_clustering_experiments_mocked_online(tmp_path: Path):
    """Verify full online clustering tracking workflow with mocked W&B calls."""
    sample_experiments = [
        {
            "algorithm": "kmeans",
            "n_clusters": 2,
            "random_state": 42,
            "silhouette": 0.3004,
            "davies_bouldin": 1.3453,
            "calinski_harabasz": 297.13,
            "inertia": 1681.94,
            "cluster_sizes": [260, 362],
            "cluster_balance_ratio": 0.7182,
        },
        {
            "algorithm": "kmeans",
            "n_clusters": 4,
            "random_state": 42,
            "silhouette": 0.3348,
            "davies_bouldin": 1.1771,
            "calinski_harabasz": 298.59,
            "inertia": 1015.73,
            "cluster_sizes": [130, 82, 161, 249],
            "cluster_balance_ratio": 0.3293,
        },
    ]

    agreement_data = [
        {
            "k": 4,
            "adjusted_rand_index": 0.7575,
            "adjusted_mutual_info": 0.7410,
            "km_silhouette": 0.3348,
            "agg_silhouette": 0.3045,
        }
    ]

    seed_stability = [
        {"seed": 42, "silhouette": 0.3348, "ari_vs_seed_42": 1.0, "ami_vs_seed_42": 1.0},
        {"seed": 123, "silhouette": 0.3348, "ari_vs_seed_42": 0.9912, "ami_vs_seed_42": 0.9890},
    ]

    profiles_df = pd.DataFrame(
        {
            "cluster_id": [0, 1, 2, 3],
            "archetype_name": ["Urban", "Foothill", "Rural", "Coastal"],
            "unit_count": [130, 82, 161, 249],
            "pct_of_total": [20.9, 13.2, 25.9, 40.0],
            "count_dk": [100, 60, 90, 119],
            "count_udupi": [30, 22, 71, 130],
            "dim_healthcare_deficit_mean": [0.15, 0.72, 0.35, 0.31],
            "dim_demographic_vulnerability_mean": [0.18, 0.28, 0.52, 0.24],
            "dim_infrastructure_deficit_mean": [0.12, 0.45, 0.58, 0.71],
            "dim_coastal_exposure_mean": [0.45, 0.12, 0.38, 0.89],
        }
    ).set_index("cluster_id")

    mock_run = MagicMock()
    mock_run.id = "mock_run_123"
    mock_run.url = "https://wandb.ai/mock/project/runs/mock_run_123"

    json_path = tmp_path / "clustering_manifest.json"

    with patch("wandb.init", return_value=mock_run) as mock_init:
        with patch("wandb.log") as mock_log:
            with patch("wandb.Table") as mock_table:
                res = log_clustering_experiments(
                    experiments_data=sample_experiments,
                    agreement_data=agreement_data,
                    seed_stability_data=seed_stability,
                    archetype_profiles_df=profiles_df,
                    project_name="cyclone-test-project",
                    mode="online",
                    fallback_json_path=json_path,
                )

                assert res["status"] == "online"
                assert res["synced"] is True
                assert res["project"] == "cyclone-test-project"
                assert len(res["runs"]) == 2
                # 2 individual model runs + 1 summary run = 3 calls to wandb.init
                assert mock_init.call_count == 3
                assert mock_run.log.call_count >= 3
                # Verified Table artifact logging
                assert mock_table.call_count >= 3

    # Verify fallback json was written
    assert json_path.exists()
    saved_data = json.loads(json_path.read_text())
    assert len(saved_data) == 2
    assert saved_data[0]["algorithm"] == "kmeans"


def test_log_clustering_experiments_disabled():
    """Verify disabled mode bypasses W&B init."""
    with patch("wandb.init") as mock_init:
        res = log_clustering_experiments(
            experiments_data=[{"algorithm": "kmeans", "n_clusters": 2}],
            mode="disabled",
        )
        assert res["status"] == "disabled"
        assert res["synced"] is False
        assert mock_init.call_count == 0


def test_log_prioritization_experiments_mocked_online(tmp_path: Path):
    """Verify full prioritization sensitivity tracking with mocked W&B calls."""
    sensitivity_metrics = {
        "total_units_evaluated": 622,
        "spearman_rank_correlations": {
            "equal_weights": {
                "equal_weights": 1.0,
                "hazard_focused": 0.704,
                "access_focused": 0.829,
                "no_coastal": 0.656,
                "with_population_scale": 0.980,
            },
            "hazard_focused": {
                "equal_weights": 0.704,
                "hazard_focused": 1.0,
                "access_focused": 0.313,
                "no_coastal": -0.025,
                "with_population_scale": 0.783,
            },
        },
        "kendall_tau_correlations": {
            "equal_weights": {
                "equal_weights": 1.0,
                "hazard_focused": 0.501,
            }
        },
        "robust_units_summary": [
            {
                "shrid2": "11-29-575-05561-617431",
                "unit_name": "Shimanthur",
                "district_name": "Dakshina Kannada",
                "rank_mean": 22.4,
                "rank_std": 14.1,
                "rank_min": 4,
                "rank_max": 42,
                "rank_shift_span": 38,
            }
        ],
    }

    dummy_df = pd.DataFrame(
        {
            "shrid2": ["1", "2", "3"],
            "priority_score": [0.70, 0.65, 0.50],
            "priority_rank": [1, 2, 3],
            "rank_equal_weights": [1, 2, 3],
            "score_equal_weights": [0.70, 0.65, 0.50],
            "rank_hazard_focused": [2, 1, 3],
            "score_hazard_focused": [0.68, 0.72, 0.45],
            "rank_access_focused": [1, 3, 2],
            "score_access_focused": [0.71, 0.55, 0.60],
            "rank_no_coastal": [10, 50, 100],
            "score_no_coastal": [0.55, 0.45, 0.35],
            "rank_with_population_scale": [1, 2, 3],
            "score_with_population_scale": [0.69, 0.64, 0.49],
        }
    )

    mock_run = MagicMock()
    mock_run.id = "mock_prioritization_run"
    mock_run.url = "https://wandb.ai/mock/project/runs/mock_prioritization_run"

    json_path = tmp_path / "sensitivity_summary.json"

    with patch("wandb.init", return_value=mock_run) as mock_init:
        with patch("wandb.log") as mock_log:
            with patch("wandb.Table") as mock_table:
                res = log_prioritization_experiments(
                    sensitivity_metrics=sensitivity_metrics,
                    scenarios_dict=DEFAULT_SCENARIOS,
                    scored_gdf=dummy_df,
                    project_name="cyclone-test-project",
                    mode="online",
                    fallback_json_path=json_path,
                )

                assert res["status"] == "online"
                assert res["synced"] is True
                # 5 individual scenarios + 1 summary run = 6 calls to wandb.init
                assert mock_init.call_count == 6
                assert mock_run.log.call_count >= 6
                assert mock_table.call_count >= 2

    assert json_path.exists()


def test_log_clustering_experiments_wandb_backward_compatibility():
    """Verify legacy log_clustering_experiments_wandb remains compatible with previous caller contract."""
    sample_exp = [{"algorithm": "kmeans", "n_clusters": 2, "silhouette": 0.30}]

    # In disabled/offline mode without online sync, returns False cleanly
    res_offline = log_clustering_experiments_wandb(
        experiments_data=sample_exp,
        mode="disabled",
    )
    assert res_offline is False

    # In mocked online mode, returns True
    mock_run = MagicMock()
    with patch("wandb.init", return_value=mock_run):
        with patch("wandb.Table"):
            with patch("src.tracking.wandb_logger.is_wandb_authenticated", return_value=True):
                res_online = log_clustering_experiments_wandb(
                    experiments_data=sample_exp,
                    mode="online",
                )
                assert res_online is True
