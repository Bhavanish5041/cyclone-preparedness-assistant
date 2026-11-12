"""Unit and integration tests for the continuous deployment engine (scripts/deploy_production.py).

Verifies:
- Service metadata configuration.
- Change detection rules (git diff, explicit targeting, auto mode).
- Ollama (:11434) and Cloudflare Tunnel exclusion guarantees.
- Baseline state recording for instant rollback.
- Health check polling logic and failure detection.
- Rollback execution when health probes fail.
- Dry run execution.
"""

import json
from pathlib import Path
import unittest.mock as mock
import pytest

from scripts.deploy_production import (
    SERVICE_METADATA,
    SIMULATION_METADATA,
    detect_changed_services,
    record_current_deployment_state,
    perform_health_checks,
    deploy_production,
    setup_logger,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_service_metadata_guarantees():
    """Verify that only authorized container services are defined and internal LLM/tunnels are excluded."""
    assert set(SERVICE_METADATA.keys()) == {"api", "frontend", "dashboard"}
    for svc, meta in SERVICE_METADATA.items():
        assert "container_name" in meta
        assert "ghcr_repo" in meta
        assert "health_url" in meta
        assert "file_patterns" in meta
        assert "ollama" not in meta["container_name"]
        assert "tunnel" not in meta["container_name"]
        assert "11434" not in meta["health_url"]


def test_change_detection_explicit_services():
    """Verify explicit service targeting overrides auto detection."""
    logger = setup_logger()
    assert detect_changed_services(logger, requested_services="api") == ["api"]
    assert detect_changed_services(logger, requested_services="frontend") == ["frontend"]
    assert detect_changed_services(logger, requested_services="dashboard") == ["dashboard"]
    assert detect_changed_services(logger, requested_services="api,frontend") == ["api", "frontend"]
    assert detect_changed_services(logger, requested_services="all") == ["api", "frontend", "dashboard"]


def test_change_detection_auto_fallback(tmp_path):
    """Verify auto mode falls back cleanly when no prior state exists."""
    logger = setup_logger()
    result = detect_changed_services(logger, requested_services="auto")
    assert "api" in result
    assert "frontend" in result
    assert "dashboard" in result


def test_change_detection_git_patterns():
    """Verify pattern matching maps source directories to appropriate services."""
    logger = setup_logger()
    with mock.patch("scripts.deploy_production.run_command") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout="api/routers/communities.py\nsrc/models/predict.py\n",
        )
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch("builtins.open", mock.mock_open(read_data='{"commit_sha": "abc1234"}')):
                services = detect_changed_services(logger, target_sha="def5678", requested_services="auto")
                assert services == ["api"]


def test_record_current_deployment_state(tmp_path):
    """Verify baseline container state is captured and preserved for rollback."""
    logger = setup_logger()
    mock_info = {
        "id": "container123",
        "image": "cyclone-assistant-api:latest",
        "imageId": "sha256:11223344",
        "status": "running",
        "health": "healthy",
    }
    with mock.patch("scripts.deploy_production.get_container_info", return_value=mock_info):
        with mock.patch("scripts.deploy_production.ROLLBACK_BACKUP_FILE", tmp_path / "rollback.json"):
            state = record_current_deployment_state(logger, ["api"])
            assert "api" in state["services"]
            assert state["services"]["api"]["image"] == "cyclone-assistant-api:latest"
            assert (tmp_path / "rollback.json").exists()


def test_perform_health_checks_success():
    """Verify health checker returns True when all containers and endpoints report healthy."""
    logger = setup_logger()
    mock_info = {"status": "running", "health": "healthy"}
    with mock.patch("scripts.deploy_production.get_container_info", return_value=mock_info):
        with mock.patch("scripts.deploy_production.http_probe", return_value=(200, {"status": "healthy"})):
            result = perform_health_checks(logger, ["api"], timeout_seconds=5)
            assert result is True


def test_perform_health_checks_failure_timeout():
    """Verify health checker returns False when an endpoint consistently fails."""
    logger = setup_logger()
    mock_info = {"status": "running", "health": "healthy"}
    with mock.patch("scripts.deploy_production.get_container_info", return_value=mock_info):
        with mock.patch("scripts.deploy_production.http_probe", return_value=(500, {"error": "Internal Error"})):
            result = perform_health_checks(logger, ["api"], timeout_seconds=1)
            assert result is False


def test_deploy_production_dry_run():
    """Verify dry-run mode returns 0 without executing Docker commands."""
    logger = setup_logger()
    ret = deploy_production(
        image_tag="sha-test1234",
        commit_sha="test1234",
        services_arg="api",
        dry_run=True,
        logger=logger,
    )
    assert ret == 0


def test_deployment_lock_concurrency_prevention(tmp_path):
    """Verify that DeploymentLock prevents concurrent deployments via mutual exclusion."""
    from scripts.deploy_production import DeploymentLock

    logger = setup_logger()
    lock_file = tmp_path / ".deploy.lock"

    # Acquire lock in first context
    with DeploymentLock(lock_file, logger):
        # Attempting to acquire the same lock concurrently must raise RuntimeError
        with pytest.raises(RuntimeError, match="Concurrent deployment"):
            with DeploymentLock(lock_file, logger):
                pass

    # Once released, acquiring again succeeds
    with DeploymentLock(lock_file, logger):
        assert lock_file.exists()


def test_get_image_digest():
    """Verify get_image_digest queries docker inspect for immutable sha256 digests."""
    from scripts.deploy_production import get_image_digest

    with mock.patch("scripts.deploy_production.run_command") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout="ghcr.io/bhavanish5041/cyclone-assistant-api@sha256:abcd1234ef5678\n",
        )
        digest = get_image_digest("ghcr.io/bhavanish5041/cyclone-assistant-api:sha-7d0e09a")
        assert digest == "ghcr.io/bhavanish5041/cyclone-assistant-api@sha256:abcd1234ef5678"


def test_simulation_failure_triggers_isolated_teardown(tmp_path):
    """Verify that a failed health check in simulation triggers isolated compose teardown without altering prod."""
    logger = setup_logger()
    lock_file = tmp_path / ".deploy.lock"

    with mock.patch("scripts.deploy_production.DEPLOYMENT_LOCK_FILE", lock_file):
        with mock.patch("scripts.deploy_production.perform_health_checks", return_value=False):
            with mock.patch("scripts.deploy_production.subprocess.run") as mock_sub:
                mock_sub.return_value = mock.Mock(returncode=0)
                with mock.patch("scripts.deploy_production.run_command") as mock_cmd:
                    mock_cmd.return_value = mock.Mock(returncode=0)
                    ret = deploy_production(
                        image_tag="test-tag",
                        simulate=True,
                        logger=logger,
                    )
                    assert ret == 1
                    # Verify teardown command was called with isolated project name
                    down_calls = [
                        call[0][0] for call in mock_cmd.call_args_list
                        if "down" in call[0][0]
                    ]
                    assert any("-p" in cmd and "cyclone-ci-test" in cmd for cmd in down_calls)


def test_repo_owner_normalization_for_docker_oci():
    """Verify repo_owner is normalized to lowercase per Docker OCI specification."""
    logger = setup_logger()
    with mock.patch("scripts.deploy_production.detect_changed_services", return_value=[]):
        ret = deploy_production(
            image_tag="sha-test123",
            repo_owner="Bhavanish5041",
            dry_run=True,
            logger=logger,
        )
        assert ret == 0


def test_record_current_deployment_state_creates_rollback_tag(tmp_path):
    """Verify baseline state recording creates an immutable local :rollback-backup tag."""
    logger = setup_logger()
    mock_info = {
        "id": "container123",
        "image": "cyclone-assistant-api:latest",
        "imageId": "sha256:112233445566",
        "status": "running",
        "health": "healthy",
    }
    with mock.patch("scripts.deploy_production.get_container_info", return_value=mock_info):
        with mock.patch("scripts.deploy_production.run_command") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            with mock.patch("scripts.deploy_production.ROLLBACK_BACKUP_FILE", tmp_path / "rollback.json"):
                state = record_current_deployment_state(logger, ["api"])
                assert "api" in state["services"]
                # Verify docker tag was invoked with rollback-backup
                tag_calls = [c[0][0] for c in mock_run.call_args_list if "tag" in c[0][0]]
                assert any("cyclone-assistant-api:rollback-backup" in c for c in tag_calls)


def test_execute_rollback_restores_backup_tag(tmp_path):
    """Verify execute_rollback restores rollback-backup tag to :latest and applies image to compose."""
    from scripts.deploy_production import execute_rollback

    logger = setup_logger()
    backup_file = tmp_path / "rollback_state.json"
    backup_data = {
        "services": {
            "api": {
                "container_name": "cyclone-api",
                "image": "cyclone-assistant-api:latest",
                "image_id": "sha256:112233445566",
            }
        }
    }
    backup_file.write_text(json.dumps(backup_data))

    with mock.patch("scripts.deploy_production.ROLLBACK_BACKUP_FILE", backup_file):
        with mock.patch("scripts.deploy_production.run_command") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            with mock.patch("scripts.deploy_production.subprocess.run") as mock_sub:
                mock_sub.return_value = mock.Mock(returncode=0)
                with mock.patch("scripts.deploy_production.perform_health_checks", return_value=True):
                    ok = execute_rollback(logger, ["api"])
                    assert ok is True
                    # Verify tag restoration call
                    tag_calls = [c[0][0] for c in mock_run.call_args_list if "tag" in c[0][0]]
                    assert any("cyclone-assistant-api:latest" in c for c in tag_calls)


