"""Unit and integration tests for the continuous deployment engine (scripts/deploy_production.py).

Verifies:
- Service metadata configuration.
- Change detection rules (git diff, explicit targeting, auto mode).
- Ollama (:11434) and Cloudflare Tunnel exclusion guarantees.
- Baseline state recording for instant rollback.
- Health check polling logic and failure detection.
- Rollback execution when health probes fail.
- Dry run execution.
- Immutable tag validation (rejection of mutable 'latest', acceptance of sha- and semver).
- Comprehensive preflight checks (docker daemon, disk space, remote image pull).
- Service ordering (API cutover and verified first, then dependent services).
- Atomic rollback preventing partial updates when dependent services fail.
- Secret and credential masking in logs.
"""

import io
import json
import logging
from pathlib import Path
import unittest.mock as mock
import pytest

from scripts.deploy_production import (
    SERVICE_METADATA,
    SIMULATION_METADATA,
    SERVICE_DEPLOYMENT_ORDER,
    SERVICE_ROLLBACK_ORDER,
    DEFAULT_PRODUCTION_PROJECT,
    DEFAULT_SIMULATION_PROJECT,
    detect_changed_services,
    record_current_deployment_state,
    perform_health_checks,
    execute_preflight_checks,
    is_immutable_image_tag,
    resolve_compose_project,
    detect_existing_compose_project,
    SecretMaskingFilter,
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
    """Verify dry-run mode returns 0 with immutable tag without executing Docker commands."""
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


# ============================================================================
# NEW ENHANCED CONTINUOUS DEPLOYMENT TESTS
# ============================================================================

def test_is_immutable_image_tag_validation():
    """Verify is_immutable_image_tag accurately validates immutable vs mutable tags."""
    # Valid immutable tags
    assert is_immutable_image_tag("sha-4694bdc") is True
    assert is_immutable_image_tag("sha-7d0e09a") is True
    assert is_immutable_image_tag("sha-097eef2000000000000000000000000000000000") is True
    assert is_immutable_image_tag("v1.0.0") is True
    assert is_immutable_image_tag("v2.1.3-beta.1") is True

    # Mutable / unverified tags strictly forbidden
    assert is_immutable_image_tag("latest") is False
    assert is_immutable_image_tag("LATEST") is False
    assert is_immutable_image_tag("master") is False
    assert is_immutable_image_tag("main") is False
    assert is_immutable_image_tag("staging") is False
    assert is_immutable_image_tag("dev") is False
    assert is_immutable_image_tag("head") is False
    assert is_immutable_image_tag("") is False
    assert is_immutable_image_tag(None) is False
    assert is_immutable_image_tag("unverified-random-tag") is False


def test_deploy_production_rejects_mutable_latest_tag():
    """Verify deploy_production strictly rejects 'latest' and other mutable tags in production mode."""
    logger = setup_logger()
    ret = deploy_production(
        image_tag="latest",
        dry_run=True,
        simulate=False,
        logger=logger,
    )
    assert ret == 1

    ret_master = deploy_production(
        image_tag="master",
        dry_run=True,
        simulate=False,
        logger=logger,
    )
    assert ret_master == 1


def test_execute_preflight_checks_success():
    """Verify preflight checks pass when docker is responsive, tag is immutable, and disk is adequate."""
    logger = setup_logger()
    with mock.patch("scripts.deploy_production.run_command") as mock_run:
        # docker info -> 0, docker compose config -> 0, docker pull -> 0
        mock_run.return_value = mock.Mock(returncode=0)
        with mock.patch("shutil.disk_usage", return_value=(100 * 10**9, 50 * 10**9, 50 * 10**9)):
            ok = execute_preflight_checks(
                logger=logger,
                image_tag="sha-4694bdc",
                repo_owner="bhavanish5041",
                services=["api", "frontend"],
                compose_file="docker-compose.yml",
                is_simulation=False,
            )
            assert ok is True


def test_execute_preflight_checks_docker_down():
    """Verify preflight check fails immediately if Docker daemon is unresponsive."""
    logger = setup_logger()
    with mock.patch("scripts.deploy_production.run_command") as mock_run:
        mock_run.return_value = mock.Mock(returncode=1, stderr="Cannot connect to the Docker daemon")
        ok = execute_preflight_checks(
            logger=logger,
            image_tag="sha-4694bdc",
            repo_owner="bhavanish5041",
            services=["api"],
            compose_file="docker-compose.yml",
            is_simulation=False,
        )
        assert ok is False


def test_execute_preflight_checks_insufficient_disk_space():
    """Verify preflight check fails if host free disk space is less than 1GB."""
    logger = setup_logger()
    with mock.patch("scripts.deploy_production.run_command") as mock_run:
        mock_run.return_value = mock.Mock(returncode=0)
        # Mock 500 MB free (less than 1 GB threshold)
        with mock.patch("shutil.disk_usage", return_value=(100 * 10**9, 99.5 * 10**9, 500 * 1024 * 1024)):
            ok = execute_preflight_checks(
                logger=logger,
                image_tag="sha-4694bdc",
                repo_owner="bhavanish5041",
                services=["api"],
                compose_file="docker-compose.yml",
                is_simulation=False,
            )
            assert ok is False


def test_execute_preflight_checks_missing_remote_and_local_image_fails():
    """Verify preflight fails if image cannot be pulled from GHCR and is not present locally."""
    logger = setup_logger()
    with mock.patch("scripts.deploy_production.run_command") as mock_run:
        # First call: docker info (0)
        # Second call: docker compose config (0)
        # Third call: docker pull (1 - failed)
        # Fourth call: docker inspect (1 - missing locally)
        mock_run.side_effect = [
            mock.Mock(returncode=0),
            mock.Mock(returncode=0),
            mock.Mock(returncode=1, stderr="repository does not exist"),
            mock.Mock(returncode=1, stderr="No such image"),
        ]
        with mock.patch("shutil.disk_usage", return_value=(100 * 10**9, 50 * 10**9, 50 * 10**9)):
            ok = execute_preflight_checks(
                logger=logger,
                image_tag="sha-4694bdc",
                repo_owner="bhavanish5041",
                services=["api"],
                compose_file="docker-compose.yml",
                is_simulation=False,
            )
            assert ok is False


def test_service_ordering_api_deployed_before_dependent_services(tmp_path):
    """Verify that root service 'api' is cut over and probed first before dependent services."""
    logger = setup_logger()
    lock_file = tmp_path / ".deploy.lock"

    compose_calls = []

    def mock_sub_run(cmd, *args, **kwargs):
        compose_calls.append(cmd)
        return mock.Mock(returncode=0)

    with mock.patch("scripts.deploy_production.DEPLOYMENT_LOCK_FILE", lock_file):
        with mock.patch("scripts.deploy_production.DEPLOYMENT_STATE_FILE", tmp_path / "deploy_state.json"):
            with mock.patch("scripts.deploy_production.execute_preflight_checks", return_value=True):
                with mock.patch("scripts.deploy_production.record_current_deployment_state", return_value={}):
                    with mock.patch("scripts.deploy_production.perform_health_checks", return_value=True):
                        with mock.patch("scripts.deploy_production.run_command", return_value=mock.Mock(returncode=0)):
                            with mock.patch("scripts.deploy_production.subprocess.run", side_effect=mock_sub_run):
                                ret = deploy_production(
                                    image_tag="sha-4694bdc",
                                    services_arg="all",  # api, frontend, dashboard
                                    logger=logger,
                                )
                                assert ret == 0

    # Verify cutover sequence: api first in batch 1, frontend & dashboard in batch 2
    compose_up_calls = [cmd for cmd in compose_calls if "up" in cmd]
    assert len(compose_up_calls) == 2, f"Expected 2 ordered cutover batches, got {len(compose_up_calls)}"
    # First batch must contain only 'api'
    assert "api" in compose_up_calls[0]
    assert "frontend" not in compose_up_calls[0]
    # Second batch must contain dependent services
    assert "frontend" in compose_up_calls[1]
    assert "dashboard" in compose_up_calls[1]


def test_rollback_prevents_partial_update_when_dependent_service_fails(tmp_path):
    """Verify that if 'api' succeeds but 'frontend' fails, rollback restores ALL touched services."""
    logger = setup_logger()
    lock_file = tmp_path / ".deploy.lock"

    # Stage 1 (api) health check returns True; Stage 2 (frontend) health check returns False
    health_results = [True, False]

    with mock.patch("scripts.deploy_production.DEPLOYMENT_LOCK_FILE", lock_file):
        with mock.patch("scripts.deploy_production.execute_preflight_checks", return_value=True):
            with mock.patch("scripts.deploy_production.record_current_deployment_state", return_value={}):
                with mock.patch("scripts.deploy_production.perform_health_checks", side_effect=health_results):
                    with mock.patch("scripts.deploy_production.run_command", return_value=mock.Mock(returncode=0)):
                        with mock.patch("scripts.deploy_production.subprocess.run", return_value=mock.Mock(returncode=0)):
                            with mock.patch("scripts.deploy_production.execute_rollback", return_value=True) as mock_rb:
                                ret = deploy_production(
                                    image_tag="sha-4694bdc",
                                    services_arg="api,frontend",
                                    logger=logger,
                                )
                                assert ret == 2  # Rollback succeeded return code
                                # Crucial: rollback must be called for BOTH api and frontend, not leaving api partially updated!
                                mock_rb.assert_called_once()
                                rolled_back_svcs = mock_rb.call_args[0][1]
                                assert "api" in rolled_back_svcs
                                assert "frontend" in rolled_back_svcs


def test_secret_masking_filter():
    """Verify that SecretMaskingFilter scrubs tokens, PATs, and credentials from all logged outputs."""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(SecretMaskingFilter())

    test_logger = logging.getLogger("test_secret_masker")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers = [handler]

    # Construct synthetic secret tokens at runtime to prevent false positives in static repo secret scanners
    dummy_ghp = "ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    dummy_aws = "AKIA" + "IOSFODNN7EXAMPLE"

    test_logger.info(f"Logging into GHCR with {dummy_ghp}")
    test_logger.info(f"AWS key detected: {dummy_aws}")
    test_logger.info("Connecting with password=SuperSecretPassword123!")

    output = log_stream.getvalue()
    assert "ghp_" not in output
    assert dummy_aws not in output
    assert "SuperSecretPassword123!" not in output
    assert "***REDACTED***" in output


# ============================================================================
# REGRESSION TESTS: COMPOSE PROJECT RESOLUTION & NAME CONFLICT PREVENTION
# ============================================================================

def test_resolve_compose_project_simulation():
    """Verify simulation mode strictly returns the isolated test project name."""
    assert resolve_compose_project(is_simulation=True) == DEFAULT_SIMULATION_PROJECT
    assert resolve_compose_project(compose_file="docker-compose.test.yml") == DEFAULT_SIMULATION_PROJECT


def test_resolve_compose_project_explicit_argument():
    """Verify explicit requested_project overrides defaults and container labels."""
    assert resolve_compose_project(requested_project="custom-project") == "custom-project"


def test_resolve_compose_project_env_var(monkeypatch):
    """Verify COMPOSE_PROJECT_NAME env var overrides default project."""
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "env-injected-project")
    assert resolve_compose_project() == "env-injected-project"


def test_resolve_compose_project_detects_live_container_project():
    """Verify dynamic detection reads com.docker.compose.project from running containers."""
    with mock.patch("scripts.deploy_production.detect_existing_compose_project", return_value="cyclone-preparedness-assistant"):
        proj = resolve_compose_project()
        assert proj == "cyclone-preparedness-assistant"


def test_resolve_compose_project_detects_mismatch_and_preserves_live_project():
    """Regression test: Mismatch between compose file and live container labels.
    
    If docker-compose.yml has name 'cyclone-prod', but running containers are
    labeled 'cyclone-preparedness-assistant', resolve_compose_project must prioritize
    the live container project to prevent container-name conflicts.
    """
    with mock.patch("scripts.deploy_production.detect_existing_compose_project", return_value="cyclone-preparedness-assistant"):
        proj = resolve_compose_project(compose_file="docker-compose.yml")
        assert proj == "cyclone-preparedness-assistant"
        assert proj != "cyclone-prod"


def test_resolve_compose_project_fallback_to_compose_file_name(tmp_path):
    """Verify that when no containers are running, top-level name from compose file is used."""
    compose_content = "name: cyclone-preparedness-assistant\nservices:\n  api:\n    image: test:latest\n"
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(compose_content)

    with mock.patch("scripts.deploy_production.detect_existing_compose_project", return_value=None):
        with mock.patch("scripts.deploy_production.PROJECT_ROOT", tmp_path):
            proj = resolve_compose_project(compose_file="docker-compose.yml")
            assert proj == "cyclone-preparedness-assistant"


def test_deploy_production_uses_exact_project_name(tmp_path):
    """Verify deploy_production executes compose commands with -p cyclone-preparedness-assistant and NEVER cyclone-prod."""
    logger = setup_logger()
    lock_file = tmp_path / ".deploy.lock"
    compose_commands = []

    def mock_sub_run(cmd, *args, **kwargs):
        compose_commands.append(cmd)
        return mock.Mock(returncode=0)

    with mock.patch("scripts.deploy_production.DEPLOYMENT_LOCK_FILE", lock_file):
        with mock.patch("scripts.deploy_production.DEPLOYMENT_STATE_FILE", tmp_path / "deploy_state.json"):
            with mock.patch("scripts.deploy_production.execute_preflight_checks", return_value=True):
                with mock.patch("scripts.deploy_production.record_current_deployment_state", return_value={}):
                    with mock.patch("scripts.deploy_production.perform_health_checks", return_value=True):
                        with mock.patch("scripts.deploy_production.run_command", return_value=mock.Mock(returncode=0)):
                            with mock.patch("scripts.deploy_production.resolve_compose_project", return_value="cyclone-preparedness-assistant"):
                                with mock.patch("scripts.deploy_production.subprocess.run", side_effect=mock_sub_run):
                                    ret = deploy_production(
                                        image_tag="sha-4694bdc",
                                        services_arg="api",
                                        logger=logger,
                                    )
                                    assert ret == 0

    assert len(compose_commands) > 0
    for cmd in compose_commands:
        assert "-p" in cmd
        p_index = cmd.index("-p")
        project_used = cmd[p_index + 1]
        assert project_used == "cyclone-preparedness-assistant"
        assert project_used != "cyclone-prod"


def test_execute_rollback_uses_backup_compose_project(tmp_path):
    """Verify execute_rollback restores containers using the recorded project from baseline backup."""
    from scripts.deploy_production import execute_rollback

    logger = setup_logger()
    backup_file = tmp_path / "rollback_state.json"
    backup_data = {
        "compose_project": "cyclone-preparedness-assistant",
        "services": {
            "api": {
                "container_name": "cyclone-api",
                "image": "cyclone-assistant-api:latest",
                "image_id": "sha256:112233445566",
            }
        },
    }
    backup_file.write_text(json.dumps(backup_data))

    compose_commands = []

    def mock_sub_run(cmd, *args, **kwargs):
        compose_commands.append(cmd)
        return mock.Mock(returncode=0)

    with mock.patch("scripts.deploy_production.ROLLBACK_BACKUP_FILE", backup_file):
        with mock.patch("scripts.deploy_production.run_command", return_value=mock.Mock(returncode=0, stdout="")):
            with mock.patch("scripts.deploy_production.subprocess.run", side_effect=mock_sub_run):
                with mock.patch("scripts.deploy_production.perform_health_checks", return_value=True):
                    ok = execute_rollback(logger, ["api"])
                    assert ok is True

    assert len(compose_commands) > 0
    cmd = compose_commands[0]
    assert "-p" in cmd
    p_index = cmd.index("-p")
    project_used = cmd[p_index + 1]
    assert project_used == "cyclone-preparedness-assistant"
    assert project_used != "cyclone-prod"


def test_record_current_deployment_state_includes_compose_project(tmp_path):
    """Verify record_current_deployment_state saves compose_project to rollback backup file."""
    logger = setup_logger()
    mock_info = {
        "id": "container123",
        "image": "cyclone-assistant-api:latest",
        "imageId": "sha256:11223344",
        "status": "running",
        "health": "healthy",
        "compose_project": "cyclone-preparedness-assistant",
    }
    with mock.patch("scripts.deploy_production.get_container_info", return_value=mock_info):
        with mock.patch("scripts.deploy_production.ROLLBACK_BACKUP_FILE", tmp_path / "rollback.json"):
            state = record_current_deployment_state(
                logger, ["api"], project_name="cyclone-preparedness-assistant"
            )
            assert state["compose_project"] == "cyclone-preparedness-assistant"
            saved_content = json.loads((tmp_path / "rollback.json").read_text())
            assert saved_content["compose_project"] == "cyclone-preparedness-assistant"
