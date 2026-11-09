"""Tests for GitHub Actions CI/CD workflow configuration and static validation.

Verifies:
- Workflow existence, YAML syntax, and schema completeness.
- Trigger conditions (push, pull_request, workflow_dispatch on default integration branch).
- Strict job dependency graph (build-and-publish strictly depends on test passing).
- Image matrix configuration (cyclone-assistant-api and cyclone-assistant-dashboard).
- Least privilege permissions model (contents: read, packages: write).
- PR security guardrails (no registry login or push on pull_request events).
- Image tagging strategy (SHA-based immutability, branch tracking, latest on default).
- Dockerfile and Compose validation consistency.
"""

from pathlib import Path
import subprocess
import yaml
import pytest

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github/workflows/ci-cd.yml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def workflow_data():
    """Load and parse the CI/CD workflow YAML file."""
    assert WORKFLOW_PATH.is_file(), f"Workflow file not found at {WORKFLOW_PATH}"
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Workflow YAML must parse into a dictionary"
    return data


def test_workflow_top_level_metadata(workflow_data):
    """Verify workflow name, top-level permissions, and triggers."""
    assert "name" in workflow_data
    assert "CI/CD" in workflow_data["name"]

    # Verify least-privilege permissions at top level
    assert "permissions" in workflow_data
    assert workflow_data["permissions"].get("contents") == "read"

    # Verify triggers
    triggers = workflow_data.get("on")
    assert isinstance(triggers, dict)
    assert "push" in triggers
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers

    # Verify branch targeting includes default integration branch 'master'
    assert "master" in triggers["push"]["branches"]
    assert "master" in triggers["pull_request"]["branches"]


def test_workflow_job_graph_and_dependencies(workflow_data):
    """Verify required jobs and strict dependency sequencing."""
    jobs = workflow_data.get("jobs", {})
    assert "test" in jobs
    assert "build-and-publish" in jobs
    assert "deployment-readiness" in jobs

    # Verify test job does not depend on other jobs
    assert "needs" not in jobs["test"]

    # Verify build-and-publish strictly depends on test passing
    assert jobs["build-and-publish"].get("needs") == "test"

    # Verify deployment-readiness depends on both test and build-and-publish
    deploy_needs = jobs["deployment-readiness"].get("needs")
    if isinstance(deploy_needs, list):
        assert "test" in deploy_needs
        assert "build-and-publish" in deploy_needs
    else:
        assert deploy_needs == "build-and-publish"


def test_test_job_configuration(workflow_data):
    """Verify CI test job installs dependencies and runs pytest and compose config."""
    test_job = workflow_data["jobs"]["test"]
    assert test_job["runs-on"] == "ubuntu-latest"
    assert test_job.get("permissions", {}).get("contents") == "read"

    steps = test_job.get("steps", [])
    step_names = [s.get("name", "") for s in steps]
    step_runs = [s.get("run", "") for s in steps]

    # Verify checkout with Git LFS and setup-python
    uses = [s.get("uses", "") for s in steps]
    checkout_steps = [s for s in steps if "actions/checkout" in s.get("uses", "")]
    assert len(checkout_steps) > 0, "actions/checkout step required"
    assert checkout_steps[0].get("with", {}).get("lfs") is True, "actions/checkout must have lfs: true"
    assert any("actions/setup-python" in u for u in uses)

    # Verify requirements.txt installation and pytest invocation
    all_runs = "\n".join(step_runs)
    assert "requirements.txt" in all_runs
    assert "pytest" in all_runs
    assert "docker compose config" in all_runs
    assert ".wandb_artifacts_cache" in all_runs
    assert "data/processed/coastal_vulnerability_archetypes.parquet" in all_runs
    assert "data/processed/modelling_dataset.parquet" in all_runs
    assert "data/processed/master_village_preparedness.parquet" in all_runs


def test_build_and_publish_matrix_configuration(workflow_data):
    """Verify build matrix defines both api and dashboard services with valid Dockerfiles."""
    build_job = workflow_data["jobs"]["build-and-publish"]
    strategy = build_job.get("strategy", {})
    matrix = strategy.get("matrix", {})
    include_list = matrix.get("include", [])

    assert len(include_list) == 2, "Matrix must define exactly two services: api and dashboard"

    services = {item["service"]: item for item in include_list}
    assert "api" in services
    assert "dashboard" in services

    # Verify image names and Dockerfiles
    assert services["api"]["image_name"] == "cyclone-assistant-api"
    assert services["api"]["dockerfile"] == "Dockerfile.api"
    assert (PROJECT_ROOT / services["api"]["dockerfile"]).is_file()

    assert services["dashboard"]["image_name"] == "cyclone-assistant-dashboard"
    assert services["dashboard"]["dockerfile"] == "Dockerfile.dashboard"
    assert (PROJECT_ROOT / services["dashboard"]["dockerfile"]).is_file()

    # Verify checkout with Git LFS and cache seeding in build job
    steps = build_job.get("steps", [])
    checkout_steps = [s for s in steps if "actions/checkout" in s.get("uses", "")]
    assert len(checkout_steps) > 0, "actions/checkout step required in build job"
    assert checkout_steps[0].get("with", {}).get("lfs") is True, "build checkout must have lfs: true"

    step_runs = "\n".join(s.get("run", "") for s in steps)
    assert ".wandb_artifacts_cache" in step_runs, "build job must seed offline W&B artifact cache"


def test_security_and_pull_request_guardrails(workflow_data):
    """Verify that images are NEVER pushed from untrusted pull requests."""
    build_job = workflow_data["jobs"]["build-and-publish"]
    steps = build_job.get("steps", [])

    # Check login step condition
    login_step = next((s for s in steps if "login-action" in s.get("uses", "")), None)
    assert login_step is not None, "Login step must be present"
    login_if = login_step.get("if", "")
    assert "pull_request" in login_if
    assert "!=" in login_if or "github.event_name != 'pull_request'" in login_if

    # Check build-push step condition
    build_step = next((s for s in steps if "build-push-action" in s.get("uses", "")), None)
    assert build_step is not None, "Build-push step must be present"
    push_condition = build_step.get("with", {}).get("push", "")
    assert "pull_request" in push_condition
    assert "!=" in push_condition

    # Verify least-privilege packages: write permission on build job
    permissions = build_job.get("permissions", {})
    assert permissions.get("packages") == "write"
    assert permissions.get("contents") == "read"


def test_image_tagging_strategy(workflow_data):
    """Verify image tagging includes SHA-based immutable tags and branch tracking."""
    build_job = workflow_data["jobs"]["build-and-publish"]
    steps = build_job.get("steps", [])
    meta_step = next((s for s in steps if "metadata-action" in s.get("uses", "")), None)
    assert meta_step is not None, "Docker metadata action must be configured"

    tags_config = meta_step.get("with", {}).get("tags", "")
    assert "type=sha" in tags_config
    assert "prefix=sha-" in tags_config
    assert "type=ref,event=branch" in tags_config
    assert "latest" in tags_config

    images_config = meta_step.get("with", {}).get("images", "")
    assert "ghcr.io" in images_config


def test_docker_compose_config_validation():
    """Verify local docker-compose.yml passes validation cleanly without errors."""
    res = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"docker compose config failed: {res.stderr}"
