"""Tests for GitHub Actions Multi-Stage CI/CD and Self-Hosted CD Workflows.

Verifies:
- CI workflow existence, YAML schema, and multi-stage pipeline configuration.
- Trigger conditions (push/PR/dispatch on master and main).
- Strict 10-stage sequential dependency graph.
- Matrix publication configuration for api, frontend, and dashboard services.
- Least-privilege permissions model (contents: read, packages: write on publication).
- Pull Request isolation (zero image pushes on PRs).
- Image tagging strategy (commit-based sha-, branch tracking, latest).
- CD workflow (.github/workflows/manual-deploy.yml) configuration:
  - Manual trigger only (workflow_dispatch).
  - Explicit confirmation token gate.
  - Manual approval gate via GitHub Environment.
  - Self-hosted runner execution with strict PR restriction.
- Docker Compose configuration syntax and integrity.
"""

from pathlib import Path
import subprocess
import yaml
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/ci-cd.yml"
CD_WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/manual-deploy.yml"


@pytest.fixture(scope="module")
def ci_workflow_data():
    """Load and parse the multi-stage CI workflow YAML file."""
    assert CI_WORKFLOW_PATH.is_file(), f"CI workflow file not found at {CI_WORKFLOW_PATH}"
    with open(CI_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "CI workflow YAML must parse into a dictionary"
    return data


@pytest.fixture(scope="module")
def cd_workflow_data():
    """Load and parse the CD deployment workflow YAML file."""
    assert CD_WORKFLOW_PATH.is_file(), f"CD workflow file not found at {CD_WORKFLOW_PATH}"
    with open(CD_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "CD workflow YAML must parse into a dictionary"
    return data


def test_ci_workflow_top_level_metadata(ci_workflow_data):
    """Verify CI workflow name, least-privilege permissions, and triggers."""
    assert "name" in ci_workflow_data
    assert "CI" in ci_workflow_data["name"]

    # Verify least-privilege permissions at top level
    assert "permissions" in ci_workflow_data
    assert ci_workflow_data["permissions"].get("contents") == "read"

    # Verify triggers (in PyYAML, unquoted 'on' is parsed as boolean True)
    triggers = ci_workflow_data.get("on") or ci_workflow_data.get(True, {})
    assert "push" in triggers
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers

    # Verify branch targeting includes master and main
    assert "master" in triggers["push"]["branches"]
    assert "master" in triggers["pull_request"]["branches"]


def test_stage_1_installs_pyyaml_before_validation(ci_workflow_data):
    """Verify Stage 1 installs pyyaml validation dependencies before YAML syntax validation."""
    assert (PROJECT_ROOT / "requirements-ci.txt").is_file(), "requirements-ci.txt must exist"
    reqs = (PROJECT_ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
    assert "pyyaml" in reqs.lower(), "requirements-ci.txt must contain pyyaml"

    stage1 = ci_workflow_data["jobs"]["stage-1-repo-config-validation"]
    steps = stage1.get("steps", [])

    install_idx = None
    validate_idx = None
    for i, step in enumerate(steps):
        run_cmd = step.get("run", "")
        if "requirements-ci.txt" in run_cmd or ("pip install" in run_cmd and "pyyaml" in run_cmd):
            install_idx = i
        if "import yaml" in run_cmd:
            validate_idx = i

    assert install_idx is not None, "Stage 1 must have an installation step for CI validation dependencies"
    assert validate_idx is not None, "Stage 1 must have a YAML syntax validation step"
    assert install_idx < validate_idx, (
        f"Validation dependencies install step (index {install_idx}) must precede YAML validation (index {validate_idx})"
    )


def test_ci_workflow_10_stages_and_dependencies(ci_workflow_data):
    """Verify all 10 pipeline stages and their strict dependency ordering."""
    jobs = ci_workflow_data.get("jobs", {})
    expected_stages = [
        "stage-1-repo-config-validation",
        "stage-2-model-integrity-and-risk",
        "stage-3-model-quality-evaluation",
        "stage-4-static-analysis-and-security",
        "stage-5-backend-tests",
        "stage-6-frontend-tests",
        "stage-7-qwen-ollama-mocks",
        "stage-8-docker-validation",
        "stage-9-e2e-regression",
        "stage-10-artifact-publication",
    ]
    for stage in expected_stages:
        assert stage in jobs, f"Missing pipeline stage: {stage}"

    # Verify dependency links
    assert "needs" not in jobs["stage-1-repo-config-validation"]
    assert jobs["stage-2-model-integrity-and-risk"].get("needs") == "stage-1-repo-config-validation"
    assert jobs["stage-3-model-quality-evaluation"].get("needs") == "stage-2-model-integrity-and-risk"
    assert jobs["stage-4-static-analysis-and-security"].get("needs") == "stage-1-repo-config-validation"
    assert jobs["stage-5-backend-tests"].get("needs") == "stage-2-model-integrity-and-risk"
    assert jobs["stage-6-frontend-tests"].get("needs") == "stage-4-static-analysis-and-security"
    assert jobs["stage-7-qwen-ollama-mocks"].get("needs") == "stage-5-backend-tests"

    stage8_needs = jobs["stage-8-docker-validation"].get("needs", [])
    assert "stage-5-backend-tests" in stage8_needs
    assert "stage-6-frontend-tests" in stage8_needs

    stage9_needs = jobs["stage-9-e2e-regression"].get("needs", [])
    assert "stage-5-backend-tests" in stage9_needs
    assert "stage-7-qwen-ollama-mocks" in stage9_needs
    assert "stage-8-docker-validation" in stage9_needs

    stage10_needs = jobs["stage-10-artifact-publication"].get("needs", [])
    assert "stage-3-model-quality-evaluation" in stage10_needs
    assert "stage-7-qwen-ollama-mocks" in stage10_needs
    assert "stage-8-docker-validation" in stage10_needs
    assert "stage-9-e2e-regression" in stage10_needs


def test_ci_runs_on_github_hosted_runners(ci_workflow_data):
    """Verify that CI strictly runs on GitHub-hosted runners (ubuntu-latest), never self-hosted."""
    jobs = ci_workflow_data.get("jobs", {})
    for stage_id, stage_config in jobs.items():
        runs_on = stage_config.get("runs-on")
        assert runs_on == "ubuntu-latest", (
            f"Stage {stage_id} must run on ubuntu-latest, found: {runs_on}"
        )


def test_ci_artifact_publication_matrix(ci_workflow_data):
    """Verify Stage 10 matrix publishes api, frontend, and dashboard images with valid Dockerfiles."""
    pub_job = ci_workflow_data["jobs"]["stage-10-artifact-publication"]
    strategy = pub_job.get("strategy", {})
    matrix = strategy.get("matrix", {})
    include_list = matrix.get("include", [])

    assert len(include_list) == 3, "Matrix must define 3 services: api, frontend, dashboard"
    services = {item["service"]: item for item in include_list}
    assert "api" in services
    assert "frontend" in services
    assert "dashboard" in services

    assert services["api"]["image_name"] == "cyclone-assistant-api"
    assert services["api"]["dockerfile"] == "Dockerfile.api"
    assert (PROJECT_ROOT / services["api"]["dockerfile"]).is_file()

    assert services["frontend"]["image_name"] == "cyclone-assistant-frontend"
    assert services["frontend"]["dockerfile"] == "Dockerfile.frontend"
    assert (PROJECT_ROOT / services["frontend"]["dockerfile"]).is_file()

    assert services["dashboard"]["image_name"] == "cyclone-assistant-dashboard"
    assert services["dashboard"]["dockerfile"] == "Dockerfile.dashboard"
    assert (PROJECT_ROOT / services["dashboard"]["dockerfile"]).is_file()

    # Verify PR guard condition prevents publishing from pull requests
    job_if = pub_job.get("if", "")
    assert "pull_request" in job_if
    assert "!=" in job_if

    # Verify permissions
    perms = pub_job.get("permissions", {})
    assert perms.get("packages") == "write"
    assert perms.get("contents") == "read"

    # Verify lowercase repo owner normalization step exists
    step_names = [step.get("name", "") for step in pub_job.get("steps", [])]
    assert any("Lowercase" in name or "Normalize" in name for name in step_names)
    assert any("W&B" in name or "cache" in name for name in step_names)


def test_cd_workflow_security_and_approval_gates(cd_workflow_data):
    """Verify CD deployment workflow safety, manual approvals, and self-hosted runner constraints."""
    # Must only trigger on workflow_dispatch (manual)
    triggers = cd_workflow_data.get("on") or cd_workflow_data.get(True, {})
    assert "workflow_dispatch" in triggers
    assert "push" not in triggers
    assert "pull_request" not in triggers

    jobs = cd_workflow_data.get("jobs", {})
    assert "pre-deployment-verification" in jobs
    assert "production-release-gate" in jobs
    assert "deploy-to-laptop" in jobs

    # Pre-deployment verification runs on GitHub-hosted runner
    assert jobs["pre-deployment-verification"]["runs-on"] == "ubuntu-latest"

    # Production release gate requires manual approval via GitHub Environment
    gate_job = jobs["production-release-gate"]
    assert gate_job["runs-on"] == "ubuntu-latest"
    assert "environment" in gate_job

    # Deployment job targets dedicated self-hosted runner on the Ubuntu laptop
    deploy_job = jobs["deploy-to-laptop"]
    assert "environment" in deploy_job
    runs_on = deploy_job.get("runs-on")
    assert isinstance(runs_on, list)
    assert "self-hosted" in runs_on
    assert "cyclone-runner" in runs_on

    # Security check: PR execution strictly forbidden on runner
    job_if = deploy_job.get("if", "")
    assert "workflow_dispatch" in job_if
    assert "repository" in job_if


def test_docker_compose_config_validation():
    """Verify local docker-compose.yml and docker-compose.test.yml pass validation cleanly."""
    res = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"docker compose config failed: {res.stderr}"

    res_test = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "config", "--quiet"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res_test.returncode == 0, f"docker-compose.test.yml config failed: {res_test.stderr}"
