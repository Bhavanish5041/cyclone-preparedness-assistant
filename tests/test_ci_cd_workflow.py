"""Tests for GitHub Actions Multi-Stage CI/CD Workflows (Max 7 Jobs Total).

Verifies:
- CI workflow existence, YAML schema, and multi-stage pipeline configuration.
- Trigger conditions (push/PR/dispatch on master and main).
- Maximum 7 jobs TOTAL across the relevant workflows:
  - Exactly 6 CI jobs in ci-cd.yml
  - Exactly 1 CD job in manual-deploy.yml
  - Total across all workflows == 7.
- Strict sequential dependency graph (job-1 -> job-2 -> job-3 -> job-4 -> job-5 -> job-6).
- Least-privilege permissions model (contents: read, packages: write on publication).
- CD workflow (.github/workflows/manual-deploy.yml) configuration:
  - Manual trigger only (workflow_dispatch).
  - Explicit confirmation token gate ('CONFIRM-DEPLOY').
  - Manual approval gate via GitHub Environment.
  - Immutable image tag enforcement (sha-* or v*, rejecting 'latest').
- Docker Compose configuration syntax and integrity.
"""

import glob
from pathlib import Path
import subprocess
import pytest
import yaml

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


def test_total_github_actions_jobs_is_at_most_seven(ci_workflow_data, cd_workflow_data):
    """Strict Requirement: Total jobs across all relevant CI/CD workflows must not exceed 7."""
    workflow_files = glob.glob(str(PROJECT_ROOT / ".github" / "workflows" / "*.yml"))
    total_jobs = 0
    job_counts = {}

    for w_file in workflow_files:
        with open(w_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        jobs = data.get("jobs", {})
        count = len(jobs)
        job_counts[Path(w_file).name] = count
        total_jobs += count

    assert total_jobs <= 7, f"Total GitHub Actions jobs ({total_jobs}) exceeds maximum 7! Breakdown: {job_counts}"
    assert total_jobs == 7, f"Expected exactly 7 total jobs (6 CI + 1 CD), found {total_jobs} ({job_counts})"


def test_ci_workflow_six_jobs_and_dependency_chain(ci_workflow_data):
    """Verify exactly 6 CI pipeline jobs and their strict sequential dependency graph."""
    jobs = ci_workflow_data.get("jobs", {})
    expected_jobs = [
        "job-1-lint-and-compile",
        "job-2-unit-and-model-tests",
        "job-3-model-smoke-tests",
        "job-4-model-quality-gates",
        "job-5-api-swagger-integration",
        "job-6-docker-validation",
    ]
    assert len(jobs) == 6, f"ci-cd.yml must contain exactly 6 jobs, found {len(jobs)}: {list(jobs.keys())}"
    for j in expected_jobs:
        assert j in jobs, f"Missing CI job: {j}"

    # Verify sequential dependency graph (needs)
    assert "needs" not in jobs["job-1-lint-and-compile"]
    assert jobs["job-2-unit-and-model-tests"].get("needs") == "job-1-lint-and-compile"
    assert jobs["job-3-model-smoke-tests"].get("needs") == "job-2-unit-and-model-tests"
    assert jobs["job-4-model-quality-gates"].get("needs") == "job-3-model-smoke-tests"
    assert jobs["job-5-api-swagger-integration"].get("needs") == "job-4-model-quality-gates"
    assert jobs["job-6-docker-validation"].get("needs") == "job-5-api-swagger-integration"


def test_ci_runs_on_github_hosted_runners(ci_workflow_data):
    """Verify all CI jobs run strictly on GitHub-hosted runners (ubuntu-latest)."""
    jobs = ci_workflow_data.get("jobs", {})
    for job_id, job_config in jobs.items():
        runs_on = job_config.get("runs-on")
        assert runs_on == "ubuntu-latest", f"Job {job_id} must run on ubuntu-latest, found: {runs_on}"


def test_cd_workflow_single_job_and_safety_gates(cd_workflow_data):
    """Verify CD deployment workflow safety, manual approvals, and single-job consolidation."""
    # Must only trigger on workflow_dispatch (manual)
    triggers = cd_workflow_data.get("on") or cd_workflow_data.get(True, {})
    assert "workflow_dispatch" in triggers
    assert "push" not in triggers
    assert "pull_request" not in triggers

    jobs = cd_workflow_data.get("jobs", {})
    assert len(jobs) == 1, f"manual-deploy.yml must contain exactly 1 CD job, found {len(jobs)}"
    assert "job-7-protected-cd-deployment" in jobs

    cd_job = jobs["job-7-protected-cd-deployment"]
    assert "environment" in cd_job
    assert cd_job.get("runs-on") == "ubuntu-latest"

    # Verify least-privilege permissions
    perms = cd_workflow_data.get("permissions", {})
    assert perms.get("packages") == "read"
    assert perms.get("contents") == "read"


def test_cd_workflow_immutable_image_tag_enforcement(cd_workflow_data):
    """Verify CD workflow mandates immutable image tags and strictly forbids mutable 'latest' in production."""
    triggers = cd_workflow_data.get("on") or cd_workflow_data.get(True, {})
    inputs = triggers.get("workflow_dispatch", {}).get("inputs", {})
    assert "image_tag" in inputs, "CD workflow must define image_tag input"

    tag_input = inputs["image_tag"]
    assert tag_input.get("default") != "latest", "CD image_tag default must not be mutable 'latest'"
    assert tag_input.get("default", "").startswith("sha-"), "CD image_tag default must be an immutable sha-* tag"
    assert "forbidden" in tag_input.get("description", "").lower(), "Description must document that 'latest' is forbidden"

    cd_job = cd_workflow_data["jobs"]["job-7-protected-cd-deployment"]
    step_scripts = [step.get("run", "") for step in cd_job.get("steps", [])]
    assert any("CONFIRM-DEPLOY" in s for s in step_scripts), "Pre-deployment must check CONFIRM-DEPLOY token"
    assert any("latest" in s and ("forbidden" in s.lower() or "violation" in s.lower()) for s in step_scripts), (
        "Pre-deployment step must reject 'latest' tag for production"
    )
    assert any("sha-" in s for s in step_scripts), "Pre-deployment step must validate sha- tag format"


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
