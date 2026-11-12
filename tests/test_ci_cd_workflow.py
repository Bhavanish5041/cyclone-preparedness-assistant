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

    # Permissions check: Least privilege read access to packages and contents
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

    # Pre-deployment verification job must enforce immutable tag check
    verify_job = cd_workflow_data["jobs"]["pre-deployment-verification"]
    step_scripts = [step.get("run", "") for step in verify_job.get("steps", [])]
    assert any("CONFIRM-DEPLOY" in s for s in step_scripts), "Pre-deployment must check CONFIRM-DEPLOY token"
    assert any("latest" in s and ("forbidden" in s.lower() or "violation" in s.lower()) for s in step_scripts), (
        "Pre-deployment step must reject 'latest' tag for production"
    )
    assert any("sha-" in s for s in step_scripts), "Pre-deployment step must validate sha- tag format"


def test_cd_workflow_checkout_ref_no_literal_head(cd_workflow_data):
    """Regression test: Ensure CD workflow never passes literal 'HEAD' as actions/checkout ref.

    Detects and prevents actions/checkout failures where git attempts to fetch
    '+refs/heads/HEAD*:refs/remotes/origin/HEAD*'.

    Verifies:
    - workflow_dispatch default for git_commit_sha is NOT literal 'HEAD'.
    - image_tag default is independently set to immutable 'sha-1221e7c'.
    - Production deployment remains protected by the production-release-gate approval gate.
    - No actions/checkout step in manual-deploy.yml passes literal 'HEAD' or raw unvalidated input.
    - All checkout steps use a safe resolved ref (target_commit or github.sha).
    - Pre-deployment verification step safely resolves empty or 'HEAD' input to ${{ github.sha }}.
    """
    triggers = cd_workflow_data.get("on") or cd_workflow_data.get(True, {})
    inputs = triggers.get("workflow_dispatch", {}).get("inputs", {})

    # 1. Verify git_commit_sha default is not 'HEAD'
    commit_input = inputs.get("git_commit_sha", {})
    assert commit_input.get("default") != "HEAD", "git_commit_sha default must not be literal 'HEAD'"

    # 2. Verify image_tag default is sha-1221e7c
    image_tag_input = inputs.get("image_tag", {})
    assert image_tag_input.get("default") == "sha-1221e7c", (
        f"image_tag default must be 'sha-1221e7c', found '{image_tag_input.get('default')}'"
    )

    jobs = cd_workflow_data.get("jobs", {})

    # 3. Verify deploy-to-laptop depends on production-release-gate approval
    deploy_job = jobs.get("deploy-to-laptop", {})
    deploy_needs = deploy_job.get("needs", [])
    if isinstance(deploy_needs, str):
        deploy_needs = [deploy_needs]
    assert "production-release-gate" in deploy_needs, (
        "deploy-to-laptop must depend on production-release-gate approval gate"
    )

    # 4. Check all checkout steps across all jobs
    for job_id, job in jobs.items():
        steps = job.get("steps", [])
        for step in steps:
            uses = step.get("uses", "")
            if "actions/checkout" in uses:
                with_params = step.get("with", {})
                ref = with_params.get("ref", "")
                assert ref != "HEAD", f"Job '{job_id}' passes literal 'HEAD' to actions/checkout"
                assert ref != "${{ github.event.inputs.git_commit_sha }}", (
                    f"Job '{job_id}' directly passes unvalidated github.event.inputs.git_commit_sha to actions/checkout"
                )
                if ref:
                    assert "github.sha" in ref or "target_commit" in ref, (
                        f"Job '{job_id}' checkout ref '{ref}' must reference github.sha or target_commit"
                    )

    # 5. Verify pre-deployment verification job resolves HEAD/empty commit to github.sha
    verify_job = jobs.get("pre-deployment-verification", {})
    step_scripts = [step.get("run", "") for step in verify_job.get("steps", [])]
    assert any("github.sha" in s and ("HEAD" in s or "head" in s) for s in step_scripts), (
        "Pre-deployment verification must have step resolving HEAD to ${{ github.sha }}"
    )


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
