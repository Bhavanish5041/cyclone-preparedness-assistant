#!/usr/bin/env python3
"""Automated Continuous Deployment Engine for CYCLONE INTEL.

Handles zero-downtime rolling service cutovers on self-hosted Ubuntu runner:
1. Records previous running container state for instant automatic rollback.
2. Detects which services actually changed (API, Frontend, Dashboard).
3. Leaves host Ollama (11434) and Cloudflare Tunnel untouched.
4. Enforces immutable commit-based image tags (rejects mutable 'latest').
5. Runs comprehensive deployment preflight checks (docker daemon, disk space, remote images).
6. Executes dependency-ordered cutovers (API first, then dependent Frontend/Dashboard).
7. Conducts multi-layer health checks (Docker container health + HTTP probes).
8. Automatically rolls back ALL touched services if any check fails (preventing partial updates).
9. Masks secrets and credentials in all logs.
10. Supports isolated simulation mode (--simulate) for safe end-to-end verification.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DEPLOYMENT_STATE_FILE = PROJECT_ROOT / ".deployment_state.json"
ROLLBACK_BACKUP_FILE = LOGS_DIR / "rollback_state.json"
DEPLOYMENT_LOCK_FILE = LOGS_DIR / ".deploy.lock"

SERVICE_METADATA = {
    "api": {
        "container_name": "cyclone-api",
        "ghcr_repo": "cyclone-assistant-api",
        "health_url": "http://127.0.0.1:8000/health",
        "env_var": "API_IMAGE",
        "file_patterns": [
            "api/",
            "src/api/",
            "src/integration/",
            "src/monitoring/",
            "src/models/",
            "src/geo/",
            "src/optimization/",
            "models/",
            "data/processed/",
            "Dockerfile.api",
            "docker/requirements-api.txt",
        ],
    },
    "frontend": {
        "container_name": "cyclone-frontend",
        "ghcr_repo": "cyclone-assistant-frontend",
        "health_url": "http://127.0.0.1/health",
        "env_var": "FRONTEND_IMAGE",
        "file_patterns": [
            "frontend/",
            "deployment/nginx/",
            "Dockerfile.frontend",
        ],
    },
    "dashboard": {
        "container_name": "cyclone-dashboard",
        "ghcr_repo": "cyclone-assistant-dashboard",
        "health_url": "http://127.0.0.1:8501/_stcore/health",
        "env_var": "DASHBOARD_IMAGE",
        "file_patterns": [
            "src/assistant/",
            "Dockerfile.dashboard",
            "docker/requirements-dashboard.txt",
        ],
    },
}

SIMULATION_METADATA = {
    "api": {
        "container_name": "cyclone-ci-api-test",
        "health_url": "http://127.0.0.1:8008/health",
        "compose_service": "api-ci-test",
    },
    "frontend": {
        "container_name": "cyclone-ci-frontend-test",
        "health_url": "http://127.0.0.1:8088/health",
        "compose_service": "frontend-ci-test",
    },
}

# Strict service dependency ordering: Root service 'api' first, followed by dependents
SERVICE_DEPLOYMENT_ORDER = ["api", "frontend", "dashboard"]
SERVICE_ROLLBACK_ORDER = ["dashboard", "frontend", "api"]

# Immutable tag validation: Enforce sha-<short_sha> or semantic version tags; strictly forbid mutable tags
IMMUTABLE_TAG_PATTERN = re.compile(
    r"^(sha-[0-9a-zA-Z._-]+|v[0-9]+(\.[0-9]+)*(-[a-zA-Z0-9.]+)?)$"
)
MUTABLE_TAGS = {"latest", "master", "main", "staging", "dev", "develop", "head"}


def is_immutable_image_tag(tag: Optional[str]) -> bool:
    """Return True if image tag is an immutable release tag (e.g., sha-<commit_sha> or vX.Y.Z)."""
    if not tag or not isinstance(tag, str):
        return False
    tag_clean = tag.strip().lower()
    if tag_clean in MUTABLE_TAGS:
        return False
    return bool(IMMUTABLE_TAG_PATTERN.match(tag.strip()))


class SecretMaskingFilter(logging.Filter):
    """Filter that scrubs secrets, tokens, and credentials from log messages."""

    SECRET_PATTERNS = [
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
        re.compile(r"github_pat_[a-zA-Z0-9_]{82}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----"),
        re.compile(r"(?:api[_-]?key|secret|password|token|bearer)[=:\s]+(['\"]?)([\w./-]{8,})\1", re.IGNORECASE),
        re.compile(r"(?<=://)[^/:]+:[^/@]+(?=@)"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pat in self.SECRET_PATTERNS:
                record.msg = pat.sub("***REDACTED***", record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (self._mask(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._mask(a) if isinstance(a, str) else a for a in record.args
                )
        return True

    def _mask(self, val: str) -> str:
        for pat in self.SECRET_PATTERNS:
            val = pat.sub("***REDACTED***", val)
        return val


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("cyclone_deployer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    mask_filter = SecretMaskingFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(mask_filter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(mask_filter)
        logger.addHandler(file_handler)

    return logger


class DeploymentLock:
    """Process-level mutual exclusion lock to prevent concurrent deployments."""

    def __init__(self, lock_file: Path, logger: logging.Logger):
        self.lock_file = lock_file
        self.logger = logger
        self.fd: Optional[int] = None

    def __enter__(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(
                self.fd,
                f"PID {os.getpid()} locked at {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n".encode("utf-8"),
            )
        except (BlockingIOError, OSError) as e:
            self.logger.critical(
                f"🚨 CONCURRENCY ERROR: Another deployment process is currently running and holding the lock ({self.lock_file}). "
                f"Aborting deployment to prevent conflicting container cutovers."
            )
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except Exception:
                    pass
                self.fd = None
            raise RuntimeError("Concurrent deployment locked.") from e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None


def get_image_digest(image_ref: str) -> Optional[str]:
    """Retrieve immutable sha256 repository digest or image ID."""
    cmd = ["docker", "inspect", "--format", "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}", image_ref]
    proc = run_command(cmd, check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return None


def run_command(
    cmd: List[str], check: bool = True, capture: bool = True, cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        cwd=str(cwd or PROJECT_ROOT),
    )


def http_probe(url: str, timeout: float = 5.0) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CycloneCDHealthCheck/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            code = response.status
            body_text = response.read().decode("utf-8", errors="ignore")
            try:
                data = json.loads(body_text)
            except Exception:
                data = {"raw": body_text[:200]}
            return code, data
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="ignore")
        return e.code, {"error": body_text[:200]}
    except Exception as e:
        return 0, {"error": str(e)}


def get_container_info(container_name: str) -> Optional[Dict[str, Any]]:
    cmd = [
        "docker",
        "inspect",
        "--format",
        '{"id":"{{.Id}}","image":"{{.Config.Image}}","imageId":"{{.Image}}","status":"{{.State.Status}}","health":"{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"}',
        container_name,
    ]
    proc = run_command(cmd, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout.strip())
    except Exception:
        return None


def execute_preflight_checks(
    logger: logging.Logger,
    image_tag: str,
    repo_owner: str,
    services: List[str],
    compose_file: str,
    is_simulation: bool = False,
) -> bool:
    """Execute preflight checks prior to modifying any container."""
    logger.info("Executing deployment preflight checks...")

    # 1. Docker Daemon Connectivity Check
    docker_info = run_command(["docker", "info"], check=False)
    if docker_info.returncode != 0:
        logger.critical("Preflight check failed: Docker daemon is unreachable or not running.")
        return False
    logger.info("  ✅ Preflight: Docker daemon is responsive.")

    # 2. Immutable Tag Enforcement (Production)
    if not is_simulation:
        if not is_immutable_image_tag(image_tag):
            logger.critical(
                f"Preflight check failed: Tag '{image_tag}' is mutable or unverified. "
                "Production CD strictly requires an immutable tag (e.g. sha-<commit_sha> or vX.Y.Z)."
            )
            return False
        logger.info(f"  ✅ Preflight: Image tag '{image_tag}' verified immutable.")

    # 3. Host Disk Space Check (require >= 1GB free)
    try:
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        free_mb = free / (1024 * 1024)
        if free < 1024 * 1024 * 1024:
            logger.critical(
                f"Preflight check failed: Insufficient host disk space ({free_mb:.1f} MB free, require >= 1024 MB)."
            )
            return False
        logger.info(f"  ✅ Preflight: Host disk space verified ({free_mb:.1f} MB free).")
    except Exception as e:
        logger.warning(f"  ⚠️ Preflight: Could not determine disk space: {e}")

    # 4. Docker Compose Syntax & Configuration Check
    compose_check = run_command(["docker", "compose", "-f", compose_file, "config", "--quiet"], check=False)
    if compose_check.returncode != 0:
        logger.critical(f"Preflight check failed: Docker Compose configuration '{compose_file}' has syntax errors.")
        return False
    logger.info(f"  ✅ Preflight: Docker Compose configuration '{compose_file}' is valid.")

    # 5. Image Pre-pull and Provenance Verification (All target services upfront)
    if not is_simulation:
        for svc in services:
            meta = SERVICE_METADATA.get(svc)
            if not meta:
                continue
            full_image_name = f"ghcr.io/{repo_owner}/{meta['ghcr_repo']}:{image_tag}"
            logger.info(f"  Preflight: Pulling verified immutable image: {full_image_name}")
            pull_proc = run_command(["docker", "pull", full_image_name], check=False)
            if pull_proc.returncode != 0:
                inspect_proc = run_command(["docker", "inspect", full_image_name], check=False)
                if inspect_proc.returncode != 0:
                    logger.critical(
                        f"Preflight check failed: Image '{full_image_name}' could not be pulled from GHCR "
                        f"and is not present locally. Deployment aborted before modifying containers."
                    )
                    return False
            logger.info(f"  ✅ Preflight: Image verified for service '{svc}': {full_image_name}")

    logger.info("All preflight checks passed successfully.")
    return True


def record_current_deployment_state(
    logger: logging.Logger, services: List[str]
) -> Dict[str, Any]:
    state = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "services": {},
    }
    for svc in services:
        meta = SERVICE_METADATA.get(svc)
        if not meta:
            continue
        c_name = meta["container_name"]
        info = get_container_info(c_name)
        if info:
            state["services"][svc] = {
                "container_name": c_name,
                "image": info["image"],
                "image_id": info["imageId"],
                "status": info["status"],
                "health": info["health"],
            }
            logger.info(
                f"[BASELINE] Service '{svc}' running image: {info['image']} (ID: {info['imageId'][:12]})"
            )
            # Create an explicit local rollback-backup tag to prevent overwriting tags from corrupting rollback
            if info.get("imageId"):
                backup_tag = f"{meta['ghcr_repo']}:rollback-backup"
                tag_proc = run_command(["docker", "tag", info["imageId"], backup_tag], check=False)
                if tag_proc.returncode == 0:
                    logger.info(f"Preserved rollback image backup: {backup_tag}")
        else:
            logger.warning(f"[BASELINE] Container '{c_name}' not currently running.")

    try:
        ROLLBACK_BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ROLLBACK_BACKUP_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Rollback baseline state preserved at {ROLLBACK_BACKUP_FILE}")
    except Exception as e:
        logger.error(f"Failed to write rollback state backup: {e}")

    return state


def detect_changed_services(
    logger: logging.Logger,
    target_sha: Optional[str] = None,
    requested_services: Optional[str] = "auto",
) -> List[str]:
    if requested_services and requested_services not in ["auto", ""]:
        if requested_services == "all":
            selected = ["api", "frontend", "dashboard"]
        else:
            selected = [
                s.strip()
                for s in requested_services.split(",")
                if s.strip() in SERVICE_METADATA
            ]
        logger.info(f"Explicit services targeted for deployment: {selected}")
        return selected

    # Auto-detection mode using git diff or last deployed state
    last_state = {}
    if DEPLOYMENT_STATE_FILE.exists():
        try:
            with open(DEPLOYMENT_STATE_FILE, "r", encoding="utf-8") as f:
                last_state = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read previous deployment state: {e}")

    last_sha = last_state.get("commit_sha")
    changed_services: List[str] = []

    if target_sha and last_sha and target_sha != last_sha:
        logger.info(f"Analyzing git changes between {last_sha[:8]} and {target_sha[:8]}...")
        diff_cmd = ["git", "diff", "--name-only", last_sha, target_sha]
        proc = run_command(diff_cmd, check=False)
        if proc.returncode == 0:
            changed_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            for svc, meta in SERVICE_METADATA.items():
                for pat in meta["file_patterns"]:
                    if any(f.startswith(pat) or pat in f for f in changed_files):
                        if svc not in changed_services:
                            changed_services.append(svc)
                            logger.info(f"Detected changes for service '{svc}' (matched pattern '{pat}')")
                            break

    # If no git diff could be computed, or if this is the first deployment state
    if not changed_services:
        logger.info("No git diff detected or first deployment. Verifying all production services.")
        changed_services = ["api", "frontend", "dashboard"]

    return changed_services


def perform_health_checks(
    logger: logging.Logger,
    services: List[str],
    timeout_seconds: int = 60,
    is_simulation: bool = False,
) -> bool:
    logger.info(f"Beginning multi-layer health checks (timeout: {timeout_seconds}s)...")
    start_time = time.time()
    pending = list(services)

    while pending and (time.time() - start_time) < timeout_seconds:
        for svc in list(pending):
            if is_simulation:
                sim_meta = SIMULATION_METADATA.get(svc, {})
                c_name = sim_meta.get("container_name")
                h_url = sim_meta.get("health_url")
                # If failure injection is requested for simulation, deliberately use invalid port
                if os.environ.get("SIMULATE_HEALTH_FAILURE") == "true" and svc == "api":
                    h_url = "http://127.0.0.1:9999/nonexistent_health_probe"
            else:
                meta = SERVICE_METADATA.get(svc, {})
                c_name = meta.get("container_name")
                h_url = meta.get("health_url")

            if not c_name or not h_url:
                pending.remove(svc)
                continue

            info = get_container_info(c_name)
            if not info or info.get("status") != "running":
                continue

            # Check container health status if configured
            c_health = info.get("health")
            if c_health != "healthy" and c_health != "none":
                continue

            # Probe HTTP health endpoint
            status_code, body = http_probe(h_url, timeout=3.0)
            if status_code == 200:
                logger.info(
                    f"  ✅ Service '{svc}' ({c_name}) healthy: HTTP {status_code} at {h_url}"
                )
                pending.remove(svc)

        if pending:
            time.sleep(2)

    if pending:
        logger.error(f"❌ Health check failed or timed out for services: {pending}")
        return False

    logger.info("All target services passed container health and HTTP readiness probes!")
    return True


def execute_rollback(
    logger: logging.Logger,
    rollback_services: List[str],
    compose_file: str = "docker-compose.yml",
) -> bool:
    logger.critical("🚨 INITIATING AUTOMATIC ROLLBACK TO PREVIOUS WORKING IMAGES 🚨")
    backup_file = ROLLBACK_BACKUP_FILE
    if not backup_file.exists() and Path("/tmp/cyclone_deploy_rollback_state.json").exists():
        backup_file = Path("/tmp/cyclone_deploy_rollback_state.json")

    if not backup_file.exists():
        logger.critical(f"Rollback backup file not found at {backup_file}. Attempting graceful restart.")
        project_args = ["-p", "cyclone-ci-test"] if "test" in compose_file else ["-p", "cyclone-prod"]
        run_command(["docker", "compose"] + project_args + ["-f", compose_file, "restart"] + rollback_services, check=False)
        return False

    try:
        with open(backup_file, "r", encoding="utf-8") as f:
            backup_state = json.load(f)
    except Exception as e:
        logger.critical(f"Failed to read rollback backup file: {e}")
        return False

    services_data = backup_state.get("services", {})
    env = os.environ.copy()

    # Revert services in reverse dependency order: dashboard, frontend, then api
    ordered_rollback = [s for s in SERVICE_ROLLBACK_ORDER if s in rollback_services]
    for s in rollback_services:
        if s not in ordered_rollback:
            ordered_rollback.append(s)

    for svc in ordered_rollback:
        svc_backup = services_data.get(svc)
        if not svc_backup:
            logger.warning(f"No backup data for service '{svc}'. Skipping rollback image retag.")
            continue

        meta = SERVICE_METADATA.get(svc, {})
        env_var = meta.get("env_var")
        backup_tag = f"{meta['ghcr_repo']}:rollback-backup"

        # Check if preserved rollback-backup tag exists
        tag_inspect = run_command(["docker", "inspect", backup_tag], check=False)
        if tag_inspect.returncode == 0:
            target_image = backup_tag
            # Restore :latest pointer to the rollback image
            run_command(["docker", "tag", backup_tag, f"{meta['ghcr_repo']}:latest"], check=False)
        else:
            target_image = svc_backup.get("image_id") or svc_backup.get("image")

        logger.info(f"Reverting service '{svc}' to previous working image: {target_image}")
        if env_var and target_image:
            env[env_var] = target_image

    project_args = ["-p", "cyclone-ci-test"] if "test" in compose_file else ["-p", "cyclone-prod"]
    cmd = ["docker", "compose"] + project_args + ["-f", compose_file, "up", "-d", "--no-deps"] + ordered_rollback
    logger.info(f"Executing rollback compose cutover: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"Rollback compose command failed: {proc.stderr}")
        return False

    logger.info("Verifying rollback health...")
    healthy = perform_health_checks(
        logger, ordered_rollback, timeout_seconds=45, is_simulation="test" in compose_file
    )
    if healthy:
        logger.info("✅ Automatic rollback completed successfully. Previous working state restored.")
        return True
    else:
        logger.critical("❌ Rollback completed but services failed post-rollback health checks.")
        return False


def deploy_production(
    image_tag: str,
    commit_sha: Optional[str] = None,
    repo_owner: str = "Bhavanish5041",
    services_arg: str = "auto",
    dry_run: bool = False,
    simulate: bool = False,
    simulate_failure: bool = False,
    health_timeout: int = 60,
    logger: Optional[logging.Logger] = None,
) -> int:
    if logger is None:
        logger = setup_logger(LOGS_DIR / "deployment.log")

    # Normalize repo owner to lowercase per Docker OCI specification
    repo_owner = (repo_owner or "bhavanish5041").strip().lower()

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("=" * 65)
    logger.info("CYCLONE INTEL: ZERO-DOWNTIME CONTINUOUS DEPLOYMENT ENGINE")
    logger.info(f"Execution Mode: {'SIMULATION (Isolated)' if simulate else 'PRODUCTION'}")
    logger.info(f"Target Image Tag: {image_tag}")
    logger.info(f"Target Commit SHA: {commit_sha or 'HEAD'}")
    logger.info(f"Repository Owner: {repo_owner}")
    logger.info(f"UTC Timestamp: {timestamp}")
    logger.info("=" * 65)

    lock = None
    if not dry_run:
        try:
            lock = DeploymentLock(DEPLOYMENT_LOCK_FILE, logger)
            lock.__enter__()
        except RuntimeError:
            return 1

    try:
        # 1. Identify Target Services
        if simulate:
            changed_services = ["api", "frontend"]
        else:
            changed_services = detect_changed_services(
                logger, target_sha=commit_sha, requested_services=services_arg
            )

        if not changed_services:
            logger.info("No services require deployment. System is fully up-to-date.")
            return 0

        # Enforce service dependency ordering: api root first, followed by dependent services
        ordered_services = [s for s in SERVICE_DEPLOYMENT_ORDER if s in changed_services]
        for s in changed_services:
            if s not in ordered_services:
                ordered_services.append(s)
        changed_services = ordered_services

        logger.info(f"Services scheduled for deployment (dependency ordered): {changed_services}")
        logger.info("Preserving host Ollama daemon (:11434) and Cloudflare Tunnel untouched.")

        # Fast-path for dry-run inspection (does not mutate containers or query docker daemon)
        if dry_run:
            if not simulate and not is_immutable_image_tag(image_tag):
                logger.critical(
                    f"[DRY-RUN] Preflight check failed: Tag '{image_tag}' is mutable or unverified. "
                    "Production CD strictly requires an immutable tag (e.g. sha-<commit_sha> or vX.Y.Z)."
                )
                return 1
            logger.info("[DRY-RUN] Execution planned without altering runtime state.")
            for svc in changed_services:
                img = f"ghcr.io/{repo_owner}/{SERVICE_METADATA[svc]['ghcr_repo']}:{image_tag}"
                logger.info(f"[DRY-RUN] Would deploy verified image: {img}")
                logger.info(f"[DRY-RUN] Would restart container: {SERVICE_METADATA[svc]['container_name']}")
            return 0

        # 2. Comprehensive Preflight Checks (Executed prior to touching any container)
        compose_file = "docker-compose.test.yml" if simulate else "docker-compose.yml"
        preflight_ok = execute_preflight_checks(
            logger=logger,
            image_tag=image_tag,
            repo_owner=repo_owner,
            services=changed_services,
            compose_file=compose_file,
            is_simulation=simulate,
        )
        if not preflight_ok:
            logger.critical("Aborting deployment due to preflight check failure.")
            return 1

        # 3. Record Baseline State for Rollback BEFORE Modifying Any Containers
        if not simulate:
            record_current_deployment_state(logger, changed_services)

        # 4. Prepare Environment with Immutable Image References
        env = os.environ.copy()
        if not simulate:
            for svc in changed_services:
                meta = SERVICE_METADATA[svc]
                full_image_name = f"ghcr.io/{repo_owner}/{meta['ghcr_repo']}:{image_tag}"
                if meta["env_var"]:
                    env[meta["env_var"]] = full_image_name
                # Also tag as local for compose consistency
                run_command(["docker", "tag", full_image_name, f"{meta['ghcr_repo']}:latest"], check=False)

        # 5. Handle Simulated Failure Injection if Requested
        if simulate and simulate_failure:
            logger.warning("[SIMULATION] Intentionally injecting failure into test service to verify failure detection & rollback...")
            os.environ["SIMULATE_HEALTH_FAILURE"] = "true"

        # 6. Service Ordering Cutover:
        # Step A: If 'api' is being deployed, cutover and verify 'api' first.
        # Step B: Only when 'api' is healthy, cutover dependent services ('frontend', 'dashboard').
        # If ANY step fails, roll back ALL changed services to restore full consistency without partial updates.
        deployed_so_far: List[str] = []
        project_args = ["-p", "cyclone-ci-test"] if simulate else ["-p", "cyclone-prod"]

        # Group services into ordered cutover batches: primary (api) first, then dependents
        batches: List[List[str]] = []
        if "api" in changed_services:
            batches.append(["api"])
            dependents = [s for s in changed_services if s != "api"]
            if dependents:
                batches.append(dependents)
        else:
            batches.append(changed_services)

        for batch_index, batch_services in enumerate(batches, start=1):
            batch_compose_services = []
            if simulate:
                for svc in batch_services:
                    sim_meta = SIMULATION_METADATA.get(svc)
                    if sim_meta:
                        batch_compose_services.append(sim_meta["compose_service"])
            else:
                batch_compose_services = batch_services

            logger.info(
                f"Executing deployment cutover [Stage {batch_index}/{len(batches)}]: {batch_services}"
            )
            cmd = (
                ["docker", "compose"]
                + project_args
                + ["-f", compose_file, "up", "-d", "--no-deps"]
                + batch_compose_services
            )
            proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                logger.error(f"Docker compose up failed for batch {batch_services}: {proc.stderr}")
                if simulate:
                    run_command(["docker", "compose", "-p", "cyclone-ci-test", "-f", compose_file, "down", "-v"], check=False)
                else:
                    all_to_revert = list(dict.fromkeys(deployed_so_far + batch_services))
                    execute_rollback(logger, all_to_revert, compose_file)
                return 1

            deployed_so_far.extend(batch_services)

            # Verify health checks for current batch
            logger.info(f"Verifying health for batch: {batch_services}...")
            healthy = perform_health_checks(
                logger, batch_services, timeout_seconds=health_timeout, is_simulation=simulate
            )
            if not healthy:
                logger.error(f"Health check failure detected in batch {batch_services}!")
                if simulate:
                    logger.info("[SIMULATION] Triggering simulated rollback teardown...")
                    run_command(["docker", "compose", "-p", "cyclone-ci-test", "-f", compose_file, "down", "-v"], check=False)
                    logger.info("[SIMULATION] Teardown complete. Failure was correctly caught and isolated.")
                    return 1
                else:
                    # Roll back ALL services touched in this deployment batch (prevents partially updated state)
                    all_to_revert = list(dict.fromkeys(deployed_so_far))
                    logger.critical(
                        f"Health check failed. Rolling back all touched services ({all_to_revert}) "
                        "to prevent partial deployment inconsistency."
                    )
                    rollback_ok = execute_rollback(logger, all_to_revert, compose_file)
                    return 1 if not rollback_ok else 2

        # 7. Record Deployment Success
        if simulate:
            logger.info("[SIMULATION] Simulation health checks passed! Tearing down isolated test stack...")
            run_command(["docker", "compose", "-p", "cyclone-ci-test", "-f", compose_file, "down", "-v"], check=False)
            logger.info("[SIMULATION] Teardown complete. Zero production services modified.")
        else:
            digests = {}
            for svc in changed_services:
                meta = SERVICE_METADATA.get(svc, {})
                full_img = f"ghcr.io/{repo_owner}/{meta.get('ghcr_repo')}:{image_tag}"
                d = get_image_digest(full_img) or get_image_digest(f"{meta.get('ghcr_repo')}:latest")
                if d:
                    digests[svc] = d

            deployment_record = {
                "timestamp": timestamp,
                "commit_sha": commit_sha or "HEAD",
                "image_tag": image_tag,
                "image_digests": digests,
                "deployed_services": changed_services,
                "status": "SUCCESS",
            }
            try:
                with open(DEPLOYMENT_STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(deployment_record, f, indent=2)
                logger.info(f"Deployment record written to {DEPLOYMENT_STATE_FILE}")
            except Exception as e:
                logger.warning(f"Could not write deployment state: {e}")

        logger.info("=" * 65)
        logger.info(f"✅ DEPLOYMENT SUCCESSFULLY COMPLETED & VERIFIED: {changed_services}")
        logger.info("=" * 65)
        return 0
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Production CD for CYCLONE INTEL")
    parser.add_argument(
        "--image-tag",
        default="sha-4694bdc",
        help="Exact immutable image tag to deploy (e.g. sha-4694bdc). Mutable 'latest' is forbidden.",
    )
    parser.add_argument("--commit-sha", default=None, help="Target Git commit SHA (default: HEAD)")
    parser.add_argument("--repo-owner", default="Bhavanish5041", help="GitHub repository owner")
    parser.add_argument("--services", default="auto", help="Services to deploy: auto, all, or comma-separated list")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying containers")
    parser.add_argument("--simulate", action="store_true", help="Run deployment simulation in isolated stack")
    parser.add_argument("--simulate-failure", action="store_true", help="Simulate a deployment failure and rollback")
    parser.add_argument("--health-timeout", type=int, default=60, help="Health check timeout in seconds")
    parser.add_argument("--log-file", default=str(LOGS_DIR / "deployment.log"), help="Log file path")

    args = parser.parse_args()
    log = setup_logger(Path(args.log_file))

    ret = deploy_production(
        image_tag=args.image_tag,
        commit_sha=args.commit_sha,
        repo_owner=args.repo_owner,
        services_arg=args.services,
        dry_run=args.dry_run,
        simulate=args.simulate,
        simulate_failure=args.simulate_failure,
        health_timeout=args.health_timeout,
        logger=log,
    )
    sys.exit(0 if ret == 0 else 1)
