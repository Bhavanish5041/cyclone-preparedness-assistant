#!/usr/bin/env python3
"""Automated Docker Stage 8 Validation Suite for CI.

Verifies:
1. Building test images (cyclone-ci-api:test, cyclone-ci-frontend:test)
2. Image verification
3. Starting isolated Compose stack (project: cyclone-ci-test)
4. Health check validation on container endpoints
5. API readiness check (port 8008)
6. Frontend Nginx static serving (port 8088)
7. Nginx-to-FastAPI reverse proxy routing (/api/...)
8. Representative endpoints through proxy:
   - /api/health
   - /api/dashboard/overview
   - /api/communities?search=Tallur
   - /api/communities/11-29-569-05522-608691
   - /api/districts
   - /api/analytics/preparedness
   - /docs
   - /openapi.json
9. Clean shutdown and teardown
"""

import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

COMPOSE_FILE = "docker-compose.test.yml"
PROJECT_NAME = "cyclone-ci-test"
API_BASE = "http://localhost:8008"
FRONTEND_BASE = "http://localhost:8088"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def http_get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "DockerCITest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except Exception as e:
        return 0, str(e)


def main():
    print("=" * 70)
    print("CYCLONE INTEL: Docker Validation Suite (Stage 8)")
    print("=" * 70)

    try:
        # Verify frontend assets exist before building container
        frontend_dist_index = Path("frontend/dist/index.html")
        if not frontend_dist_index.exists():
            print("\nNotice: frontend/dist not found. Compiling production bundle...")
            run_cmd(["npm", "--prefix", "frontend", "run", "build"])

        # Verify W&B artifact cache exists before building API container
        wandb_cache_dir = Path(".wandb_artifacts_cache/bhavanishreddy69-na_cyclone-preparedness-assistant_village-preparedness-classifier_v0")
        if not wandb_cache_dir.exists() or not any(wandb_cache_dir.iterdir()):
            print("\nNotice: .wandb_artifacts_cache not found or empty. Seeding offline cache from models/render_artifact...")
            wandb_cache_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            render_dir = Path("models/render_artifact")
            if render_dir.exists():
                for item in render_dir.iterdir():
                    dest = wandb_cache_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)

        # 1. Build and verify images
        print("\n[1/5] Building isolated test Docker images...")
        run_cmd(["docker", "compose", "-p", PROJECT_NAME, "-f", COMPOSE_FILE, "build"])

        # 2. Start isolated test compose stack
        print("\n[2/5] Starting isolated test stack...")
        run_cmd(["docker", "compose", "-p", PROJECT_NAME, "-f", COMPOSE_FILE, "up", "-d"])

        # 3. Wait for containers to become healthy
        print("\n[3/5] Waiting for containers to become healthy...")
        healthy = False
        for attempt in range(25):
            time.sleep(3)
            status_api, body_api = http_get(f"{API_BASE}/health")
            status_fe, _ = http_get(f"{FRONTEND_BASE}/")
            if status_api == 200 and status_fe == 200:
                print(f"Containers healthy after {(attempt + 1) * 3}s.")
                healthy = True
                break
            print(f"Waiting... (API: {status_api}, Frontend: {status_fe})")

        if not healthy:
            print("ERROR: Containers failed to become healthy within timeout.")
            # Print logs for diagnostics
            logs = run_cmd(["docker", "compose", "-p", PROJECT_NAME, "-f", COMPOSE_FILE, "logs"], check=False)
            print(logs.stdout)
            sys.exit(1)

        # 4. Test representative endpoints through Nginx proxy
        print("\n[4/5] Testing Nginx-to-FastAPI proxy and endpoints...")
        test_endpoints = [
            (f"{FRONTEND_BASE}/", 200, "Frontend index.html"),
            (f"{FRONTEND_BASE}/health", 200, "Direct Nginx /health proxy"),
            (f"{FRONTEND_BASE}/ready", 200, "Direct Nginx /ready proxy"),
            (f"{FRONTEND_BASE}/api/health", 200, "API health status"),
            (f"{FRONTEND_BASE}/api/dashboard/overview", 200, "Regional overview KPIs"),
            (f"{FRONTEND_BASE}/api/communities?search=Tallur", 200, "Community search"),
            (f"{FRONTEND_BASE}/api/communities/11-29-569-05522-608691", 200, "Tallur dossier"),
            (f"{FRONTEND_BASE}/api/districts", 200, "Districts list"),
            (f"{FRONTEND_BASE}/api/analytics/preparedness", 200, "Preparedness distributions"),
            (f"{FRONTEND_BASE}/docs", 200, "Swagger UI docs"),
            (f"{FRONTEND_BASE}/openapi.json", 200, "OpenAPI schema"),
        ]

        failed = 0
        for url, expected_status, label in test_endpoints:
            code, resp_text = http_get(url)
            if code == expected_status:
                print(f"  PASS: [{code}] {label} -> {url}")
            else:
                print(f"  FAIL: Expected [{expected_status}], got [{code}] for {label} ({url})")
                failed += 1

        if failed > 0:
            print(f"\nERROR: {failed} proxy endpoint test(s) failed.")
            sys.exit(1)

        print("\nAll Docker proxy and endpoint checks passed!")

    finally:
        # 5. Tear down and cleanup isolated test stack
        print("\n[5/5] Tearing down isolated test stack and cleaning up...")
        run_cmd(["docker", "compose", "-p", PROJECT_NAME, "-f", COMPOSE_FILE, "down", "-v"], check=False)
        print("Cleanup complete.")

    print("\nStage 8 Docker validation PASSED successfully.")


if __name__ == "__main__":
    main()
