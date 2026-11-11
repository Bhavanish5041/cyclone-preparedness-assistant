#!/usr/bin/env python3
"""Lightweight Operations and Health Monitor for CYCLONE INTEL Production Stack.

Monitors:
- Container status, health, and restart counts
- Process and container CPU/Memory consumption
- API health and readiness endpoints (/health, /ready, /api/models/registry)
- Ollama and Bridge connectivity
- Public Cloudflare Tunnel reachability
- Deterministic response latency

Zero heavy dependencies, suitable for laptop-hosted production.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    """Run subprocess command safely."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode, res.stdout.strip()
    except Exception as e:
        return 1, str(e)


def check_http_endpoint(url: str, timeout: float = 3.0) -> dict:
    """Check HTTP endpoint status and measure latency."""
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CycloneOpsMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            data = resp.read().decode("utf-8")
            try:
                body = json.loads(data)
            except Exception:
                body = data[:200]
            return {
                "url": url,
                "status": resp.status,
                "latency_ms": round(latency_ms, 2),
                "healthy": resp.status == 200,
                "data": body,
            }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "url": url,
            "status": getattr(exc, "code", 0),
            "latency_ms": round(latency_ms, 2),
            "healthy": False,
            "error": str(exc),
        }


def get_docker_stats() -> list[dict]:
    """Retrieve Docker container stats and restart counts."""
    if not shutil.which("docker"):
        return []

    code, out = run_cmd(["docker", "ps", "-a", "--filter", "name=cyclone-", "--format", "{{.Names}}\t{{.Status}}\t{{.ID}}"])
    if code != 0 or not out:
        return []

    containers = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            name, status, cid = parts[0], parts[1], parts[2]
            # Inspect restart count
            _, inspect_out = run_cmd(["docker", "inspect", "--format", "{{.RestartCount}}\t{{.State.Health.Status}}", cid])
            restarts = 0
            health = "n/a"
            if inspect_out:
                insp_parts = inspect_out.split("\t")
                try:
                    restarts = int(insp_parts[0])
                except ValueError:
                    pass
                if len(insp_parts) > 1 and insp_parts[1]:
                    health = insp_parts[1]

            containers.append({
                "name": name,
                "id": cid[:12],
                "status": status,
                "health": health,
                "restart_count": restarts,
            })

    # Get memory/cpu stats
    code, stats_out = run_cmd(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"])
    if code == 0 and stats_out:
        stats_map = {}
        for line in stats_out.splitlines():
            sp = line.split("\t")
            if len(sp) >= 4:
                stats_map[sp[0]] = {"cpu": sp[1], "mem": sp[2], "mem_perc": sp[3]}
        for c in containers:
            if c["name"] in stats_map:
                c.update(stats_map[c["name"]])

    return containers


def get_systemd_status() -> dict[str, str]:
    """Check status of relevant systemd user units."""
    units = ["cyclone-tunnel.service", "cyclone-ollama-bridge.service", "cyclone-api.service"]
    status_map = {}
    for unit in units:
        code, out = run_cmd(["systemctl", "--user", "is-active", unit])
        status_map[unit] = out if code == 0 else "inactive"
    return status_map


def get_public_url() -> str | None:
    """Find current trycloudflare.com URL from journalctl."""
    code, out = run_cmd(["journalctl", "--user", "-u", "cyclone-tunnel.service", "-n", "40", "--no-pager"])
    if code == 0 and out:
        for line in reversed(out.splitlines()):
            if "trycloudflare.com" in line:
                for token in line.split():
                    if "trycloudflare.com" in token:
                        url = token.strip("|+* \t\r\n")
                        if not url.startswith("http"):
                            url = f"https://{url}"
                        return url
    return None


def collect_monitoring_report() -> dict:
    """Gather complete operational telemetry."""
    # Endpoints
    api_health = check_http_endpoint("http://127.0.0.1:8000/health")
    api_ready = check_http_endpoint("http://127.0.0.1:8000/ready")
    proxy_health = check_http_endpoint("http://127.0.0.1:80/health")
    registry_health = check_http_endpoint("http://127.0.0.1:8000/api/models/registry")

    # Ollama checks
    host_ollama = check_http_endpoint("http://127.0.0.1:11434/api/tags")
    bridge_ollama = check_http_endpoint("http://172.18.0.1:11434/api/tags")

    # Cloudflare public check
    public_url = get_public_url()
    public_health = None
    if public_url:
        public_health = check_http_endpoint(f"{public_url}/health", timeout=5.0)

    # Containers
    containers = get_docker_stats()

    # Systemd
    systemd = get_systemd_status()

    # Overall health assessment
    is_healthy = (
        api_health.get("healthy", False)
        and api_ready.get("healthy", False)
        and proxy_health.get("healthy", False)
        and host_ollama.get("healthy", False)
        and bridge_ollama.get("healthy", False)
    )

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "overall_healthy": is_healthy,
        "endpoints": {
            "api_internal_health": api_health,
            "api_internal_ready": api_ready,
            "proxy_health": proxy_health,
            "model_registry": registry_health,
        },
        "llm_infrastructure": {
            "host_ollama": host_ollama,
            "docker_bridge_ollama": bridge_ollama,
        },
        "public_ingress": {
            "tunnel_url": public_url,
            "tunnel_reachability": public_health,
        },
        "containers": containers,
        "systemd_user_services": systemd,
    }


def print_cli_report(data: dict) -> None:
    """Format and print terminal dashboard."""
    print("=" * 68)
    print("  CYCLONE INTEL — PRODUCTION OPERATIONS & HEALTH MONITOR")
    print("=" * 68)
    status_str = "🟢 HEALTHY & OPERATIONAL" if data["overall_healthy"] else "🔴 DEGRADED / ATTENTION REQUIRED"
    print(f"Status:    {status_str}")
    print(f"Timestamp: {data['timestamp']}\n")

    print("[1] CONTAINER FLEET STATUS (Docker Compose)")
    print("-" * 68)
    print(f"{'CONTAINER':<20} {'STATUS':<22} {'HEALTH':<10} {'CPU':<8} {'MEMORY'}")
    print("-" * 68)
    for c in data["containers"]:
        name = c.get("name", "n/a")
        status = c.get("status", "n/a")[:20]
        health = c.get("health", "n/a")
        cpu = c.get("cpu", "n/a")
        mem = c.get("mem", "n/a")
        restarts = c.get("restart_count", 0)
        restart_flag = f" (restarts: {restarts})" if restarts > 0 else ""
        print(f"{name:<20} {status:<22} {health:<10} {cpu:<8} {mem}{restart_flag}")
    print()

    print("[2] SERVICE ENDPOINTS & LATENCY")
    print("-" * 68)
    for key, ep in data["endpoints"].items():
        ok = "PASS" if ep.get("healthy") else "FAIL"
        lat = f"{ep.get('latency_ms', 0):.1f} ms"
        status_code = ep.get("status", "err")
        print(f"  [{ok}] {key:<24} : code={status_code:<4} latency={lat}")
    print()

    print("[3] LLM RUNTIME CONNECTIVITY")
    print("-" * 68)
    for key, ep in data["llm_infrastructure"].items():
        ok = "PASS" if ep.get("healthy") else "FAIL"
        lat = f"{ep.get('latency_ms', 0):.1f} ms"
        print(f"  [{ok}] {key:<24} : code={ep.get('status')} latency={lat}")
    print()

    print("[4] PUBLIC INGRESS & SYSTEMD SERVICES")
    print("-" * 68)
    tunnel_url = data["public_ingress"].get("tunnel_url") or "Unavailable"
    pub_health = data["public_ingress"].get("tunnel_reachability")
    pub_status = "PASS" if pub_health and pub_health.get("healthy") else "FAIL"
    pub_lat = f"{pub_health.get('latency_ms', 0):.1f} ms" if pub_health else "n/a"
    print(f"  Tunnel URL: {tunnel_url}")
    print(f"  [{pub_status}] Public Ingress Reachability : latency={pub_lat}")
    print("  Systemd Units:")
    for svc, state in data["systemd_user_services"].items():
        flag = "active" if state == "active" else "inactive"
        print(f"    - {svc:<32} : {flag}")
    print()

    print("=" * 68)
    print("OPERATIONAL HELPER COMMANDS:")
    print("  View API logs:        docker logs -f cyclone-api")
    print("  View Frontend logs:   docker logs -f cyclone-frontend")
    print("  View Tunnel logs:     journalctl --user -u cyclone-tunnel.service -f")
    print("  Restart Stack:        docker compose restart")
    print("  Full Validation:      .venv/bin/python scripts/validate_deployment.py")
    print("  Rollback to Host:     bash scripts/rollback_to_host.sh")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Cyclone Intel Production Operations Monitor")
    parser.add_argument("--json", action="store_true", help="Output raw JSON metrics")
    args = parser.parse_args()

    report = collect_monitoring_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_cli_report(report)

    sys.exit(0 if report["overall_healthy"] else 1)


if __name__ == "__main__":
    main()
