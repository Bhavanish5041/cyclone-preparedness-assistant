"""Automated script to capture page screenshots and measure exact Docker memory usage."""

import json
import os
import subprocess
import time
from pathlib import Path

PAGES = [
    ("overview", "Overview", "http://localhost:10002/?page=overview"),
    ("gis_explorer", "Regional GIS Explorer", "http://localhost:10002/?page=gis_explorer"),
    ("community_profile", "Community Profiles", "http://localhost:10002/?page=community_profile"),
    ("scenario_analysis", "Scenario Analysis", "http://localhost:10002/?page=scenario_analysis"),
    ("archetypes", "Vulnerability Archetypes", "http://localhost:10002/?page=archetypes"),
    ("reports", "Reports and Export", "http://localhost:10002/?page=reports"),
    ("ai_readiness", "AI Readiness (Preview)", "http://localhost:10002/?page=ai_readiness"),
]

SCREENSHOT_DIR = Path("docs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def get_docker_memory(container_name="test-render-app"):
    """Query docker stats for exact memory usage and limit."""
    cmd = [
        "docker", "stats", "--no-stream", "--format",
        "{{.MemUsage}}|||{{.MemPerc}}|||{{.CPUPerc}}", container_name
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = res.stdout.strip().split("|||")
    return {
        "mem_usage": parts[0] if len(parts) > 0 else "N/A",
        "mem_perc": parts[1] if len(parts) > 1 else "N/A",
        "cpu_perc": parts[2] if len(parts) > 2 else "N/A",
    }


def capture_screenshot(url, output_path):
    """Use headless Google Chrome to capture high-resolution screenshot."""
    cmd = [
        "/usr/bin/google-chrome",
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--window-size=1600,1100",
        "--virtual-time-budget=10000",
        f"--screenshot={output_path}",
        url,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    print("=== Starting Render Container Memory & Screenshot Measurement ===")
    time.sleep(3)
    startup_mem = get_docker_memory()
    print(f"Startup / Idle Memory: {startup_mem['mem_usage']} ({startup_mem['mem_perc']})")

    results = []
    results.append({
        "page_key": "startup_idle",
        "page_name": "Container Startup & Idle",
        "mem_usage": startup_mem["mem_usage"],
        "mem_perc": startup_mem["mem_perc"],
    })

    peak_bytes = 0

    for key, name, url in PAGES:
        print(f"\n--- Loading and Measuring: {name} ({url}) ---")
        screenshot_file = SCREENSHOT_DIR / f"{key}.png"

        # Capture screenshot
        capture_screenshot(url, str(screenshot_file))
        time.sleep(2)

        mem = get_docker_memory()
        print(f"Page: {name} | Memory: {mem['mem_usage']} ({mem['mem_perc']}) | Screenshot: {screenshot_file}")

        results.append({
            "page_key": key,
            "page_name": name,
            "url": url,
            "mem_usage": mem["mem_usage"],
            "mem_perc": mem["mem_perc"],
            "screenshot": str(screenshot_file),
        })

    # Summary JSON
    summary_path = Path("docs/memory_measurement_phase3_1.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Measurement Summary Saved to docs/memory_measurement_phase3_1.json ===")
    for r in results:
        print(f"- {r['page_name']:<30}: {r['mem_usage']:<20} ({r['mem_perc']})")


if __name__ == "__main__":
    main()
