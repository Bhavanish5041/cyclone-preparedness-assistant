"""Phase 4.4 Docker Container Memory and Acceptance Benchmark.

Executes within a Docker container hard-capped at 512 MiB memory:
- Traverses all Streamlit platform views via AppTest
- Executes single-turn and multi-turn conversational assistant queries
- Monitors process VmRSS, VmHWM, and Linux cgroup v2 memory.current & memory.peak
- Verifies stability, response latency, and headroom against Render Free Tier limits
"""

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import time

from streamlit.testing.v1 import AppTest

from src.api.schemas import ChatMessage, ChatRequest, MessageRole
from src.assistant.api_client import AdapterMode, DecisionSupportApiClient, get_api_client

RENDER_LIMIT_MIB = 512.0
OUTPUT_JSON = Path("docs/memory_measurement_phase4_4_docker.json")


def get_cgroup_memory_mib() -> float:
    """Read Linux cgroup v2 current memory usage in MiB."""
    for path in ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return round(int(f.read().strip()) / (1024.0 * 1024.0), 2)
            except Exception:
                pass
    return 0.0


def get_cgroup_peak_mib() -> float:
    """Read Linux cgroup v2 peak memory usage in MiB."""
    for path in ["/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return round(int(f.read().strip()) / (1024.0 * 1024.0), 2)
            except Exception:
                pass
    return 0.0


def get_process_rss_mib() -> float:
    """Read current process Resident Set Size (RSS) in MiB."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(float(line.split()[1]) / 1024.0, 2)
    except Exception:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / 1024.0, 2)


def get_process_peak_mib() -> float:
    """Read process peak RSS (VmHWM) in MiB."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(float(line.split()[1]) / 1024.0, 2)
    except Exception:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / 1024.0, 2)


def record_stage(stage_id: str, desc: str, duration_ms: float = 0.0) -> dict:
    rss = get_process_rss_mib()
    hwm = get_process_peak_mib()
    cgroup = get_cgroup_memory_mib()
    cgroup_peak = get_cgroup_peak_mib()

    pct_512 = round((cgroup / RENDER_LIMIT_MIB) * 100, 2)
    headroom = round(RENDER_LIMIT_MIB - cgroup, 2)

    print(f"[{stage_id}] RSS: {rss:.1f} MB | Cgroup: {cgroup:.1f} MB ({pct_512:.1f}%) | Peak Cgroup: {cgroup_peak:.1f} MB | Duration: {duration_ms:.1f} ms")

    return {
        "stage": stage_id,
        "description": desc,
        "process_rss_mib": rss,
        "process_peak_mib": hwm,
        "cgroup_current_mib": cgroup,
        "cgroup_peak_mib": cgroup_peak,
        "percent_of_512mib": pct_512,
        "headroom_mib": headroom,
        "duration_ms": round(duration_ms, 2),
    }


def main():
    print("=" * 80)
    print("PHASE 4.4 DOCKER CONTAINER 512 MiB MEMORY & ACCEPTANCE BENCHMARK")
    print(f"Hardware Constraint: 512.0 MiB Hard Limit (Render Free-Tier Specification)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    stages = []

    # 1. Baseline
    gc.collect()
    stages.append(record_stage("baseline_idle", "Baseline container runtime before app initialization"))

    # 2. AppTest Initialization & Overview View
    t0 = time.perf_counter()
    app_path = Path("/app/src/assistant/app.py")
    if not app_path.exists():
        app_path = Path(__file__).resolve().parent.parent / "src" / "assistant" / "app.py"

    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    stages.append(record_stage("view_overview", "Streamlit initial render (Overview page)", dur))

    # 3. View: Regional GIS Explorer
    t0 = time.perf_counter()
    at.sidebar.radio(key="platform_nav_radio").set_value("Regional GIS Explorer")
    at.run(timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    stages.append(record_stage("view_gis_explorer", "Streamlit render Regional GIS Explorer", dur))

    # 4. View: Community Profiles
    t0 = time.perf_counter()
    at.sidebar.radio(key="platform_nav_radio").set_value("Community Profiles")
    at.run(timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    stages.append(record_stage("view_community_profiles", "Streamlit render Community Profiles", dur))

    # 5. View: Scenario Analysis
    t0 = time.perf_counter()
    at.sidebar.radio(key="platform_nav_radio").set_value("Scenario Analysis")
    at.run(timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    stages.append(record_stage("view_scenario_analysis", "Streamlit render Scenario Analysis", dur))

    # 6. View: Vulnerability Archetypes
    t0 = time.perf_counter()
    at.sidebar.radio(key="platform_nav_radio").set_value("Vulnerability Archetypes")
    at.run(timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    stages.append(record_stage("view_vulnerability_archetypes", "Streamlit render Vulnerability Archetypes", dur))

    # 7. View: Reports & Export
    t0 = time.perf_counter()
    at.sidebar.radio(key="platform_nav_radio").set_value("Reports and Export")
    at.run(timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    stages.append(record_stage("view_reports_export", "Streamlit render Reports and Export", dur))

    # 8. View: Cyclone Assistant Initial View
    t0 = time.perf_counter()
    at.sidebar.radio(key="platform_nav_radio").set_value("Cyclone Assistant")
    at.run(timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    stages.append(record_stage("view_cyclone_assistant_init", "Streamlit render Cyclone Assistant initial view", dur))

    # 9. Conversational Multi-Turn Execution via Direct-Mode API Client
    client = get_api_client(mode="direct")

    # Conversation A - Turn 1: Tallur assessment
    t0 = time.perf_counter()
    res_a1 = client.chat(ChatRequest(message="Explain why Tallur has its current priority classification."))
    dur = (time.perf_counter() - t0) * 1000
    assert "74.34" in res_a1.answer or "BAND_4_ADEQUATE" in res_a1.answer
    stages.append(record_stage("conv_a_turn1", "Conversation A Turn 1: Tallur priority classification", dur))

    # Conversation A - Turn 2: Infrastructure follow-up
    hist_a = [
        ChatMessage(role=MessageRole.USER, content="Explain why Tallur has its current priority classification."),
        ChatMessage(role=MessageRole.ASSISTANT, content=res_a1.answer),
    ]
    t0 = time.perf_counter()
    res_a2 = client.chat(ChatRequest(
        message="What infrastructure indicators contribute to that score?",
        history=hist_a,
    ))
    dur = (time.perf_counter() - t0) * 1000
    assert "Tallur" in res_a2.answer or "PHC" in res_a2.answer or "1.68 km" in res_a2.answer
    stages.append(record_stage("conv_a_turn2", "Conversation A Turn 2: Tallur infrastructure indicators", dur))

    # Conversation A - Turn 3: Interventions follow-up
    hist_a.extend([
        ChatMessage(role=MessageRole.USER, content="What infrastructure indicators contribute to that score?"),
        ChatMessage(role=MessageRole.ASSISTANT, content=res_a2.answer),
    ])
    t0 = time.perf_counter()
    res_a3 = client.chat(ChatRequest(
        message="What interventions are recommended?",
        history=hist_a,
    ))
    dur = (time.perf_counter() - t0) * 1000
    assert "Tallur" in res_a3.answer or "Advisories" in res_a3.answer or "inspection" in res_a3.answer.lower()
    stages.append(record_stage("conv_a_turn3", "Conversation A Turn 3: Tallur recommended interventions", dur))

    # Conversation B - Turn 1: District Comparison
    t0 = time.perf_counter()
    res_b1 = client.chat(ChatRequest(message="Compare Dakshina Kannada and Udupi."))
    dur = (time.perf_counter() - t0) * 1000
    assert "369" in res_b1.answer and "253" in res_b1.answer
    stages.append(record_stage("conv_b_turn1", "Conversation B Turn 1: Compare DK (369) vs Udupi (253)", dur))

    # Conversation B - Turn 2: Band 1 comparison
    hist_b = [
        ChatMessage(role=MessageRole.USER, content="Compare Dakshina Kannada and Udupi."),
        ChatMessage(role=MessageRole.ASSISTANT, content=res_b1.answer),
    ]
    t0 = time.perf_counter()
    res_b2 = client.chat(ChatRequest(
        message="Which district has more Band 1 communities?",
        history=hist_b,
    ))
    dur = (time.perf_counter() - t0) * 1000
    assert "Dakshina Kannada" in res_b2.answer or "Band 1" in res_b2.answer
    stages.append(record_stage("conv_b_turn2", "Conversation B Turn 2: Band 1 comparison follow-up", dur))

    # Conversation C - Turn 1: Ambiguous Community Disambiguation
    t0 = time.perf_counter()
    res_c1 = client.chat(ChatRequest(message="Assess Siddapur."))
    dur = (time.perf_counter() - t0) * 1000
    assert "Disambiguation Required" in res_c1.answer or "602812" in res_c1.answer
    stages.append(record_stage("conv_c_turn1", "Conversation C Turn 1: Ambiguous Siddapur disambiguation table", dur))

    # Final Garbage Collection & Peak Audit
    gc.collect()
    peak_cgroup = get_cgroup_peak_mib()
    final_cgroup = get_cgroup_memory_mib()
    peak_proc = get_process_peak_mib()
    final_proc = get_process_rss_mib()

    headroom_mib = round(RENDER_LIMIT_MIB - peak_cgroup, 2)
    headroom_pct = round((headroom_mib / RENDER_LIMIT_MIB) * 100, 2)

    print("\n" + "=" * 80)
    print("DOCKER 512 MiB BENCHMARK SUMMARY")
    print(f"Absolute Peak Container Memory (cgroup): {peak_cgroup:.2f} MiB ({peak_cgroup/RENDER_LIMIT_MIB*100:.1f}%)")
    print(f"Absolute Peak Process Memory (VmHWM):    {peak_proc:.2f} MiB ({peak_proc/RENDER_LIMIT_MIB*100:.1f}%)")
    print(f"Available Headroom under 512 MiB limit:  {headroom_mib:.2f} MiB ({headroom_pct:.1f}% remaining)")
    print(f"Safe for Render Free Tier (<= 90% cap):  {'YES (GO)' if peak_cgroup < (RENDER_LIMIT_MIB * 0.90) else 'NO (RISK)'}")
    print("=" * 80)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "container_limit_mib": RENDER_LIMIT_MIB,
        "peak_cgroup_mib": peak_cgroup,
        "final_cgroup_mib": final_cgroup,
        "peak_process_rss_mib": peak_proc,
        "final_process_rss_mib": final_proc,
        "headroom_mib": headroom_mib,
        "headroom_percent": headroom_pct,
        "render_free_compliant": peak_cgroup < (RENDER_LIMIT_MIB * 0.90),
        "stages": stages,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Benchmark results written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
