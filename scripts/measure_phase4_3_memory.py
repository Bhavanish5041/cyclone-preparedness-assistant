"""Memory and Performance Benchmark for Phase 4.3 (Streamlit Assistant Integration).

Measures:
1. Idle Memory: Baseline Python runtime RSS.
2. Overview Page Memory: Streamlit AppTest rendering Overview.
3. GIS Explorer Memory: Streamlit AppTest rendering GIS map & controls.
4. Community Profile Memory: Streamlit AppTest rendering Granular Community Profiles.
5. Assistant Initialization Memory: Streamlit AppTest navigating to Cyclone Assistant.
6. Assistant Request Memory (Mocked LLM): Executing a mocked Qwen tool-calling interaction.
7. Assistant Request Memory (Deterministic Fallback): Executing direct local decision-matrix evaluation.
8. Peak Memory: Process VmHWM (High Water Mark) and margin against Render Free 512 MiB ceiling.
"""

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from src.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    ToolExecutionRecord,
)
from src.assistant.api_client import AdapterMode, DecisionSupportApiClient
from src.assistant.decision_support import DecisionSupportService
from src.assistant.views.chat_assistant import execute_assistant_turn

RENDER_FREE_LIMIT_MIB = 512.0
OUTPUT_JSON = Path("docs/memory_measurement_phase4_3.json")


def get_current_rss_mib() -> float:
    """Read current Resident Set Size (RSS) in MiB from /proc/self/status."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: VmRSS:     123456 kB
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
    except Exception:
        pass
    # Fallback to getrusage (ru_maxrss is in KiB on Linux)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / 1024.0, 2)


def get_peak_rss_mib() -> float:
    """Read peak Resident Set Size (VmHWM) in MiB from /proc/self/status."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
    except Exception:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / 1024.0, 2)


def main():
    print("=" * 70)
    print("PHASE 4.3 STREAMLIT ASSISTANT MEMORY AND PERFORMANCE BENCHMARK")
    print(f"Render Free-Tier Budget: {RENDER_FREE_LIMIT_MIB:.1f} MiB")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    measurements = []

    # 1. Idle Memory
    gc.collect()
    idle_mib = get_current_rss_mib()
    print(f"1. Idle Baseline Memory: {idle_mib:.2f} MiB ({idle_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%)")
    measurements.append({
        "stage": "idle_baseline",
        "description": "Baseline Python runtime before loading platform components",
        "rss_mib": idle_mib,
        "percent_of_512mib": round(idle_mib / RENDER_FREE_LIMIT_MIB * 100, 2),
    })

    # 2. AppTest: Overview Page
    app_path = Path(__file__).resolve().parent.parent / "src" / "assistant" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)
    overview_mib = get_current_rss_mib()
    print(f"2. Overview Page Memory: {overview_mib:.2f} MiB ({overview_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%)")
    measurements.append({
        "stage": "overview_page",
        "description": "Streamlit AppTest initial run rendering Overview page",
        "rss_mib": overview_mib,
        "percent_of_512mib": round(overview_mib / RENDER_FREE_LIMIT_MIB * 100, 2),
    })

    # 3. AppTest: Regional GIS Explorer
    at.sidebar.radio(key="platform_nav_radio").set_value("Regional GIS Explorer")
    at.run(timeout=30)
    gis_mib = get_current_rss_mib()
    print(f"3. GIS Explorer Memory: {gis_mib:.2f} MiB ({gis_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%)")
    measurements.append({
        "stage": "gis_explorer_page",
        "description": "Streamlit AppTest rendering GIS Explorer map and layer controls",
        "rss_mib": gis_mib,
        "percent_of_512mib": round(gis_mib / RENDER_FREE_LIMIT_MIB * 100, 2),
    })

    # 4. AppTest: Community Profiles
    at.sidebar.radio(key="platform_nav_radio").set_value("Community Profiles")
    at.run(timeout=30)
    community_mib = get_current_rss_mib()
    print(f"4. Community Profiles Memory: {community_mib:.2f} MiB ({community_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%)")
    measurements.append({
        "stage": "community_profiles_page",
        "description": "Streamlit AppTest rendering granular Community Profiles",
        "rss_mib": community_mib,
        "percent_of_512mib": round(community_mib / RENDER_FREE_LIMIT_MIB * 100, 2),
    })

    # 5. AppTest: Assistant Initialization
    at.sidebar.radio(key="platform_nav_radio").set_value("Cyclone Assistant")
    at.run(timeout=30)
    assistant_init_mib = get_current_rss_mib()
    print(f"5. Assistant Initialization Memory: {assistant_init_mib:.2f} MiB ({assistant_init_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%)")
    measurements.append({
        "stage": "assistant_initialization",
        "description": "Streamlit AppTest navigating to Cyclone Assistant view",
        "rss_mib": assistant_init_mib,
        "percent_of_512mib": round(assistant_init_mib / RENDER_FREE_LIMIT_MIB * 100, 2),
    })

    # 6. Assistant Request Memory: Mocked LLM Request Turn
    # Reuse cached client/service to mirror production single-container behavior
    from src.assistant.api_client import get_api_client
    cached_client = get_api_client(mode="direct")

    mock_orch = MagicMock()
    async def _mock_process(req):
        return ChatResponse(
            answer="Tallur (PC11: 608691) is in Band 4 Adequate with AHP preparedness score 74.34.",
            tools_executed=["lookup_community", "explain_preparedness_assessment"],
            tool_records=[
                ToolExecutionRecord(
                    tool_name="lookup_community",
                    arguments={"query": "Tallur"},
                    success=True,
                    result_summary="Resolved PC11: 608691",
                    execution_time_ms=1.1,
                )
            ],
            structured_evidence={
                "explain_preparedness_assessment": {
                    "name": "Tallur",
                    "district": "Udupi",
                    "ahp_score": 74.34,
                    "priority_band": "BAND_4_ADEQUATE",
                    "ml_surrogate_assessment": {
                        "tier": "Tier 1 - High Preparedness",
                        "scientific_disclaimer": "Logistic Regression is an empirical surrogate.",
                    },
                }
            },
            is_fallback=False,
            conversation_status=ConversationStatus.COMPLETED,
            model_name="qwen2.5:0.5b-instruct",
            total_duration_ms=420.0,
            scientific_precedence_note="AHP is authoritative.",
        )
    mock_orch.process_chat.side_effect = _mock_process

    mock_client = DecisionSupportApiClient(
        mode=AdapterMode.DIRECT,
        service=cached_client.service,
        orchestrator=mock_orch,
    )

    t0 = time.perf_counter()
    res_mock = mock_client.chat(ChatRequest(message="Assess Tallur in Udupi"))
    mock_req_dur_ms = round((time.perf_counter() - t0) * 1000, 2)
    assistant_mock_req_mib = get_current_rss_mib()
    print(f"6. Assistant Mocked LLM Request Memory: {assistant_mock_req_mib:.2f} MiB ({assistant_mock_req_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%) [Duration: {mock_req_dur_ms:.1f} ms]")
    measurements.append({
        "stage": "assistant_mock_llm_request",
        "description": "Executing conversational turn with mocked Qwen tool-calling response (reusing singleton service)",
        "rss_mib": assistant_mock_req_mib,
        "percent_of_512mib": round(assistant_mock_req_mib / RENDER_FREE_LIMIT_MIB * 100, 2),
        "duration_ms": mock_req_dur_ms,
    })

    # 7. Assistant Request Memory: Deterministic Fallback Request Turn
    # Uses real local deterministic tool execution reusing singleton service
    t0 = time.perf_counter()
    res_fallback = cached_client.chat(ChatRequest(message="Assess Edmoge in Dakshina Kannada"))
    fallback_dur_ms = round((time.perf_counter() - t0) * 1000, 2)
    assistant_fallback_req_mib = get_current_rss_mib()
    print(f"7. Assistant Fallback Request Memory: {assistant_fallback_req_mib:.2f} MiB ({assistant_fallback_req_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%) [Duration: {fallback_dur_ms:.1f} ms]")
    measurements.append({
        "stage": "assistant_fallback_request",
        "description": "Executing conversational turn with local deterministic fallback matrix (reusing singleton service)",
        "rss_mib": assistant_fallback_req_mib,
        "percent_of_512mib": round(assistant_fallback_req_mib / RENDER_FREE_LIMIT_MIB * 100, 2),
        "duration_ms": fallback_dur_ms,
    })

    # 8. Peak Memory Measurement
    peak_mib = get_peak_rss_mib()
    headroom_mib = round(RENDER_FREE_LIMIT_MIB - peak_mib, 2)
    headroom_pct = round((headroom_mib / RENDER_FREE_LIMIT_MIB) * 100, 2)
    print(f"\n8. PEAK PROCESS MEMORY (VmHWM): {peak_mib:.2f} MiB ({peak_mib/RENDER_FREE_LIMIT_MIB*100:.1f}%)")
    print(f"   Available Headroom: {headroom_mib:.2f} MiB ({headroom_pct}% remaining)")

    summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "memory_limit_mib": RENDER_FREE_LIMIT_MIB,
        "peak_rss_mib": peak_mib,
        "headroom_mib": headroom_mib,
        "headroom_percent": headroom_pct,
        "remains_suitable_for_render_free": peak_mib < (RENDER_FREE_LIMIT_MIB * 0.90),
        "concurrency_caveat": "Note: Measured in single-process Direct Mode. Concurrent request memory has not been benchmarked.",
        "measurements": measurements,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStructured benchmark results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
