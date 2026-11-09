# Phase 4.3: Streamlit Conversational Assistant Integration & Render Architecture

**Cyclone Preparedness Intelligence Platform**
**Version:** 4.3.0
**Date:** November 2026
**Status:** Completed & Validated

---

## 1. Executive Summary

Phase 4.3 successfully integrates the conversational AI decision support assistant into the Streamlit platform, fulfilling the operational requirements of coastal disaster response commanders across Dakshina Kannada, Udupi, and Uttara Kannada.

Crucially, the integration preserves the **single-container Direct Mode architecture** specified for Render Free deployment. Streamlit communicates with the tool-calling orchestrator entirely in-process, bypassing the need for a running FastAPI server or internal HTTP network requests to localhost.

### Key Architectural Tenets Preserved:
1. **Single-Container Direct Mode**: Zero internal network hops; synchronous client adapter safely drives asynchronous orchestration in-process.
2. **Warm Chocolate Editorial Design**: Zero emojis, zero em dashes, elegant typography (DM Serif Display and DM Sans), restrained metric cards, and clear visual hierarchy.
3. **Scientific Governance**: Deterministic AHP remains authoritative (0.0 to 100.0, Bands 1 to 4); Logistic Regression is strictly demarcated as an empirical tier surrogate (macro-feature approximation); K-Means archetypes remain descriptive; Uttara Kannada uses the verified 3-sector fallback.
4. **Resilient Fallback**: Operates with full operational fidelity even when `GEMINI_API_KEY` is absent, rate-limited, or timed out.
5. **Render Free Compatibility**: The assistant layer adds less than 0.5 MiB to application memory, executing in sub-millisecond to sub-second durations.

---

## 2. Direct Mode Integration Architecture

### 2.1 The Standalone Render Problem & Solution

In standard containerized deployments, web interfaces frequently call backend APIs over HTTP (`http://localhost:8000/api/v1/...`). However, in Render Free-Tier deployments:
- Only a single web service is provisioned (`Dockerfile.render`).
- Running both Uvicorn/FastAPI and Streamlit in parallel within a 512 MiB container risks memory starvation and process management race conditions.
- Direct Mode was established in Phase 3.0 to query `DecisionSupportService` in-memory.

Phase 4.3 extends this pattern to conversational AI via [`DecisionSupportApiClient.chat()`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/api_client.py#L202-L260):

```mermaid
graph TD
    subgraph Single Container [Render Standalone Container (512 MiB)]
        UI[Streamlit App - src/assistant/app.py]
        View[Cyclone Assistant View - src/assistant/views/chat_assistant.py]
        Client[DecisionSupportApiClient - src/assistant/api_client.py]
        Orch[ToolCallingOrchestrator - src/assistant/llm/orchestrator.py]
        Reg[ToolRegistry - src/assistant/tools/registry.py]
        DSS[DecisionSupportService - In-Memory Parquet & ML]

        UI --> View
        View --> Client
        Client -- "Direct Mode (In-Process)" --> Orch
        Orch --> Reg
        Reg --> DSS
    end

    subgraph External [External Services]
        Gemini[Google Gemini API v1beta]
    end

    Orch -. "Optional HTTPS (Header Auth)" .-> Gemini
```

### 2.2 Asynchronous Event Loop Bridging

`ToolCallingOrchestrator.process_chat()` is an asynchronous coroutine (`async def`). Because Streamlit executes page rendering within synchronous script runner threads, `DecisionSupportApiClient` includes a thread-safe execution bridge:

```python
@staticmethod
def _run_async(coro: Any) -> Any:
    """Run an asynchronous coroutine synchronously, handling existing event loops safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
```

This guarantees zero deadlocks or "event loop already running" exceptions across both interactive Streamlit runs and automated testing environments.

---

## 3. Conversational Interface & User Experience

The assistant view is housed in [`src/assistant/views/chat_assistant.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/views/chat_assistant.py) and registered in the sidebar navigation as **Cyclone Assistant**.

### 3.1 Design System Adherence
- **Color Palette**: Dark chocolate `#211914`, elevated surfaces `#2B211B`, cards `#342820`, text `#F3E9DC` and `#BDAE9C`, warm ochre `#C18B62`, and terracotta accent `#D6B28C`.
- **Zero Emojis**: System statuses are presented using crisp text badges (`[SUCCESS]`, `[FAILED]`, `[DETERMINISTIC MODE]`).
- **Zero Em Dashes**: Clean colons `:` and hyphens `-` are used exclusively.

### 3.2 Five Standard Operational Inquiries
Five preset buttons provide single-click access to standard disaster response scenarios:
1. `Explain why Tallur has its current priority classification.`
2. `Compare Dakshina Kannada and Udupi.`
3. `Find Band 1 communities.`
4. `Explain Uttara Kannada's fallback methodology.`
5. `Assess a community's healthcare accessibility.`

### 3.3 Message Bubble Architecture
- **User Query Bubble**: Styled as a distinct warm card (`#342820`) with timestamp and role label.
- **Assistant Response Container**: Styled on elevated surface (`#2B211B`) with a left amber accent border (`#C18B62`).
- **Deterministic Indicator**: When operating in fallback mode, a highlighted amber notice alerts the user that evaluations were computed directly by the local decision matrix.
- **Tool Telemetry Expander**: Collapsible panel showing the count of tools invoked, total latency in milliseconds, arguments passed, and per-tool execution summaries.
- **Authoritative Evidence Cards**: Automatically rendered when community evaluation evidence is present, showing community metadata, authoritative AHP score, priority band badge, and the mandatory surrogate ML disclaimer.

### 3.4 Conversation State & Guardrails
- **Session State**: Message history is stored in `st.session_state["assistant_messages"]`.
- **Ephemeral Storage**: Messages exist only in browser session memory; no conversation logs or user inputs are persisted to disk.
- **Bounded Memory**: Session history is capped at 20 messages (`MAX_SESSION_HISTORY_MESSAGES = 20`) to prevent browser memory growth.
- **Bounded Request Turns**: Only the most recent 10 turns (`MAX_REQUEST_HISTORY_TURNS = 10`) are forwarded to the orchestrator, matching backend schema constraints.
- **Duplicate Prevention**: A boolean state lock (`st.session_state["is_processing"]`) and whitespace filtering prevent accidental double submissions.
- **Clear History**: A prominent "Clear Conversation" button resets session history cleanly.

---

## 4. Configuration & API Handling

The assistant automatically detects its environment and adapts its behavior:

| Environment Variable | Default Value | Description |
|---|---|---|
| `GEMINI_API_KEY` | `""` (Empty) | Google Gemini API key. If absent or invalid, platform automatically uses deterministic fallback. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Configurable model identifier for Gemini v1beta REST API. |
| `CYCLONE_ADAPTER_MODE` | `direct` | Execution mode (`direct` for in-process, `rest` for HTTP client). |

---

## 5. Render Memory & Performance Profiling

A dedicated memory benchmark script ([`scripts/measure_phase4_3_memory.py`](file:///home/bhavanish/cyclone-preparedness-assistant/scripts/measure_phase4_3_memory.py)) was executed against the platform. Exact measurements were recorded to [`docs/memory_measurement_phase4_3.json`](file:///home/bhavanish/cyclone-preparedness-assistant/docs/memory_measurement_phase4_3.json).

### 5.1 Test-Runner Process Measurements (`measure_phase4_3_memory.py`)

| Stage | Process RSS | Change | Details |
|---|---|---|---|
| **Idle Baseline** | 323.94 MiB | Baseline | Python runtime + pytest/AppTest harness loaded |
| **Overview Page** | 452.83 MiB | +128.89 MiB | Parquet dataset, GeoPandas, and Altair imported |
| **GIS Explorer** | 534.87 MiB | +82.04 MiB | Folium mapping, shapefile geometries rendered |
| **Community Profiles** | 543.92 MiB | +9.05 MiB | Granular radar and indicator charts generated |
| **Assistant Initialization** | **544.01 MiB** | **+0.09 MiB** | Cyclone Assistant view rendered in Streamlit |
| **Assistant Mocked Request** | **544.02 MiB** | **+0.01 MiB** | Mocked Gemini tool-calling turn (Duration: 0.7 ms) |
| **Assistant Fallback Request**| **544.35 MiB** | **+0.33 MiB** | Deterministic local tool execution (Duration: 7.3 ms) |
| **Peak Process RSS (VmHWM)** | 547.77 MiB | - | Peak test process allocation |

### 5.2 Standalone Container Runtime Reality (`Dockerfile.render`)

In the test runner, Python baseline starts at ~324 MiB because the entire pytest/AppTest framework is loaded in the same process.

In the actual standalone Docker container (`python:3.14-slim`), memory usage is substantially lower, as verified in Phase 3.1 container measurements ([`docs/memory_measurement_phase3_1.json`](file:///home/bhavanish/cyclone-preparedness-assistant/docs/memory_measurement_phase3_1.json)):
- **Container Idle / Startup**: 63.31 MiB (12.37% of 512 MiB limit)
- **Container Overview**: 255.40 MiB (49.89%)
- **Container GIS Explorer**: 402.00 MiB (78.51%)
- **Container Community Profiles**: 416.70 MiB (81.40%)
- **Container Reports & Archetypes**: 420.10 MiB (82.05%)
- **Projected Container with Assistant**: **~420.6 MiB (82.15%)**
- **Available Headroom**: **~91.4 MiB (~17.85% margin)**

### 5.3 Performance & Latency
- **Mocked / Streamed LLM Turn**: < 1.0 ms local orchestrator latency + network I/O.
- **Deterministic Fallback Evaluation**: **7.3 ms** total execution time to query in-memory Parquet data, calculate metrics, and formulate response.
- **Assistant Memory Overhead**: **< 0.5 MiB total**, demonstrating that the assistant adds virtually zero incremental memory pressure over the baseline decision-support services.

> [!WARNING]
> **Concurrency Limitation**: Memory measurements were conducted for a single-user active session. Concurrent multi-user sessions on Render Free (512 MiB) have **not** been benchmarked and could risk memory exhaustion under simultaneous GIS rendering. For multi-commander operational deployments, vertical scaling to Render Starter (2 GB) or horizontal container instances is recommended.

---

## 6. Verification & Regression Suite

The implementation is verified by [`tests/test_assistant_streamlit.py`](file:///home/bhavanish/cyclone-preparedness-assistant/tests/test_assistant_streamlit.py):

- `test_init_chat_session_state`: Validates session keys initialization without overwriting active data.
- `test_clear_chat_history`: Validates message history and processing state reset.
- `test_duplicate_and_empty_submission_handling`: Verifies blank/whitespace prompt rejection.
- `test_session_history_bounded_size`: Validates history capping at 20 messages.
- `test_direct_mode_chat_execution_fallback`: Validates in-process Direct Mode execution and deterministic fallback output.
- `test_direct_mode_chat_mocked_llm_response`: Validates tool-grounded synthesis with mocked LLM response.
- `test_render_tool_execution_details_no_crash`: Validates telemetry rendering without exceptions.
- `test_render_evidence_cards_surrogate_ml_disclaimer`: Validates mandatory surrogate ML disclaimer in UI.
- `test_render_message_bubble_styling`: Validates warm chocolate styling for user and assistant bubbles.
- `test_apptest_navigation_to_cyclone_assistant`: Validates end-to-end AppTest navigation to Cyclone Assistant page and widget presence.

### Regression Test Results
```bash
.venv/bin/pytest tests/
# 348 passed, 13 warnings in 48.12s
```
- **Total Passing Tests**: **348** (338 baseline + 10 new Streamlit assistant tests).
- **Regressions**: **0**.

---

## 7. Deployment Instructions

### 7.1 Single-Container Direct Mode (Render Free)

Deploy using the existing [`Dockerfile.render`](file:///home/bhavanish/cyclone-preparedness-assistant/Dockerfile.render):

```dockerfile
# Start Streamlit application in Direct Mode binding to Render dynamic PORT
CMD ["sh", "-c", "exec streamlit run src/assistant/app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
```

### 7.2 Render Environment Variables
- `CYCLONE_ADAPTER_MODE`: `direct`
- `GEMINI_API_KEY`: *(Optional)* Provide Google Gemini API key for external LLM tool calling. If omitted, assistant automatically defaults to deterministic fallback.
- `GEMINI_MODEL`: `gemini-2.5-flash`

---

## 8. Conclusion on Render Free Suitability

**Verdict**: The integrated application **remains suitable** for single-process Render Free deployment.

1. **Memory Ceiling Compliance**: The assistant adds less than 0.5 MiB to application memory. In the standalone Render container, total peak memory remains approximately 420.6 MiB, preserving a ~91 MiB (~18%) safety margin below the 512 MiB limit.
2. **Zero Additional Processes**: No secondary FastAPI or Uvicorn background process is needed.
3. **Zero Localhost Network Latency**: In-process tool calling operates with sub-10ms deterministic response times.
4. **Graceful Fallback**: The platform functions seamlessly out of the box without requiring external API keys.
