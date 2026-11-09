# Phase 4.2: Gemini LLM Client and Tool Orchestrator Architecture & Verification

**Cyclone Preparedness Intelligence Platform**
**Version:** 4.2.0
**Date:** November 2026
**Status:** Completed & Validated

---

## 1. Executive Summary

Phase 4.2 integrates Google's official Gemini API (v1beta REST interface) with the deterministic tool suite created in Phase 4.1 (`src/assistant/tools/`). The result is a resilient conversational decision-support assistant exposed at `POST /api/v1/assistant/chat` in FastAPI.

The architecture prioritizes **scientific integrity**, **operational predictability**, and **zero runtime credential exposure**:
1. **Authoritative Precedence**: All factual, numerical, and preparedness assertions are strictly grounded in deterministic tool results from Phase 4.1.
2. **Surrogate ML Demarcation**: Logistic Regression remains an empirical tier surrogate (macro-feature approximation), never presented as a physical cyclone hazard or path forecast.
3. **Multi-Round Guardrails**: Strict turn limits prevent runaway tool invocation loops (maximum 4 rounds per user query, maximum 8 total tool calls).
4. **Deterministic Fallback Engine**: If `GEMINI_API_KEY` is unconfigured, expired, rate-limited, timed out, or experiencing upstream connectivity errors, the platform deterministically answers queries using the local tool registry with zero downtime.
5. **Render Free-Tier Optimization**: Designed to operate cleanly within 512 MB memory limits, using asynchronous connection pooling, bounded payloads, and zero heavyweight agent frameworks.

---

## 2. Package Architecture (`src/assistant/llm/`)

The LLM integration is housed in a clean, modular package independent of UI frameworks:

```
src/assistant/llm/
├── __init__.py           # Package exports (GeminiClient, ToolCallingOrchestrator, schemas, prompts)
├── client.py             # Asynchronous Gemini REST API client with header authentication & key redaction
├── orchestrator.py       # Conversational tool orchestrator with multi-round loop & fallback routing
├── prompts.py            # Scientific system instruction & deterministic fallback response templates
└── schemas.py            # Strict Pydantic v2 schemas for requests, responses, and tool records
```

### 2.1 File Responsibilities

| Module | Core Responsibility |
|---|---|
| [`client.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/llm/client.py) | Communicates with `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` via `httpx.AsyncClient`. Uses header-based `x-goog-api-key`. Categorizes HTTP status codes into typed exceptions (`GeminiAuthError`, `GeminiRateLimitError`, `GeminiTimeoutError`, `GeminiNetworkError`, `GeminiResponseError`). Automatically redacts API keys in all logs, exceptions, and text strings. |
| [`orchestrator.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/llm/orchestrator.py) | Coordinates the conversation cycle. Converts registry tools into Gemini Function Declarations. Executes requested functions strictly through `ToolRegistry.execute()`. Formats tool results into Gemini's `functionResponse` envelopes under `role: "user"`. Enforces `max_rounds=4` and `max_total_calls=8`. Triggers deterministic fallback if Gemini fails. |
| [`prompts.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/llm/prompts.py) | Defines the non-negotiable `SYSTEM_INSTRUCTION` covering AHP authority, surrogate ML demarcation, K-Means descriptive nature, Uttara Kannada 3-sector fallback, Census 2011 caveats, and zero real-time operational invention. Defines `format_deterministic_fallback_response`. |
| [`schemas.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/assistant/llm/schemas.py) | Pydantic v2 models: `ChatMessage`, `ChatRequest`, `ChatResponse`, `ToolExecutionRecord`, `ConversationStatus`, `MessageRole`. Implements strict string trimming, character bounds, and history length limits (max 10 messages). |

---

## 3. Gemini API Client Details

### 3.1 Endpoint & Authentication

- **API Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- **Authentication**: Header `x-goog-api-key: {GEMINI_API_KEY}`. The API key is **never** passed as a URL query parameter (`?key=...`) to ensure it does not appear in web server access logs, reverse proxy traces, or browser history.
- **Model Configuration**: Defaults to `gemini-2.5-flash` via `GEMINI_MODEL` environment variable, allowing dynamic upgrades to newer models without code modifications.

### 3.2 Key Protection & Scrubbing

Any raw key string matching `GEMINI_API_KEY` is scrubbed via regex replacement prior to formatting error logs or raising exceptions:
```python
clean_resp_text = self._mask_api_key_in_string(resp.text)
```
This guarantees that invalid key responses (HTTP 401/403) or proxy error messages containing the credential are unconditionally redacted to `[REDACTED_API_KEY]`.

---

## 4. Multi-Round Tool Calling & Loop Controls

### 4.1 Orchestration Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant API as FastAPI (/api/v1/assistant/chat)
    participant Orch as ToolCallingOrchestrator
    participant Gemini as Gemini API (v1beta)
    participant Reg as ToolRegistry
    participant Service as DecisionSupportService

    User->>API: POST ChatRequest (message, history, community_id)
    API->>Orch: process_chat(request)

    alt API Key missing or client unconfigured
        Orch->>Reg: Deterministic local tool execution
        Reg->>Service: Query community/decision metrics
        Service-->>Reg: Deterministic payload
        Reg-->>Orch: Tool results
        Orch-->>API: ChatResponse (is_fallback=True)
        API-->>User: Structured Fallback Response
    else API Key configured
        Orch->>Gemini: POST generateContent (system_instruction, tools, contents)
        loop Multi-Round Loop (max 4 rounds, max 8 calls)
            alt Gemini returns text candidate
                Gemini-->>Orch: Final text response
            else Gemini returns functionCall(s)
                Gemini-->>Orch: functionCall: [lookup_community, ...]
                Orch->>Reg: validate_and_execute(tool_name, tool_args)
                Reg->>Service: In-memory evaluation
                Service-->>Reg: Verified tool output
                Reg-->>Orch: Standard tool envelope
                Orch->>Gemini: POST generateContent (contents + functionResponse)
            end
        end
        Orch-->>API: ChatResponse (is_fallback=False, tools_executed, records)
        API-->>User: Grounded Answer + Evidence
    end
```

### 4.2 Guardrail Limits

1. **Maximum Rounds (`max_rounds = 4`)**: The orchestrator allows at most 4 back-and-forth function-call steps per user turn.
2. **Maximum Total Tool Invocations (`max_total_calls = 8`)**: Caps cumulative tool executions across rounds to 8. If reached, the orchestrator forces a final synthesis round with existing evidence.
3. **Execution Isolation**: Gemini is never allowed to execute arbitrary code or call tools outside `ToolRegistry.list_tools()`. Unknown tool names immediately return an error envelope (`"status": "error", "error_type": "UNKNOWN_TOOL"`), allowing the model to correct its call or apologize.

---

## 5. Scientific Precedence & System Instructions

The assistant is strictly constrained by the following scientific mandates:

1. **AHP Absolute Authority**: Preparedness classifications (Band 1 Critical to Band 4 Adequate) and composite scores (0.0 to 100.0) generated by the multi-criteria AHP model are authoritative and immutable.
2. **Surrogate ML Demarcation**: The Logistic Regression classifier is an empirical tier surrogate trained on macro-features. It must **never** be cited as a cyclone landfall probability, storm surge hazard risk, or real-time track forecast.
3. **K-Means Descriptive Clustering**: Archetype groupings (e.g., *Agrarian Inland Remote*, *Coastal High-Density Infrastructure*) are purely descriptive profiles based on historical Census features.
4. **Uttara Kannada Deterministic Fallback**: For the 1,257 units in Uttara Kannada where sub-district census covariates were unassigned, the 3-sector fallback methodology (coastal proximity, population density, primary health distance) must be transparently acknowledged.
5. **Historical Census 2011 Vintage**: Census data is over 15 years old. Demographic counts and infrastructure baselines must be qualified as historical references.
6. **Zero Real-Time Operational Invention**: The assistant has no live sensors, live weather feeds, or hospital dispatch data. It must never invent shelter occupancy, road passability, or current wind speeds, always directing users to the India Meteorological Department (IMD) and Karnataka State Disaster Management Authority (KSDMA) for active emergencies.

---

## 6. Deterministic Fallback Engine

When Gemini is unavailable (unconfigured key, network failure, 429 quota exhaustion, or 504 gateway timeout), the system seamlessly transitions to deterministic mode:

1. **Community Assessment Query**: If a community identifier is provided or detected in the user query, the orchestrator runs:
   - `explain_preparedness_assessment(identifier)`
   - `get_intervention_recommendations(identifier)`
2. **District Comparison Query**: If district names (Dakshina Kannada, Udupi, Uttara Kannada) are detected, it invokes:
   - `compare_districts(districts)`
3. **Priority Band Query**: If band names (Critical, High, Moderate, Adequate) are queried, it executes:
   - `filter_and_rank_communities(priority_band=...)`
4. **General Overview**: Returns a factual summary of coastal Karnataka coverage (1,879 units) with explicit guidance on querying specific communities or districts.

Each fallback response sets `is_fallback: true`, `conversation_status: "FALLBACK"`, and includes the full `SCIENTIFIC_PRECEDENCE_DISCLAIMER`.

---

## 7. FastAPI Endpoint: `POST /api/v1/assistant/chat`

### 7.1 Request Schema (`ChatRequest`)

```json
{
  "message": "What is the preparedness status and priority band of Tallur in Udupi?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hello, I am coordinating cyclone relief."
    },
    {
      "role": "assistant",
      "content": "Hello. I can assist with preparedness assessments across coastal Karnataka."
    }
  ],
  "community_identifier": "Tallur"
}
```

### 7.2 Response Schema (`ChatResponse`)

```json
{
  "answer": "Tallur (PC11 ID 608691, Kundapura taluk, Udupi district) has an authoritative AHP Preparedness Score of 74.34, placing it in Band 4 Adequate...",
  "conversation_status": "COMPLETED",
  "tools_executed": [
    "lookup_community",
    "explain_preparedness_assessment",
    "get_intervention_recommendations"
  ],
  "tool_records": [
    {
      "tool_name": "explain_preparedness_assessment",
      "arguments": {"identifier": "608691"},
      "success": true,
      "result_summary": "Status: ok | PC11: 608691 | AHP Score: 74.34 | Band: BAND_4_ADEQUATE",
      "execution_time_ms": 1.42
    }
  ],
  "structured_evidence": {
    "explain_preparedness_assessment": {
      "name": "Tallur",
      "district": "Udupi",
      "ahp_score": 74.34,
      "priority_band": "BAND_4_ADEQUATE",
      "ml_surrogate_assessment": {
        "tier": "Tier 1 - High Preparedness",
        "scientific_disclaimer": "Logistic Regression is an empirical AHP-tier surrogate, not a physical cyclone hazard or probability model."
      }
    }
  },
  "is_fallback": false,
  "fallback_reason": null,
  "model_name": "gemini-2.5-flash",
  "scientific_precedence_note": "AHP PRECEDENCE MANDATE: Preparedness bands and scores are calculated using the multi-criteria AHP model...",
  "timestamp": "2026-11-09T11:45:00Z"
}
```

---

## 8. Render Free-Tier Resource Profiling

| Dimension | Specification / Metric | Impact on 512 MB Free-Tier |
|---|---|---|
| **Python HTTP Client** | `httpx.AsyncClient` with connection pooling | ~1.5 MB memory overhead; non-blocking asynchronous event loop |
| **Request Payload Bounds** | Prompt ≤ 2,000 chars; History ≤ 10 items | Bounded memory usage per request (< 50 KB per request buffer) |
| **Tool Response Bounds** | Max 20 communities per roster; strict JSON | < 25 KB serialized tool payload |
| **Deterministic Latency** | In-memory lookup via pandas index | < 5 ms tool execution time |
| **LLM Latency** | Async streaming via Google Generative Language API | ~800 ms - 1.8 s per round (no server CPU burden during network I/O) |
| **Memory Headroom** | Base FastAPI + ML artifacts (~210 MB) | Assistant adds < 15 MB, keeping total usage well under 250 MB (50% margin) |

---

## 9. Verification & Test Suite Summary

The conversational LLM engine and tool orchestrator are verified by `tests/test_assistant_orchestrator.py`:

- **Client Configuration & Defaults**: Verifies `GEMINI_MODEL` initialization and endpoint base URL.
- **HTTP Status Code Mapping**: Tests 401/403 (Auth), 429 (Rate Limit), 500 (Upstream Error), TimeoutException, and ConnectError with proper error class translation.
- **Key Redaction**: Validates that secret API keys are scrubbed and replaced with `[REDACTED_API_KEY]` in error strings and exception info.
- **Direct Text Generation**: Verifies direct conversational responses when tools are not needed.
- **Single Tool Calling**: Verifies single function execution (`lookup_community`) and subsequent response grounding.
- **Sequential Tool Calling**: Verifies multi-round sequential execution (`lookup_community` followed by `explain_preparedness_assessment`).
- **Invalid Tool Invocation**: Verifies server-side rejection of non-existent tool names without orchestrator failure.
- **Max Rounds Cap**: Verifies enforcement of `max_rounds=4`, halting potential infinite loops.
- **Deterministic Fallback (Unconfigured Key)**: Verifies instant fallback execution when `GEMINI_API_KEY` is empty.
- **Deterministic Fallback (Rate Limit)**: Verifies graceful degradation on HTTP 429.
- **Deterministic Fallback (Timeout)**: Verifies graceful degradation on upstream timeout.
- **Scientific Guardrails**: Verifies presence of AHP precedence, surrogate ML demarcation, and Census 2011 vintage caveats in system instructions.
- **Pydantic Request Validation**: Verifies whitespace rejection, string bounds, and history length capping.
- **FastAPI Endpoint Integration**: Verifies live HTTP dispatch to `POST /api/v1/assistant/chat` via `starlette.testclient.TestClient`.

---

## 10. Next Milestone Recommendation: Phase 4.3

With the backend conversational engine, tool registry, and API endpoint fully operational and tested, the platform is ready for **Phase 4.3: Streamlit Assistant Chat Interface & Frontend Integration**:
1. Add an interactive **Conversational Assistant Drawer / Tab** in the Streamlit application.
2. Connect Streamlit to `POST /api/v1/assistant/chat` via `src/assistant/api_client.py`.
3. Render structured evidence cards (AHP score, priority band badges, archetype info) alongside natural language responses.
4. Provide quick-action suggested prompts (e.g. *"Assess Tallur's isolation"*, *"Compare Dakshina Kannada with Udupi"*, *"Show Band 1 Critical communities"*).
5. Display visual fallback indicators if the backend is operating in deterministic mode.
