# CYCLONE INTEL: Production MLOps Architecture & Operations Guide

**System Name:** CYCLONE INTEL — Cyclone Preparedness Decision Intelligence Platform  
**Target Region:** Coastal Karnataka (Dakshina Kannada, Udupi, Uttara Kannada across 1,879 administrative units)  
**Host Hardware Target:** Intel Core i5-13500H, ~12 GiB usable DDR5 RAM, Intel Iris Xe Graphics (No discrete GPU)  
**Host Operating System:** Ubuntu Linux 24.04 LTS (x86_64)  
**ML Frameworks:** scikit-learn 1.5+, Ollama / Qwen2.5-0.5B-Instruct, sentence-transformers, FastAPI 0.159+  

---

## 1. System Overview & Core Scientific Precedence

CYCLONE INTEL operates on a foundational operational safety principle:
**Deterministic Analytical Hierarchy Process (AHP) composite scoring and the 4-tier operational priority allocation maintain absolute operational precedence over all machine learning surrogate predictions and generative language model outputs.**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INCOMING USER QUERY                                  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PHASE 1: SUB-MS REFUSAL GATE                            │
│  - Evaluates dynamic real-time operational telemetry (wind speed, hospital beds, etc.) │
│  - Gated via high-precision compiled regex and sentence-transformer embeddings        │
│  - Returns immediate authoritative DDMA / IMD referral without LLM invocation         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (If query is valid baseline intelligence)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 2: QWEN INTENT PARSING                              │
│  - Local Qwen2.5-0.5B-Instruct via Snap Ollama (127.0.0.1:11434)                       │
│  - Maps query into structured QwenIntentParse schema (tool selection + arguments)      │
│  - Resolves ambiguous village names (e.g. Ullal in DK vs UK) via ArgumentExtractor     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 3: DETERMINISTIC TOOL SUITE (7 TOOLS)                      │
│  1. lookup_community: Demographic and spatial lookup                                   │
│  2. explain_preparedness_assessment: Exact AHP scores (0.0-100.0) & tiers (High/Med/Low)│
│  3. analyze_infrastructure_and_exposure: Road access & distance to coast / medical    │
│  4. get_intervention_recommendations: Prioritized policy actions based on deficits   │
│  5. compare_districts: Comparative totals across DK, Udupi, and Uttara Kannada         │
│  6. filter_and_rank_communities: 4 operational priority bands (Band 1 Critical to 4)  │
│  7. explain_vulnerability_archetype: 4 K-Means clusters + UK fallback heuristics      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (Structured Evidence Envelope)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: GROUNDED SYNTHESIS & AUDIT GUARD                         │
│  - Qwen synthesizes natural language briefing strictly grounded in tool data           │
│  - audit_and_guard_answer verifies AHP score and priority band in model prose          │
│  - If omission or discrepancy is detected, appends authoritative reconciliation block  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Lifecycle & Governance

The platform manages three distinct model families, each with strict operational scopes:

| Model Family | Artifact / Identifier | Framework | Operational Role | Governance Mandate |
| :--- | :--- | :--- | :--- | :--- |
| **Surrogate Classifier** | `village-preparedness-classifier:v1.0.0` | scikit-learn (RobustScaler + Logistic Regression) | Predicts calibrated tier probabilities across 26 features; detects boundary cases | **NOT hazard risk:** Probabilities approximate AHP tiers; they do NOT forecast physical storm surge or casualties. |
| **Vulnerability Archetypes** | `coastal-vulnerability-kmeans:v1.0.0` | scikit-learn (K-Means k=4) | Groups 622 units into 4 descriptive vulnerability archetypes | **Descriptive only:** Does not predict cyclone damage. Unassigned Uttara Kannada units use fallback heuristics. |
| **Orchestration LLM** | `qwen2.5:0.5b-instruct` | Ollama / llama.cpp (4-bit quantized) | Query parsing, intent selection, structured argument extraction, grounded synthesis | **Tool-grounded:** Outputs must reflect deterministic tool results; audit guard enforces ground truth. |

---

## 3. Local Model Registry Architecture

To eliminate reliance on external cloud services (W&B, MLflow) while maintaining enterprise-grade auditability, CYCLONE INTEL implements a lightweight, local, Git-compatible Model Registry (`models/model_registry.json`).

### Manifest Capabilities:
- **Cryptographic SHA256 Verification:** Every serialized artifact on disk is hashed and verified before promotion.
- **Strict Versioning & Lifecycle States:** Tracks `candidate`, `validated`, `production`, and `retired` states.
- **Complete Provenance:** Links each model version to training dataset splits, input/output schemas, and evaluation dates.
- **Audit History:** Full log of promotion and rollback events with timestamps, authors, and rationale notes.

### Model Registry CLI Operations:
```bash
# 1. List all registered models and active versions
python scripts/model_registry_cli.py list

# 2. Inspect metadata and schema for a model
python scripts/model_registry_cli.py show village-preparedness-classifier

# 3. Validate SHA256 checksums of all model files on disk
python scripts/model_registry_cli.py validate

# 4. Promote a candidate model to production
python scripts/model_registry_cli.py promote village-preparedness-classifier v1.1.0 --status production --note "Passed quality gates"

# 5. Execute an emergency rollback to a previous version
python scripts/model_registry_cli.py rollback village-preparedness-classifier v1.0.0 --note "Restoring validated baseline"
```

---

## 4. Evaluation Pipeline & Quality Gates

The model quality evaluation suite (`src/evaluation/model_quality_pipeline.py`) evaluates model performance across an independently authored, comprehensive regression dataset (`tests/fixtures/model_test_dataset.json`) containing **130 curated test cases** covering all operational domains:

- Community lookup & identification
- Multi-district comparative analysis (DK, Udupi, Uttara Kannada)
- Granular preparedness assessments & domain deficit triggers
- Infrastructure exposure & evacuation shelter connectivity
- Intervention recommendation heuristics
- Complex multi-intent requests
- Ambiguous community and district names (e.g., Belur, Ullal)
- Unsupported live meteorological & operational telemetry requests
- Adversarial prompt injections & jailbreak attempts
- Boundary cases near AHP tier thresholds
- Contradictory evidence and malformed schemas

> [!NOTE]
> Ground-truth AHP scores, operational priority bands, and Census baseline metrics are dynamically derived directly from authoritative datasets (`DecisionSupportService.ranked_df`). Zero synthetic or fabricated AHP scores are used.

### Regression Gates & Acceptance Metrics (130 Cases):

| Quality Gate Metric | Acceptance Threshold | Measured Baseline | Status | Evaluation Scope & Notes |
| :--- | :--- | :--- | :--- | :--- |
| **AHP Score Invariance Gate** | $= 1.0000$ (100%) | **1.0000** | PASS ✅ | Ground-truth AHP scores are perfectly preserved across all turns |
| **No ML Downgrade of AHP Low**| $= 1.0000$ (100%) | **1.0000** | PASS ✅ | Absolute invariant: ML surrogate never downgrades Low preparedness |
| **Intent Macro F1** | $\ge 0.8500$ | **0.9575** | PASS ✅ | Balanced precision/recall across all 7 tool intents + refusal |
| **Tool Argument Accuracy** | $\ge 0.8500$ | **0.9701** | PASS ✅ | Evaluates identifier extraction, district pairs, and filters |
| **Entity Resolution Accuracy** | $\ge 0.9000$ | **0.9701** | PASS ✅ | Disambiguates duplicate village names across coastal taluks |
| **Multi-Intent Completeness** | $\ge 0.8500$ | **0.9630** | PASS ✅ | Extracts complete tool sequence for complex composite queries |
| **JSON Schema Validity Rate** | $\ge 0.9500$ | **1.0000** | PASS ✅ | Pydantic schema conformance across all structured payloads |
| **Grounded Factual Correctness**| $\ge 0.9000$ | **1.0000** | PASS ✅ | Strict alignment with authoritative baseline data |
| **Unsupported Claim Frequency** | $= 0.0000$ (0%) | **0.0000** | PASS ✅ | Zero hallucinated metrics, hospital beds, or dynamic telemetry |
| **Refusal Precision** | $\ge 0.9000$ | **1.0000** | PASS ✅ | 16 positive dynamic queries, 0 false positive refusals |
| **Refusal Recall** | $\ge 0.9000$ | **1.0000** | PASS ✅ | 16 positive dynamic queries, 0 false negative bypasses |
| **Deterministic Pipeline Latency (p95)** | $\le 100\text{ ms}$ | **24.30 ms** | PASS ✅ | Offline gate, regex/embedding scan, and tool routing (mean: 19.35 ms) |
| **Deterministic Tool Latency (p95)** | $\le 25\text{ ms}$ | **3.82 ms** | PASS ✅ | Execution time of pure Python analytical tools (mean: 2.20 ms) |
| **Live Qwen Generation Latency (mean)** | $< 15\text{ s}$ | **364.83 ms** | PASS ✅ | Ollama CPU generation on Intel Core i5-13500H (~70.35 tok/s) |
| **Process RSS Memory Delta** | $\le 250\text{ MB}$ | **6.79 MB** | PASS ✅ | Initial RSS: 872.66 MB → Final RSS: 879.45 MB across full suite |

> [!IMPORTANT]
> **Sample Size & Statistical Scope Disclosure:**
> The evaluation dataset (`tests/fixtures/model_test_dataset.json`) contains **130 curated test cases**. While substantially expanded to cover 16 distinct error modes and edge cases, this dataset remains a targeted functional and scientific invariant test harness; it does **not** represent an open-ended natural distribution sample for production concept drift. Refusal recall is evaluated on 16 live telemetry edge cases.
>
> **Latency Measurement Clarification:**
> The automated evaluation reports three disaggregated latency metrics:
> 1. *Deterministic Pipeline Latency* (mean: 19.35 ms, p95: 24.30 ms): Refusal regex/embedding evaluation, Qwen intent parsing, entity resolution, and structured envelope construction.
> 2. *Deterministic Tool Execution Latency* (mean: 2.20 ms, p95: 3.82 ms): Exact pandas/numpy calculations for dossiers and comparisons.
> 3. *Live Qwen Generation Latency* (mean: 364.83 ms, p95: 579.52 ms): Time spent purely generating response tokens in Ollama.
>
> **Surrogate Probability Semantics:**
> Logistic Regression class probabilities represent empirical class likelihoods approximating AHP decision boundaries across 26 Census indicators. They are **never** described or utilized as physical cyclone hazard probabilities, storm-surge inundation risks, or casualty forecasts.

Automated regression checks run via:
```bash
# Backend unit, integration, and model quality tests (517 passed, 1 skipped)
pytest tests/test_model_quality_gates.py tests/test_model_inference_suite.py

# Frontend React test suite (10 passed)
cd frontend && npm test -- --run
```

---

## 5. FastAPI Inference Service Architecture

The FastAPI backend (`api/main.py`) provides high-performance asynchronous endpoints under `/api`:

### Endpoints:
- `POST /api/models/predict`: Typed surrogate inference with 26-feature validation, calibrated probabilities, boundary case detection, and authoritative AHP comparison.
- `GET /api/models/registry`: Inspect registered models and checksum verification status.
- `GET /api/models/registry/{model_id}`: Detailed model schema and evaluation history.
- `GET /api/health` and `GET /health`: Liveness probe reporting unit counts and model load state.
- `GET /api/health/ready` and `GET /ready`: Readiness probe verifying dataset, surrogate model, and Ollama connectivity.
- `POST /api/assistant/chat`: Conversational decision support with tool grounding and audit guards.

### Middleware & Observability:
- **`RequestIDMiddleware`:** Automatically injects or propagates unique `X-Request-ID` headers across all incoming requests and logs structured access traces with elapsed milliseconds.
- **Performance Timing:** Returns `X-Process-Time` on all HTTP responses.
- **Centralized Error Handling:** Formats unexpected exceptions as consistent JSON envelopes with zero internal Python trace leakage.

---

## 6. Docker Containerization & Service Topology

The production containerized stack is composed of 4 modular microservices supervised via Docker Compose, alongside a lightweight, non-privileged systemd user bridge for secure host LLM connectivity:

```
┌───────────────────────────────── DOCKER NETWORK (cyclone-net) ─────────────────────────────────┐
│                                                                                                │
│  [ cyclone-frontend ] (Nginx + React SPA bundle, port 80)                                      │
│     ├── Serves static assets with 1-year immutable caching                                     │
│     ├── Proxies /api/assistant/chat with 5 req/s rate limit & 90s timeout                      │
│     ├── Proxies general /api/ with 30 req/s rate limit & 30s timeout                           │
│     └── Blocks /ollama, /api/generate, /api/tags (Strict 404)                                  │
│          │                                                                                     │
│          ▼ (HTTP / Port 8000)                                                                  │
│  [ cyclone-api ] (FastAPI Python 3.14 backend)                                                 │
│     ├── Bound to internal bridge network (no unnecessary public port exposure)                │
│     ├── Memory Limit: 1.5 GiB RAM, CPU Limit: 2.0 Cores                                        │
│     └── Mounts model cache and persistent monitoring volumes                                   │
│          │                                                                                     │
│          ▼ (host.docker.internal:11434 -> 172.18.0.1:11434)                                    │
│  [ cyclone-ollama-bridge.service ] (Internal Docker Bridge Proxy, ~1.7 MB RAM)                 │
│     ├── Bound strictly to internal Docker bridge interfaces (172.18.0.1 / 172.17.0.1)          │
│     └── Forwards to Host Snap Ollama Daemon (127.0.0.1:11434, qwen2.5:0.5b-instruct)           │
│                                                                                                │
│  [ cyclone-prometheus ] (Observability TSDB, port 9090)                                        │
│  [ cyclone-dashboard ] (Streamlit regional command dashboard, port 8501)                       │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Security & Hardware Isolation:
- **Zero External LLM Exposure:** Ollama is bound to loopback `127.0.0.1:11434`. The bridge proxy binds strictly to internal Docker interfaces (`172.18.0.1`). Port 11434 is never bound to `0.0.0.0` or reachable externally.
- **Resource Constraints (Tailored for 12 GiB System RAM):**
  - **Frontend Nginx (`cyclone-frontend`):** 256 MB RAM limit, 1.0 CPU core (measured: ~13 MiB).
  - **FastAPI Backend (`cyclone-api`):** 1,536 MB RAM limit, 2.0 CPU cores (measured: ~284 MiB).
  - **Streamlit Dashboard (`cyclone-dashboard`):** 1,024 MB RAM limit, 1.0 CPU core (measured: ~56 MiB).
  - **Prometheus TSDB (`cyclone-prometheus`):** 512 MB RAM limit, 0.5 CPU core (measured: ~24 MiB).
  - **Total Active Fleet Memory:** ~377 MiB across all 4 containers (over 95% headroom available on 12 GiB host).

---

## 7. CI/CD Pipeline Architecture

GitHub Actions workflows are partitioned into continuous integration and controlled deployment:

### Workflow 1: `.github/workflows/ci-cd.yml` (Automated PR/Push CI)
1. **Python Linting:** Syntax verification across all Python files and `flake8` critical error scanning (`E9,F63,F7,F82`).
2. **Frontend Validation:** Node.js 22 setup, deterministic `npm ci` (no fallbacks), Vitest test suite execution, and production Vite bundle build.
3. **Git LFS Validation:** Byte-level validation of Parquet headers (`PAR1`) for all 6 required datasets.
4. **Model Quality Regression Gates:** Deterministic pytest execution verifying model inference, AHP invariance, and 13 scientific quality gates (`test_model_quality_gates.py`) on GitHub CPU runners without downloading Ollama.
5. **Security Auditing:** `pip-audit` automated dependency vulnerability scanning.
6. **Docker Compose Validation:** Static validation of Compose topology (`docker compose config --quiet`).
7. **Container Image Build & Publish:** Builds and publishes immutable SHA-tagged images (`ghcr.io/bhavanish5041/cyclone-assistant-api:sha-<short_sha>`) to GitHub Container Registry (GHCR) on `master` pushes only (never on untrusted PRs).

### Workflow 2: `.github/workflows/manual-deploy.yml` (Controlled CD)
- **Manual Trigger Only (`workflow_dispatch`):** Requires human operator confirmation (`YES-DEPLOY`).
- **Environment Selection:** Controlled staging vs production targeting.
- **Pre-Deployment Safety Gate:** Runs local model artifact SHA256 validation and quality regression suite before authorizing cutover.

---

## 8. Safe Rollback & Emergency Operating Procedures

### Scenario A: Instant Automated Rollback to Host Services
If any containerized service fails or health probes degrade:
```bash
# Execute the tested 1-command rollback script:
bash scripts/rollback_to_host.sh
```
This automatically stops Docker Compose, re-enables and starts host `cyclone-api.service`, restores host `cyclone-nginx`, restarts the Cloudflare Tunnel, and verifies `http://127.0.0.1/health`.

### Scenario B: Roll Back a Machine Learning Model Version
If a newly deployed surrogate model demonstrates unexpected distribution drift:
```bash
# 1. Roll back active version in local registry
python scripts/model_registry_cli.py rollback village-preparedness-classifier v1.0.0 --note "Reverting to validated baseline"

# 2. Restart backend container
docker compose restart api

# 3. Verify restored model version
curl -s http://127.0.0.1:8000/api/models/registry | jq .
```

### Scenario C: Day-to-Day Operations & Monitoring
```bash
# Run real-time operational monitor:
.venv/bin/python scripts/ops_monitor.py

# Run full end-to-end 14-point production verification:
.venv/bin/python scripts/e2e_production_test.py

# View live container logs:
docker logs -f cyclone-api
docker logs -f cyclone-frontend

# View Cloudflare Tunnel logs:
journalctl --user -u cyclone-tunnel.service -f
```
