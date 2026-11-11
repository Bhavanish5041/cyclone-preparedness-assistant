# CYCLONE INTEL: Self-Hosted Public Deployment Architecture Guide

## 1. System Overview

**CYCLONE INTEL** is a production-grade, self-hosted cyclone preparedness decision support platform for the coastal Karnataka region (Dakshina Kannada, Udupi, and Uttara Kannada across 1,879 administrative units).

The system runs entirely on the host laptop hardware (Intel Core i5-13500H, 16 GB RAM, Ubuntu Linux) without relying on paid external LLM APIs, cloud hosting fees, or third-party servers.

```
                                  [ PUBLIC INTERNET ]
                                           │
                                  (HTTPS / TLS 1.3)
                                           ▼
                                 [ Cloudflare Edge ]
                                           │
                        (Encrypted QUIC / HTTP/2 Tunnel)
                                           ▼
┌────────────────────────────────── LOCAL HOST ──────────────────────────────────┐
│                                                                                │
│  [ cloudflared daemon ] (systemd user service: cyclone-tunnel.service)         │
│          │                                                                     │
│          ▼ (HTTP / Port 80)                                                    │
│  [ Nginx Reverse Proxy ] (Docker container: cyclone-nginx, --restart always)   │
│     ├── Static SPA Routing (try_files -> /usr/share/nginx/html/index.html)     │
│     ├── Security Headers (X-Content-Type-Options, Frame-Options, XSS, CSP)     │
│     ├── Rate Limiting (30 req/s API, 5 req/s Chat)                             │
│     └── Block Rules (Strict 404 for /ollama, /api/generate, /api/tags)         │
│          │                                                                     │
│          ▼ (HTTP / Port 8000)                                                  │
│  [ FastAPI Application ] (systemd user service: cyclone-api.service)           │
│     ├── 8 API Routers (/communities, /districts, /maps, /analytics, etc.)      │
│     └── /api/assistant/chat                                                   │
│             │                                                                  │
│             ▼                                                                  │
│  [ Phase 8A IntentRouter & Refusal Gate ]                                      │
│     ├── Refusal Gate (Regex & rule checks for dynamic live telemetry)          │
│     ├── Intent Classifier (Logistic Regression on all-MiniLM-L6-v2, 100% Acc)  │
│     └── Argument Extractor & Disambiguator (Pydantic models, gazetteer)        │
│             │                                                                  │
│             ▼                                                                  │
│  [ Authoritative Deterministic Tool Suite ] (7 authoritative tools)            │
│     ├── AHP Multi-Criteria Preparedness Engine (Composite Scores 0.0 - 100.0)  │
│     ├── Operational 4-Band Prioritization (Band 1 Critical to Band 4 Adequate) │
│     ├── Vulnerability Archetypes (4 K-Means clusters + UK fallback heuristics) │
│     └── Empirical Logistic Regression Tier Surrogate (26 static features)     │
│             │                                                                  │
│             ▼ (Structured Evidence Package)                                    │
│  [ Local Ollama LLM Service ] (snap service: snap.ollama.ollama.service)       │
│     ├── Model: Qwen2.5-0.5B-Instruct (4-bit quantized, ~397 MB blob)           │
│     ├── Binding: 127.0.0.1:11434 ONLY (Blocked from public exposure)           │
│     ├── Concurrency: asyncio.Semaphore(1) bounded CPU scheduling               │
│     └── Scientific Precedence Reconciliation (audit_and_guard_answer)          │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Network Topology & Port Bindings

| Component | Port | Network Interface | Publicly Reachable? | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Cloudflare Edge** | 443 | Public Anycast DNS | Yes | Free, valid SSL/TLS termination |
| **Nginx Reverse Proxy** | 80 | `0.0.0.0:80` | Via Tunnel only | Serves React SPA & proxies `/api/` |
| **FastAPI Backend** | 8000 | `127.0.0.1:8000` | No (Loopback only) | Managed by systemd `cyclone-api` |
| **Ollama Daemon** | 11434 | `127.0.0.1:11434` | **NO (STRICTLY BLOCKED)** | Snap daemon; zero external exposure |
| **Prometheus (Metrics)** | 9090 | `0.0.0.0:9090` | Internal/Local only | Model monitoring container |
| **Streamlit (Dashboard)**| 8501 | `0.0.0.0:8501` | Internal/Local only | Decision support dashboard container |

---

## 3. Key Invariants & Scientific Guardrails

1. **Deterministic AHP Operational Precedence:**
   - Deterministic AHP Composite Preparedness Scores (0.0 to 100.0) and 4-tier priority classifications maintain absolute operational authority.
   - The language model (`qwen2.5:0.5b-instruct`) is strictly an evidence summarizer; it is mathematically incapable of overriding, altering, or re-ranking scores.

2. **Empirical Surrogate Demarcation:**
   - Multinomial Logistic Regression class probabilities approximate AHP tier boundaries on static Census 2011 indicators.
   - They do NOT represent physical hazard strike risk, flood inundation depth, storm surge, or casualty probability.

3. **Total Refusal of Dynamic Operational Telemetry:**
   - The Refusal Gate immediately intercepts and refuses queries regarding live wind speeds, cyclone tracks, shelter vacancy, hospital staffing, or road blockages.
   - Users are deterministically redirected to official IMD and district DDMA bulletin channels.

4. **Zero Paid LLM Dependencies:**
   - Model inference is 100% self-hosted on the laptop CPU using Ollama and Qwen2.5-0.5B-Instruct. No OpenAI, Anthropic, or paid Gemini API keys are required or called.
