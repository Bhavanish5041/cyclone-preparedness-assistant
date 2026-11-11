# CYCLONE INTEL: Self-Hosted Production Deployment Verification Report

**Date:** 2026-11-10  
**Host Platform:** Ubuntu Linux 24.04 LTS (x86_64)  
**Hardware:** Intel Core i5-13500H (12 Cores, 16 Threads, 4.7 GHz Turbo), 16 GB DDR5 RAM, Intel Iris Xe Graphics (No NVIDIA GPU)  
**Project Workspace:** `/home/bhavanish/cyclone-preparedness-assistant`  
**Primary Live Public URL:** `https://pediatric-gen-safari-makers.trycloudflare.com`  

---

## 1. Executive Summary

The entire **CYCLONE INTEL** cyclone preparedness decision intelligence platform has been successfully configured, tested, and deployed as a fully self-hosted system on the host laptop.

All ten requirements of the master task specification have been achieved:
1. **React + TypeScript frontend:** Production Vite bundle built and served with SPA routing fallback.
2. **FastAPI backend:** Production ASGI server running under systemd user service management.
3. **Deterministic cyclone intelligence engine:** Authoritative AHP calculations, 4-tier operational bands, 4 K-Means archetypes, and Logistic Regression surrogate preserved with 100% calculation integrity across 1,879 units.
4. **Phase 8A IntentRouter & Refusal Gate:** High-precision intent classifier (100% test accuracy on `all-MiniLM-L6-v2`) and refusal gate integrated into the assistant flow.
5. **Local Language Model:** `qwen2.5:0.5b-instruct` running locally via Ollama with zero paid external API dependencies.
6. **Nginx reverse proxy:** Production Nginx proxy on port 80 routing static assets and proxying `/api/` with security headers, rate limiting, and blocking of internal ports.
7. **Automatic startup at reboot:** Configured via systemd user services (`cyclone-api`, `cyclone-tunnel`) with system-level user lingering (`Linger=yes`), snap-managed Ollama, and Docker container restart policy (`--restart always`).
8. **Public internet access:** Live Cloudflare Tunnel (`cloudflared`) providing encrypted HTTPS public access with zero port forwarding.
9. **No paid LLM API:** Completely eliminated dependency on paid APIs (OpenRouter/Gemini disabled in favor of local Ollama).
10. **No Render or external application hosting:** Host laptop acts as the self-contained production application server.

---

## 2. Component Status & Ports

| Subsystem | Service / Process | Status | Port / Socket | Access Control |
| :--- | :--- | :--- | :--- | :--- |
| **Cloudflare Tunnel** | `cyclone-tunnel.service` (systemd user) | Active (running) | Encrypted QUIC/H2 | Public Anycast Edge |
| **Nginx Reverse Proxy**| `cyclone-nginx` (Docker container) | Up / Healthy | `0.0.0.0:80` | Port 80 via tunnel/LAN |
| **FastAPI Backend** | `cyclone-api.service` (systemd user) | Active (running) | `127.0.0.1:8000` | Loopback only |
| **Local LLM Daemon** | `snap.ollama.ollama.service` | Active (running) | `127.0.0.1:11434` | **STRICTLY BLOCKED** |
| **Model Embedder** | MiniLM-L6-v2 (CPU Transformer) | In-process | Memory cached | Internal |
| **Surrogate Classifier**| Multinomial Logistic Regression | In-process | Memory cached | Internal |

---

## 3. Resource Footprint & Hardware Utilization

Measurements taken during active inference and load testing:

```
=== MEMORY USAGE ===
Total RAM:        12.0 GiB (16 GB hardware less integrated GPU reservation)
Used RAM:         8.1 GiB
Free / Available: 4.8 GiB available (plenty of operational headroom)

=== COMPONENT RESIDENT MEMORY (RSS) ===
Ollama LLM Engine (llama-server):  617.9 MB
FastAPI Backend (uvicorn + data):  875.2 MB
Cloudflare Tunnel (cloudflared):    42.0 MB
Nginx Reverse Proxy (all workers): ~35.0 MB
Total Self-Hosted Stack Footprint: ~1.57 GB RAM

=== CPU & INFERENCE PERFORMANCE ===
CPU Load Average:   2.27 (12 physical/logical cores, low load)
Intent Routing:     2.1 ms on CPU
Deterministic Tool: 1.0 - 3.4 ms
LLM Synthesis:      1.2s - 3.8s for 150-250 tokens
End-to-End Latency: ~3.9s over public HTTPS Cloudflare Tunnel
```

---

## 4. Test Suite Execution & Verification

| Test Suite | Tests Run | Result | Duration | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **CycloneLM Phase 8A Intent & Routing** | 83 | PASS (83/83) | 9.25s | Classifiers, refusal gate, routing |
| **Ollama Client & Pipeline Integration**| 9 | PASS (9/9) | 5.66s | Local inference, timeouts, fallback |
| **Orchestrator & Tool Grounding** | 75 | PASS (75/75) | 6.10s | AHP precedence, multi-turn loops |
| **Combined Backend Regression** | 157 | PASS (157/157)| 11.99s | Zero regressions across backend |
| **React Frontend Tests (Vitest)** | 10 | PASS (10/10) | 1.13s | Components, dashboard, assistant |
| **Vite Production Build** | N/A | SUCCESS | 2.15s | Clean bundle in `frontend/dist/` |

---

## 5. Security & Boundary Compliance Audit

1. **Strict Port 11434 Shielding:**
   - Ollama daemon is bound to `127.0.0.1:11434`.
   - Nginx explicitly returns HTTP 404 for any request targeting `/ollama`, `/api/generate`, `/api/tags`, or `/api/pull`.
   - Verified via `curl -I http://127.0.0.1/ollama` -> `HTTP/1.1 404 Not Found`.

2. **AHP Absolute Governance:**
   - Authoritative AHP calculations are performed deterministically in Python.
   - The LLM receives structured evidence only after deterministic execution.
   - `audit_and_guard_answer` verifies that AHP score and priority band in prose match ground truth; if omitted, an explicit authoritative reconciliation block is appended.

3. **Dynamic Operational Refusal:**
   - Real-time weather, live wind speeds, current hospital beds, and road closures trigger immediate `REFUSED` status.
   - Tested: `What is the live wind speed right now in Mangalore?` -> Immediate refusal pointing to IMD/DDMA; 0 tool executions.

4. **Multi-District Disambiguation:**
   - Homonymous communities across districts (e.g., Ullal in Uttara Kannada vs Ullal in Dakshina Kannada) return `AMBIGUOUS_ARGUMENT` and display a disambiguation table.
   - Explicitly qualified queries (e.g. `Ullal in Dakshina Kannada`) route cleanly to the intended unit.

5. **Rate Limiting & Security Headers:**
   - Nginx enforces `limit_req_zone` (30 req/s on `/api/`, 5 req/s on `/api/assistant/chat`).
   - Production security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy: strict-origin-when-cross-origin`) attached to all responses.
