# 02. System Architecture & Data Flow

## 1. End-to-End System Architecture

CYCLONE INTEL employs a multi-tiered, microservices-oriented architecture designed for edge deployment, air-gapped resilience, and strict separation between automated cloud CI and human-authorized self-hosted deployment.

```mermaid
flowchart TD
    subgraph INGRESS["Ingress & Edge Security"]
        CLIENT["Browser Client / Incident Commander"] -->|"HTTPS Traffic"| CF_TUNNEL["Cloudflare Quick Tunnel (*.trycloudflare.com)\nOutbound Encrypted Tunnel (PID 715104)"]
        CF_TUNNEL -->|"Proxy to Localhost:80"| NGINX["Nginx Container (:80)\nReverse Proxy & Static Asset Server"]
    end

    subgraph DOCKER_STACK["Docker Compose Production Stack"]
        NGINX -->|"Static Routes /"| REACT["cyclone-frontend\nReact 18 TypeScript Single Page App"]
        NGINX -->|"/api/* Reverse Proxy"| FASTAPI["cyclone-api (:8000)\nFastAPI Application Server (Uvicorn)"]
        FASTAPI -->|"Metrics Export /metrics"| PROM["cyclone-prometheus (:9090)\nTime-Series Metrics Scraper"]
        STREAMLIT["cyclone-dashboard (:8501)\nStreamlit Decision Center (Internal)"]
    end

    subgraph ANALYTICAL_CORE["Analytical Core (In-Process inside FastAPI)"]
        FASTAPI --> DSS["DecisionSupportService\n(src/assistant/decision_support.py)"]
        DSS --> AHP_ENG["AHP Decision Matrix Engine\n(src/integration/decision_matrix.py)"]
        DSS --> LR_SURR["Multinomial Logistic Surrogate\n(models/logistic_regression_tier.joblib)"]
        DSS --> KM_ARCH["K-Means Archetype Engine\n(models/kmeans_k4_model.joblib)"]
    end

    subgraph LOCAL_HOST_SERVICES["Preserved Local Host Daemons (Exempt from Container Lifecycle)"]
        FASTAPI -.->|"HTTP POST http://127.0.0.1:11434"| OLLAMA["Ollama LLM Daemon (PID 2790)\nModel: Qwen 2.5:0.5B-instruct (Q4_K_M)\nRAM: 618 MB / CPU Only"]
        RUNNER["GitHub Actions Runner (PID 751811)\nDedicated Self-Hosted CD Daemon\nService: actions.runner...service"]
    end
```

---

## 2. Ingress & Security Boundary Matrix

To protect critical infrastructure, CYCLONE INTEL implements a **Zero Open Port** perimeter:

| Boundary Layer | Enforcing Mechanism | Security Guarantee |
| :--- | :--- | :--- |
| **Public Ingress** | Cloudflare Quick Tunnel (`cloudflared`) | **Zero open inbound firewall ports** (TCP 22, 80, 8000, 11434 closed to WAN). All incoming requests pass through Cloudflare edge proxying. |
| **Container Reverse Proxy** | Nginx (`deployment/nginx/nginx.container.conf`) | Enforces strict CORS headers, rate limiting, and internal path rewriting (`/api/` forwarded to internal Docker network `api:8000`). |
| **Docker Daemon Socket** | `/var/run/docker.sock` | Restricted to local root/docker group. No insecure TCP Docker socket exposed. |
| **LLM Inference Isolation** | Localhost Binding (`127.0.0.1:11434`) | Ollama daemon listens strictly on host loopback. Cannot be probed or accessed directly from external networks. |
| **CI/CD Execution Boundary** | Ephemeral GitHub Runners vs. Self-Hosted | **Untrusted Pull Request code NEVER executes on the self-hosted production host.** CI runs entirely in isolated GitHub cloud containers. |

---

## 3. Data Processing & Analytical Pipeline Flow

```mermaid
sequenceDiagram
    autonumber
    actor Commander as Incident Commander
    participant Web as React Web UI
    participant API as FastAPI Backend
    participant Tool as Deterministic Tool Registry
    participant Engine as Decision Support Engine
    participant LLM as Local Qwen 2.5 LLM

    Commander->>Web: Submits: "Why is Tallur classified as Band 4?"
    Web->>API: POST /api/assistant/chat {message, history}
    
    API->>LLM: Pass prompt to Qwen Intent Router
    alt LLM Available & Intent Detected
        LLM-->>API: Returns Tool Call: lookup_community(query="Tallur")
        API->>Tool: Execute Tool with validation
        Tool->>Engine: Search administrative records
        Engine-->>Tool: Found Tallur (PC11: 608691, Udupi, Score: 74.34, Tier: High)
        Tool-->>API: Bounded structured payload (JSON)
        API->>LLM: Pass tool result for synthesis
        LLM-->>API: Synthesizes explainable response
    else LLM Offline or Out-of-Scope Query
        API->>Engine: Deterministic Fallback Pipeline
        Engine-->>API: Assembles structured rule-based briefing
    end

    API->>API: audit_and_guard_answer() checks for factual consistency
    API-->>Web: JSON Response with AHP Score, Priority Band, and Advisories
    Web-->>Commander: Displays grounded response with interactive metric badges
```

---

## 4. Production Container Stack Specification

The runtime stack is defined in [`docker-compose.yml`](file:///home/bhavanish/cyclone-preparedness-assistant/docker-compose.yml):

| Service Name | Container Name | Image Repository | Base Port | Health Probe URL | Memory Limit |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **`frontend`** | `cyclone-frontend` | `ghcr.io/bhavanish5041/cyclone-assistant-frontend` | `:80` | `http://127.0.0.1:80/` | 256 MiB |
| **`api`** | `cyclone-api` | `ghcr.io/bhavanish5041/cyclone-assistant-api` | `:8000` | `http://127.0.0.1:8000/health` | 512 MiB |
| **`dashboard`** | `cyclone-dashboard` | `ghcr.io/bhavanish5041/cyclone-assistant-dashboard` | `:8501` | `http://127.0.0.1:8501/_stcore/health` | 512 MiB |
| **`prometheus`** | `cyclone-prometheus`| `prom/prometheus:v2.54.1` | `:9090` | `http://127.0.0.1:9090/-/healthy` | 256 MiB |

> [!IMPORTANT]
> **Preserved Host Processes:**  
> The **Ollama inference daemon** (`:11434`) and the **Cloudflare tunnel daemon** (`cloudflared`) run as native systemd/host processes. They are deliberately excluded from Docker Compose to prevent heavy model reloading overhead and tunnel connection resets during rolling service cutovers.
