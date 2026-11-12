# Deployment & Operational Guide: React + FastAPI Migration
**Platform:** Cyclone Preparedness Intelligence Platform  
**Target Architecture:** Decoupled React Frontend + FastAPI REST Backend  
**Date:** November 2026  

---

## 1. Architectural Architecture & Strategy

The migration decouples the presentation layer from the analytical core:
1. **Frontend**: Standalone Single-Page Application (SPA) built with React 18, TypeScript, Tailwind CSS, React Router, TanStack Query, Recharts, and Leaflet.
2. **Backend**: Lightweight, asynchronous FastAPI REST service (`api/main.py`) directly consuming the existing authoritative decision support engine (`src.assistant.decision_support.DecisionSupportService`), model surrogate, and deterministic tool orchestrator.
3. **Legacy Preservation**: The original Streamlit platform (`src/assistant/app.py`) and legacy API endpoints (`src/api/main.py`) remain 100% operational, fully preserved, and passing all regression suites.

---

## 2. Resource & Hosting Feasibility Analysis

### Memory & Compute Profile
| Component | Memory Consumption | Hosting Recommendation | Cost Tier |
| :--- | :--- | :--- | :--- |
| **React Frontend (SPA)** | 0 MB (Client-side execution) | Render Static Site / Cloudflare Pages / Vercel | Free ($0/mo) |
| **FastAPI REST Service** | ~120 MB - 160 MB RSS | Render Web Service (Docker / Python) | Free (512 MB tier) |
| **Streamlit (Legacy)** | ~380 MB - 450 MB RSS | Existing Render Container | Free (512 MB tier) |
| **CycloneLM (Local LLM)** | 8 GB - 16 GB VRAM | Dedicated GPU Worker (RunPod / vLLM / Vertex AI) | Paid GPU tier |

### Key Hosting Assessment: Render 512 MB Free Tier
- **Old Monolithic Streamlit**: Streamlit's in-memory session states, reactive re-runs, and Matplotlib figure caches frequently consumed 380–460 MB, risking Out-Of-Memory (OOM) eviction near 512 MB.
- **New Decoupled Architecture**: 
  - Hosting the compiled React static assets on a **Render Static Site** consumes **0 MB** of server memory.
  - The **FastAPI backend** maintains an immutable singleton of the 1,879 records in Pandas/Arrow and the 3 KB Logistic Regression pipeline, consuming only **~140 MB RAM**. This provides **>350 MB of safety headroom** on the Render free tier!
  - **CycloneLM Consideration**: A large language model (e.g., 7B parameter CycloneLM) **cannot** run in-process on Render's 512 MB free instance. It must be hosted on an external GPU inference service and called via HTTP using the backend's provider abstraction (`src/assistant/llm/client.py`).

---

## 3. Environment Variables Reference

### Backend (`api/main.py`)
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | Optional | `8000` | Port for Uvicorn server |
| `ALLOWED_ORIGINS` | Optional | `http://localhost:5173,http://localhost:3000` | Comma-separated CORS origins |
| `WANDB_MODEL_ARTIFACT` | Optional | `village-preparedness-classifier:v0` | Artifact identifier for W&B model |
| `WANDB_CACHE_DIR` | Optional | `.wandb_artifacts_cache` | Local directory for cached model artifacts |
| `LLM_PROVIDER` | Optional | `ollama` | Production conversational AI provider |
| `OLLAMA_BASE_URL` | Optional | `http://127.0.0.1:11434` | Endpoint for local Ollama daemon (Qwen 2.5) |
| `OLLAMA_MODEL` | Optional | `qwen2.5:0.5b-instruct` | Local model identifier |

### Frontend (`frontend/`)
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Optional | `/api` | Base path for FastAPI backend |

---

## 4. Build & Run Instructions

### 4.1 Running Locally

#### Step 1: Start FastAPI Backend
```bash
# Activate virtual environment
source .venv/bin/activate

# Run FastAPI backend with live reload
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: `http://localhost:8000/docs`

#### Step 2: Start React Frontend
```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start Vite development server
npm run dev
```
Frontend will be available at: `http://localhost:5173`

---

### 4.2 Production Build Commands

#### Frontend Production Build
```bash
cd frontend
npm run build
# Outputs optimized production assets to frontend/dist/
```

#### Production Verification / Preview
```bash
cd frontend
npm run preview
```

---

## 5. Health Checks & Monitoring

- **Container Readiness Check**: `GET http://localhost:8000/health`  
  Returns HTTP 200 with `{ "status": "healthy", "total_units": 1879, "version": "2.0.0" }`.
- **API Detailed Status Check**: `GET http://localhost:8000/api/health`  
  Returns full status including model version and district availability.
- **Performance Profiling**: Every API response includes the `X-Process-Time` HTTP header (typically `<0.0150s` for filtered queries).

---

## 6. Testing Verification Summary

- **Python Backend Regression**: `447 passing`, `1 skipped` (35 test suites).
- **React Frontend Unit & Page Tests**: `10 passing` in Vitest.
- **Production Build Status**: Clean Vite bundle (`dist/index.html`, `dist/assets/*.js`, `dist/assets/*.css`) with zero TypeScript errors.
