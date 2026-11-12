# CYCLONE INTEL: Production Deployment & Operations Runbook

## 1. System Architecture & Topology

The production CYCLONE INTEL platform is self-hosted on a secured Linux environment and exposed to emergency responders via an outbound Cloudflare Quick Tunnel.

```
                              ┌─────────────────────────────────────────────────────┐
                              │                   Local Linux Host                  │
                              │                                                     │
[Stakeholders / Web Browser]  │  ┌─────────────────┐       ┌─────────────────────┐  │
              │               │  │  cyclone-       │       │    cyclone-api      │  │
              ▼               │  │  frontend       │ /api/ │   FastAPI:8000      │  │
   [Cloudflare Quick Tunnel] ───►│  Nginx:80       ├──────►│ (Decision Engine &  │  │
  (*.trycloudflare.com)       │  │ (SPA + Proxy)   │       │  Surrogate Model)   │  │
                              │  └─────────────────┘       └──────────┬──────────┘  │
                              │                                       │             │
                              │  ┌─────────────────┐                  ▼             │
                              │  │  cyclone-       │       ┌─────────────────────┐  │
                              │  │  prometheus     │       │    Local Ollama     │  │
                              │  │  Port 9090      │       │    Port 11434       │  │
                              │  └─────────────────┘       │ (Qwen2.5-0.5B LLM)  │  │
                              │                            └─────────────────────┘  │
                              └─────────────────────────────────────────────────────┘
```

### Security Boundaries
- **Inbound Access**: Standard inbound TCP ports (22, 80, 8000, 11434) are NOT opened to the public internet. All traffic enters through the Cloudflare Quick Tunnel.
- **Ollama Isolation**: Port 11434 is bound strictly to `127.0.0.1` and `host.docker.internal`. Nginx actively blocks any external requests to `/ollama`, `/api/generate`, `/api/tags`, or `/api/pull`.
- **API Documentation**: The interactive OpenAPI `/docs` and `/redoc` interfaces are configurable via `ENABLE_DOCS=true` and do not expose internal administrative tokens or endpoints.

---

## 2. Pre-Deployment Verification Checklist

Before initiating any production deployment or cutover:

- [ ] **CI Pipeline Passed**: Ensure all 10 stages in `.github/workflows/ci-cd.yml` have passed on the target commit.
- [ ] **Image Published to GHCR**: Ensure versioned Docker images exist:
  - `ghcr.io/<owner>/cyclone-assistant-api:sha-<commit_sha>`
  - `ghcr.io/<owner>/cyclone-assistant-frontend:sha-<commit_sha>`
- [ ] **Local Model Registry Valid**: Run `python scripts/model_registry_cli.py validate` on the host to verify artifact SHA256 checksums.
- [ ] **Model Quality Gates Verified**: Run `pytest tests/test_model_quality_gates.py` to confirm 100% invariance and $\ge 85\%$ concordance.
- [ ] **Ollama Service Running**: Verify Ollama daemon is active:
  ```bash
  curl -s http://127.0.0.1:11434/api/tags | grep -q "qwen2.5:0.5b-instruct" && echo "Ollama ready"
  ```
- [ ] **Backup Verified**: Check that rollback script `scripts/rollback_to_host.sh` is executable.

---

## 3. Self-Hosted Runner Registration & Auto-Start Configuration

The self-hosted runner executes on the Ubuntu laptop and automates production deployments while ensuring no external SSH/Docker access is opened.

### Step 1: Obtain GitHub Runner Token
1. Navigate to: `https://github.com/Bhavanish5041/cyclone-preparedness-assistant/settings/actions/runners/new`
2. Copy the registration token string (e.g., `BXXXXXXXXXXXXXXXXXXXX`).

### Step 2: Register Runner & Install Auto-Start Service
Execute the automated setup script on your laptop:

```bash
# Register runner with GitHub and install system service (auto-starts on reboot)
./scripts/setup_runner.sh <YOUR_GITHUB_RUNNER_TOKEN>
```

Alternatively, if you prefer user-level systemd (non-sudo):
```bash
cd /home/bhavanish/actions-runner
./config.sh --url https://github.com/Bhavanish5041/cyclone-preparedness-assistant --token <TOKEN> --name cyclone-laptop-runner --labels self-hosted,linux,x64,cyclone-runner --work _work --replace --unattended
cd /home/bhavanish/cyclone-preparedness-assistant
./scripts/install_runner_user_service.sh
```

---

## 4. How to Trigger an Approved Deployment

1. **Navigate to GitHub Actions**:
   Go to repository **Actions** tab -> **CD Pipeline - Controlled Production Deployment (Self-Hosted Runner)**.
2. **Click "Run workflow"**:
   - **Target environment**: `production` (or `simulation` for dry-run testing)
   - **Exact Docker image tag**: `sha-<short_sha>` (e.g. `sha-7d0e09a` or `latest`)
   - **Services to deploy**: `auto` (deploys only services that changed) or `api`, `frontend`, `dashboard`, `all`
   - **Explicit confirmation phrase**: Type `CONFIRM-DEPLOY`
   - **Public Cloudflare Tunnel URL**: (Optional) e.g. `https://your-tunnel.trycloudflare.com`
3. **Approve Production Environment**:
   - The workflow runs pre-deployment verification, then pauses at the **Production Release Gate**.
   - Designated maintainer clicks **Review deployments** -> selects **production** -> clicks **Approve and deploy**.
4. **Automated Cutover & Monitoring**:
   - The self-hosted runner on your laptop picks up the approved deployment job.
   - It captures the rollback baseline, pulls verified GHCR images, cuts over changed containers, probes healthchecks, logs results, and automatically rolls back if health checks fail.

---

## 5. Local Automated / Simulation Deployment Commands

To run an isolated deployment simulation or manual cutover directly on the laptop:

```bash
# Isolated simulation on test ports (8008, 8088) without touching production:
python3 scripts/deploy_production.py --simulate

# Simulate failure detection and automatic rollback:
python3 scripts/deploy_production.py --simulate --simulate-failure

# Production cutover with changed-service auto detection:
python3 scripts/deploy_production.py --image-tag latest --services auto

# Emergency manual rollback if needed:
./scripts/rollback_to_host.sh
```

### Representative Endpoint Verification
```bash
# 1. Health and readiness
curl -f http://127.0.0.1/health
curl -f http://127.0.0.1/ready

# 2. Command-center overview KPIs
curl -f http://127.0.0.1/api/dashboard/overview

# 3. Tallur community regression dossier
curl -f http://127.0.0.1/api/communities/11-29-569-05522-608691

# 4. Interactive OpenAPI Swagger UI
curl -f http://127.0.0.1/docs
```

---

## 4. Emergency Rollback Procedures

If smoke tests fail, memory limits are exceeded, or an application defect is identified:

### Option A: Automated Immediate Rollback to Previous Stable Container Images
```bash
# Set tag to previous known-good release
export PREVIOUS_TAG="sha-<previous_sha>"

docker tag ghcr.io/<owner>/cyclone-assistant-api:${PREVIOUS_TAG} cyclone-assistant-api:latest
docker tag ghcr.io/<owner>/cyclone-assistant-frontend:${PREVIOUS_TAG} cyclone-assistant-frontend:latest

docker compose up -d --no-deps api frontend
```

### Option B: Host Process Emergency Fallback
If Docker engine itself experiences failure, run the emergency host rollback script:
```bash
./scripts/rollback_to_host.sh
```
This terminates container processes and restores host-based systemd services for the API and frontend.

---

## 5. Cloudflare Quick Tunnel Operations

### Monitoring the Public URL
The tunnel runs via systemd or local daemon. To extract the current active public URL:
```bash
journalctl -u cloudflared -n 50 --no-pager | grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' | tail -n 1
```

### Testing Public Connectivity
```bash
PUBLIC_URL=$(journalctl -u cloudflared -n 50 --no-pager | grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' | tail -n 1)
curl -f "$PUBLIC_URL/health"
curl -f "$PUBLIC_URL/api/dashboard/overview"
```

---

## 6. Common Troubleshooting Scenarios

| Issue | Symptom | Remediation |
| :--- | :--- | :--- |
| **Qwen/Ollama Unavailable** | Assistant responses show `Fallback mode` chip in UI; latency $< 25\text{ ms}$. | Check Ollama daemon: `curl http://127.0.0.1:11434/api/tags`. If stopped, run `ollama serve`. Deterministic fallback continues serving grounded responses with 100% uptime. |
| **404 on API Routes** | Frontend shows network error or retry button. | Verify Nginx proxy rules in `deployment/nginx/nginx.container.conf`. Confirm `api` service DNS resolves within Docker network `cyclone-net`. |
| **Port Conflict on 80/8000** | Container fails to start with `bind: address already in use`. | Check existing host processes: `sudo lsof -i :80` or `sudo lsof -i :8000`. Kill duplicate uvicorn instances before starting Compose. |
| **Out-of-Memory (OOM)** | Container restarts unexpectedly; `docker inspect` shows exit code 137. | Increase container memory limit in `docker-compose.yml` (default: 1536M for API, 256M for frontend). |
