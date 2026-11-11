# CYCLONE INTEL: Production Server Operations & Maintenance Guide

## 1. Quick Reference: Service Architecture & Commands

The production application runs as a containerized Docker Compose stack behind a Cloudflare Quick Tunnel, with local LLM inference mediated by an internal non-root systemd bridge proxy.

| Component | Management Command | Status Command | Logs Command |
| :--- | :--- | :--- | :--- |
| **Docker Stack (All Services)** | `docker compose restart` | `docker compose ps` | `docker compose logs -f` |
| **FastAPI Backend Container** | `docker compose restart api` | `docker ps -f name=cyclone-api` | `docker logs -f cyclone-api` |
| **Frontend Nginx Container** | `docker compose restart frontend`| `docker ps -f name=cyclone-frontend`| `docker logs -f cyclone-frontend`|
| **Ollama Bridge Proxy (Host)** | `systemctl --user restart cyclone-ollama-bridge` | `systemctl --user status cyclone-ollama-bridge` | `journalctl --user -u cyclone-ollama-bridge -f` |
| **Cloudflare Tunnel (Host)** | `systemctl --user restart cyclone-tunnel` | `systemctl --user status cyclone-tunnel` | `journalctl --user -u cyclone-tunnel -f` |
| **Host Ollama Daemon** | `sudo systemctl restart snap.ollama.ollama.service` | `systemctl status snap.ollama.ollama.service` | `journalctl -u snap.ollama.ollama.service -f` |

---

## 2. Real-Time Operational Monitoring

### Comprehensive Production Health Dashboard
```bash
# Display live container statuses, CPU/memory, latencies, and service states:
.venv/bin/python scripts/ops_monitor.py

# Output raw JSON metrics:
.venv/bin/python scripts/ops_monitor.py --json
```

### Complete End-to-End Production Verification (14/14 Gates)
```bash
# Validates live public tunnel, real Qwen inference, AHP invariance, and security isolation:
.venv/bin/python scripts/e2e_production_test.py
```

---

## 3. Service Lifecycle Management

### Starting the Production Stack
```bash
# 1. Ensure Host Ollama is active
systemctl status snap.ollama.ollama.service

# 2. Ensure internal Ollama bridge proxy is running
systemctl --user start cyclone-ollama-bridge.service

# 3. Start the Docker Compose stack (detached)
docker compose up -d

# 4. Start Cloudflare Tunnel
systemctl --user start cyclone-tunnel.service
```

### Graceful Shutdown
```bash
systemctl --user stop cyclone-tunnel.service
docker compose down
systemctl --user stop cyclone-ollama-bridge.service
```

---

## 4. Emergency Rollback to Host-Based Services

If container orchestration or Docker networking encounters an unresolvable issue, execute the tested one-command rollback:

```bash
bash scripts/rollback_to_host.sh
```

**What this does automatically:**
1. Shuts down the Docker Compose stack (`docker compose down`).
2. Re-enables and starts host `cyclone-api.service`.
3. Starts host `cyclone-nginx` container on port 80.
4. Restarts `cyclone-tunnel.service`.
5. Verifies `http://127.0.0.1/health`.

---

## 5. Rebuilding Frontend or Updating Code

### Updating Frontend Assets
```bash
cd frontend
npm ci
npm test -- --run
npm run build

# Rebuild and restart the container:
cd ..
docker compose build frontend
docker compose up -d frontend
```

### Updating Backend Python API
```bash
# Rebuild and restart the API container:
docker compose build api
docker compose up -d api
```

---

## 6. Cloudflare Tunnel Configuration & Public URL

The tunnel service is configured in `~/.config/systemd/user/cyclone-tunnel.service`:
```ini
[Unit]
Description=CYCLONE INTEL Cloudflare Tunnel Service
After=network.target docker.service

[Service]
Type=simple
ExecStart=/home/bhavanish/.local/bin/cloudflared tunnel --protocol http2 --edge-ip-version 4 --url http://127.0.0.1:80
Restart=always
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

> [!NOTE]
> The flags `--protocol http2 --edge-ip-version 4` guarantee connection stability by preventing QUIC timeouts on local IPv6-disabled interfaces.

To retrieve the current public URL at any time:
```bash
journalctl --user -u cyclone-tunnel.service -n 50 --no-pager | grep "trycloudflare.com" | awk '{print $NF}' | tail -n 1
```

---

## 7. Security Isolation Verification

Ollama and internal LLM management endpoints are strictly isolated from the public network:
- Port 11434 is bound only to `127.0.0.1` and internal bridge `172.18.0.1`.
- Nginx blocks `/ollama`, `/api/generate`, `/api/tags` with HTTP 404.
- Verify from outside:
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" https://<PUBLIC_URL>/ollama          # Returns 404
  curl -s -o /dev/null -w "%{http_code}\n" https://<PUBLIC_URL>/api/generate    # Returns 404
  curl -s -o /dev/null -w "%{http_code}\n" https://<PUBLIC_URL>/api/tags        # Returns 404
  ```
