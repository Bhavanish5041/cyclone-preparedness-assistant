# 11. Troubleshooting & Live Demo Recovery Guide

This runbook outlines rapid triage procedures and recovery commands for common issues encountered during live demonstrations.

---

## 1. Issue: Ollama LLM Daemon is Unresponsive or Offline

### Symptoms:
- Assistant chat returns responses marked with `"is_fallback": true`.
- Terminal shows `ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 11434)`.

### Immediate Live Demo Action (No panic needed):
1. **Explain the Fallback Feature:** Inform the audience:  
   *"Notice that our deterministic fallback engine has activated seamlessly. The assistant continues to return exact AHP scores, priority bands, and advisories without crashing."*
2. **Restart the Ollama Daemon:**
```bash
# Check if Ollama is running
curl -s http://127.0.0.1:11434/api/tags || echo "Ollama offline"

# Restart the snap or native service
sudo systemctl restart snap.ollama.ollama.service || sudo systemctl restart ollama.service

# Verify Qwen model is loaded
curl -s http://127.0.0.1:11434/api/tags | jq .
```

---

## 2. Issue: Port 80, 8000, or 8501 Already in Use

### Symptoms:
- `docker compose up -d` fails with:  
  `Bind for 0.0.0.0:8000 failed: port is already allocated`.

### Recovery Steps:
```bash
# 1. Identify which process is holding the port (e.g. 8000)
sudo lsof -i :8000 || sudo ss -tulpn | grep :8000

# 2. Terminate rogue background python or uvicorn processes
kill $(lsof -t -i :8000) 2>/dev/null || true

# 3. Cleanly restart the docker stack
docker compose down --remove-orphans
docker compose up -d
```

---

## 3. Issue: Cloudflare Tunnel Connection Drop

### Symptoms:
- Public URL (`*.trycloudflare.com`) returns HTTP 502 or Connection Timed Out.
- Localhost works, but remote access fails.

### Recovery Steps:
```bash
# 1. Check if cloudflared host process is alive
pgrep -fl cloudflared

# 2. If dead, start a new temporary Quick Tunnel in the background
cloudflared tunnel --url http://127.0.0.1:80 > /tmp/cftunnel.log 2>&1 &

# 3. Retrieve the new public URL
sleep 3
grep -o 'https://[-a-z0-9.]*trycloudflare.com' /tmp/cftunnel.log | tail -1
```
*Note: If network access is lost, switch immediately to demonstrating on `http://127.0.0.1:80/`.*

---

## 4. Issue: Container Fails Health Check or Crash Loops

### Symptoms:
- `docker compose ps` shows `cyclone-api (unhealthy)` or `Restarting (1)`.

### Rapid Diagnosis & Fix:
```bash
# 1. Inspect recent container logs
docker logs cyclone-api --tail 50

# 2. Verify dataset LFS integrity (make sure datasets are not Git LFS pointer files)
head -c 4 data/processed/master_village_preparedness.parquet | od -c
# Expected output must contain "P A R 1" magic bytes.

# 3. If LFS pointers are corrupt, pull datasets via git-lfs
git lfs pull
docker compose restart api
```

---

## 5. Issue: Weights & Biases Authentication Warning

### Symptoms:
- Script prints `wandb: WARNING W&B online mode requested, but no active authentication found`.

### Explanation & Resolution:
CYCLONE INTEL's tracking module ([`src/tracking/wandb_logger.py`](file:///home/bhavanish/cyclone-preparedness-assistant/src/tracking/wandb_logger.py)) includes an automatic fallback mechanism:
- When unauthenticated, the tracking engine automatically switches from `online` to `offline` mode.
- Experiments are recorded locally into `./wandb/offline-run-*` without interrupting execution or crashing pipelines.
- To silence warnings explicitly, append `--wandb-mode offline` or `--mode offline` to commands.

---

## 6. Issue: Self-Hosted Runner Shows "Offline" on GitHub

### Symptoms:
- Manual CD dispatch stays queued with message: `Waiting for a runner to pick up this job...`.

### Recovery Steps:
```bash
# 1. Check systemd runner service status
systemctl status actions.runner.Bhavanish5041-cyclone-preparedness-assistant.cyclone-laptop-runner.service --no-pager

# 2. Restart the runner daemon
sudo systemctl restart actions.runner.Bhavanish5041-cyclone-preparedness-assistant.cyclone-laptop-runner.service

# 3. Verify Runner.Listener process is active
ps aux | grep -E "Runner.Listener" | grep -v grep
```
