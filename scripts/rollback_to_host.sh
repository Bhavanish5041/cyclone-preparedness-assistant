#!/usr/bin/env bash
# ==============================================================================
# CYCLONE INTEL: Emergency Rollback Script
# Reverts from Containerized Docker Stack to Host-Based Production Services
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo "Initiating Emergency Rollback to Host-Based Deployment..."
echo "=========================================================="

cd "$PROJECT_ROOT"

# Step 1: Shut down Docker Compose services
echo "[1/4] Stopping Docker Compose stack..."
docker compose down || true

# Step 2: Re-enable and start host FastAPI systemd service
echo "[2/4] Restoring host FastAPI service (cyclone-api.service)..."
systemctl --user enable cyclone-api.service
systemctl --user start cyclone-api.service

# Step 3: Start host Nginx container
echo "[3/4] Restoring host Nginx container (cyclone-nginx)..."
if docker ps -a --format '{{.Names}}' | grep -q "^cyclone-nginx$"; then
    docker start cyclone-nginx
else
    docker run -d \
        --name cyclone-nginx \
        --restart always \
        --net=host \
        -v "$PROJECT_ROOT/deployment/nginx/nginx.conf":/etc/nginx/nginx.conf:ro \
        -v "$PROJECT_ROOT/frontend/dist":/usr/share/nginx/html:ro \
        nginx:alpine
fi

# Step 4: Restart Cloudflare Tunnel
echo "[4/4] Restarting Cloudflare Tunnel..."
systemctl --user restart cyclone-tunnel.service

# Verification
echo "Waiting 5 seconds for host services to stabilize..."
sleep 5

echo "Verifying host health endpoint..."
if curl -s -f http://127.0.0.1/health > /dev/null; then
    echo "=========================================================="
    echo "ROLLBACK COMPLETE: Host services operational at http://127.0.0.1"
    echo "=========================================================="
    exit 0
else
    echo "=========================================================="
    echo "WARNING: Health check failed after rollback. Inspect:"
    echo "  systemctl --user status cyclone-api.service"
    echo "  docker logs cyclone-nginx"
    echo "=========================================================="
    exit 1
fi
