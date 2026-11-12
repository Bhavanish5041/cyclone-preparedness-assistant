#!/usr/bin/env bash
# ==============================================================================
# CYCLONE INTEL: Non-Sudo User Systemd Auto-Start Service Setup
# Enables runner to start automatically on system boot without requiring root
# ==============================================================================
set -euo pipefail

RUNNER_DIR="/home/bhavanish/actions-runner"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/cyclone-actions-runner.service"

mkdir -p "$SERVICE_DIR"

cat << EOF > "$SERVICE_FILE"
[Unit]
Description=GitHub Actions Self-Hosted Runner (CYCLONE INTEL)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=$RUNNER_DIR/run.sh
WorkingDirectory=$RUNNER_DIR
Restart=always
RestartSec=10s
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=1min
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF

echo "Reloading systemd user daemon..."
systemctl --user daemon-reload

echo "Enabling and starting cyclone-actions-runner.service..."
systemctl --user enable --now cyclone-actions-runner.service

echo "Enabling lingering for persistent background operation on boot..."
loginctl enable-linger "$USER" 2>/dev/null || true

echo "=========================================================="
echo "User systemd service installed and enabled:"
systemctl --user status cyclone-actions-runner.service --no-pager || true
echo "=========================================================="
