#!/usr/bin/env bash
# ==============================================================================
# CYCLONE INTEL: Dedicated Self-Hosted GitHub Actions Runner Setup
# Configures the runner and registers it as an auto-starting system service
# ==============================================================================
set -euo pipefail

RUNNER_DIR="/home/bhavanish/actions-runner"
REPO_URL="https://github.com/Bhavanish5041/cyclone-preparedness-assistant"
RUNNER_NAME="cyclone-laptop-runner"
RUNNER_LABELS="self-hosted,linux,x64,cyclone-runner"

echo "=========================================================="
echo "CYCLONE INTEL: GitHub Actions Self-Hosted Runner Setup"
echo "Repository: $REPO_URL"
echo "Runner Directory: $RUNNER_DIR"
echo "Runner Name: $RUNNER_NAME"
echo "Labels: $RUNNER_LABELS"
echo "=========================================================="

if [ $# -lt 1 ]; then
    echo "ERROR: Missing runner registration token."
    echo ""
    echo "Usage:"
    echo "  $0 <GITHUB_RUNNER_REGISTRATION_TOKEN>"
    echo ""
    echo "To obtain a registration token:"
    echo "  1. Navigate to: $REPO_URL/settings/actions/runners/new"
    echo "  2. Copy the token provided in the registration command"
    echo "  3. Run: $0 <COPIED_TOKEN>"
    exit 1
fi

TOKEN="$1"

if [ ! -d "$RUNNER_DIR" ]; then
    echo "Creating runner directory: $RUNNER_DIR"
    mkdir -p "$RUNNER_DIR"
fi

cd "$RUNNER_DIR"

# Verify runner binaries exist
if [ ! -f "./config.sh" ]; then
    echo "Downloading runner binaries..."
    curl -o actions-runner-linux-x64-2.338.2.tar.gz -L https://github.com/actions/runner/releases/download/v2.338.2/actions-runner-linux-x64-2.338.2.tar.gz
    tar xzf ./actions-runner-linux-x64-2.338.2.tar.gz
fi

# Remove previous configuration if re-registering
if [ -f ".runner" ]; then
    echo "Removing previous runner configuration..."
    ./config.sh remove --token "$TOKEN" 2>/dev/null || rm -f .runner .credentials .credentials_rsaparams
fi

# Configure runner non-interactively
echo "Registering runner with GitHub..."
./config.sh \
    --url "$REPO_URL" \
    --token "$TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --work "_work" \
    --replace \
    --unattended

echo ""
echo "=========================================================="
echo "Runner successfully configured with GitHub!"
echo "Now configuring auto-start on system boot..."
echo "=========================================================="

# Service Installation (Auto-start on reboot)
if command -v sudo >/dev/null 2>&1; then
    echo "Installing systemd service via ./svc.sh (requires sudo)..."
    sudo ./svc.sh install bhavanish
    sudo ./svc.sh start
    echo "Runner systemd service status:"
    sudo ./svc.sh status
else
    echo "Configuring user-level systemd service (non-sudo)..."
    mkdir -p ~/.config/systemd/user/
    cat << 'EOF' > ~/.config/systemd/user/cyclone-actions-runner.service
[Unit]
Description=GitHub Actions Runner (CYCLONE INTEL)
After=network-online.target docker.service

[Service]
ExecStart=/home/bhavanish/actions-runner/run.sh
WorkingDirectory=/home/bhavanish/actions-runner
Restart=always
RestartSec=5s

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now cyclone-actions-runner.service
    loginctl enable-linger bhavanish || true
    echo "User systemd service active and persistent."
fi

echo "=========================================================="
echo "Runner setup complete. The runner will auto-start after reboot."
echo "=========================================================="
