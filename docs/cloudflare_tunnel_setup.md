# Public Internet Access via Cloudflare Tunnel (`cloudflared`)

## 1. Overview

**Cloudflare Tunnel** securely connects your self-hosted CYCLONE INTEL application to Cloudflare’s global edge network without:
- Opening inbound router ports
- Configuring public IP port forwarding
- Managing local SSL/TLS certificate renewals (Let's Encrypt / Certbot)
- Exposing the host laptop IP address to DDoS attacks

All inbound traffic is encrypted over an outbound QUIC / HTTP/2 connection established from the laptop to Cloudflare’s edge data centers (e.g. BLR/BOM edge nodes).

---

## 2. Cloudflare Binary & Service

The `cloudflared` binary is located at `/home/bhavanish/.local/bin/cloudflared`.

### Systemd User Service
The tunnel is orchestrated by systemd user unit `cyclone-tunnel.service`:

```ini
[Unit]
Description=CYCLONE INTEL Cloudflare Tunnel Service
After=network.target cyclone-api.service

[Service]
Type=simple
ExecStart=/home/bhavanish/.local/bin/cloudflared tunnel --url http://127.0.0.1:80
Restart=always
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

### Auto-Start at Boot (Without Login)
Because `loginctl enable-linger bhavanish` is enabled on Ubuntu:
- The user systemd manager initializes at system boot.
- `cyclone-tunnel.service` and `cyclone-api.service` automatically start on machine reboot without requiring interactive graphical or SSH login.

---

## 3. Finding the Active Public URL

When running a quick tunnel (`--url http://127.0.0.1:80`), Cloudflare assigns a randomized hostname on `*.trycloudflare.com`.

To retrieve the current active URL:

```bash
journalctl --user -u cyclone-tunnel.service -n 50 --no-pager | grep "trycloudflare.com"
```

Example active endpoint:
```
https://pediatric-gen-safari-makers.trycloudflare.com
```

---

## 4. Optional: Binding to a Permanent Custom Domain

If you have a domain managed by Cloudflare (e.g., `cycloneintel.org`):

1. **Login to Cloudflare:**
   ```bash
   /home/bhavanish/.local/bin/cloudflared tunnel login
   ```
2. **Create Named Tunnel:**
   ```bash
   /home/bhavanish/.local/bin/cloudflared tunnel create cyclone-prod
   ```
3. **Configure Routing:**
   Create `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: <TUNNEL_UUID>
   credentials-file: /home/bhavanish/.cloudflared/<TUNNEL_UUID>.json

   ingress:
     - hostname: app.yourdomain.com
       service: http://127.0.0.1:80
     - service: http_status:404
   ```
4. **Route DNS:**
   ```bash
   /home/bhavanish/.local/bin/cloudflared tunnel route dns cyclone-prod app.yourdomain.com
   ```
5. **Update Service:**
   Update `ExecStart` in `~/.config/systemd/user/cyclone-tunnel.service` to:
   ```ini
   ExecStart=/home/bhavanish/.local/bin/cloudflared tunnel run cyclone-prod
   ```
   Then reload and restart:
   ```bash
   systemctl --user daemon-reload && systemctl --user restart cyclone-tunnel.service
   ```
