# VPS Deployment Guide (No Domain Yet)

This project is already Dockerized and now separated for production-minded VPS usage.

## Files Added/Updated

- `Dockerfile`: non-root runtime, healthcheck against `readyz`, cleaner defaults
- `docker-compose.yml`: production compose (app internal, Caddy public)
- `docker-compose.dev.yml`: local override (direct app port + reload)
- `Caddyfile`: IP-based reverse proxy defaults, security headers, no forced TLS
- `.env.example`: production-ready environment template
- `deploy.sh`: simple deploy/update helper for VPS
- `diabetes-api.service`: optional systemd unit for auto-start and managed restarts

## 1) VPS First-Time Setup

1. Install Docker Engine + Compose plugin.
2. Clone repo into VPS, e.g. `/opt/diabetesmealplanpredictionapi`.
3. Copy env template:

```bash
cp .env.example .env
```

4. Edit `.env` for production values.

## 2) Build and Run (Production)

```bash
docker compose -f docker-compose.yml up -d --build
```

API becomes available from VPS IP on port `80`:

```text
http://<VPS_PUBLIC_IP>/api/v1/healthz
```

## 3) Update / Redeploy

```bash
bash deploy.sh
```

Or manually:

```bash
docker compose -f docker-compose.yml up -d --build --remove-orphans
```

## 4) Daily Operations

Check status:

```bash
docker compose -f docker-compose.yml ps
```

Follow logs:

```bash
docker compose -f docker-compose.yml logs -f app
docker compose -f docker-compose.yml logs -f proxy
```

Restart services:

```bash
docker compose -f docker-compose.yml restart app
docker compose -f docker-compose.yml restart proxy
```

Health check:

```bash
curl http://<VPS_PUBLIC_IP>/api/v1/readyz
```

## 5) Optional systemd Integration

Install unit:

```bash
sudo cp diabetes-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now diabetes-api.service
```

Managed actions:

```bash
sudo systemctl status diabetes-api.service
sudo systemctl reload diabetes-api.service
sudo systemctl restart diabetes-api.service
```

## 6) Firewall (Recommended)

UFW baseline:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw enable
sudo ufw status
```

Do not open port `8000` publicly; the app is internal behind proxy.

## 7) Local Development Compose

Use dev override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

This exposes app directly at `http://localhost:8000` with reload enabled.

## 8) Later Upgrade When Domain Is Available

1. Set DNS A record to VPS IP.
2. Update `.env`:
   - `CADDY_SITE_ADDRESS=api.your-domain.com`
3. Enable HTTPS in `Caddyfile` by removing `auto_https off` (or deleting the global block).
4. If needed, expose `443:443` in `docker-compose.yml` for TLS traffic.
5. Redeploy: `docker compose -f docker-compose.yml up -d --build`.

After this, Caddy can manage certificates automatically.
