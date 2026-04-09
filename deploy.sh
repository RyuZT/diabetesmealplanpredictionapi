#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/diabetesmealplanpredictionapi}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

cd "$APP_DIR"

echo "[deploy] Building and starting services..."
docker compose -f "$COMPOSE_FILE" up -d --build --remove-orphans

echo "[deploy] Current status:"
docker compose -f "$COMPOSE_FILE" ps
