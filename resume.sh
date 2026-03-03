#!/usr/bin/env bash
set -e

cd ~/agent/openclawd-infra

echo "[1/3] Starting services..."
docker compose up -d

echo "[2/3] Container status:"
docker compose ps

echo "[3/3] Health checks:"
curl -s http://localhost:8000/health | sed 's/^/API: /'
echo
curl -s http://localhost:6333/healthz && echo "Qdrant: healthz passed"
