#!/usr/bin/env bash
set -e

PORT=${1:-8765}
TUNNEL_PROVIDER=${2:-cloudflared}

echo "=========================================================="
echo "Starting Public Tunnel for Voice Assistant Gateway (Port: $PORT)"
echo "=========================================================="

if command -v cloudflared &> /dev/null && [ "$TUNNEL_PROVIDER" = "cloudflared" ]; then
    echo "[Tunnel] Launching Cloudflare Tunnel..."
    exec cloudflared tunnel --url "http://localhost:$PORT"
elif command -v ngrok &> /dev/null; then
    echo "[Tunnel] Launching ngrok Tunnel..."
    exec ngrok http "$PORT"
else
    echo "[ERROR] Neither 'cloudflared' nor 'ngrok' binary was found in PATH."
    echo "Please install cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) or ngrok."
    exit 1
fi
