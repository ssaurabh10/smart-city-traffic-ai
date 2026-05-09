#!/usr/bin/env bash
# scripts/launch_dashboard.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launches the Smart City Traffic AI dashboard in Chrome using the
# NVIDIA RTX 4050 instead of Intel UHD integrated graphics.
#
# Usage:
#   ./scripts/launch_dashboard.sh           # dashboard only
#   ./scripts/launch_dashboard.sh --backend # also start FastAPI backend
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_URL="http://localhost:5173"
BACKEND_URL="http://localhost:8000"

# ── Optional: start backend + frontend servers ────────────────────────────────
if [[ "$1" == "--backend" ]]; then
  echo "Starting FastAPI backend…"
  cd "$PROJECT_DIR"
  source venv/bin/activate
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
  BACKEND_PID=$!
  echo "Backend PID: $BACKEND_PID"

  echo "Starting Vite frontend…"
  cd "$PROJECT_DIR/frontend"
  npm run dev &
  FRONTEND_PID=$!
  echo "Frontend PID: $FRONTEND_PID"

  # Wait for servers to be ready
  echo "Waiting for servers…"
  sleep 3
fi

# ── Launch Chrome on RTX 4050 ────────────────────────────────────────────────
echo ""
echo "Launching Chrome on NVIDIA RTX 4050 → $FRONTEND_URL"
echo ""

CHROME_ARGS="--no-first-run --no-default-browser-check $FRONTEND_URL"

if [[ "$XDG_SESSION_TYPE" == "wayland" ]]; then
  # Wayland: use switcherooctl (sets correct NVIDIA env vars for Wayland)
  echo "Session: Wayland → using switcherooctl"
  switcherooctl launch google-chrome $CHROME_ARGS 2>/dev/null &
else
  # X11: use PRIME offload env vars
  echo "Session: X11 → using PRIME offload"
  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  DRI_PRIME=1 \
  google-chrome --ozone-platform=x11 --use-gl=angle --use-angle=gl \
    $CHROME_ARGS 2>/dev/null &
fi

CHROME_PID=$!
echo "Chrome PID: $CHROME_PID"
echo ""
echo "✓ To verify GPU: open chrome://gpu → check GL_RENDERER for 'RTX 4050'"

