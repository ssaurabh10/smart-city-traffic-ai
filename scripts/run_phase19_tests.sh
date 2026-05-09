#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

echo "== Phase 19: Python system tests =="
"$PYTHON_BIN" -m unittest discover -s tests -p "test_*.py" -v

echo ""
echo "== Phase 19: Backend syntax checks =="
"$PYTHON_BIN" -m py_compile \
  ai-engine/bus_priority.py \
  ai-engine/emergency.py \
  ai-engine/emissions.py \
  ai-engine/multi_agent.py \
  backend/main.py \
  backend/streamer.py

echo ""
echo "== Phase 19: Dashboard rendering build =="
cd frontend
npm run build

echo ""
echo "Static test suite passed."
echo "For live <200ms latency, start the simulation and run:"
echo "  $PYTHON_BIN scripts/test_websocket_latency.py --target-ms 200"
