#!/usr/bin/env bash
# Start the CAM Dashboard
# Usage: ./start.sh [--bg]
#   --bg  Run in background (logs to data/logs/dashboard.log)

set -euo pipefail

CAM_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$CAM_DIR/.venv/bin/python"
LOG_DIR="$CAM_DIR/data/logs"

if [ ! -f "$VENV" ]; then
    echo "Error: Virtual environment not found at $CAM_DIR/.venv"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

mkdir -p "$LOG_DIR"

cd "$CAM_DIR"

if [ "${1:-}" = "--bg" ]; then
    echo "Starting CAM Dashboard in background..."
    nohup "$VENV" -m interfaces.dashboard.server > "$LOG_DIR/dashboard.log" 2>&1 &
    PID=$!
    echo "$PID" > "$LOG_DIR/dashboard.pid"
    echo "Dashboard running (PID $PID)"
    echo "Logs: $LOG_DIR/dashboard.log"
    echo "Stop: ./stop.sh"
else
    echo "Starting CAM Dashboard..."
    exec "$VENV" -m interfaces.dashboard.server
fi
