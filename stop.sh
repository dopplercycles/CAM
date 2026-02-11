#!/usr/bin/env bash
# Stop the CAM Dashboard
set -euo pipefail

CAM_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$CAM_DIR/data/logs/dashboard.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Dashboard may not be running in background mode."
    echo "Checking for running dashboard process..."
    PIDS=$(pgrep -f "interfaces.dashboard.server" || true)
    if [ -n "$PIDS" ]; then
        echo "Found dashboard process(es): $PIDS"
        kill $PIDS
        echo "Stopped."
    else
        echo "No dashboard process found."
    fi
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    rm -f "$PID_FILE"
    echo "Dashboard stopped (PID $PID)"
else
    rm -f "$PID_FILE"
    echo "Dashboard was not running (stale PID $PID)"
fi
