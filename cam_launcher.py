#!/usr/bin/env python3
"""CAM Launcher — unified entry point for all CAM subsystems.

Thin wrapper around server.py that adds:
- Pre-flight self-test (warn-only, never blocks startup)
- Data directory verification
- Boot time tracking
- Signal handling for graceful shutdown
- Logging setup

Run directly:
    python3 cam_launcher.py

Or via systemd:
    sudo systemctl start cam
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Record boot start time before any heavy imports
BOOT_START = time.monotonic()
BOOT_TIME = datetime.now(timezone.utc)

logger = logging.getLogger("cam.launcher")


def setup_logging():
    """Configure root logger for startup messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_preflight(config):
    """Quick pre-flight checks. Warn on failure, never block startup."""
    checks = []

    # Config sections present
    for section in ["dashboard", "models", "memory"]:
        ok = hasattr(config, section)
        checks.append(("config." + section, ok))

    # Data directories writable
    for d in ["data", "data/tasks", "data/memory", "data/logs"]:
        p = Path(d)
        ok = p.exists() and os.access(p, os.W_OK)
        checks.append((f"dir:{d}", ok))

    # SQLite databases exist (init happens in server.py, just check presence)
    for db in ["data/analytics.db", "data/memory/episodic.db"]:
        checks.append((f"db:{db}", Path(db).exists()))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    logger.info("Pre-flight: %d/%d checks passed", passed, total)
    for name, ok in checks:
        if not ok:
            logger.warning("Pre-flight FAILED: %s", name)


def ensure_data_dirs():
    """Create required data directories if they don't exist."""
    dirs = [
        "data",
        "data/tasks",
        "data/memory",
        "data/logs",
        "data/audio",
        "data/backups",
        "data/knowledge",
        "data/knowledge/inbox",
        "data/reports",
        "data/transfers",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Data directories verified")


def install_signal_handlers():
    """Install SIGTERM/SIGINT handlers for graceful shutdown."""
    def handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown", sig_name)
        # uvicorn catches these and triggers lifespan shutdown
        # We just log here; the actual cleanup is in server.py lifespan
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    logger.info("Signal handlers installed (SIGTERM, SIGINT)")


def main():
    """Entry point — runs the full boot sequence then starts uvicorn."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("CAM Launcher starting")
    logger.info("=" * 60)

    # Load and validate config
    from core.config import get_config
    config = get_config()
    logger.info("Config loaded")

    # Run pre-flight self-test (warn-only, never blocks startup)
    run_preflight(config)

    # Create required data directories
    ensure_data_dirs()

    # Set boot_time on the server module so dashboard can access it
    import interfaces.dashboard.server as server
    server.boot_time = BOOT_TIME.isoformat()
    server.boot_duration = round(time.monotonic() - BOOT_START, 2)

    # Install signal handlers for graceful shutdown
    install_signal_handlers()

    # Start uvicorn (blocks until shutdown)
    import uvicorn
    host = getattr(config.dashboard, "host", "0.0.0.0")
    port = getattr(config.dashboard, "port", 8080)
    logger.info("Starting uvicorn on %s:%d", host, port)
    uvicorn.run(
        "interfaces.dashboard.server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
