"""
CAM Event Logger

Centralized event logging for the CAM system. Captures all significant
activity — task submissions, OATI phase transitions, agent connections,
model router calls, kill switch activations, and errors — in a structured,
timestamped format.

Events are stored in-memory in a ring buffer (default 1000 entries) and
can be exported to a JSON file for the audit trail.

CLAUDE.md: "Comprehensive logging — if it happened, there's a record"

Usage:
    from core.event_logger import EventLogger

    event_logger = EventLogger(max_events=1000)
    event_logger.info("task", "Task submitted", task_id="abc123")
    event_logger.warn("agent", "Heartbeat missed", agent_id="nova")
    event_logger.error("system", "Model router failed", error="connection refused")
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.events")


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

SEVERITY_INFO = "INFO"
SEVERITY_WARN = "WARN"
SEVERITY_ERROR = "ERROR"


@dataclass
class Event:
    """A single logged event in the CAM system.

    Attributes:
        timestamp:  When the event occurred (UTC).
        severity:   INFO, WARN, or ERROR.
        category:   Event category (task, agent, model, system, etc.).
        message:    Human-readable description of what happened.
        details:    Arbitrary key-value metadata (task_id, agent_id, etc.).
    """
    timestamp: datetime
    severity: str
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / dashboard broadcast."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Event Logger
# ---------------------------------------------------------------------------

class EventLogger:
    """Centralized event logger for the CAM system.

    Stores events in a bounded deque (ring buffer). When the buffer is
    full, oldest events are dropped. Events are also forwarded to
    Python's logging module so they appear in the server logs.

    An optional broadcast callback pushes new events to connected
    dashboard clients in real time.

    Args:
        max_events: Maximum number of events to keep in memory.
    """

    def __init__(self, max_events: int = 1000):
        self._events: deque[Event] = deque(maxlen=max_events)
        self._on_new_event = None  # async callback for dashboard broadcast
        logger.info("EventLogger initialized (max_events=%d)", max_events)

    # -------------------------------------------------------------------
    # Broadcast callback — set by server.py
    # -------------------------------------------------------------------

    def set_broadcast_callback(self, callback):
        """Set the async callback for broadcasting new events to dashboards.

        Args:
            callback: async function(event_dict) that sends to dashboards.
        """
        self._on_new_event = callback

    # -------------------------------------------------------------------
    # Logging methods
    # -------------------------------------------------------------------

    def info(self, category: str, message: str, **details):
        """Log an INFO-level event."""
        self._log(SEVERITY_INFO, category, message, details)

    def warn(self, category: str, message: str, **details):
        """Log a WARN-level event."""
        self._log(SEVERITY_WARN, category, message, details)

    def error(self, category: str, message: str, **details):
        """Log an ERROR-level event."""
        self._log(SEVERITY_ERROR, category, message, details)

    def _log(self, severity: str, category: str, message: str, details: dict):
        """Create an event, store it, log it, and broadcast it."""
        event = Event(
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            category=category,
            message=message,
            details=details,
        )
        self._events.append(event)

        # Forward to Python logging
        log_msg = f"[{category}] {message}"
        if details:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            log_msg += f" ({detail_str})"

        if severity == SEVERITY_ERROR:
            logger.error(log_msg)
        elif severity == SEVERITY_WARN:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Broadcast to dashboards (non-blocking, fire-and-forget)
        if self._on_new_event is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._safe_broadcast(event))
            except RuntimeError:
                pass  # No running event loop — skip broadcast

    async def _safe_broadcast(self, event: Event):
        """Broadcast an event to dashboards, swallowing errors."""
        try:
            await self._on_new_event(event.to_dict())
        except Exception:
            pass  # Never let broadcast errors disrupt logging

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------

    def get_recent(self, count: int = 100) -> list[dict]:
        """Return the most recent N events as dicts.

        Args:
            count: Maximum number of events to return.

        Returns:
            List of event dicts, newest last.
        """
        events = list(self._events)
        return [e.to_dict() for e in events[-count:]]

    def get_all(self) -> list[dict]:
        """Return all events in the buffer as dicts."""
        return [e.to_dict() for e in self._events]

    @property
    def count(self) -> int:
        """Number of events currently in the buffer."""
        return len(self._events)

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def export_json(self, filepath: str | Path) -> int:
        """Export all events to a JSON file.

        Args:
            filepath: Path to write the JSON file.

        Returns:
            Number of events exported.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        events = self.get_all()
        with open(filepath, "w") as f:
            json.dump({
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "event_count": len(events),
                "events": events,
            }, f, indent=2)

        logger.info("Exported %d events to %s", len(events), filepath)
        return len(events)
