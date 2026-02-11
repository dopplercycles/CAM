"""
CAM Notification Manager

Evaluates events from the EventLogger against configurable rules and pushes
user-facing alerts to the dashboard as toast popups with a bell icon showing
unread count.

Rules are gated by config toggles in [notifications] — changes to settings.toml
take effect immediately (config is read at evaluation time, not cached).

Usage:
    from core.notifications import NotificationManager

    nm = NotificationManager()
    nm.set_broadcast_callback(broadcast_notification)
    nm.evaluate_event(event_dict)
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from core.config import get_config

logger = logging.getLogger("cam.notifications")


# ---------------------------------------------------------------------------
# Notification model
# ---------------------------------------------------------------------------

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"


@dataclass
class Notification:
    """A single user-facing notification.

    Attributes:
        id:         Unique identifier (UUID4 hex, first 12 chars).
        timestamp:  When the notification was created (UTC).
        severity:   info, warn, error, or critical.
        title:      Short headline.
        message:    Longer description of what happened.
        category:   Source category (agent, task, system, cost).
        read:       Whether the user has dismissed/read this notification.
    """
    id: str
    timestamp: datetime
    severity: str
    title: str
    message: str
    category: str
    read: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / dashboard broadcast."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "category": self.category,
            "read": self.read,
        }


# ---------------------------------------------------------------------------
# Notification Manager
# ---------------------------------------------------------------------------

class NotificationManager:
    """Evaluates events against notification rules and broadcasts alerts.

    Rules read config at evaluation time so changes to settings.toml take
    effect immediately after a config reload — no restart needed.

    Args:
        max_history: Maximum notifications to keep (overridden by config on
                     first evaluate_event call, but used as initial capacity).
    """

    def __init__(self, max_history: int = 200):
        self._notifications: deque[Notification] = deque(maxlen=max_history)
        self._on_notification: Callable[..., Coroutine] | None = None

        # High error rate tracking: rolling window of task outcomes (True=ok, False=fail)
        self._task_outcomes: deque[bool] = deque(maxlen=100)
        self._last_error_rate_alert: float = 0.0  # monotonic timestamp

        # Cost threshold tracking: accumulated model cost
        self._total_cost_usd: float = 0.0
        self._cost_alert_fired: bool = False

        logger.info("NotificationManager initialized (max_history=%d)", max_history)

    # -------------------------------------------------------------------
    # Broadcast callback — set by server.py
    # -------------------------------------------------------------------

    def set_broadcast_callback(self, callback):
        """Set the async callback for broadcasting notifications to dashboards.

        Args:
            callback: async function(notification_dict) called for each new notification.
        """
        self._on_notification = callback

    # -------------------------------------------------------------------
    # Event evaluation — runs all rule checks
    # -------------------------------------------------------------------

    def evaluate_event(self, event_dict: dict):
        """Evaluate an event against all notification rules.

        Reads config each time so toggles are hot-reloadable.

        Args:
            event_dict: Serialized event from EventLogger (has severity,
                        category, message, details keys).
        """
        config = get_config()
        notif_cfg = config.notifications

        if not notif_cfg.enabled:
            return

        # Update the ring buffer size if config changed
        configured_max = notif_cfg.max_history
        if self._notifications.maxlen != configured_max:
            old = list(self._notifications)
            self._notifications = deque(old, maxlen=configured_max)

        # Run each rule check
        self._check_agent_disconnect(event_dict, notif_cfg)
        self._check_task_failure(event_dict, notif_cfg)
        self._check_kill_switch(event_dict, notif_cfg)
        self._check_high_error_rate(event_dict, notif_cfg)
        self._check_cost_threshold(event_dict, notif_cfg)

    # -------------------------------------------------------------------
    # Rule checks
    # -------------------------------------------------------------------

    def _check_agent_disconnect(self, event: dict, cfg):
        """Alert when an agent disconnects or goes offline."""
        if not cfg.agent_disconnect:
            return
        if event.get("category") != "agent":
            return
        msg = (event.get("message") or "").lower()
        if "disconnected" in msg or "went offline" in msg:
            agent_id = event.get("details", {}).get("agent_id", "unknown")
            self._emit(
                SEVERITY_ERROR,
                "Agent Disconnected",
                f"Agent '{agent_id}' has gone offline.",
                "agent",
            )

    def _check_task_failure(self, event: dict, cfg):
        """Alert when a task fails."""
        if not cfg.task_failure:
            return
        if event.get("category") != "task":
            return
        severity = (event.get("severity") or "").upper()
        msg = (event.get("message") or "").upper()
        if severity == "ERROR" and "FAILED" in msg:
            task_id = event.get("details", {}).get("task_id", "")
            self._emit(
                SEVERITY_ERROR,
                "Task Failed",
                f"Task {task_id} has failed." if task_id else "A task has failed.",
                "task",
            )

    def _check_kill_switch(self, event: dict, cfg):
        """Alert when the kill switch is activated."""
        if not cfg.kill_switch:
            return
        if event.get("category") != "system":
            return
        msg = (event.get("message") or "").upper()
        if "KILL SWITCH" in msg:
            self._emit(
                SEVERITY_CRITICAL,
                "Kill Switch Activated",
                "All agents have been halted. The kill switch was activated.",
                "system",
            )

    def _check_high_error_rate(self, event: dict, cfg):
        """Alert when the task error rate exceeds the configured threshold.

        Tracks a rolling window of task outcomes and rate-limits alerts
        to at most one every 5 minutes.
        """
        if not cfg.high_error_rate:
            return
        if event.get("category") != "task":
            return

        msg = (event.get("message") or "").upper()
        severity = (event.get("severity") or "").upper()

        # Track task outcomes
        if "COMPLETED" in msg and severity != "ERROR":
            self._task_outcomes.append(True)
        elif "FAILED" in msg and severity == "ERROR":
            self._task_outcomes.append(False)
        else:
            return  # Not a terminal task event

        window = int(cfg.error_rate_window)
        if len(self._task_outcomes) < window:
            return  # Not enough data yet

        recent = list(self._task_outcomes)[-window:]
        fail_count = recent.count(False)
        error_rate = (fail_count / window) * 100

        if error_rate < cfg.error_rate_threshold:
            return

        # Rate limit: 1 alert per 5 minutes
        now = time.monotonic()
        if now - self._last_error_rate_alert < 300:
            return

        self._last_error_rate_alert = now
        self._emit(
            SEVERITY_WARN,
            "High Error Rate",
            f"Task error rate is {error_rate:.0f}% "
            f"({fail_count}/{window} recent tasks failed).",
            "task",
        )

    def _check_cost_threshold(self, event: dict, cfg):
        """Alert when cumulative model cost exceeds the configured threshold.

        Fires once — resets if the threshold is raised in config.
        """
        if not cfg.cost_threshold:
            return
        if event.get("category") != "model":
            return

        cost = event.get("details", {}).get("cost_usd")
        if cost is None:
            return

        self._total_cost_usd += float(cost)
        threshold = float(cfg.cost_threshold_usd)

        # Allow re-alerting if threshold was raised past current total
        if self._total_cost_usd < threshold:
            self._cost_alert_fired = False
            return

        if self._cost_alert_fired:
            return

        self._cost_alert_fired = True
        self._emit(
            SEVERITY_WARN,
            "Cost Threshold Reached",
            f"Cumulative model cost has reached ${self._total_cost_usd:.4f} "
            f"(threshold: ${threshold:.2f}).",
            "cost",
        )

    # -------------------------------------------------------------------
    # Emit — create notification and broadcast
    # -------------------------------------------------------------------

    def _emit(self, severity: str, title: str, message: str, category: str):
        """Create a notification, store it, and fire the broadcast callback."""
        notif = Notification(
            id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            title=title,
            message=message,
            category=category,
        )
        self._notifications.append(notif)
        logger.info(
            "Notification [%s] %s: %s", severity.upper(), title, message,
        )

        if self._on_notification is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._safe_broadcast(notif))
            except RuntimeError:
                pass  # No running event loop — skip broadcast

    async def _safe_broadcast(self, notif: Notification):
        """Broadcast a notification to dashboards, swallowing errors."""
        try:
            await self._on_notification(notif.to_dict())
        except Exception:
            pass  # Never let broadcast errors disrupt the notification system

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------

    def get_recent(self, count: int = 50) -> list[dict]:
        """Return the most recent N notifications as dicts, newest first.

        Args:
            count: Maximum number of notifications to return.

        Returns:
            List of notification dicts, newest first.
        """
        notifications = list(self._notifications)
        return [n.to_dict() for n in reversed(notifications[-count:])]

    def get_unread_count(self) -> int:
        """Return the number of unread notifications."""
        return sum(1 for n in self._notifications if not n.read)

    # -------------------------------------------------------------------
    # Dismiss
    # -------------------------------------------------------------------

    def dismiss(self, notification_id: str) -> bool:
        """Mark a notification as read.

        Args:
            notification_id: The notification ID to dismiss.

        Returns:
            True if found and marked, False if not found.
        """
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    def dismiss_all(self):
        """Mark all notifications as read."""
        for n in self._notifications:
            n.read = True
