"""
CAM Agent Health Monitor

Tracks per-agent health metrics: heartbeat latency, missed heartbeats,
task success/failure rates, and ping RTT. Provides green/yellow/red
health indicators for the dashboard.

Decoupled from server internals — receives events via method calls,
same pattern as the Orchestrator.

Usage:
    from core.health_monitor import HealthMonitor

    health_monitor = HealthMonitor(registry=registry, heartbeat_interval=30.0)
    health_monitor.on_agent_connected("firehorseclawd")
    health_monitor.on_heartbeat("firehorseclawd")
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.agent_registry import AgentRegistry


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.health")


# ---------------------------------------------------------------------------
# Per-agent health metrics
# ---------------------------------------------------------------------------

@dataclass
class AgentHealthMetrics:
    """Health metrics for a single agent.

    Updated by the HealthMonitor as events arrive (heartbeats, task
    completions, pings). The dashboard reads these to render health
    indicators on each agent card.

    Attributes:
        missed_heartbeats:   Consecutive missed heartbeats (0 = on time).
        last_heartbeat_at:   When the last heartbeat arrived.
        heartbeat_latency_ms: How late vs expected interval (delta from 30s).
        connected_since:     When the current session started.
        total_uptime_seconds: Accumulated uptime across reconnects.
        tasks_completed:     Lifetime completed task count.
        tasks_failed:        Lifetime failed task count.
        active_tasks:        Currently running tasks on this agent.
        last_ping_rtt_ms:    Most recent ping round-trip time.
    """
    missed_heartbeats: int = 0
    last_heartbeat_at: datetime | None = None
    heartbeat_latency_ms: float = 0.0
    connected_since: datetime | None = None
    total_uptime_seconds: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    active_tasks: int = 0
    last_ping_rtt_ms: float | None = None

    @property
    def health_level(self) -> str:
        """Green / yellow / red health indicator.

        - green: 0 missed heartbeats AND error rate < 25%
        - yellow: 1-2 missed OR error rate >= 25%
        - red: 3+ missed heartbeats
        """
        if self.missed_heartbeats >= 3:
            return "red"
        if self.missed_heartbeats >= 1:
            return "yellow"
        if self.success_rate < 75.0 and (self.tasks_completed + self.tasks_failed) > 0:
            return "yellow"
        return "green"

    @property
    def success_rate(self) -> float:
        """Task success percentage (0-100). Returns 100 if no tasks yet."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 100.0
        return (self.tasks_completed / total) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / dashboard broadcast."""
        return {
            "missed_heartbeats": self.missed_heartbeats,
            "last_heartbeat_at": (
                self.last_heartbeat_at.isoformat()
                if self.last_heartbeat_at else None
            ),
            "heartbeat_latency_ms": round(self.heartbeat_latency_ms, 1),
            "connected_since": (
                self.connected_since.isoformat()
                if self.connected_since else None
            ),
            "total_uptime_seconds": round(self.total_uptime_seconds, 1),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "active_tasks": self.active_tasks,
            "last_ping_rtt_ms": (
                round(self.last_ping_rtt_ms, 1)
                if self.last_ping_rtt_ms is not None else None
            ),
            "health_level": self.health_level,
            "success_rate": round(self.success_rate, 1),
        }


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Monitors agent health across the CAM network.

    Holds a reference to the AgentRegistry so it can mark agents offline
    when 3 consecutive heartbeats are missed. Otherwise self-contained.

    The dashboard server calls event methods (on_heartbeat, on_task_completed,
    etc.) as things happen, and reads health state via to_broadcast_dict().

    Args:
        registry:           The shared AgentRegistry instance.
        heartbeat_interval: Expected seconds between heartbeats (default 30).
    """

    def __init__(self, registry: AgentRegistry, heartbeat_interval: float = 30.0):
        self._registry = registry
        self._heartbeat_interval = heartbeat_interval

        # agent_id -> AgentHealthMetrics
        self._metrics: dict[str, AgentHealthMetrics] = {}

        # ping_id -> (agent_id, sent_timestamp)
        self._pending_pings: dict[str, tuple[str, float]] = {}

        logger.info(
            "HealthMonitor initialized (heartbeat_interval=%.0fs)",
            heartbeat_interval,
        )

    # -------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------

    def _ensure_metrics(self, agent_id: str) -> AgentHealthMetrics:
        """Get or create metrics for an agent."""
        if agent_id not in self._metrics:
            self._metrics[agent_id] = AgentHealthMetrics()
        return self._metrics[agent_id]

    # -------------------------------------------------------------------
    # Event receivers — called by server.py
    # -------------------------------------------------------------------

    def on_agent_connected(self, agent_id: str):
        """Record that an agent connected. Resets missed heartbeat count."""
        metrics = self._ensure_metrics(agent_id)
        metrics.connected_since = datetime.now(timezone.utc)
        metrics.missed_heartbeats = 0
        logger.info("Health tracking started for agent '%s'", agent_id)

    def on_agent_disconnected(self, agent_id: str):
        """Record that an agent disconnected. Accumulates uptime."""
        metrics = self._metrics.get(agent_id)
        if metrics and metrics.connected_since:
            elapsed = (
                datetime.now(timezone.utc) - metrics.connected_since
            ).total_seconds()
            metrics.total_uptime_seconds += elapsed
            metrics.connected_since = None
            logger.info(
                "Agent '%s' disconnected — session uptime %.0fs, total %.0fs",
                agent_id, elapsed, metrics.total_uptime_seconds,
            )

    def on_heartbeat(self, agent_id: str):
        """Record a heartbeat arrival. Computes latency and resets miss count."""
        metrics = self._ensure_metrics(agent_id)
        now = datetime.now(timezone.utc)

        # Compute latency: how late was this heartbeat vs expected interval
        if metrics.last_heartbeat_at:
            actual_delta = (now - metrics.last_heartbeat_at).total_seconds()
            latency_ms = (actual_delta - self._heartbeat_interval) * 1000.0
            # Clamp negative latency to 0 — early is fine
            metrics.heartbeat_latency_ms = max(0.0, latency_ms)
        else:
            metrics.heartbeat_latency_ms = 0.0

        metrics.last_heartbeat_at = now
        metrics.missed_heartbeats = 0

    def on_task_dispatched(self, agent_id: str):
        """Record that a task was dispatched to an agent."""
        metrics = self._ensure_metrics(agent_id)
        metrics.active_tasks += 1

    def on_task_completed(self, agent_id: str):
        """Record a successful task completion."""
        metrics = self._ensure_metrics(agent_id)
        metrics.tasks_completed += 1
        metrics.active_tasks = max(0, metrics.active_tasks - 1)

    def on_task_failed(self, agent_id: str):
        """Record a task failure (timeout or exception)."""
        metrics = self._ensure_metrics(agent_id)
        metrics.tasks_failed += 1
        metrics.active_tasks = max(0, metrics.active_tasks - 1)

    # -------------------------------------------------------------------
    # Core logic — heartbeat checking
    # -------------------------------------------------------------------

    def check_heartbeats(self) -> list[str]:
        """Check all online agents for missed heartbeats.

        For each online agent, computes how many heartbeat intervals
        have elapsed since the last heartbeat. If >= 3, marks the agent
        offline in the registry and logs a warning.

        Returns:
            List of agent_ids that were newly marked offline.
        """
        now = datetime.now(timezone.utc)
        newly_offline = []

        for agent in self._registry.list_online():
            agent_id = agent.agent_id
            metrics = self._ensure_metrics(agent_id)

            if not metrics.last_heartbeat_at:
                continue

            elapsed = (now - metrics.last_heartbeat_at).total_seconds()
            missed = int(elapsed / self._heartbeat_interval)
            metrics.missed_heartbeats = missed

            if missed >= 3:
                self._registry.deregister(agent_id)
                newly_offline.append(agent_id)
                logger.warning(
                    "Agent '%s' missed %d heartbeats (%.0fs elapsed) — marked offline",
                    agent_id, missed, elapsed,
                )

        return newly_offline

    # -------------------------------------------------------------------
    # Ping / Pong — manual RTT measurement
    # -------------------------------------------------------------------

    def create_ping(self, agent_id: str) -> dict:
        """Create a ping message to send to an agent.

        Stores the send timestamp so we can compute RTT when the
        pong comes back.

        Args:
            agent_id: Which agent we're pinging.

        Returns:
            The ping message dict to send via WebSocket.
        """
        import time

        ping_id = str(uuid.uuid4())
        self._pending_pings[ping_id] = (agent_id, time.monotonic())

        return {
            "type": "ping",
            "ping_id": ping_id,
        }

    def resolve_pong(self, ping_id: str) -> float | None:
        """Handle a pong response, computing RTT.

        Args:
            ping_id: The ping_id from the pong message.

        Returns:
            RTT in milliseconds, or None if the ping_id wasn't found.
        """
        import time

        entry = self._pending_pings.pop(ping_id, None)
        if entry is None:
            return None

        agent_id, sent_time = entry
        rtt_ms = (time.monotonic() - sent_time) * 1000.0

        metrics = self._ensure_metrics(agent_id)
        metrics.last_ping_rtt_ms = rtt_ms

        logger.info(
            "Ping RTT for agent '%s': %.1fms",
            agent_id, rtt_ms,
        )
        return rtt_ms

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_broadcast_dict(self) -> dict[str, dict]:
        """All agents' health metrics as {agent_id: metrics_dict}.

        Used for broadcasting to all dashboard clients.
        """
        return {
            agent_id: metrics.to_dict()
            for agent_id, metrics in self._metrics.items()
        }

    def get_agent_health(self, agent_id: str) -> dict | None:
        """Get health metrics for a single agent.

        Returns:
            The metrics dict, or None if no metrics exist for this agent.
        """
        metrics = self._metrics.get(agent_id)
        if metrics is None:
            return None
        return metrics.to_dict()
