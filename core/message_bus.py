"""
CAM Inter-Agent Message Bus

Lightweight pub/sub message bus for agent-to-agent communication.
Agents publish messages to named channels, and subscribers (other agents,
the orchestrator, the dashboard) receive them in real time.

No external dependencies — stdlib only.

Usage:
    from core.message_bus import MessageBus

    bus = MessageBus()
    bus.subscribe("research_results", my_callback)
    await bus.publish("research_agent", "research_results", {"finding": "..."})
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.message_bus")


# ---------------------------------------------------------------------------
# BusMessage — a single message on the bus
# ---------------------------------------------------------------------------

@dataclass
class BusMessage:
    """A single message published to the bus.

    Attributes:
        message_id: Short unique identifier (uuid4 hex[:12]).
        sender:     Agent ID or "system".
        channel:    Which channel this was published to.
        payload:    Arbitrary data dict.
        timestamp:  When the message was published (UTC ISO string).
    """
    message_id: str
    sender: str
    channel: str
    payload: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / dashboard broadcast."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "channel": self.channel,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# MessageBus — pub/sub with per-channel ring buffers
# ---------------------------------------------------------------------------

# Predefined channels — agents publish to these, subscribers listen
CHANNELS = [
    "task_updates",        # Task lifecycle: created, dispatched, completed, failed
    "content_pipeline",    # Content agent: script drafts, TTS jobs, publish events
    "research_results",    # Research agent: findings, summaries, scout alerts
    "alerts",              # High-priority notifications across agents
    "system",              # System-level: startup, shutdown, config changes
    "robot_events",        # ROS 2 bridge: robot status, navigation, e-stop events
]


class MessageBus:
    """Pub/sub message bus for inter-agent communication.

    Each channel has its own ring buffer (bounded deque). Subscribers
    are notified asynchronously via fire-and-forget tasks.

    Args:
        max_messages: Max messages to retain per channel.
        event_logger: Optional EventLogger instance for audit trail.
    """

    def __init__(self, max_messages: int = 500, event_logger=None):
        self._max_messages = max_messages
        self._event_logger = event_logger

        # Per-channel message ring buffers
        self._buffers: dict[str, deque[BusMessage]] = {
            ch: deque(maxlen=max_messages) for ch in CHANNELS
        }

        # Per-channel subscriber lists (async callables)
        self._subscribers: dict[str, list[Callable]] = {
            ch: [] for ch in CHANNELS
        }

        # Global subscribers (receive ALL channel messages — orchestrator use)
        self._global_subscribers: list[Callable] = []

        # Stats tracking
        self._stats: dict[str, int] = {ch: 0 for ch in CHANNELS}
        self._sender_stats: dict[str, int] = {}
        self._sender_channel_counts: dict[str, dict[str, int]] = {}

        # Dashboard broadcast callback (set by server.py)
        self._on_broadcast: Callable | None = None

        logger.info(
            "MessageBus initialized (%d channels, max_messages=%d)",
            len(CHANNELS), max_messages,
        )

    # -------------------------------------------------------------------
    # Broadcast callback — set by server.py
    # -------------------------------------------------------------------

    def set_broadcast_callback(self, callback: Callable):
        """Set the async callback for broadcasting bus messages to dashboards.

        Args:
            callback: async function(message_dict) that sends to dashboards.
        """
        self._on_broadcast = callback

    # -------------------------------------------------------------------
    # Subscribe
    # -------------------------------------------------------------------

    def subscribe(self, channel: str, callback: Callable) -> bool:
        """Subscribe to messages on a specific channel.

        Args:
            channel:  Channel name (must be in CHANNELS).
            callback: Async callable receiving a BusMessage.

        Returns:
            True if subscribed, False if channel doesn't exist.
        """
        if channel not in self._subscribers:
            logger.warning("Cannot subscribe to unknown channel: %s", channel)
            return False
        self._subscribers[channel].append(callback)
        return True

    def subscribe_all(self, callback: Callable):
        """Subscribe to ALL channels (orchestrator / monitor use).

        Args:
            callback: Async callable receiving a BusMessage.
        """
        self._global_subscribers.append(callback)

    # -------------------------------------------------------------------
    # Publish
    # -------------------------------------------------------------------

    async def publish(self, sender: str, channel: str, payload: dict) -> BusMessage | None:
        """Publish a message to a channel.

        Args:
            sender:  Agent ID or "system".
            channel: Target channel (must be in CHANNELS).
            payload: Arbitrary data dict.

        Returns:
            The created BusMessage, or None if the channel is invalid.
        """
        if channel not in self._buffers:
            logger.warning("Publish to unknown channel '%s' from '%s'", channel, sender)
            return None

        # Create message
        msg = BusMessage(
            message_id=uuid4().hex[:12],
            sender=sender,
            channel=channel,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Store in channel buffer
        self._buffers[channel].append(msg)

        # Update stats
        self._stats[channel] += 1
        self._sender_stats[sender] = self._sender_stats.get(sender, 0) + 1
        if sender not in self._sender_channel_counts:
            self._sender_channel_counts[sender] = {}
        sc = self._sender_channel_counts[sender]
        sc[channel] = sc.get(channel, 0) + 1

        # Notify channel subscribers + global subscribers (fire-and-forget)
        all_callbacks = self._subscribers.get(channel, []) + self._global_subscribers
        for cb in all_callbacks:
            try:
                asyncio.get_running_loop().create_task(self._safe_notify(cb, msg))
            except RuntimeError:
                pass  # No running event loop

        # Log to EventLogger
        if self._event_logger is not None:
            summary = str(payload)[:80]
            self._event_logger.info(
                "message_bus",
                f"[{channel}] {sender}: {summary}",
                channel=channel, sender=sender, message_id=msg.message_id,
            )

        # Broadcast to dashboards
        if self._on_broadcast is not None:
            try:
                asyncio.get_running_loop().create_task(
                    self._safe_broadcast(msg)
                )
            except RuntimeError:
                pass

        return msg

    async def _safe_notify(self, callback: Callable, msg: BusMessage):
        """Call a subscriber callback, swallowing errors."""
        try:
            await callback(msg)
        except Exception as e:
            logger.warning("Bus subscriber error: %s", e)

    async def _safe_broadcast(self, msg: BusMessage):
        """Broadcast a bus message to dashboards, swallowing errors."""
        try:
            await self._on_broadcast(msg.to_dict())
        except Exception:
            pass  # Never let broadcast errors disrupt the bus

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------

    def get_recent(self, channel: str | None = None, count: int = 50) -> list[dict]:
        """Return recent messages, optionally filtered by channel.

        Args:
            channel: If set, only return messages from this channel.
                     If None, return from all channels merged by time.
            count:   Max messages to return.

        Returns:
            List of message dicts, newest last.
        """
        if channel and channel in self._buffers:
            msgs = list(self._buffers[channel])
            return [m.to_dict() for m in msgs[-count:]]

        # Merge all channels, sort by timestamp
        all_msgs: list[BusMessage] = []
        for buf in self._buffers.values():
            all_msgs.extend(buf)
        all_msgs.sort(key=lambda m: m.timestamp)
        return [m.to_dict() for m in all_msgs[-count:]]

    def get_stats(self) -> dict:
        """Return per-channel message counts and total.

        Returns:
            {"channels": {channel: count, ...}, "total": N}
        """
        total = sum(self._stats.values())
        return {"channels": dict(self._stats), "total": total}

    def get_sender_channel_matrix(self) -> dict:
        """Return sender→channel count matrix for dashboard visualization.

        Returns:
            {sender: {channel: count, ...}, ...}
        """
        return {
            sender: dict(channels)
            for sender, channels in self._sender_channel_counts.items()
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for initial WS connect to dashboard.

        Returns:
            Dict with recent messages, stats, and sender matrix.
        """
        return {
            "messages": self.get_recent(count=100),
            "stats": self.get_stats(),
            "matrix": self.get_sender_channel_matrix(),
            "channels": CHANNELS,
        }
