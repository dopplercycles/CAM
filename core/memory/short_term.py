"""
CAM Short-Term Memory

Holds the current session's conversation context as a list of messages
with timestamps. When the list exceeds a configurable max length, older
messages are compressed into a summary so the context stays manageable.

This is the "what's happening right now" memory — it gets cleared when
the session ends or CAM restarts. Think of it like a mechanic's mental
notepad while working on a bike.

Usage:
    from core.memory.short_term import ShortTermMemory

    stm = ShortTermMemory(max_messages=100)
    stm.add("user", "What year was the Sportster introduced?")
    stm.add("assistant", "The Harley-Davidson Sportster debuted in 1957.")

    context = stm.get_context()         # list of message dicts
    status = stm.get_status()           # {message_count, max_messages, ...}
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.memory.short_term")


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in the short-term memory buffer.

    Attributes:
        role:       Who sent it — "user", "assistant", "system", "summary"
        content:    The message text
        timestamp:  When the message was recorded (UTC)
        metadata:   Optional extra data (task_id, source, etc.)
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON/dashboard use."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """In-memory conversation context for the current session.

    Messages are stored in chronological order. When the buffer exceeds
    max_messages, the oldest messages (keeping the most recent half) are
    compressed into a single summary message. This keeps context size
    bounded while preserving the most relevant recent history.

    The summary_ratio controls how much of the buffer gets summarized:
    when triggered, the oldest (1 - summary_ratio) of messages become
    a summary, keeping the newest summary_ratio fraction intact.

    Args:
        max_messages:   Maximum messages before summarization kicks in.
        summary_ratio:  Fraction of messages to keep after summarization (0.0-1.0).
                        Default 0.5 means keep the newest half.
    """

    def __init__(self, max_messages: int = 100, summary_ratio: float = 0.5):
        self._messages: list[Message] = []
        self._max_messages = max(10, max_messages)  # floor at 10
        self._summary_ratio = max(0.1, min(0.9, summary_ratio))
        self._summary_count = 0   # how many times we've summarized
        self._total_added = 0     # lifetime count of messages added

        logger.info(
            "ShortTermMemory initialized (max=%d, summary_ratio=%.1f)",
            self._max_messages, self._summary_ratio,
        )

    # -------------------------------------------------------------------
    # Adding messages
    # -------------------------------------------------------------------

    def add(self, role: str, content: str, metadata: dict | None = None) -> Message:
        """Add a message to the short-term memory buffer.

        If this pushes the buffer over max_messages, the oldest messages
        are compressed into a summary before the new message is appended.

        Args:
            role:     Who sent it — "user", "assistant", "system"
            content:  The message text
            metadata: Optional extra data (task_id, source, etc.)

        Returns:
            The created Message object.
        """
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        self._total_added += 1

        # Check if we need to summarize
        if len(self._messages) > self._max_messages:
            self._summarize()

        logger.debug(
            "STM message added (role=%s, buffer=%d/%d): %.80s",
            role, len(self._messages), self._max_messages, content,
        )
        return msg

    # -------------------------------------------------------------------
    # Retrieving context
    # -------------------------------------------------------------------

    def get_context(self) -> list[dict]:
        """Return all messages as a list of dicts, oldest first.

        This is what gets fed to the model as conversation context.
        """
        return [m.to_dict() for m in self._messages]

    def get_recent(self, count: int = 10) -> list[dict]:
        """Return the most recent N messages as dicts."""
        return [m.to_dict() for m in self._messages[-count:]]

    @property
    def messages(self) -> list[Message]:
        """Direct access to the message list (read-only intent)."""
        return list(self._messages)

    @property
    def message_count(self) -> int:
        """Current number of messages in the buffer."""
        return len(self._messages)

    # -------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------

    def _summarize(self):
        """Compress older messages into a summary to stay under the limit.

        Takes the oldest (1 - summary_ratio) messages, builds a plain-text
        summary, and replaces them with a single "summary" message. The
        newest messages are preserved intact.

        This is a simple extractive summary — it lists the key points
        from each compressed message. A future enhancement could use
        the model router to generate an abstractive summary.
        """
        keep_count = int(len(self._messages) * self._summary_ratio)
        # Always keep at least 5 messages
        keep_count = max(5, keep_count)
        compress_count = len(self._messages) - keep_count

        if compress_count <= 0:
            return

        to_compress = self._messages[:compress_count]
        to_keep = self._messages[compress_count:]

        # Build a simple extractive summary
        summary_lines = []
        for msg in to_compress:
            # Skip existing summaries — just carry their content forward
            if msg.role == "summary":
                summary_lines.append(msg.content)
            else:
                # Truncate long messages in the summary
                short = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                summary_lines.append(f"[{msg.role}] {short}")

        summary_text = (
            f"[Session summary — {compress_count} messages compressed]\n"
            + "\n".join(summary_lines)
        )

        summary_msg = Message(
            role="summary",
            content=summary_text,
            metadata={
                "compressed_count": compress_count,
                "summary_number": self._summary_count + 1,
            },
        )

        self._messages = [summary_msg] + to_keep
        self._summary_count += 1

        logger.info(
            "STM summarized: %d messages compressed → summary #%d (buffer now %d)",
            compress_count, self._summary_count, len(self._messages),
        )

    # -------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------

    def clear(self):
        """Clear all messages. Called on session end or restart."""
        count = len(self._messages)
        self._messages.clear()
        self._summary_count = 0
        self._total_added = 0
        logger.info("STM cleared (%d messages removed)", count)

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of the short-term memory state.

        Used by the dashboard memory panel and orchestrator status.
        """
        return {
            "message_count": len(self._messages),
            "max_messages": self._max_messages,
            "summary_count": self._summary_count,
            "total_added": self._total_added,
            "usage_pct": round(len(self._messages) / self._max_messages * 100, 1),
            "oldest_message": (
                self._messages[0].timestamp.isoformat() if self._messages else None
            ),
            "newest_message": (
                self._messages[-1].timestamp.isoformat() if self._messages else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(messages={len(self._messages)}/{self._max_messages}, "
            f"summaries={self._summary_count})"
        )


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stm = ShortTermMemory(max_messages=10, summary_ratio=0.5)

    # Add some test messages
    stm.add("user", "What year was the Sportster introduced?")
    stm.add("assistant", "The Harley-Davidson Sportster debuted in 1957.")
    stm.add("user", "What engine did it use?")
    stm.add("assistant", "The original Sportster used the Ironhead engine.")
    stm.add("system", "Task completed: research query")

    print(f"\n{stm}")
    print(f"Status: {stm.get_status()}")

    # Fill past the limit to trigger summarization
    for i in range(8):
        stm.add("user", f"Follow-up question #{i + 1} about motorcycle history")

    print(f"\nAfter overflow: {stm}")
    print(f"Status: {stm.get_status()}")

    print("\nContext:")
    for msg in stm.get_context():
        role = msg["role"]
        text = msg["content"][:100]
        print(f"  [{role}] {text}{'...' if len(msg['content']) > 100 else ''}")
