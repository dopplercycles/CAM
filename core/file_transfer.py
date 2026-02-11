"""
CAM File Transfer Manager

Server-side state tracking and utilities for file transfers between
the dashboard and remote agents over WebSocket.

Files are base64-encoded and sent in 64KB chunks with per-chunk acks.
SHA-256 checksums verify integrity after assembly.

Usage:
    from core.file_transfer import FileTransferManager

    ft_manager = FileTransferManager(on_progress=broadcast_transfer_progress)
    transfer = ft_manager.create_transfer(...)
"""

import base64
import hashlib
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

logger = logging.getLogger("cam.file_transfer")


def generate_transfer_id() -> str:
    """Generate a unique transfer ID: ft-<12 hex chars>."""
    return "ft-" + os.urandom(6).hex()


def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum of raw bytes. Returns 'sha256:<hex>'."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def chunk_file_data(data: bytes, chunk_size: int = 65536) -> list[str]:
    """Split raw bytes into base64-encoded chunks.

    Args:
        data: Raw file bytes.
        chunk_size: Size of each chunk in bytes (before base64 encoding).

    Returns:
        List of base64 strings, one per chunk.
    """
    chunks = []
    for offset in range(0, len(data), chunk_size):
        raw_chunk = data[offset:offset + chunk_size]
        chunks.append(base64.b64encode(raw_chunk).decode("ascii"))
    return chunks


# ---------------------------------------------------------------------------
# FileTransfer — tracks a single transfer
# ---------------------------------------------------------------------------

@dataclass
class FileTransfer:
    """Represents a single file transfer (server-to-agent or agent-to-server)."""

    transfer_id: str
    direction: str                          # "to_agent" or "from_agent"
    agent_id: str
    agent_name: str
    filename: str
    file_size: int
    chunk_count: int
    chunks_done: int = 0
    status: str = "pending"                 # pending | active | completed | failed | cancelled
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    checksum: str = ""
    dest_path: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()

    @property
    def progress(self) -> float:
        """Progress as a float 0.0–1.0."""
        if self.chunk_count == 0:
            return 1.0
        return self.chunks_done / self.chunk_count

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for dashboard broadcast."""
        return {
            "transfer_id": self.transfer_id,
            "direction": self.direction,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "filename": self.filename,
            "file_size": self.file_size,
            "chunk_count": self.chunk_count,
            "chunks_done": self.chunks_done,
            "status": self.status,
            "progress": round(self.progress, 4),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "checksum": self.checksum,
            "dest_path": self.dest_path,
        }


# ---------------------------------------------------------------------------
# FileTransferManager — server-side state for all transfers
# ---------------------------------------------------------------------------

class FileTransferManager:
    """Manages active and historical file transfers.

    Args:
        on_progress: Async callback called whenever a transfer's state changes.
                     Signature: async def callback(transfer: FileTransfer)
        history_size: Max completed transfers to keep in the ring buffer.
        max_active: Max concurrent active transfers.
    """

    def __init__(
        self,
        on_progress: Callable[[FileTransfer], Coroutine] | None = None,
        history_size: int = 100,
        max_active: int = 5,
    ):
        self._active: dict[str, FileTransfer] = {}
        self._history: deque[FileTransfer] = deque(maxlen=history_size)
        self._on_progress = on_progress
        self._max_active = max_active
        # Receive buffers for from_agent transfers: transfer_id → {chunks: {index: bytes}}
        self._receive_buffers: dict[str, dict[int, bytes]] = {}

    def create_transfer(
        self,
        direction: str,
        agent_id: str,
        agent_name: str,
        filename: str,
        file_size: int,
        chunk_count: int,
        checksum: str = "",
        dest_path: str = "",
    ) -> FileTransfer:
        """Create and register a new transfer.

        Returns the FileTransfer object, or raises ValueError if at capacity.
        """
        if len(self._active) >= self._max_active:
            raise ValueError(
                f"Max active transfers ({self._max_active}) reached. "
                "Wait for a transfer to complete."
            )

        transfer = FileTransfer(
            transfer_id=generate_transfer_id(),
            direction=direction,
            agent_id=agent_id,
            agent_name=agent_name,
            filename=filename,
            file_size=file_size,
            chunk_count=chunk_count,
            checksum=checksum,
            dest_path=dest_path,
            status="active",
        )
        self._active[transfer.transfer_id] = transfer
        logger.info(
            "Transfer %s created: %s %s → %s (%d bytes, %d chunks)",
            transfer.transfer_id, direction, filename, agent_name,
            file_size, chunk_count,
        )
        return transfer

    async def update_progress(self, transfer_id: str, chunks_done: int):
        """Update chunk progress and broadcast to dashboards."""
        transfer = self._active.get(transfer_id)
        if not transfer:
            return
        transfer.chunks_done = chunks_done
        if self._on_progress:
            await self._on_progress(transfer)

    async def complete_transfer(self, transfer_id: str):
        """Mark transfer as completed, move to history."""
        transfer = self._active.pop(transfer_id, None)
        if not transfer:
            return
        transfer.status = "completed"
        transfer.completed_at = datetime.now(timezone.utc).isoformat()
        transfer.chunks_done = transfer.chunk_count
        self._history.appendleft(transfer)
        self._receive_buffers.pop(transfer_id, None)
        logger.info("Transfer %s completed: %s", transfer_id, transfer.filename)
        if self._on_progress:
            await self._on_progress(transfer)

    async def fail_transfer(self, transfer_id: str, error: str):
        """Mark transfer as failed, move to history."""
        transfer = self._active.pop(transfer_id, None)
        if not transfer:
            return
        transfer.status = "failed"
        transfer.error = error
        transfer.completed_at = datetime.now(timezone.utc).isoformat()
        self._history.appendleft(transfer)
        self._receive_buffers.pop(transfer_id, None)
        logger.warning("Transfer %s failed: %s — %s", transfer_id, transfer.filename, error)
        if self._on_progress:
            await self._on_progress(transfer)

    async def cancel_transfer(self, transfer_id: str):
        """Cancel an active transfer, move to history."""
        transfer = self._active.pop(transfer_id, None)
        if not transfer:
            return
        transfer.status = "cancelled"
        transfer.completed_at = datetime.now(timezone.utc).isoformat()
        self._history.appendleft(transfer)
        self._receive_buffers.pop(transfer_id, None)
        logger.info("Transfer %s cancelled: %s", transfer_id, transfer.filename)
        if self._on_progress:
            await self._on_progress(transfer)

    def get_transfer(self, transfer_id: str) -> FileTransfer | None:
        """Look up an active transfer by ID."""
        return self._active.get(transfer_id)

    def init_receive_buffer(self, transfer_id: str):
        """Initialize a receive buffer for an incoming from_agent transfer."""
        self._receive_buffers[transfer_id] = {}

    def store_chunk(self, transfer_id: str, chunk_index: int, data: bytes):
        """Store a received chunk in the buffer."""
        buf = self._receive_buffers.get(transfer_id)
        if buf is not None:
            buf[chunk_index] = data

    def assemble_chunks(self, transfer_id: str, chunk_count: int) -> bytes | None:
        """Assemble all received chunks into a complete file. Returns None if incomplete."""
        buf = self._receive_buffers.get(transfer_id)
        if buf is None or len(buf) < chunk_count:
            return None
        parts = []
        for i in range(chunk_count):
            chunk = buf.get(i)
            if chunk is None:
                return None
            parts.append(chunk)
        return b"".join(parts)

    def get_active_list(self) -> list[dict]:
        """Return all active transfers as dicts for JSON serialization."""
        return [t.to_dict() for t in self._active.values()]

    def get_history(self) -> list[dict]:
        """Return transfer history as dicts for JSON serialization."""
        return [t.to_dict() for t in self._history]

    def get_all_transfers(self) -> list[dict]:
        """Return active + history for dashboard init."""
        return self.get_active_list() + self.get_history()
