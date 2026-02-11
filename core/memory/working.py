"""
CAM Working Memory

Persistent storage for active task state that survives restarts.
Backed by a JSON file so the orchestrator can pick up where it
left off after a crash or reboot.

Think of this as the clipboard on the workbench — notes about what
you're in the middle of, so you don't lose your place when you
step away and come back.

Usage:
    from core.memory.working import WorkingMemory

    wm = WorkingMemory()
    wm.save_task("abc-123", {"description": "Research M-8 recall", "phase": "THINK"})

    # After restart:
    active = wm.get_all_active()   # returns all saved task states
    task_state = wm.get_task("abc-123")
    wm.remove_task("abc-123")      # clear when task completes
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.memory.working")


# ---------------------------------------------------------------------------
# Default persist path
# ---------------------------------------------------------------------------

DEFAULT_PERSIST_PATH = "data/tasks/working_memory.json"


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """JSON-backed persistent storage for active task state.

    Each entry represents a task that was in progress when the system
    last ran. On startup, the orchestrator reads this to find tasks
    that need to be resumed.

    The JSON file is written atomically (write to temp, then rename)
    to avoid corruption from crashes mid-write.

    Args:
        persist_path: Path to the JSON file for persistence.
    """

    def __init__(self, persist_path: str = DEFAULT_PERSIST_PATH):
        self._path = Path(persist_path)
        self._entries: dict[str, dict] = {}   # task_id → state dict
        self._load()
        logger.info(
            "WorkingMemory initialized (path=%s, active_entries=%d)",
            self._path, len(self._entries),
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _load(self):
        """Load state from the JSON file, if it exists.

        If the file is missing or corrupt, start with empty state.
        A corrupt file is logged as a warning but doesn't crash —
        we'd rather lose working memory than fail to start.
        """
        if not self._path.exists():
            self._entries = {}
            return

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                self._entries = data
            else:
                logger.warning("Working memory file has unexpected format, starting fresh")
                self._entries = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load working memory (%s), starting fresh", e)
            self._entries = {}

    def _save(self):
        """Persist current state to the JSON file atomically.

        Writes to a temp file first, then renames — so a crash mid-write
        doesn't corrupt the existing file.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")

        try:
            tmp_path.write_text(
                json.dumps(self._entries, indent=2, default=str),
                encoding="utf-8",
            )
            tmp_path.replace(self._path)
        except OSError as e:
            logger.error("Failed to save working memory: %s", e)

    # -------------------------------------------------------------------
    # Task state management
    # -------------------------------------------------------------------

    def save_task(self, task_id: str, state: dict[str, Any]) -> None:
        """Save or update the state for an active task.

        Call this when a task enters the OATI loop so its state is
        preserved in case of a restart.

        Args:
            task_id:  The task's UUID
            state:    Dict with task state — description, phase, complexity,
                      assigned_agent, etc. Must be JSON-serializable.
        """
        # Stamp when this entry was last updated
        state["_updated_at"] = datetime.now(timezone.utc).isoformat()
        if "_saved_at" not in self._entries.get(task_id, {}):
            state["_saved_at"] = datetime.now(timezone.utc).isoformat()

        self._entries[task_id] = state
        self._save()

        logger.debug(
            "Working memory saved task %s (phase=%s)",
            task_id[:8], state.get("phase", "unknown"),
        )

    def get_task(self, task_id: str) -> dict | None:
        """Retrieve the saved state for a specific task.

        Args:
            task_id: The task's UUID or short ID prefix.

        Returns:
            The state dict, or None if not found.
        """
        # Try exact match first
        if task_id in self._entries:
            return self._entries[task_id]

        # Try prefix match (short ID)
        for full_id, state in self._entries.items():
            if full_id.startswith(task_id):
                return state

        return None

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from working memory (it's done or failed).

        Call this when a task completes or fails so it doesn't get
        resumed on the next startup.

        Args:
            task_id: The task's UUID.

        Returns:
            True if the task was found and removed, False otherwise.
        """
        if task_id in self._entries:
            del self._entries[task_id]
            self._save()
            logger.debug("Working memory removed task %s", task_id[:8])
            return True

        return False

    def get_all_active(self) -> dict[str, dict]:
        """Return all active task states.

        Used by the orchestrator on startup to find tasks that need
        to be resumed.

        Returns:
            Dict of task_id → state for all entries.
        """
        return dict(self._entries)

    def update_phase(self, task_id: str, phase: str) -> None:
        """Quick helper to update just the phase of a task.

        Convenience method so the orchestrator doesn't need to re-save
        the entire state dict just to update the current phase.

        Args:
            task_id: The task's UUID.
            phase:   The new phase ("OBSERVE", "THINK", "ACT", "ITERATE").
        """
        if task_id in self._entries:
            self._entries[task_id]["phase"] = phase
            self._entries[task_id]["_updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def clear(self):
        """Remove all entries. Use with caution — tasks will not be resumed."""
        count = len(self._entries)
        self._entries.clear()
        self._save()
        logger.info("Working memory cleared (%d entries removed)", count)

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of working memory state.

        Used by the dashboard memory panel.
        """
        entries_summary = []
        for task_id, state in self._entries.items():
            entries_summary.append({
                "task_id": task_id,
                "short_id": task_id[:8],
                "description": state.get("description", "")[:100],
                "phase": state.get("phase", "unknown"),
                "complexity": state.get("complexity", "unknown"),
                "saved_at": state.get("_saved_at"),
                "updated_at": state.get("_updated_at"),
            })

        return {
            "active_count": len(self._entries),
            "persist_path": str(self._path),
            "entries": entries_summary,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"WorkingMemory(active={len(self._entries)}, path={self._path})"


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    # Use a temp file so we don't clobber real data
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()

    try:
        wm = WorkingMemory(persist_path=tmp.name)

        # Save some task states
        wm.save_task("aaaa-1111-2222-3333", {
            "description": "Research Harley M-8 oil pump recall",
            "phase": "THINK",
            "complexity": "medium",
        })
        wm.save_task("bbbb-4444-5555-6666", {
            "description": "Draft YouTube script for barn find CB750",
            "phase": "ACT",
            "complexity": "high",
        })

        print(f"\n{wm}")
        print(f"Status: {json.dumps(wm.get_status(), indent=2)}")

        # Simulate restart — load from the same file
        wm2 = WorkingMemory(persist_path=tmp.name)
        print(f"\nAfter 'restart': {wm2}")
        print(f"Active tasks: {list(wm2.get_all_active().keys())}")

        # Retrieve and remove
        state = wm2.get_task("aaaa-1111")
        print(f"\nLookup 'aaaa-1111': {state}")

        wm2.remove_task("aaaa-1111-2222-3333")
        print(f"After remove: {wm2}")

    finally:
        os.unlink(tmp.name)
