"""
CAM Task Definition System

Defines what a "task" is in CAM and provides the queue that holds them.

A Task is a unit of work — anything from "research this recall notice"
to "draft a YouTube script" to "schedule an appointment." Tasks flow
through the orchestrator's OBSERVE → THINK → ACT → ITERATE loop.

The TaskQueue is the single source of truth for all pending, running,
and completed work. Right now it's in-memory; persistent storage
(JSON/SQLite) will come with core/memory/working.py.

Usage:
    from core.task import Task, TaskQueue, TaskStatus, TaskComplexity

    queue = TaskQueue()
    task = queue.add_task(
        description="research Harley-Davidson M-8 oil pump recall",
        source="cli",
        complexity=TaskComplexity.MEDIUM,
    )
    next_task = queue.get_next()
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.task")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Lifecycle states for a task.

    Tasks move through these states as the orchestrator processes them:

        PENDING → RUNNING → COMPLETED
                         └→ FAILED
    """
    PENDING = "pending"        # Waiting in the queue
    RUNNING = "running"        # Currently being processed by the orchestrator
    COMPLETED = "completed"    # Finished successfully
    FAILED = "failed"          # Finished with an error


class TaskComplexity(Enum):
    """How complex a task is — determines which model handles it.

    Maps directly to the model router's routing table:
        LOW    → Local Ollama (glm-4.7-flash) — fast, free
        MEDIUM → Local Ollama (gpt-oss:20b) or Kimi K2.5 — balanced
        HIGH   → Claude API — multi-step reasoning, quality matters
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A unit of work for CAM to process.

    Tasks come from many sources: CLI commands, dashboard messages,
    scheduled jobs, agent connector commands, or follow-up tasks
    generated during the ITERATE phase.

    Each task gets a UUID so it's globally unique — no collisions
    even if we eventually distribute tasks across multiple agents
    on the Pi cluster.

    Attributes:
        task_id:        Unique identifier (UUID string)
        description:    What needs to be done, in plain English
        complexity:     How complex the task is (routes to different models)
        source:         Where the task came from ("cli", "dashboard", "schedule", etc.)
        assigned_agent: Which agent is handling this (None = unassigned)
        status:         Current lifecycle state
        created_at:     When the task was added to the queue
        completed_at:   When the task finished (None until done)
        result:         Output from the ACT phase (None until completed/failed)
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    complexity: TaskComplexity = TaskComplexity.LOW
    source: str = "internal"
    assigned_agent: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    result: Any = None

    @property
    def short_id(self) -> str:
        """First 8 chars of the UUID — easier to read in logs."""
        return self.task_id[:8]

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON/dashboard use."""
        return {
            "task_id": self.task_id,
            "short_id": self.short_id,
            "description": self.description,
            "complexity": self.complexity.value,
            "source": self.source,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
        }


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------

class TaskQueue:
    """In-memory task queue for the orchestrator.

    The single source of truth for all work in CAM. Provides FIFO
    ordering, status filtering, and status summaries.

    Thread-safe enough for our use case (single orchestrator loop
    with async tasks). If we move to multi-agent concurrent access,
    we'll add asyncio.Lock or move to a proper message broker.

    Future: persist to disk via core/memory/working.py so tasks
    survive restarts.
    """

    def __init__(self):
        # All tasks, in insertion order
        self._tasks: list[Task] = []

        logger.info("TaskQueue initialized")

    # -------------------------------------------------------------------
    # Adding tasks
    # -------------------------------------------------------------------

    def add_task(
        self,
        description: str,
        source: str = "internal",
        complexity: TaskComplexity = TaskComplexity.LOW,
        assigned_agent: str | None = None,
    ) -> Task:
        """Create a new task and add it to the queue.

        This is how work enters CAM — the CLI, dashboard, scheduled
        jobs, and sub-agents all call this.

        Args:
            description:    What needs to be done, in plain English.
            source:         Where this task came from (for logging/audit).
            complexity:     Task complexity (determines model routing).
            assigned_agent: Optionally assign to a specific agent.

        Returns:
            The created Task object.
        """
        task = Task(
            description=description,
            source=source,
            complexity=complexity,
            assigned_agent=assigned_agent,
        )
        self._tasks.append(task)

        logger.info(
            "Task %s added (source=%s, complexity=%s): %s",
            task.short_id, task.source, task.complexity.value,
            task.description,
        )
        return task

    # -------------------------------------------------------------------
    # Retrieving tasks
    # -------------------------------------------------------------------

    def get_next(self) -> Task | None:
        """Get the next pending task from the queue (FIFO).

        Returns the oldest task with PENDING status, or None if the
        queue is empty. Does NOT change the task's status — the
        orchestrator does that when it starts working on it.

        Returns:
            The next Task to work on, or None.
        """
        for task in self._tasks:
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def get_task(self, task_id: str) -> Task | None:
        """Look up a specific task by its UUID.

        Args:
            task_id: Full UUID string or short ID prefix.

        Returns:
            The Task if found, None otherwise.
        """
        for task in self._tasks:
            if task.task_id == task_id or task.task_id.startswith(task_id):
                return task
        return None

    # -------------------------------------------------------------------
    # Listing and filtering
    # -------------------------------------------------------------------

    def list_all(self) -> list[Task]:
        """Return all tasks in the queue, in insertion order."""
        return list(self._tasks)

    def list_by_status(self, status: TaskStatus) -> list[Task]:
        """Return all tasks with a specific status."""
        return [t for t in self._tasks if t.status == status]

    @property
    def pending(self) -> list[Task]:
        """All tasks waiting to be picked up."""
        return self.list_by_status(TaskStatus.PENDING)

    @property
    def running(self) -> list[Task]:
        """All tasks currently being worked on."""
        return self.list_by_status(TaskStatus.RUNNING)

    @property
    def completed(self) -> list[Task]:
        """All tasks that finished successfully."""
        return self.list_by_status(TaskStatus.COMPLETED)

    @property
    def failed(self) -> list[Task]:
        """All tasks that finished with errors."""
        return self.list_by_status(TaskStatus.FAILED)

    # -------------------------------------------------------------------
    # Status summary
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a summary of the queue state.

        Useful for the dashboard status panel and orchestrator reporting.
        """
        return {
            "total": len(self._tasks),
            "pending": len(self.pending),
            "running": len(self.running),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }

    def __len__(self) -> int:
        """Total number of tasks (all statuses)."""
        return len(self._tasks)

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"TaskQueue(total={status['total']}, pending={status['pending']}, "
            f"running={status['running']}, completed={status['completed']}, "
            f"failed={status['failed']})"
        )


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    queue = TaskQueue()

    # Add some test tasks
    t1 = queue.add_task(
        "research Harley-Davidson M-8 oil pump recall",
        source="cli",
        complexity=TaskComplexity.MEDIUM,
    )
    t2 = queue.add_task(
        "draft YouTube script for barn find CB750",
        source="dashboard",
        complexity=TaskComplexity.HIGH,
    )
    t3 = queue.add_task(
        "check appointment schedule for tomorrow",
        source="schedule",
        complexity=TaskComplexity.LOW,
    )

    print(f"\nQueue: {queue}")
    print(f"Status: {queue.get_status()}")

    # Simulate processing
    next_task = queue.get_next()
    print(f"\nNext task: {next_task.short_id} — {next_task.description}")
    next_task.status = TaskStatus.RUNNING
    print(f"After starting: {queue}")

    next_task.status = TaskStatus.COMPLETED
    next_task.completed_at = datetime.now(timezone.utc)
    next_task.result = "Found recall notice TSB-1234"
    print(f"After completing: {queue}")

    # Look up by short ID
    found = queue.get_task(t2.short_id)
    print(f"\nLookup '{t2.short_id}': {found.description}")

    # List all
    print("\nAll tasks:")
    for task in queue.list_all():
        print(f"  [{task.status.value:>9}] {task.short_id} — {task.description}")
