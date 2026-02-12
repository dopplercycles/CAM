"""
CAM Agent Swarm Coordination

Enables parallel task execution across multiple agents. A SwarmTask
breaks a complex objective into subtasks that run simultaneously,
then assembles the combined results.

Example: "Research DR650 maintenance" spawns research (technical data),
content (outline), and business (parts pricing) tasks in parallel.
When all finish, an assembly task synthesizes the results.

Swarm subtasks are plain Task objects processed by the existing OATI
loop. The orchestrator's iterate() phase checks if a completed task
belongs to a swarm, and when all subtasks finish, queues an assembly
task to synthesize results.

Usage:
    from core.swarm import SwarmTask, SwarmStatus

    swarm = SwarmTask(
        name="DR650 Research",
        objective="Research DR650 maintenance for a YouTube video",
        subtasks=[task1, task2, task3],
    )
    task_queue.add_swarm(swarm)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from core.task import Task, TaskStatus, TaskComplexity


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cam.swarm")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SwarmStatus(Enum):
    """Lifecycle states for a swarm.

    Swarms move through these states as the orchestrator processes them:

        PENDING → RUNNING → ASSEMBLING → COMPLETED
                                       └→ FAILED

    Partial failure is allowed — if some subtasks fail but others
    succeed, assembly synthesizes the available results.
    """
    PENDING = "pending"          # Created, no subtasks dispatched yet
    RUNNING = "running"          # At least one subtask picked up by OATI loop
    ASSEMBLING = "assembling"    # All subtasks done, synthesis task running
    COMPLETED = "completed"      # Assembly finished, final result available
    FAILED = "failed"            # Assembly failed or ALL subtasks failed


# ---------------------------------------------------------------------------
# SwarmTask
# ---------------------------------------------------------------------------

@dataclass
class SwarmTask:
    """A parallel execution group — subtasks run simultaneously, then assemble.

    Unlike TaskChain (sequential), SwarmTask queues ALL subtasks at once
    so they can be picked up by different agents in parallel. When all
    subtasks finish, an assembly task synthesizes their results.

    Attributes:
        swarm_id:         Unique identifier (UUID string)
        name:             Human-readable name for the swarm
        objective:        High-level goal that subtasks contribute to
        subtasks:         Plain Task objects, all queued at once for parallel execution
        assembly_task:    Created during ASSEMBLING phase for synthesis
        status:           Current lifecycle state
        timeout_seconds:  Max time before swarm is considered stale (not enforced in v1)
        source:           Where the swarm was created from
        created_at:       When the swarm was created
        completed_at:     When the swarm finished (None until done)
        final_result:     Output from the assembly task (None until completed)
    """
    swarm_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    objective: str = ""
    subtasks: list[Task] = field(default_factory=list)
    assembly_task: Task | None = None
    status: SwarmStatus = SwarmStatus.PENDING
    timeout_seconds: float = 300.0
    source: str = "dashboard"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    final_result: Any = None

    # -------------------------------------------------------------------
    # Short ID
    # -------------------------------------------------------------------

    @property
    def short_id(self) -> str:
        """First 8 chars of the UUID — easier to read in logs."""
        return self.swarm_id[:8]

    # -------------------------------------------------------------------
    # Subtask filtering
    # -------------------------------------------------------------------

    @property
    def completed_subtasks(self) -> list[Task]:
        """All subtasks that finished successfully."""
        return [t for t in self.subtasks if t.status == TaskStatus.COMPLETED]

    @property
    def failed_subtasks(self) -> list[Task]:
        """All subtasks that finished with errors."""
        return [t for t in self.subtasks if t.status == TaskStatus.FAILED]

    @property
    def pending_subtasks(self) -> list[Task]:
        """All subtasks still waiting or running."""
        return [t for t in self.subtasks
                if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]

    @property
    def all_subtasks_done(self) -> bool:
        """True when every subtask is COMPLETED or FAILED."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self.subtasks
        )

    @property
    def progress_pct(self) -> float:
        """Percentage of subtasks that are done (completed or failed)."""
        if not self.subtasks:
            return 0.0
        done = sum(
            1 for t in self.subtasks
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        )
        return round((done / len(self.subtasks)) * 100, 1)

    @property
    def has_partial_failure(self) -> bool:
        """True if some subtasks failed but others succeeded."""
        return bool(self.failed_subtasks) and bool(self.completed_subtasks)

    # -------------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------------

    def build_assembly_prompt(self) -> str:
        """Build a synthesis prompt from all subtask results.

        Collects each subtask's description, status, assigned agent,
        and result into a structured prompt that asks the model to
        synthesize a coherent final answer.
        """
        parts = [
            f"## Swarm Objective\n{self.objective}\n",
            "## Subtask Results\n",
        ]

        for i, task in enumerate(self.subtasks, 1):
            status = task.status.value
            agent = task.assigned_agent or "unassigned"
            result = task.result or "(no result)"
            parts.append(
                f"### Subtask {i}: {task.description}\n"
                f"- **Status:** {status}\n"
                f"- **Agent:** {agent}\n"
                f"- **Result:** {result}\n"
            )

        parts.append(
            "## Instructions\n"
            "Synthesize the above subtask results into a coherent, "
            "comprehensive response that addresses the original objective. "
            "If some subtasks failed, work with what's available and note "
            "any gaps. Be concise but thorough."
        )

        return "\n".join(parts)

    def mark_assembling(self):
        """Transition to ASSEMBLING and create the assembly task.

        The assembly task uses HIGH complexity so it gets routed to a
        capable model (Claude API) for quality synthesis.
        """
        self.status = SwarmStatus.ASSEMBLING
        prompt = self.build_assembly_prompt()

        self.assembly_task = Task(
            description=prompt,
            source="swarm_assembly",
            complexity=TaskComplexity.HIGH,
        )

        logger.info(
            "Swarm %s (%s) → ASSEMBLING: %d/%d subtasks completed, "
            "assembly task %s created",
            self.short_id, self.name,
            len(self.completed_subtasks), len(self.subtasks),
            self.assembly_task.short_id,
        )

    def mark_completed(self, result: Any):
        """Mark the swarm as completed with the assembly result."""
        self.status = SwarmStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.final_result = result
        logger.info(
            "Swarm %s (%s) COMPLETED — %d subtasks, result: %.200s",
            self.short_id, self.name, len(self.subtasks),
            str(result)[:200],
        )

    def mark_failed(self, reason: str = ""):
        """Mark the swarm as failed."""
        self.status = SwarmStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.final_result = f"FAILED: {reason}" if reason else "FAILED"
        logger.info(
            "Swarm %s (%s) FAILED: %s",
            self.short_id, self.name, reason or "unknown",
        )

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON/dashboard use."""
        return {
            "swarm_id": self.swarm_id,
            "short_id": self.short_id,
            "name": self.name,
            "objective": self.objective,
            "status": self.status.value,
            "source": self.source,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_result": self.final_result,
            "progress_pct": self.progress_pct,
            "has_partial_failure": self.has_partial_failure,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "assembly_task": self.assembly_task.to_dict() if self.assembly_task else None,
        }
