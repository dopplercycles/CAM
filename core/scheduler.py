"""
CAM Task Scheduler

Runs scheduled tasks — one-time, recurring interval, or daily at a fixed time.
Schedules persist to a JSON file so they survive restarts. Due tasks are
submitted to the existing TaskQueue for the orchestrator to process.

Usage:
    from core.scheduler import Scheduler, ScheduleType

    scheduler = Scheduler(task_queue, persist_path="data/schedules.json")
    scheduler.add_schedule(
        name="Health Check All Agents",
        description="status_report",
        complexity=TaskComplexity.LOW,
        schedule_type=ScheduleType.INTERVAL,
        interval_seconds=300,
    )
    await scheduler.run()   # background loop
    scheduler.stop()
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

from core.task import TaskQueue, TaskComplexity


logger = logging.getLogger("cam.scheduler")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScheduleType(Enum):
    """How a scheduled task repeats."""
    ONCE = "once"            # Fire once, then auto-disable
    INTERVAL = "interval"    # Fire every N seconds
    DAILY = "daily"          # Fire once per day at a fixed HH:MM UTC


# ---------------------------------------------------------------------------
# ScheduledTask dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScheduledTask:
    """A task definition that fires on a schedule.

    Attributes:
        schedule_id:           Unique identifier (UUID string)
        name:                  Human-readable schedule name
        description:           Task description sent to the queue when fired
        complexity:            TaskComplexity for the submitted task
        required_capabilities: Capabilities needed by the executing agent
        schedule_type:         ONCE, INTERVAL, or DAILY
        interval_seconds:      Seconds between runs (INTERVAL type, default 300)
        run_at_time:           HH:MM string for DAILY type (UTC, default "09:00")
        next_run_at:           When this schedule next fires (ISO string or None)
        last_run_at:           When this schedule last fired (ISO string or None)
        enabled:               Whether the schedule is active
        run_count:             How many times this schedule has fired
        created_at:            When the schedule was created (ISO string)
    """
    schedule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    complexity: TaskComplexity = TaskComplexity.LOW
    required_capabilities: list[str] = field(default_factory=list)
    schedule_type: ScheduleType = ScheduleType.INTERVAL
    interval_seconds: int = 300
    run_at_time: str = "09:00"
    next_run_at: str | None = None
    last_run_at: str | None = None
    enabled: bool = True
    run_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def short_id(self) -> str:
        """First 8 characters of the schedule ID for display."""
        return self.schedule_id[:8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "schedule_id": self.schedule_id,
            "short_id": self.short_id,
            "name": self.name,
            "description": self.description,
            "complexity": self.complexity.value,
            "required_capabilities": self.required_capabilities,
            "schedule_type": self.schedule_type.value,
            "interval_seconds": self.interval_seconds,
            "run_at_time": self.run_at_time,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "enabled": self.enabled,
            "run_count": self.run_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        """Deserialize from a dictionary (loaded from JSON)."""
        try:
            complexity = TaskComplexity(data.get("complexity", "low"))
        except ValueError:
            complexity = TaskComplexity.LOW

        try:
            schedule_type = ScheduleType(data.get("schedule_type", "interval"))
        except ValueError:
            schedule_type = ScheduleType.INTERVAL

        return cls(
            schedule_id=data.get("schedule_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            complexity=complexity,
            required_capabilities=data.get("required_capabilities", []),
            schedule_type=schedule_type,
            interval_seconds=data.get("interval_seconds", 300),
            run_at_time=data.get("run_at_time", "09:00"),
            next_run_at=data.get("next_run_at"),
            last_run_at=data.get("last_run_at"),
            enabled=data.get("enabled", True),
            run_count=data.get("run_count", 0),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Manages scheduled tasks — persistence, CRUD, and background execution.

    Args:
        task_queue:       The shared TaskQueue to submit due tasks into.
        persist_path:     Path to the JSON file for schedule persistence.
        check_interval:   Seconds between schedule checks in the run loop.
        on_schedule_change: Async callback invoked after any schedule mutation
                           (add/remove/update/toggle). Used to push updates
                           to dashboard clients.
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        persist_path: str = "data/schedules.json",
        check_interval: int = 30,
        on_schedule_change: Callable[[], Coroutine] | None = None,
    ):
        self._task_queue = task_queue
        self._persist_path = Path(persist_path)
        self._check_interval = check_interval
        self._on_change = on_schedule_change
        self._schedules: dict[str, ScheduledTask] = {}
        self._running = False

        self._load()

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _load(self):
        """Load schedules from the JSON file, or seed defaults on first run."""
        if self._persist_path.exists():
            try:
                with open(self._persist_path, "r") as f:
                    data = json.load(f)
                for item in data:
                    sched = ScheduledTask.from_dict(item)
                    self._schedules[sched.schedule_id] = sched
                logger.info(
                    "Loaded %d schedule(s) from %s",
                    len(self._schedules), self._persist_path,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load schedules from %s: %s — starting empty",
                    self._persist_path, e,
                )
                self._schedules = {}
        else:
            # First run — seed default health check schedule
            logger.info("No schedules file found — seeding default health check")
            self._seed_defaults()
            self._save()

    def _save(self):
        """Persist schedules to JSON with atomic write (tmp + rename)."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._persist_path.with_suffix(".json.tmp")
        try:
            data = [s.to_dict() for s in self._schedules.values()]
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp_path), str(self._persist_path))
        except Exception as e:
            logger.error("Failed to save schedules: %s", e)

    def _seed_defaults(self):
        """Create the default "Health Check All Agents" schedule."""
        sched = ScheduledTask(
            name="Health Check All Agents",
            description="status_report",
            complexity=TaskComplexity.LOW,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=300,
            enabled=True,
        )
        sched.next_run_at = self._compute_next_run(sched)
        self._schedules[sched.schedule_id] = sched

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def add_schedule(
        self,
        name: str,
        description: str,
        complexity: TaskComplexity = TaskComplexity.LOW,
        schedule_type: ScheduleType = ScheduleType.INTERVAL,
        interval_seconds: int = 300,
        run_at_time: str = "09:00",
        required_capabilities: list[str] | None = None,
    ) -> ScheduledTask:
        """Create a new scheduled task, persist, and return it."""
        sched = ScheduledTask(
            name=name,
            description=description,
            complexity=complexity,
            required_capabilities=required_capabilities or [],
            schedule_type=schedule_type,
            interval_seconds=interval_seconds,
            run_at_time=run_at_time,
            enabled=True,
        )
        sched.next_run_at = self._compute_next_run(sched)
        self._schedules[sched.schedule_id] = sched
        self._save()
        logger.info(
            "Schedule created: '%s' (%s) type=%s next=%s",
            name, sched.short_id, schedule_type.value, sched.next_run_at,
        )
        return sched

    def remove_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule by ID. Returns True if found and removed."""
        if schedule_id in self._schedules:
            name = self._schedules[schedule_id].name
            del self._schedules[schedule_id]
            self._save()
            logger.info("Schedule removed: '%s' (%s)", name, schedule_id[:8])
            return True
        return False

    def update_schedule(self, schedule_id: str, **kwargs) -> ScheduledTask | None:
        """Update fields on an existing schedule. Returns updated schedule or None."""
        sched = self._schedules.get(schedule_id)
        if not sched:
            return None

        for key, value in kwargs.items():
            if key == "complexity" and isinstance(value, str):
                try:
                    value = TaskComplexity(value)
                except ValueError:
                    continue
            if key == "schedule_type" and isinstance(value, str):
                try:
                    value = ScheduleType(value)
                except ValueError:
                    continue
            if hasattr(sched, key) and key not in ("schedule_id", "created_at", "run_count"):
                setattr(sched, key, value)

        # Recompute next run if schedule type or timing changed
        if sched.enabled:
            sched.next_run_at = self._compute_next_run(sched)

        self._save()
        logger.info("Schedule updated: '%s' (%s)", sched.name, sched.short_id)
        return sched

    def toggle_enable(self, schedule_id: str) -> ScheduledTask | None:
        """Flip the enabled state. Re-enabling recomputes next_run_at."""
        sched = self._schedules.get(schedule_id)
        if not sched:
            return None

        sched.enabled = not sched.enabled
        if sched.enabled:
            sched.next_run_at = self._compute_next_run(sched)
        else:
            sched.next_run_at = None

        self._save()
        logger.info(
            "Schedule '%s' (%s) %s",
            sched.name, sched.short_id,
            "enabled" if sched.enabled else "disabled",
        )
        return sched

    def get_schedule(self, schedule_id: str) -> ScheduledTask | None:
        """Return a single schedule by ID, or None."""
        return self._schedules.get(schedule_id)

    def list_all(self) -> list[ScheduledTask]:
        """Return all schedules as a list."""
        return list(self._schedules.values())

    def to_broadcast_list(self) -> list[dict]:
        """Return all schedules as JSON-serializable dicts for the dashboard."""
        return [s.to_dict() for s in self._schedules.values()]

    # -------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------

    def _compute_next_run(self, sched: ScheduledTask) -> str:
        """Compute the next run time for a schedule.

        - ONCE: fires immediately (now)
        - INTERVAL: now + interval_seconds
        - DAILY: next occurrence of run_at_time in UTC
        """
        now = datetime.now(timezone.utc)

        if sched.schedule_type == ScheduleType.ONCE:
            return now.isoformat()

        if sched.schedule_type == ScheduleType.INTERVAL:
            next_dt = now + timedelta(seconds=sched.interval_seconds)
            return next_dt.isoformat()

        if sched.schedule_type == ScheduleType.DAILY:
            try:
                hour, minute = map(int, sched.run_at_time.split(":"))
            except (ValueError, AttributeError):
                hour, minute = 9, 0

            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target.isoformat()

        return now.isoformat()

    # -------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------

    def _submit_due_tasks(self):
        """Check all enabled schedules and submit tasks for any that are due."""
        now = datetime.now(timezone.utc)
        fired = []

        for sched in self._schedules.values():
            if not sched.enabled or not sched.next_run_at:
                continue

            try:
                next_run = datetime.fromisoformat(sched.next_run_at)
            except (ValueError, TypeError):
                continue

            if next_run > now:
                continue

            # Due — submit task to queue
            task = self._task_queue.add_task(
                description=sched.description,
                source="scheduler",
                complexity=sched.complexity,
                required_capabilities=sched.required_capabilities,
            )
            sched.run_count += 1
            sched.last_run_at = now.isoformat()

            logger.info(
                "Schedule '%s' (%s) fired → task %s (run #%d)",
                sched.name, sched.short_id, task.short_id, sched.run_count,
            )
            fired.append(sched)

            # Update next_run or disable for ONCE type
            if sched.schedule_type == ScheduleType.ONCE:
                sched.enabled = False
                sched.next_run_at = None
            else:
                sched.next_run_at = self._compute_next_run(sched)

        if fired:
            self._save()

        return fired

    async def run(self):
        """Background loop: check for due schedules every check_interval seconds."""
        import asyncio

        self._running = True
        logger.info(
            "Scheduler started (check every %ds, %d schedule(s))",
            self._check_interval, len(self._schedules),
        )

        while self._running:
            await asyncio.sleep(self._check_interval)
            try:
                fired = self._submit_due_tasks()
                if fired and self._on_change:
                    await self._on_change()
            except Exception as e:
                logger.error("Scheduler tick error: %s", e)

    def stop(self):
        """Signal the background loop to exit."""
        self._running = False
        logger.info("Scheduler stopped")
