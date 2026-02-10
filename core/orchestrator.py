"""
CAM Core Orchestrator

The brain of CAM. Runs the main OBSERVE → THINK → ACT → ITERATE loop
that drives all autonomous behavior.

Right now this is the skeleton — placeholders where the model router,
memory system, and tool framework will plug in later. But the loop
structure and task flow are real and ready to build on.

Usage:
    from core.orchestrator import Orchestrator

    orch = Orchestrator()
    orch.queue.add_task("research Harley-Davidson M-8 oil pump recall")
    await orch.run()
"""

import asyncio
import logging
from datetime import datetime, timezone

from core.task import Task, TaskQueue, TaskStatus, TaskComplexity


# ---------------------------------------------------------------------------
# Logging — every action gets a record (CLAUDE.md coding conventions)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.orchestrator")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Main agent loop for CAM.

    The orchestrator runs a continuous loop with four phases:

        OBSERVE  — Check for new tasks from the queue
        THINK    — Analyze the task, plan an approach
        ACT      — Execute the plan (tools, model calls, sub-agents)
        ITERATE  — Log results, update memory, check for follow-ups

    Right now THINK and ACT are placeholders. As we build out the model
    router, memory system, and tool framework, they'll gain real logic.

    The TaskQueue is in-memory for now. Phase 1 gets this loop running;
    persistent task storage (JSON/SQLite) comes with the working memory
    module in core/memory/working.py.
    """

    def __init__(self, queue: TaskQueue | None = None):
        # Task queue — shared with other components (dashboard, CLI, etc.)
        self.queue = queue or TaskQueue()

        # Flag to stop the loop gracefully (kill switch, shutdown, etc.)
        self._running: bool = False

        # How long to wait between loop iterations when idle (seconds).
        # Keeps CPU usage near zero when there's nothing to do.
        self._poll_interval: float = 1.0

        logger.info("Orchestrator initialized")

    # -------------------------------------------------------------------
    # The four phases
    # -------------------------------------------------------------------

    async def observe(self) -> Task | None:
        """OBSERVE — Check for new work.

        Scans the task queue for the next pending task. In the future
        this will also check:
        - CLI input buffer
        - Dashboard WebSocket commands
        - Scheduled task timers
        - Mobile app messages
        - Voice call transcriptions

        Returns:
            The next Task to work on, or None if the queue is empty.
        """
        task = self.queue.get_next()
        if task is None:
            return None

        # Claim the task — mark it as running so nobody else grabs it
        task.status = TaskStatus.RUNNING
        logger.info(
            "[OBSERVE] Picked up task %s: %s",
            task.short_id, task.description,
        )
        return task

    async def think(self, task: Task) -> dict:
        """THINK — Analyze the task and plan an approach.

        This is where the model router will be called to:
        - Classify the task complexity (low/medium/high)
        - Retrieve relevant memory (long-term knowledge, past episodes)
        - Choose the right model for the job
        - Break complex tasks into sub-steps
        - Determine which tools or sub-agents are needed

        For now: logs the task and returns a placeholder plan.

        Args:
            task: The task to analyze.

        Returns:
            A plan dictionary describing what to do.
        """
        logger.info(
            "[THINK] Analyzing task %s: %s",
            task.short_id, task.description,
        )

        # Placeholder — will be replaced by model router + memory lookup
        plan = {
            "task_id": task.task_id,
            "description": task.description,
            "complexity": task.complexity.value,
            "model": "placeholder",       # Future: route to Ollama/Kimi/Claude
            "tools_needed": [],           # Future: identify required tools
            "sub_tasks": [],              # Future: break into steps
        }

        logger.info(
            "[THINK] Plan for task %s: complexity=%s",
            task.short_id, plan["complexity"],
        )
        return plan

    async def act(self, task: Task, plan: dict) -> str:
        """ACT — Execute the plan.

        This is where tools get called, sub-agents get dispatched,
        and model responses get generated. The security framework
        will classify every action before execution (safe/logged/
        gated/blocked per the Constitution).

        For now: returns a mock result string.

        Args:
            task: The task being executed.
            plan: The plan from the THINK phase.

        Returns:
            A result string describing what was done.
        """
        logger.info(
            "[ACT] Executing task %s: %s",
            task.short_id, task.description,
        )

        # Placeholder — will be replaced by tool execution + model calls
        result = f"Completed: {task.description}"

        logger.info(
            "[ACT] Task %s result: %s",
            task.short_id, result,
        )
        return result

    async def iterate(self, task: Task, result: str):
        """ITERATE — Wrap up and learn.

        After a task is done, this phase:
        - Updates the task status and stores the result
        - Logs the completed action to the audit trail
        - Updates memory (episodic log, working state)
        - Checks if the result spawns follow-up tasks

        For now: marks the task complete and logs it.

        Args:
            task:   The completed task.
            result: The output from the ACT phase.
        """
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        task.result = result

        logger.info(
            "[ITERATE] Task %s completed: %s",
            task.short_id, result,
        )

        # Future: save to episodic memory
        # Future: update working memory state
        # Future: check for follow-up tasks and add them to the queue

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    async def run(self):
        """Run the orchestrator loop.

        Continuously cycles through OBSERVE → THINK → ACT → ITERATE
        until stopped. When there's nothing to do, sleeps briefly to
        avoid burning CPU.

        Stop the loop by calling stop() or setting _running = False.
        """
        self._running = True
        logger.info("Orchestrator loop started")

        try:
            while self._running:
                # OBSERVE — look for work
                task = await self.observe()

                if task is None:
                    # Nothing to do — sleep and check again
                    await asyncio.sleep(self._poll_interval)
                    continue

                try:
                    # THINK — plan the approach
                    plan = await self.think(task)

                    # ACT — execute the plan
                    result = await self.act(task, plan)

                    # ITERATE — wrap up, log, learn
                    await self.iterate(task, result)

                except Exception as e:
                    # Task failed — log it, mark it, keep the loop running.
                    # Constitution failure hierarchy: stop action, secure,
                    # notify George, log, wait for direction.
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now(timezone.utc)
                    task.result = str(e)
                    logger.error(
                        "Task %s failed: %s", task.short_id, e, exc_info=True,
                    )

        except asyncio.CancelledError:
            logger.info("Orchestrator loop cancelled")
        finally:
            self._running = False
            logger.info("Orchestrator loop stopped")

    def stop(self):
        """Signal the orchestrator to stop after the current iteration.

        This is the graceful shutdown path — the loop finishes its
        current task (if any) and then exits. Used by the kill switch
        and clean shutdown handlers.
        """
        logger.info("Orchestrator stop requested")
        self._running = False

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the orchestrator loop is currently active."""
        return self._running

    def get_status(self) -> dict:
        """Return a snapshot of the orchestrator's current state.

        Useful for the dashboard status panel.
        """
        queue_status = self.queue.get_status()
        return {
            "running": self._running,
            **queue_status,
        }


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

async def _test():
    """Quick smoke test: add tasks via the queue and run the loop."""
    orch = Orchestrator()

    # Add tasks through the queue (not the orchestrator directly)
    orch.queue.add_task(
        "research Harley-Davidson M-8 oil pump recall",
        source="cli",
        complexity=TaskComplexity.MEDIUM,
    )
    orch.queue.add_task(
        "draft YouTube script for barn find CB750",
        source="dashboard",
        complexity=TaskComplexity.HIGH,
    )

    # Run the loop — it will process both tasks and then idle
    # We'll stop it after a short delay
    async def stop_after_delay():
        await asyncio.sleep(3)
        orch.stop()

    await asyncio.gather(orch.run(), stop_after_delay())

    # Print final status
    print(f"\nOrchestrator status: {orch.get_status()}")
    print(f"Queue: {orch.queue}")
    for task in orch.queue.list_all():
        print(f"  [{task.status.value:>9}] {task.short_id} — {task.description} → {task.result}")


if __name__ == "__main__":
    asyncio.run(_test())
