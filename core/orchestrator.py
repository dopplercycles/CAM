"""
CAM Core Orchestrator

The brain of CAM. Runs the main OBSERVE → THINK → ACT → ITERATE loop
that drives all autonomous behavior.

THINK uses the ModelRouter to send the task description to the right
model based on complexity. ACT is still a placeholder — tool execution
and sub-agent dispatch come later.

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
from core.model_router import ModelRouter, ModelResponse


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
        THINK    — Send task to the model router, get analysis/plan
        ACT      — Execute the plan (placeholder — tools come later)
        ITERATE  — Log results, update memory, check for follow-ups

    THINK uses the ModelRouter to route tasks to the right model based
    on complexity. The TaskQueue is in-memory for now; persistent task
    storage (JSON/SQLite) comes with core/memory/working.py.
    """

    # Map TaskComplexity → model router complexity tier
    COMPLEXITY_MAP = {
        TaskComplexity.LOW: "simple",       # → glm-4.7-flash (local, free)
        TaskComplexity.MEDIUM: "routine",   # → gpt-oss:20b (local, free)
        TaskComplexity.HIGH: "complex",     # → Claude API (quality matters)
    }

    def __init__(
        self,
        queue: TaskQueue | None = None,
        router: ModelRouter | None = None,
    ):
        # Task queue — shared with other components (dashboard, CLI, etc.)
        self.queue = queue or TaskQueue()

        # Model router — sends prompts to the right model by complexity
        self.router = router or ModelRouter()

        # Flag to stop the loop gracefully (kill switch, shutdown, etc.)
        self._running: bool = False

        # How long to wait between loop iterations when idle (seconds).
        # Keeps CPU usage near zero when there's nothing to do.
        self._poll_interval: float = 1.0

        logger.info("Orchestrator initialized (router=%s)", type(self.router).__name__)

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
        """THINK — Analyze the task using the model router.

        Sends the task description to the appropriate model (determined
        by task complexity) and returns the model's analysis as the plan.

        Future enhancements:
        - Retrieve relevant memory before prompting
        - Ask the model to break complex tasks into sub-steps
        - Identify which tools or sub-agents are needed

        Args:
            task: The task to analyze.

        Returns:
            A plan dictionary with the model's response and metadata.
        """
        router_complexity = self.COMPLEXITY_MAP.get(task.complexity, "simple")

        logger.info(
            "[THINK] Analyzing task %s (complexity=%s → %s): %s",
            task.short_id, task.complexity.value, router_complexity,
            task.description,
        )

        # Build the prompt — ask the model to analyze and plan
        system_prompt = (
            "You are CAM, an AI assistant for Doppler Cycles, a motorcycle "
            "diagnostics and content creation business. Analyze the task and "
            "provide a concise, actionable response. Be specific and practical."
        )

        response: ModelResponse = await self.router.route(
            prompt=task.description,
            task_complexity=router_complexity,
            system_prompt=system_prompt,
        )

        plan = {
            "task_id": task.task_id,
            "description": task.description,
            "complexity": task.complexity.value,
            "model": response.model,
            "backend": response.backend,
            "model_response": response.text,
            "tokens": response.total_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
            "tools_needed": [],           # Future: parse from model response
            "sub_tasks": [],              # Future: parse from model response
        }

        logger.info(
            "[THINK] Task %s plan ready — model=%s, tokens=%d, "
            "latency=%.0fms, cost=$%.6f",
            task.short_id, response.model, response.total_tokens,
            response.latency_ms, response.cost_usd,
        )
        return plan

    async def act(self, task: Task, plan: dict) -> str:
        """ACT — Execute the plan.

        This is where tools get called, sub-agents get dispatched,
        and model responses get generated. The security framework
        will classify every action before execution (safe/logged/
        gated/blocked per the Constitution).

        For now: returns the model's response from the THINK phase.
        Tool execution and sub-agent dispatch come later.

        Args:
            task: The task being executed.
            plan: The plan from the THINK phase.

        Returns:
            A result string describing what was done.
        """
        logger.info(
            "[ACT] Executing task %s (model=%s): %s",
            task.short_id, plan.get("model", "unknown"), task.description,
        )

        # For now, the model's response IS the result.
        # When tools are added, this is where we'd parse the plan
        # for tool calls and execute them.
        result = plan.get("model_response", f"Completed: {task.description}")

        logger.info(
            "[ACT] Task %s result (%.0f tokens): %.200s%s",
            task.short_id, plan.get("tokens", 0),
            result, "..." if len(result) > 200 else "",
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

        Useful for the dashboard status panel. Includes queue status
        and session cost tracking from the model router.
        """
        queue_status = self.queue.get_status()
        cost_status = self.router.get_session_costs()
        return {
            "running": self._running,
            **queue_status,
            "session_costs": cost_status,
        }


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

async def _test():
    """Smoke test: submit a task and watch it flow through the full loop.

    Uses a LOW complexity task so it routes to the local Ollama model
    (glm-4.7-flash) — fast, free, no API keys needed.
    """
    orch = Orchestrator()

    # Add a task — LOW complexity routes to local Ollama
    orch.queue.add_task(
        "What year was the Harley-Davidson Sportster first introduced? "
        "Reply in one sentence.",
        source="cli",
        complexity=TaskComplexity.LOW,
    )

    # Run the loop — stop after it processes the task
    async def stop_when_done():
        while orch.queue.pending or orch.queue.running:
            await asyncio.sleep(0.5)
        # Give ITERATE a moment to finish
        await asyncio.sleep(0.5)
        orch.stop()

    await asyncio.gather(orch.run(), stop_when_done())

    # Print results
    print("\n" + "=" * 60)
    print("ORCHESTRATOR TEST RESULTS")
    print("=" * 60)

    status = orch.get_status()
    print(f"\nQueue: {status['completed']} completed, {status['failed']} failed")

    costs = status["session_costs"]
    print(f"Session: {costs['call_count']} model calls, "
          f"{costs['total_tokens']} tokens, ${costs['total_cost_usd']}")

    for task in orch.queue.list_all():
        print(f"\n[{task.status.value}] {task.short_id} — {task.description}")
        print(f"  Result: {task.result}")


if __name__ == "__main__":
    asyncio.run(_test())
