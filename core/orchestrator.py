"""
CAM Core Orchestrator

The brain of CAM. Runs the main OBSERVE → THINK → ACT → ITERATE loop
that drives all autonomous behavior.

THINK uses the ModelRouter to send the task description to the right
model based on complexity. ACT dispatches tasks to remote agents when
available, falling back to the model response from THINK.

Usage:
    from core.orchestrator import Orchestrator

    orch = Orchestrator()
    orch.queue.add_task("research Harley-Davidson M-8 oil pump recall")
    await orch.run()
"""

import asyncio
import logging
from datetime import datetime, timezone

from core.task import Task, TaskQueue, TaskStatus, TaskComplexity, TaskChain, ChainStatus
from core.model_router import ModelRouter, ModelResponse
from core.memory.short_term import ShortTermMemory
from core.memory.working import WorkingMemory
from core.memory.long_term import LongTermMemory
from core.memory.episodic import EpisodicMemory
from core.persona import Persona
from core.task_classifier import classify as classify_task


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

    # Map TaskComplexity → model router complexity string.
    # AUTO is handled separately via the task classifier (see think()).
    COMPLEXITY_MAP = {
        TaskComplexity.LOW: "simple",       # → glm-4.7-flash (local, free)
        TaskComplexity.MEDIUM: "routine",   # → gpt-oss:20b (local, free)
        TaskComplexity.HIGH: "complex",     # → Claude API (quality matters)
    }

    # Map classifier tier number → router complexity string
    TIER_MAP = {
        1: "tier1",     # → small/fast local model
        2: "tier2",     # → medium local model
        3: "tier3",     # → large local model
    }

    def __init__(
        self,
        queue: TaskQueue | None = None,
        router: ModelRouter | None = None,
        short_term_memory: ShortTermMemory | None = None,
        working_memory: WorkingMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        persona: Persona | None = None,
        on_phase_change=None,
        on_task_update=None,
        on_dispatch_to_agent=None,
        on_model_call=None,
        on_chain_update=None,
    ):
        # Task queue — shared with other components (dashboard, CLI, etc.)
        # Note: can't use `queue or TaskQueue()` because an empty queue is falsy
        self.queue = queue if queue is not None else TaskQueue()

        # Model router — sends prompts to the right model by complexity
        self.router = router if router is not None else ModelRouter()

        # Memory systems — session context, persistent task state, long-term knowledge
        self.short_term = (
            short_term_memory if short_term_memory is not None
            else ShortTermMemory()
        )
        self.working = (
            working_memory if working_memory is not None
            else WorkingMemory()
        )
        self.long_term = (
            long_term_memory if long_term_memory is not None
            else LongTermMemory()
        )
        self.episodic = (
            episodic_memory if episodic_memory is not None
            else EpisodicMemory()
        )
        self.persona = persona if persona is not None else Persona()

        # Optional callbacks for real-time dashboard updates.
        # on_phase_change(task, phase, detail) — called at each OATI boundary
        # on_task_update() — called when any task's status changes
        self._on_phase_change = on_phase_change
        self._on_task_update = on_task_update

        # Optional callback for dispatching tasks to remote agents.
        # Signature: async (task: Task, plan: dict) -> str | None
        # Returns the agent's response string, or None to fall back to model response.
        self._on_dispatch_to_agent = on_dispatch_to_agent

        # Optional callback for logging model router calls.
        # Signature: (model, backend, tokens, latency_ms, cost_usd, task_short_id)
        self._on_model_call = on_model_call

        # Optional callback for chain status changes.
        # Signature: async (chain: TaskChain) -> None
        self._on_chain_update = on_chain_update

        # Flag to stop the loop gracefully (kill switch, shutdown, etc.)
        self._running: bool = False

        # How long to wait between loop iterations when idle (seconds).
        # Read from config (hot-reloadable — checked each loop iteration).
        try:
            from core.config import get_config
            self._poll_interval: float = get_config().orchestrator.poll_interval
        except Exception:
            self._poll_interval: float = 1.0

        logger.info(
            "Orchestrator initialized (router=%s, stm=%d/%d, working=%d active)",
            type(self.router).__name__,
            self.short_term.message_count, self.short_term._max_messages,
            len(self.working),
        )

    # -------------------------------------------------------------------
    # Callback helpers — errors here must never break the OATI loop
    # -------------------------------------------------------------------

    async def _notify_phase(self, task: Task, phase: str, detail: str = ""):
        """Push a phase change to the dashboard (if callback is set)."""
        if self._on_phase_change is None:
            return
        try:
            result = self._on_phase_change(task, phase, detail)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.debug("Phase callback error (non-fatal)", exc_info=True)

    async def _notify_task_update(self):
        """Push a task status change to the dashboard (if callback is set)."""
        if self._on_task_update is None:
            return
        try:
            result = self._on_task_update()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.debug("Task update callback error (non-fatal)", exc_info=True)

    async def _notify_chain_update(self, chain: "TaskChain"):
        """Push a chain status change to the dashboard (if callback is set)."""
        if self._on_chain_update is None:
            return
        try:
            result = self._on_chain_update(chain)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.debug("Chain update callback error (non-fatal)", exc_info=True)

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
        await self._notify_phase(task, "OBSERVE", "Picked up task")
        await self._notify_task_update()
        logger.info(
            "[OBSERVE] Picked up task %s: %s",
            task.short_id, task.description,
        )
        return task

    async def think(self, task: Task) -> dict:
        """THINK — Analyze the task using the model router.

        Sends the task description to the appropriate model (determined
        by task complexity) and returns the model's analysis as the plan.
        Records the task in working memory so it can be resumed after a
        restart. Adds the task to short-term memory for session context.

        Args:
            task: The task to analyze.

        Returns:
            A plan dictionary with the model's response and metadata.
        """
        # --- Auto-classify if complexity is AUTO ---
        if task.complexity == TaskComplexity.AUTO:
            classification = classify_task(task.description)
            router_complexity = self.TIER_MAP.get(classification.tier, "tier2")
            logger.info(
                "[THINK] Auto-classified task %s: type=%s, tier=%d → %s (%s)",
                task.short_id, classification.task_type, classification.tier,
                router_complexity, classification.reason,
            )
        else:
            # Explicit override — use the old mapping
            classification = None
            router_complexity = self.COMPLEXITY_MAP.get(task.complexity, "simple")

        # Save task state to working memory — survives restarts
        self.working.save_task(task.task_id, {
            "description": task.description,
            "phase": "THINK",
            "complexity": task.complexity.value,
            "source": task.source,
            "assigned_agent": task.assigned_agent,
        })

        # Record in short-term memory for session context
        classify_detail = ""
        if classification is not None:
            classify_detail = (
                f" [auto: {classification.task_type}, tier {classification.tier}]"
            )
        self.short_term.add("system", f"Processing task: {task.description}{classify_detail}", {
            "task_id": task.task_id,
            "phase": "THINK",
        })

        # Record in episodic memory — permanent conversation history
        self.episodic.record(
            "system",
            f"Processing task: {task.description}{classify_detail}",
            context_tags=["think", "task_start"],
            task_id=task.task_id,
        )

        await self._notify_phase(task, "THINK", f"Routing to {router_complexity} model")

        logger.info(
            "[THINK] Analyzing task %s (complexity=%s → %s): %s",
            task.short_id, task.complexity.value, router_complexity,
            task.description,
        )

        # --- Retrieve relevant long-term memories ---
        ltm_context = ""
        try:
            ltm_results = self.long_term.query(task.description, top_k=3)
            # Only include reasonably relevant results (score > 0.3)
            relevant = [r for r in ltm_results if r.score > 0.3]
            if relevant:
                ltm_lines = [f"- [{r.category}] {r.content}" for r in relevant]
                ltm_context = (
                    "\n\nRelevant knowledge from memory:\n"
                    + "\n".join(ltm_lines)
                )
                logger.info(
                    "[THINK] Retrieved %d relevant memories for task %s",
                    len(relevant), task.short_id,
                )
        except Exception:
            logger.debug("LTM retrieval failed (non-fatal)", exc_info=True)

        # Build the prompt — persona system prompt from YAML config
        system_prompt = self.persona.build_system_prompt()

        prompt = task.description + ltm_context

        response: ModelResponse = await self.router.route(
            prompt=prompt,
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

        # Notify model call listener (event logger in server.py)
        if self._on_model_call is not None:
            try:
                self._on_model_call(
                    model=response.model,
                    backend=response.backend,
                    tokens=response.total_tokens,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    task_short_id=task.short_id,
                )
            except Exception:
                logger.debug("Model call callback error (non-fatal)", exc_info=True)

        return plan

    async def act(self, task: Task, plan: dict) -> str:
        """ACT — Execute the plan.

        Tries to dispatch the task to a remote agent first. If an agent
        handles it, its response becomes the result. If no agents are
        available (or no dispatch callback is set), falls back to the
        model response from the THINK phase.

        Args:
            task: The task being executed.
            plan: The plan from the THINK phase.

        Returns:
            A result string describing what was done.
        """
        # Update working memory phase
        self.working.update_phase(task.task_id, "ACT")

        # Try agent dispatch first
        if self._on_dispatch_to_agent is not None:
            await self._notify_phase(task, "ACT", "Checking for available agents")
            logger.info(
                "[ACT] Task %s — attempting agent dispatch",
                task.short_id,
            )
            try:
                agent_result = await self._on_dispatch_to_agent(task, plan)
            except Exception as e:
                logger.warning(
                    "[ACT] Agent dispatch failed for task %s: %s",
                    task.short_id, e,
                )
                agent_result = None

            if agent_result is not None:
                logger.info(
                    "[ACT] Task %s handled by agent '%s': %.200s%s",
                    task.short_id, task.assigned_agent or "unknown",
                    agent_result, "..." if len(agent_result) > 200 else "",
                )
                return agent_result

        # Fallback — use the model response from THINK
        await self._notify_phase(task, "ACT", f"Executing via {plan.get('model', 'unknown')}")
        logger.info(
            "[ACT] Task %s — no agent available, using model response (model=%s)",
            task.short_id, plan.get("model", "unknown"),
        )

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
        await self._notify_phase(task, "ITERATE", "Wrapping up")

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        task.result = result

        await self._notify_phase(task, "COMPLETED", "Done")
        await self._notify_task_update()
        logger.info(
            "[ITERATE] Task %s completed: %s",
            task.short_id, result,
        )

        # Clear from working memory — task is done, no need to resume
        self.working.remove_task(task.task_id)

        # Record completion in short-term memory
        result_preview = result[:200] + "..." if len(result) > 200 else result
        self.short_term.add("assistant", f"Task completed: {result_preview}", {
            "task_id": task.task_id,
            "phase": "COMPLETED",
        })

        # --- Store substantial results in long-term memory ---
        if len(result) > 50:
            try:
                category = self._classify_ltm_category(task.description)
                ltm_content = f"Task: {task.description}\nResult: {result[:500]}"
                self.long_term.store(
                    content=ltm_content,
                    category=category,
                    metadata={
                        "task_id": task.task_id,
                        "source": "iterate",
                    },
                )
                logger.info(
                    "[ITERATE] Stored result in LTM (category=%s) for task %s",
                    category, task.short_id,
                )
            except Exception:
                logger.debug("LTM store failed (non-fatal)", exc_info=True)

        # Record completion in episodic memory — permanent history
        self.episodic.record(
            "assistant",
            f"Task completed: {task.description}\nResult: {result_preview}",
            context_tags=["iterate", "task_complete"],
            task_id=task.task_id,
        )

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

        # --- Resume tasks from working memory ---
        # On startup, check if any tasks were in progress when we last
        # shut down. Re-queue them so they get processed again.
        resumed = self._resume_from_working_memory()
        if resumed:
            logger.info("Resumed %d task(s) from working memory", resumed)

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

                    # --- Chain advancement ---
                    chain = self.queue.get_chain_for_task(task.task_id)
                    if chain is not None:
                        if chain.status == ChainStatus.PENDING:
                            chain.status = ChainStatus.RUNNING
                        next_task = chain.advance()
                        if next_task is not None:
                            # Override Rule 4: chain inherits highest tier.
                            # If any step in the chain was classified at a
                            # higher tier, subsequent AUTO steps inherit it.
                            if next_task.complexity == TaskComplexity.AUTO:
                                highest = self._chain_highest_tier(chain)
                                if highest is not None:
                                    next_task.complexity = highest
                                    logger.info(
                                        "Chain %s: next step %s inherits %s "
                                        "(Rule 4: chain highest tier)",
                                        chain.short_id, next_task.short_id,
                                        highest.value,
                                    )
                            # Queue the next step for the orchestrator to pick up
                            self.queue._tasks.append(next_task)
                            await self._notify_task_update()
                        else:
                            # All steps done
                            chain.mark_completed()
                        await self._notify_chain_update(chain)

                except Exception as e:
                    # Task failed — log it, mark it, keep the loop running.
                    # Constitution failure hierarchy: stop action, secure,
                    # notify George, log, wait for direction.
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now(timezone.utc)
                    task.result = str(e)
                    await self._notify_phase(task, "FAILED", str(e))
                    await self._notify_task_update()
                    logger.error(
                        "Task %s failed: %s", task.short_id, e, exc_info=True,
                    )

                    # Clear from working memory — failed tasks don't resume
                    self.working.remove_task(task.task_id)

                    # Record failure in short-term memory
                    self.short_term.add("system", f"Task failed: {e}", {
                        "task_id": task.task_id,
                        "phase": "FAILED",
                    })

                    # Record failure in episodic memory — permanent history
                    self.episodic.record(
                        "system",
                        f"Task failed: {task.description}\nError: {e}",
                        context_tags=["iterate", "task_failed"],
                        task_id=task.task_id,
                    )

                    # --- Chain failure cascade ---
                    chain = self.queue.get_chain_for_task(task.task_id)
                    if chain is not None:
                        chain.mark_failed()
                        await self._notify_chain_update(chain)

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
    # Chain tier inheritance (Override Rule 4)
    # -------------------------------------------------------------------

    # Complexity ranking for chain inheritance — higher index = higher tier
    _COMPLEXITY_RANK = {
        TaskComplexity.AUTO: 0,
        TaskComplexity.LOW: 1,
        TaskComplexity.MEDIUM: 2,
        TaskComplexity.HIGH: 3,
    }

    def _chain_highest_tier(self, chain: TaskChain) -> TaskComplexity | None:
        """Find the highest complexity assigned to any completed step in a chain.

        Returns the highest non-AUTO complexity, or None if all steps are AUTO.
        Used by Override Rule 4: chained tasks inherit the highest tier.
        """
        highest = None
        highest_rank = 0
        for step in chain.steps[:chain.current_step]:
            rank = self._COMPLEXITY_RANK.get(step.complexity, 0)
            if rank > highest_rank:
                highest_rank = rank
                highest = step.complexity
        return highest

    # -------------------------------------------------------------------
    # Working memory resume
    # -------------------------------------------------------------------

    def _resume_from_working_memory(self) -> int:
        """Check working memory for in-progress tasks and re-queue them.

        Called once at the start of run(). Any tasks that were saved in
        working memory (because they were in progress when we shut down)
        get re-added to the task queue as new PENDING tasks.

        Returns:
            Number of tasks resumed.
        """
        active = self.working.get_all_active()
        if not active:
            return 0

        resumed = 0
        for task_id, state in active.items():
            description = state.get("description", "")
            if not description:
                # No description = can't resume meaningfully
                self.working.remove_task(task_id)
                continue

            complexity_str = state.get("complexity", "low")
            try:
                complexity = TaskComplexity(complexity_str)
            except ValueError:
                complexity = TaskComplexity.LOW

            # Re-queue with the original description intact — agents
            # parse the description as a command, so prefixes break them.
            task = self.queue.add_task(
                description=description,
                source=state.get("source", "working_memory"),
                complexity=complexity,
            )

            # Record in short-term memory
            self.short_term.add("system", f"Resumed task from working memory: {description}", {
                "original_task_id": task_id,
                "new_task_id": task.task_id,
                "phase": state.get("phase", "unknown"),
            })

            # Clean the old entry — the new task gets its own working memory entry
            self.working.remove_task(task_id)

            logger.info(
                "Resumed task from working memory: %s (was in phase %s) → new task %s",
                task_id[:8], state.get("phase", "?"), task.short_id,
            )
            resumed += 1

        return resumed

    # -------------------------------------------------------------------
    # LTM category classification
    # -------------------------------------------------------------------

    # Keyword → category mappings for auto-classifying task results
    _LTM_CATEGORY_KEYWORDS = {
        "user_preference": ["prefer", "setting", "always", "never"],
        "system_learning": ["learn", "pattern", "discovered"],
        "knowledge": ["research", "fact", "history", "spec"],
    }

    def _classify_ltm_category(self, description: str) -> str:
        """Determine the LTM category for a task result based on keywords.

        Scans the task description for keyword patterns and returns
        the best matching category. Falls back to "task_result".

        Args:
            description: The task description to classify.

        Returns:
            One of: "user_preference", "system_learning", "knowledge", "task_result"
        """
        desc_lower = description.lower()
        for category, keywords in self._LTM_CATEGORY_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                return category
        return "task_result"

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the orchestrator loop is currently active."""
        return self._running

    def get_status(self) -> dict:
        """Return a snapshot of the orchestrator's current state.

        Useful for the dashboard status panel. Includes queue status,
        session cost tracking, and memory system status.
        """
        queue_status = self.queue.get_status()
        cost_status = self.router.get_session_costs()
        return {
            "running": self._running,
            **queue_status,
            "session_costs": cost_status,
            "persona": self.persona.get_status(),
            "memory": {
                "short_term": self.short_term.get_status(),
                "working": self.working.get_status(),
                "long_term": self.long_term.get_status(),
                "episodic": self.episodic.get_status(),
            },
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
