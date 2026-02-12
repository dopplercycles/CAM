"""
CAM Launch Readiness — Integration Test Suite

End-to-end validation of the full task lifecycle: submission through OATI
through storage, plus resilience checks (kill switch, backup) and
communication checks (notifications, Telegram).

Unlike self_test.py (which probes individual subsystems in isolation), this
suite tests that subsystems actually work *together*.

Usage (CLI):
    cd ~/CAM
    python -m tests.integration_test

Usage (from server.py):
    from tests.integration_test import LaunchReadiness
    lr = LaunchReadiness(config=config, ..., on_progress=callback)
    results = await lr.run_all()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Awaitable

logger = logging.getLogger("cam.integration_test")


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single integration check.

    Attributes:
        name:        Human-readable check name (e.g. "Task Submission")
        passed:      Whether the check succeeded
        message:     Detail on pass reason or failure description
        duration_ms: How long the check took in milliseconds
        category:    Grouping key: prerequisite | task_flow | agent | data |
                     resilience | communication
    """
    name: str
    passed: bool
    message: str
    duration_ms: float
    category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# LaunchReadiness
# ---------------------------------------------------------------------------

class LaunchReadiness:
    """Integration test runner — validates that subsystems work together.

    Constructor takes all shared state refs from server.py.  When None,
    the corresponding check is skipped gracefully.

    Args:
        config:             Parsed settings.toml config object
        event_logger:       EventLogger instance
        router:             ModelRouter instance
        registry:           AgentRegistry for agent tests
        task_queue:         TaskQueue for task lifecycle tests
        short_term_memory:  ShortTermMemory instance
        working_memory:     WorkingMemory instance
        episodic_memory:    EpisodicMemory instance
        long_term_memory:   LongTermMemory instance
        analytics:          Analytics instance (SQLite)
        security_audit_log: SecurityAuditLog instance
        agent_websockets:   Dict of agent_id -> WebSocket connections
        port:               Dashboard port
        orchestrator:       Orchestrator instance (OATI loop)
        notification_manager: NotificationManager for notification tests
        backup_manager:     BackupManager for backup tests
        telegram_bot:       TelegramBot instance (or None)
        content_agent:      ContentAgent instance (or None)
        research_agent:     ResearchAgent instance (or None)
        business_agent:     BusinessAgent instance (or None)
        self_test_class:    SelfTest class reference (for prerequisite check)
        on_progress:        Async callback(completed, total, result) for live updates
    """

    def __init__(
        self,
        config=None,
        event_logger=None,
        router=None,
        registry=None,
        task_queue=None,
        short_term_memory=None,
        working_memory=None,
        episodic_memory=None,
        long_term_memory=None,
        analytics=None,
        security_audit_log=None,
        agent_websockets=None,
        port=None,
        orchestrator=None,
        notification_manager=None,
        backup_manager=None,
        telegram_bot=None,
        content_agent=None,
        research_agent=None,
        business_agent=None,
        self_test_class=None,
        on_progress: Callable[[int, int, dict], Awaitable[None]] | None = None,
    ):
        self.config = config
        self.event_logger = event_logger
        self.router = router
        self.registry = registry
        self.task_queue = task_queue
        self.short_term_memory = short_term_memory
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.long_term_memory = long_term_memory
        self.analytics = analytics
        self.security_audit_log = security_audit_log
        self.agent_websockets = agent_websockets or {}
        self.port = port or 8080
        self.orchestrator = orchestrator
        self.notification_manager = notification_manager
        self.backup_manager = backup_manager
        self.telegram_bot = telegram_bot
        self.content_agent = content_agent
        self.research_agent = research_agent
        self.business_agent = business_agent
        self.self_test_class = self_test_class
        self.on_progress = on_progress

    # -------------------------------------------------------------------
    # Test runner
    # -------------------------------------------------------------------

    async def run_all(self) -> dict:
        """Run all integration checks sequentially, returning a summary dict.

        After each check, calls on_progress(completed, total, result) if set.

        Returns:
            {
                "total": int,
                "passed": int,
                "failed": int,
                "duration_ms": float,
                "results": [CheckResult.to_dict(), ...],
                "timestamp": str,
            }
        """
        checks = [
            self.check_self_test,
            self.check_task_submission,
            self.check_oati_loop,
            self.check_model_router,
            self.check_agent_dispatch,
            self.check_analytics_recording,
            self.check_event_log,
            self.check_notification,
            self.check_task_chain,
            self.check_kill_switch,
            self.check_backup,
            self.check_telegram,
        ]

        results: list[CheckResult] = []
        suite_start = time.perf_counter()

        for i, check_fn in enumerate(checks):
            start = time.perf_counter()
            try:
                result = await check_fn()
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                result = CheckResult(
                    name=check_fn.__name__.replace("check_", "").replace("_", " ").title(),
                    passed=False,
                    message=f"Crashed: {type(e).__name__}: {e}",
                    duration_ms=round(elapsed, 1),
                    category="unknown",
                )

            results.append(result)

            # Log each result
            if self.event_logger:
                level = "info" if result.passed else "error"
                log_fn = getattr(self.event_logger, level)
                log_fn(
                    "launch_readiness",
                    f"{'PASS' if result.passed else 'FAIL'}: {result.name} — {result.message}",
                    check=result.name,
                    check_category=result.category,
                    duration_ms=result.duration_ms,
                )

            # Progress callback
            if self.on_progress:
                try:
                    await self.on_progress(i + 1, len(checks), result.to_dict())
                except Exception:
                    pass  # Don't let progress callback failures stop checks

        suite_duration = (time.perf_counter() - suite_start) * 1000
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        from datetime import datetime, timezone
        summary = {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "duration_ms": round(suite_duration, 1),
            "results": [r.to_dict() for r in results],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.event_logger:
            self.event_logger.info(
                "launch_readiness",
                f"Launch readiness complete: {passed}/{len(results)} passed in {suite_duration:.0f}ms",
                passed=passed,
                failed=failed,
                total=len(results),
            )

        return summary

    # -------------------------------------------------------------------
    # The 12 Checks
    # -------------------------------------------------------------------

    async def check_self_test(self) -> CheckResult:
        """#1 Prerequisite: Run SelfTest.run_all(), pass if 0 failures."""
        start = time.perf_counter()
        try:
            if self.self_test_class is None:
                from tests.self_test import SelfTest
                st_class = SelfTest
            else:
                st_class = self.self_test_class

            # Build a SelfTest with all the shared state we have
            st = st_class(
                config=self.config,
                event_logger=self.event_logger,
                router=self.router,
                registry=self.registry,
                task_queue=self.task_queue,
                short_term_memory=self.short_term_memory,
                working_memory=self.working_memory,
                episodic_memory=self.episodic_memory,
                long_term_memory=self.long_term_memory,
                analytics=self.analytics,
                security_audit_log=self.security_audit_log,
                agent_websockets=self.agent_websockets,
                port=self.port,
            )

            results = await st.run_all()
            elapsed = (time.perf_counter() - start) * 1000

            failed = results.get("failed", -1)
            total = results.get("total", 0)
            passed = results.get("passed", 0)

            if failed == 0:
                return CheckResult("Self-Test Suite", True,
                                   f"All {total} subsystem tests passed",
                                   round(elapsed, 1), "prerequisite")
            else:
                return CheckResult("Self-Test Suite", False,
                                   f"{failed} of {total} subsystem tests failed",
                                   round(elapsed, 1), "prerequisite")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Self-Test Suite", False, str(e),
                               round(elapsed, 1), "prerequisite")

    async def check_task_submission(self) -> CheckResult:
        """#2 Task flow: add_task() → verify in pending → cleanup."""
        start = time.perf_counter()
        try:
            if self.task_queue is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Task Submission", False,
                                   "No task_queue instance provided",
                                   round(elapsed, 1), "task_flow")

            before_count = len(self.task_queue.pending)
            task = self.task_queue.add_task(
                "_launchtest_submission_probe_",
                source="launch_readiness",
            )

            # Verify it appears in pending
            found = any(
                t.task_id == task.task_id for t in self.task_queue.pending
            )

            # Clean up
            self.task_queue._tasks = [
                t for t in self.task_queue._tasks
                if t.task_id != task.task_id
            ]

            elapsed = (time.perf_counter() - start) * 1000

            if found:
                return CheckResult("Task Submission", True,
                                   "add_task → pending OK",
                                   round(elapsed, 1), "task_flow")
            else:
                return CheckResult("Task Submission", False,
                                   "Task not found in pending after add_task",
                                   round(elapsed, 1), "task_flow")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Task Submission", False, str(e),
                               round(elapsed, 1), "task_flow")

    async def check_oati_loop(self) -> CheckResult:
        """#3 Task flow: Submit task, wait up to 15s for orchestrator to complete it."""
        start = time.perf_counter()
        try:
            if self.orchestrator is None or self.task_queue is None:
                elapsed = (time.perf_counter() - start) * 1000
                missing = []
                if self.orchestrator is None:
                    missing.append("orchestrator")
                if self.task_queue is None:
                    missing.append("task_queue")
                return CheckResult("OATI Loop", False,
                                   f"Missing: {', '.join(missing)}",
                                   round(elapsed, 1), "task_flow")

            # Check if orchestrator is running
            if not getattr(self.orchestrator, '_running', False):
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("OATI Loop", False,
                                   "Orchestrator is not running",
                                   round(elapsed, 1), "task_flow")

            # Submit a test task
            task = self.task_queue.add_task(
                "_launchtest_oati_probe_",
                source="launch_readiness",
            )
            task_id = task.task_id

            # Wait up to 15 seconds for the orchestrator to pick it up and complete it
            deadline = time.perf_counter() + 15
            completed = False
            while time.perf_counter() < deadline:
                # Check if task status changed from pending
                current = next(
                    (t for t in self.task_queue._tasks if t.task_id == task_id),
                    None,
                )
                if current is None:
                    # Task was removed (cleaned up) — treat as completed
                    completed = True
                    break
                if current.status.value in ("completed", "failed"):
                    completed = True
                    break
                await asyncio.sleep(0.5)

            # Clean up — remove the test task
            self.task_queue._tasks = [
                t for t in self.task_queue._tasks
                if t.task_id != task_id
            ]

            elapsed = (time.perf_counter() - start) * 1000

            if completed:
                return CheckResult("OATI Loop", True,
                                   f"Task processed by orchestrator in {elapsed:.0f}ms",
                                   round(elapsed, 1), "task_flow")
            else:
                return CheckResult("OATI Loop", False,
                                   "Task still pending after 15s — orchestrator may be stuck",
                                   round(elapsed, 1), "task_flow")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("OATI Loop", False, str(e),
                               round(elapsed, 1), "task_flow")

    async def check_model_router(self) -> CheckResult:
        """#4 Task flow: route("ping", "simple") → verify non-empty response."""
        start = time.perf_counter()
        try:
            if self.router is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Model Router", False,
                                   "No router instance provided",
                                   round(elapsed, 1), "task_flow")

            response = await self.router.route("ping", task_complexity="simple")
            elapsed = (time.perf_counter() - start) * 1000

            if response.text and "[error]" not in response.text:
                return CheckResult("Model Router", True,
                                   f"Response from {response.model} ({response.total_tokens} tokens)",
                                   round(elapsed, 1), "task_flow")
            else:
                return CheckResult("Model Router", False,
                                   f"Empty or error response: {(response.text or '')[:100]}",
                                   round(elapsed, 1), "task_flow")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Model Router", False, str(e),
                               round(elapsed, 1), "task_flow")

    async def check_agent_dispatch(self) -> CheckResult:
        """#5 Agent: Check local agents exist + registry for remote agents."""
        start = time.perf_counter()
        try:
            local_agents = []
            if self.content_agent is not None:
                local_agents.append("ContentAgent")
            if self.research_agent is not None:
                local_agents.append("ResearchAgent")
            if self.business_agent is not None:
                local_agents.append("BusinessAgent")

            remote_agents = []
            if self.registry:
                all_agents = self.registry.list_all()
                remote_agents = [
                    a.name for a in all_agents
                    if not a.agent_id.startswith("_launchtest_")
                ]

            connected_remote = [
                name for name in remote_agents
                # Check by looking up the agent's ID in websockets
            ]
            # Count remote agents with active WebSocket connections
            ws_count = 0
            if self.registry:
                for agent in self.registry.list_all():
                    if (not agent.agent_id.startswith("_launchtest_")
                            and agent.agent_id in self.agent_websockets):
                        ws_count += 1

            elapsed = (time.perf_counter() - start) * 1000

            parts = []
            if local_agents:
                parts.append(f"{len(local_agents)} local ({', '.join(local_agents)})")
            if remote_agents:
                parts.append(f"{len(remote_agents)} registered, {ws_count} connected")

            if not local_agents and not remote_agents:
                return CheckResult("Agent Dispatch", False,
                                   "No local agents and no registered remote agents",
                                   round(elapsed, 1), "agent")

            return CheckResult("Agent Dispatch", True,
                               "; ".join(parts),
                               round(elapsed, 1), "agent")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Agent Dispatch", False, str(e),
                               round(elapsed, 1), "agent")

    async def check_analytics_recording(self) -> CheckResult:
        """#6 Data: record_task() → get_summary() → verify → cleanup."""
        start = time.perf_counter()
        try:
            if self.analytics is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Analytics Recording", False,
                                   "No analytics instance provided",
                                   round(elapsed, 1), "data")

            # Create a minimal task-like object for recording
            from core.task import Task, TaskStatus, TaskComplexity
            from datetime import datetime, timezone

            test_task = Task(
                description="_launchtest_analytics_probe_",
                source="launch_readiness",
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc),
            )

            self.analytics.record_task(test_task)
            summary = self.analytics.get_summary()

            # Clean up — delete the test record
            try:
                conn = getattr(self.analytics, '_conn', None)
                if conn:
                    conn.execute(
                        "DELETE FROM task_records WHERE description = ?",
                        ("_launchtest_analytics_probe_",),
                    )
                    conn.commit()
            except Exception:
                pass  # Best-effort cleanup

            elapsed = (time.perf_counter() - start) * 1000

            if summary and isinstance(summary, dict):
                return CheckResult("Analytics Recording", True,
                                   f"record → summary OK (total_tasks: {summary.get('total_tasks', '?')})",
                                   round(elapsed, 1), "data")
            else:
                return CheckResult("Analytics Recording", False,
                                   "get_summary returned unexpected result",
                                   round(elapsed, 1), "data")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Analytics Recording", False, str(e),
                               round(elapsed, 1), "data")

    async def check_event_log(self) -> CheckResult:
        """#7 Data: event_logger.info() → verify event in get_recent()."""
        start = time.perf_counter()
        try:
            if self.event_logger is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Event Log", False,
                                   "No event_logger instance provided",
                                   round(elapsed, 1), "data")

            marker = "_launchtest_eventlog_probe_"
            self.event_logger.info("launch_readiness", marker)

            recent = self.event_logger.get_recent(50)
            found = any(marker in str(ev.get("message", "")) for ev in recent)

            elapsed = (time.perf_counter() - start) * 1000

            if found:
                return CheckResult("Event Log", True,
                                   "info() → get_recent() OK",
                                   round(elapsed, 1), "data")
            else:
                return CheckResult("Event Log", False,
                                   "Logged event not found in get_recent()",
                                   round(elapsed, 1), "data")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Event Log", False, str(e),
                               round(elapsed, 1), "data")

    async def check_notification(self) -> CheckResult:
        """#8 Communication: emit() → verify count increased → dismiss."""
        start = time.perf_counter()
        try:
            if self.notification_manager is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Notification", False,
                                   "No notification_manager instance provided",
                                   round(elapsed, 1), "communication")

            before = self.notification_manager.get_unread_count()
            self.notification_manager.emit(
                "info",
                "_launchtest_ Integration Check",
                "This notification is part of the launch readiness check",
                "system",
            )
            after = self.notification_manager.get_unread_count()

            # Clean up — dismiss the test notification
            # Find it by title match and dismiss
            notifs = getattr(self.notification_manager, '_notifications', [])
            for n in reversed(notifs):
                if "_launchtest_" in getattr(n, 'title', ''):
                    self.notification_manager.dismiss(n.id)
                    break

            elapsed = (time.perf_counter() - start) * 1000

            if after > before:
                return CheckResult("Notification", True,
                                   f"emit → count {before} → {after} OK",
                                   round(elapsed, 1), "communication")
            else:
                return CheckResult("Notification", False,
                                   f"Unread count did not increase ({before} → {after})",
                                   round(elapsed, 1), "communication")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Notification", False, str(e),
                               round(elapsed, 1), "communication")

    async def check_task_chain(self) -> CheckResult:
        """#9 Task flow: Create 2-step chain → add_chain() → verify lookup → cleanup."""
        start = time.perf_counter()
        try:
            if self.task_queue is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Task Chain", False,
                                   "No task_queue instance provided",
                                   round(elapsed, 1), "task_flow")

            from core.task import Task, TaskChain

            step1 = Task(
                description="_launchtest_chain_step1_",
                source="launch_readiness",
            )
            step2 = Task(
                description="_launchtest_chain_step2_",
                source="launch_readiness",
            )
            chain = TaskChain(
                name="_launchtest_chain_",
                steps=[step1, step2],
                source="launch_readiness",
            )

            self.task_queue.add_chain(chain)

            # Verify the chain is findable
            found = self.task_queue.get_chain(chain.chain_id)

            # Clean up — remove chain and its tasks
            self.task_queue._chains = [
                c for c in self.task_queue._chains
                if c.chain_id != chain.chain_id
            ]
            self.task_queue._tasks = [
                t for t in self.task_queue._tasks
                if t.task_id not in (step1.task_id, step2.task_id)
            ]

            elapsed = (time.perf_counter() - start) * 1000

            if found and found.chain_id == chain.chain_id:
                return CheckResult("Task Chain", True,
                                   f"add_chain → get_chain OK ({found.total_steps} steps)",
                                   round(elapsed, 1), "task_flow")
            else:
                return CheckResult("Task Chain", False,
                                   "Chain not found after add_chain",
                                   round(elapsed, 1), "task_flow")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Task Chain", False, str(e),
                               round(elapsed, 1), "task_flow")

    async def check_kill_switch(self) -> CheckResult:
        """#10 Resilience: orchestrator.stop() → verify stopped → restart → verify running.

        Safety: uses orchestrator.stop()/_running, NOT activate_kill_switch()
        which would shut down real agents.
        """
        start = time.perf_counter()
        try:
            if self.orchestrator is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Kill Switch", False,
                                   "No orchestrator instance provided",
                                   round(elapsed, 1), "resilience")

            was_running = getattr(self.orchestrator, '_running', False)

            # Stop the orchestrator
            self.orchestrator.stop()
            stopped = not getattr(self.orchestrator, '_running', True)

            # Restart it — set _running back and re-launch the loop
            if was_running:
                # Re-enable the flag so the orchestrator can be started again
                self.orchestrator._running = False
                # Spawn the orchestrator loop as a background task again
                loop = asyncio.get_running_loop()
                loop.create_task(self.orchestrator.run())
                # Brief wait for the loop to set _running = True
                await asyncio.sleep(0.3)

            restarted = getattr(self.orchestrator, '_running', False)

            elapsed = (time.perf_counter() - start) * 1000

            if stopped and (restarted or not was_running):
                msg = "stop → verify stopped → restart OK"
                if not was_running:
                    msg = "stop OK (orchestrator was already stopped, skip restart)"
                return CheckResult("Kill Switch", True, msg,
                                   round(elapsed, 1), "resilience")
            else:
                return CheckResult("Kill Switch", False,
                                   f"stopped={stopped}, restarted={restarted}, was_running={was_running}",
                                   round(elapsed, 1), "resilience")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Kill Switch", False, str(e),
                               round(elapsed, 1), "resilience")

    async def check_backup(self) -> CheckResult:
        """#11 Resilience: backup_manager.backup() → verify ok: True."""
        start = time.perf_counter()
        try:
            if self.backup_manager is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Backup", False,
                                   "No backup_manager instance provided",
                                   round(elapsed, 1), "resilience")

            result = self.backup_manager.backup()
            elapsed = (time.perf_counter() - start) * 1000

            if result.get("ok"):
                archive = result.get("archive", "?")
                size = result.get("size_bytes", 0)
                size_kb = size / 1024 if size else 0
                return CheckResult("Backup", True,
                                   f"Backup OK ({size_kb:.0f} KB)",
                                   round(elapsed, 1), "resilience")
            else:
                error = result.get("error", "unknown error")
                return CheckResult("Backup", False,
                                   f"Backup failed: {error}",
                                   round(elapsed, 1), "resilience")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Backup", False, str(e),
                               round(elapsed, 1), "resilience")

    async def check_telegram(self) -> CheckResult:
        """#12 Communication: If configured, check is_connected. If disabled, pass with note."""
        start = time.perf_counter()
        try:
            if self.telegram_bot is None:
                elapsed = (time.perf_counter() - start) * 1000
                return CheckResult("Telegram", True,
                                   "Telegram bot not configured — skipped",
                                   round(elapsed, 1), "communication")

            connected = getattr(self.telegram_bot, 'is_connected', False)
            elapsed = (time.perf_counter() - start) * 1000

            if connected:
                return CheckResult("Telegram", True,
                                   "Bot is connected and polling",
                                   round(elapsed, 1), "communication")
            else:
                return CheckResult("Telegram", False,
                                   "Bot is configured but not connected",
                                   round(elapsed, 1), "communication")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return CheckResult("Telegram", False, str(e),
                               round(elapsed, 1), "communication")

    # -------------------------------------------------------------------
    # Standalone CLI runner
    # -------------------------------------------------------------------

    @classmethod
    async def run_standalone(cls):
        """Run integration checks from the command line with colored output.

        Creates its own config, memories, router, etc.
        Prints a colored checklist with PASS/FAIL markers.
        Exits 0 if all pass, 1 if any fail.
        """
        import sys

        # ANSI colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        DIM = "\033[2m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{BOLD}CAM Launch Readiness — Integration Test{RESET}")
        print("=" * 55)

        # Create lightweight standalone instances
        config = None
        try:
            from core.config import get_config
            config = get_config()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not load config: {e}{RESET}")

        router = None
        try:
            from core.model_router import ModelRouter
            router = ModelRouter()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create ModelRouter: {e}{RESET}")

        registry = None
        try:
            from core.agent_registry import AgentRegistry
            registry = AgentRegistry()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create AgentRegistry: {e}{RESET}")

        task_queue = None
        try:
            from core.task import TaskQueue
            task_queue = TaskQueue()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create TaskQueue: {e}{RESET}")

        short_term = None
        try:
            from core.memory.short_term import ShortTermMemory
            short_term = ShortTermMemory()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create ShortTermMemory: {e}{RESET}")

        working = None
        try:
            from core.memory.working import WorkingMemory
            working = WorkingMemory()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create WorkingMemory: {e}{RESET}")

        episodic = None
        try:
            from core.memory.episodic import EpisodicMemory
            episodic = EpisodicMemory()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create EpisodicMemory: {e}{RESET}")

        long_term = None
        try:
            from core.memory.long_term import LongTermMemory
            long_term = LongTermMemory()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create LongTermMemory: {e}{RESET}")

        analytics = None
        try:
            from core.analytics import Analytics
            db_path = "data/analytics.db"
            if config and hasattr(config, 'analytics'):
                db_path = getattr(config.analytics, 'db_path', db_path)
            analytics = Analytics(db_path=db_path)
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create Analytics: {e}{RESET}")

        security_audit_log = None
        try:
            from security.audit import SecurityAuditLog
            db_path = "data/security_audit.db"
            if config and hasattr(config, 'security'):
                db_path = getattr(config.security, 'audit_db_path', db_path)
            security_audit_log = SecurityAuditLog(db_path=db_path)
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create SecurityAuditLog: {e}{RESET}")

        event_logger = None
        try:
            from core.event_logger import EventLogger
            event_logger = EventLogger()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create EventLogger: {e}{RESET}")

        notification_manager = None
        try:
            from core.notifications import NotificationManager
            notification_manager = NotificationManager()
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create NotificationManager: {e}{RESET}")

        backup_manager = None
        try:
            from core.backup import BackupManager
            if config:
                backup_manager = BackupManager(config=config, event_logger=event_logger)
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create BackupManager: {e}{RESET}")

        tester = cls(
            config=config,
            event_logger=event_logger,
            router=router,
            registry=registry,
            task_queue=task_queue,
            short_term_memory=short_term,
            working_memory=working,
            episodic_memory=episodic,
            long_term_memory=long_term,
            analytics=analytics,
            security_audit_log=security_audit_log,
            notification_manager=notification_manager,
            backup_manager=backup_manager,
            # orchestrator, telegram_bot, agents not available in standalone
        )

        results: list[CheckResult] = []
        current_category = ""
        suite_start = time.perf_counter()

        # Run all checks
        checks = [
            tester.check_self_test,
            tester.check_task_submission,
            tester.check_oati_loop,
            tester.check_model_router,
            tester.check_agent_dispatch,
            tester.check_analytics_recording,
            tester.check_event_log,
            tester.check_notification,
            tester.check_task_chain,
            tester.check_kill_switch,
            tester.check_backup,
            tester.check_telegram,
        ]

        for check_fn in checks:
            start = time.perf_counter()
            try:
                result = await check_fn()
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                result = CheckResult(
                    name=check_fn.__name__.replace("check_", "").replace("_", " ").title(),
                    passed=False,
                    message=f"Crashed: {type(e).__name__}: {e}",
                    duration_ms=round(elapsed, 1),
                    category="unknown",
                )
            results.append(result)

            # Print category header
            if result.category != current_category:
                current_category = result.category
                print(f"\n  {BOLD}{current_category.upper()}{RESET}")

            # Print result
            if result.passed:
                marker = f"{GREEN}PASS{RESET}"
            else:
                marker = f"{RED}FAIL{RESET}"

            duration = f"{DIM}{result.duration_ms:.0f}ms{RESET}"
            print(f"  [{marker}] {result.name} — {result.message} {duration}")

        # Summary
        suite_duration = (time.perf_counter() - suite_start) * 1000
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        print(f"\n{'=' * 55}")
        color = GREEN if failed == 0 else RED
        print(f"  {color}{BOLD}{passed}/{len(results)} passed{RESET}"
              f" {DIM}({suite_duration:.0f}ms){RESET}")

        if failed > 0:
            print(f"\n  {RED}Failed checks:{RESET}")
            for r in results:
                if not r.passed:
                    print(f"    - {r.name}: {r.message}")

        print()
        sys.exit(0 if failed == 0 else 1)


# ---------------------------------------------------------------------------
# CLI entry point: python -m tests.integration_test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(LaunchReadiness.run_standalone())
