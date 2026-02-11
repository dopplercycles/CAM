"""
CAM System Self-Test Suite

Probes every subsystem and reports pass/fail status. Can be:
- Imported by server.py and triggered via dashboard button
- Run standalone via `python -m tests.self_test` for CLI checklist

Each test is a read-only probe — ephemeral test data is cleaned up
immediately. No subsystem state is modified permanently.

Usage (CLI):
    cd ~/CAM
    python -m tests.self_test

Usage (from server.py):
    from tests.self_test import SelfTest
    tester = SelfTest(config=config, event_logger=event_logger, ...)
    results = await tester.run_all()
"""

import asyncio
import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Any

logger = logging.getLogger("cam.self_test")


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result of a single self-test probe.

    Attributes:
        name:        Human-readable test name (e.g. "Config Loading")
        passed:      Whether the test succeeded
        message:     Detail on pass reason or failure description
        duration_ms: How long the test took in milliseconds
        category:    Grouping key: config | database | model | memory | core | network
    """
    name: str
    passed: bool
    message: str
    duration_ms: float
    category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# SelfTest
# ---------------------------------------------------------------------------

class SelfTest:
    """System self-test runner.

    Constructor takes optional shared state references from the dashboard
    server. When None, creates standalone instances for CLI mode.

    Args:
        config:             Parsed settings.toml config object
        event_logger:       EventLogger instance for logging results
        router:             ModelRouter instance for model tests
        registry:           AgentRegistry for agent registration tests
        task_queue:         TaskQueue for task queue tests
        short_term_memory:  ShortTermMemory instance
        working_memory:     WorkingMemory instance
        episodic_memory:    EpisodicMemory instance
        long_term_memory:   LongTermMemory instance
        analytics:          Analytics instance (SQLite)
        content_calendar:   ContentCalendar instance (SQLite)
        research_store:     ResearchStore instance (SQLite)
        security_audit_log: SecurityAuditLog instance (SQLite)
        agent_websockets:   Dict of agent_id -> WebSocket connections
        port:               Dashboard port for network tests
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
        content_calendar=None,
        research_store=None,
        security_audit_log=None,
        agent_websockets=None,
        port=None,
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
        self.content_calendar = content_calendar
        self.research_store = research_store
        self.security_audit_log = security_audit_log
        self.agent_websockets = agent_websockets or {}
        self.port = port or 8080

    # -------------------------------------------------------------------
    # Test runner
    # -------------------------------------------------------------------

    async def run_all(self) -> dict:
        """Run all tests sequentially, returning a summary dict.

        Each test is wrapped in try/except so a crashing test is recorded
        as a failure rather than aborting the entire suite.

        Returns:
            {
                "total": int,
                "passed": int,
                "failed": int,
                "duration_ms": float,
                "results": [TestResult.to_dict(), ...]
            }
        """
        tests = [
            self.test_config,
            self.test_db_analytics,
            self.test_db_episodic,
            self.test_db_content,
            self.test_db_research,
            self.test_db_security,
            self.test_ollama,
            self.test_model_query,
            self.test_task_queue,
            self.test_memory_short_term,
            self.test_memory_working,
            self.test_memory_episodic,
            self.test_memory_long_term,
            self.test_agent_registry,
            self.test_file_transfer,
            self.test_rest_api,
            self.test_websocket,
            self.test_agent_connectivity,
        ]

        results: list[TestResult] = []
        suite_start = time.perf_counter()

        for test_fn in tests:
            start = time.perf_counter()
            try:
                result = await test_fn()
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                result = TestResult(
                    name=test_fn.__name__.replace("test_", "").replace("_", " ").title(),
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
                    "self_test",
                    f"{'PASS' if result.passed else 'FAIL'}: {result.name} — {result.message}",
                    test=result.name,
                    category=result.category,
                    duration_ms=result.duration_ms,
                )

        suite_duration = (time.perf_counter() - suite_start) * 1000
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        summary = {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "duration_ms": round(suite_duration, 1),
            "results": [r.to_dict() for r in results],
        }

        if self.event_logger:
            self.event_logger.info(
                "self_test",
                f"Self-test complete: {passed}/{len(results)} passed in {suite_duration:.0f}ms",
                passed=passed,
                failed=failed,
                total=len(results),
            )

        return summary

    # -------------------------------------------------------------------
    # Individual tests
    # -------------------------------------------------------------------

    async def test_config(self) -> TestResult:
        """Parse settings.toml, verify key sections exist."""
        start = time.perf_counter()
        try:
            if self.config is None:
                from core.config import get_config
                cfg = get_config()
            else:
                cfg = self.config

            # Check key sections
            required = ["dashboard", "models", "auth", "security"]
            missing = []
            for section in required:
                if not hasattr(cfg, section):
                    missing.append(section)

            elapsed = (time.perf_counter() - start) * 1000
            if missing:
                return TestResult("Config Loading", False,
                                  f"Missing sections: {', '.join(missing)}",
                                  round(elapsed, 1), "config")
            return TestResult("Config Loading", True,
                              f"All {len(required)} required sections present",
                              round(elapsed, 1), "config")
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Config Loading", False, str(e),
                              round(elapsed, 1), "config")

    async def test_db_analytics(self) -> TestResult:
        """Open analytics.db, query task_records table."""
        return await self._test_sqlite_table(
            "Analytics DB", "analytics", "task_records",
            self.analytics, "database"
        )

    async def test_db_episodic(self) -> TestResult:
        """Open episodic.db, query episodes table."""
        return await self._test_sqlite_table(
            "Episodic DB", "episodic_memory", "episodes",
            self.episodic_memory, "database"
        )

    async def test_db_content(self) -> TestResult:
        """Open content_calendar.db, query content_entries table."""
        return await self._test_sqlite_table(
            "Content Calendar DB", "content_calendar", "content_entries",
            self.content_calendar, "database"
        )

    async def test_db_research(self) -> TestResult:
        """Open research.db, query research_entries table."""
        return await self._test_sqlite_table(
            "Research DB", "research_store", "research_entries",
            self.research_store, "database"
        )

    async def test_db_security(self) -> TestResult:
        """Open security_audit.db, query audit_log table."""
        return await self._test_sqlite_table(
            "Security Audit DB", "security_audit_log", "audit_log",
            self.security_audit_log, "database"
        )

    async def test_ollama(self) -> TestResult:
        """Check Ollama is running by fetching the model list."""
        start = time.perf_counter()
        try:
            # Get Ollama URL from config or router
            ollama_url = "http://localhost:11434"
            if self.config and hasattr(self.config, 'models'):
                ollama_url = getattr(self.config.models, 'ollama_url', ollama_url)
            elif self.router and hasattr(self.router, '_ollama_url'):
                ollama_url = self.router._ollama_url

            url = f"{ollama_url}/api/tags"
            req = urllib.request.Request(url, method="GET")

            def _fetch():
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            body = await asyncio.to_thread(_fetch)
            models = [m["name"] for m in body.get("models", [])]
            elapsed = (time.perf_counter() - start) * 1000

            return TestResult("Ollama Connection", True,
                              f"{len(models)} model(s) available",
                              round(elapsed, 1), "model")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Ollama Connection", False, str(e),
                              round(elapsed, 1), "model")

    async def test_model_query(self) -> TestResult:
        """Send a simple ping prompt through the model router."""
        start = time.perf_counter()
        try:
            if self.router is None:
                from core.model_router import ModelRouter
                router = ModelRouter()
            else:
                router = self.router

            response = await router.route("ping", task_complexity="simple")
            elapsed = (time.perf_counter() - start) * 1000

            if response.text and "[error]" not in response.text:
                return TestResult("Model Query", True,
                                  f"Response from {response.model} ({response.total_tokens} tokens)",
                                  round(elapsed, 1), "model")
            else:
                return TestResult("Model Query", False,
                                  f"Empty or error response: {response.text[:100]}",
                                  round(elapsed, 1), "model")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Model Query", False, str(e),
                              round(elapsed, 1), "model")

    async def test_task_queue(self) -> TestResult:
        """Add a task to the queue, verify it appears, then remove it."""
        start = time.perf_counter()
        try:
            if self.task_queue is None:
                from core.task import TaskQueue
                queue = TaskQueue()
            else:
                queue = self.task_queue

            before_count = len(queue.pending)
            task = queue.add_task(
                "_selftest_probe_",
                source="self_test",
            )
            after_count = len(queue.pending)

            # Clean up — remove the test task from the internal list
            queue._tasks = [t for t in queue._tasks if t.task_id != task.task_id]

            elapsed = (time.perf_counter() - start) * 1000

            if after_count > before_count:
                return TestResult("Task Queue", True,
                                  f"Add/remove OK ({before_count} -> {after_count} pending)",
                                  round(elapsed, 1), "core")
            else:
                return TestResult("Task Queue", False,
                                  f"Pending count did not increase ({before_count} -> {after_count})",
                                  round(elapsed, 1), "core")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Task Queue", False, str(e),
                              round(elapsed, 1), "core")

    async def test_memory_short_term(self) -> TestResult:
        """Add a message, verify context contains it, then clear."""
        start = time.perf_counter()
        try:
            if self.short_term_memory is None:
                from core.memory.short_term import ShortTermMemory
                stm = ShortTermMemory()
            else:
                stm = self.short_term_memory

            stm.add("system", "_selftest_probe_")
            context = stm.get_context()
            found = any("_selftest_probe_" in m.get("content", "") for m in context)

            # Clean up — remove the test message (it's the last one added)
            stm._messages = [m for m in stm._messages if "_selftest_probe_" not in m.content]

            elapsed = (time.perf_counter() - start) * 1000

            if found:
                return TestResult("Short-Term Memory", True,
                                  "Add/retrieve OK",
                                  round(elapsed, 1), "memory")
            else:
                return TestResult("Short-Term Memory", False,
                                  "Message not found in context after add",
                                  round(elapsed, 1), "memory")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Short-Term Memory", False, str(e),
                              round(elapsed, 1), "memory")

    async def test_memory_working(self) -> TestResult:
        """Save a task, retrieve it, then remove it."""
        start = time.perf_counter()
        try:
            if self.working_memory is None:
                from core.memory.working import WorkingMemory
                wm = WorkingMemory()
            else:
                wm = self.working_memory

            test_id = "_selftest_working_"
            wm.save_task(test_id, {"description": "self-test probe", "phase": "TEST"})
            retrieved = wm.get_task(test_id)
            wm.remove_task(test_id)

            elapsed = (time.perf_counter() - start) * 1000

            if retrieved and retrieved.get("description") == "self-test probe":
                return TestResult("Working Memory", True,
                                  "Save/retrieve/remove OK",
                                  round(elapsed, 1), "memory")
            else:
                return TestResult("Working Memory", False,
                                  "Retrieved data did not match saved data",
                                  round(elapsed, 1), "memory")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Working Memory", False, str(e),
                              round(elapsed, 1), "memory")

    async def test_memory_episodic(self) -> TestResult:
        """Record a test episode, search for it by keyword."""
        start = time.perf_counter()
        try:
            if self.episodic_memory is None:
                from core.memory.episodic import EpisodicMemory
                em = EpisodicMemory()
            else:
                em = self.episodic_memory

            ep = em.record("system", "_selftest_episodic_probe_",
                           context_tags=["self_test"])
            results = em.search(keyword="_selftest_episodic_probe_")
            found = any("_selftest_episodic_probe_" in r.content for r in results)

            # Clean up — delete the test episode
            cur = em._conn.cursor()
            cur.execute("DELETE FROM episodes WHERE episode_id = ?", (ep.episode_id,))
            em._conn.commit()

            elapsed = (time.perf_counter() - start) * 1000

            if found:
                return TestResult("Episodic Memory", True,
                                  "Record/search OK",
                                  round(elapsed, 1), "memory")
            else:
                return TestResult("Episodic Memory", False,
                                  "Recorded episode not found via search",
                                  round(elapsed, 1), "memory")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Episodic Memory", False, str(e),
                              round(elapsed, 1), "memory")

    async def test_memory_long_term(self) -> TestResult:
        """Store a test entry, query for it, then delete it."""
        start = time.perf_counter()
        try:
            if self.long_term_memory is None:
                from core.memory.long_term import LongTermMemory
                ltm = LongTermMemory()
            else:
                ltm = self.long_term_memory

            entry = ltm.store(
                "_selftest_ long term memory probe",
                category="system_learning",
                entry_id="_selftest_ltm_probe_",
            )

            if entry is None:
                elapsed = (time.perf_counter() - start) * 1000
                error = ltm._error if hasattr(ltm, '_error') and ltm._error else "store returned None"
                return TestResult("Long-Term Memory", False,
                                  f"Store failed: {error}",
                                  round(elapsed, 1), "memory")

            results = ltm.query("_selftest_", top_k=3)
            found = any("_selftest_" in r.content for r in results)

            # Clean up
            ltm.forget("_selftest_ltm_probe_")

            elapsed = (time.perf_counter() - start) * 1000

            if found:
                return TestResult("Long-Term Memory", True,
                                  "Store/query/forget OK",
                                  round(elapsed, 1), "memory")
            else:
                return TestResult("Long-Term Memory", False,
                                  "Stored entry not found via query",
                                  round(elapsed, 1), "memory")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Long-Term Memory", False, str(e),
                              round(elapsed, 1), "memory")

    async def test_agent_registry(self) -> TestResult:
        """Register a test agent, look it up, then deregister."""
        start = time.perf_counter()
        try:
            if self.registry is None:
                from core.agent_registry import AgentRegistry
                reg = AgentRegistry()
            else:
                reg = self.registry

            agent = reg.register(
                "_selftest_agent_",
                name="SelfTest Agent",
                ip_address="127.0.0.1",
                capabilities=["test"],
            )
            looked_up = reg.get_by_id("_selftest_agent_")
            reg.deregister("_selftest_agent_")

            # Clean up — remove from internal dict entirely
            reg._agents.pop("_selftest_agent_", None)

            elapsed = (time.perf_counter() - start) * 1000

            if looked_up and looked_up.name == "SelfTest Agent":
                return TestResult("Agent Registry", True,
                                  "Register/lookup/deregister OK",
                                  round(elapsed, 1), "core")
            else:
                return TestResult("Agent Registry", False,
                                  "Registered agent not found via get_by_id",
                                  round(elapsed, 1), "core")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Agent Registry", False, str(e),
                              round(elapsed, 1), "core")

    async def test_file_transfer(self) -> TestResult:
        """Verify compute_checksum and chunk_file_data work correctly."""
        start = time.perf_counter()
        try:
            from core.file_transfer import compute_checksum, chunk_file_data

            test_data = b"CAM self-test probe data"
            checksum = compute_checksum(test_data)
            chunks = chunk_file_data(test_data, chunk_size=8)

            elapsed = (time.perf_counter() - start) * 1000

            if not checksum.startswith("sha256:"):
                return TestResult("File Transfer Utils", False,
                                  f"Checksum format invalid: {checksum[:30]}",
                                  round(elapsed, 1), "core")

            if len(chunks) < 1:
                return TestResult("File Transfer Utils", False,
                                  "chunk_file_data returned no chunks",
                                  round(elapsed, 1), "core")

            return TestResult("File Transfer Utils", True,
                              f"Checksum OK, {len(chunks)} chunk(s) from {len(test_data)} bytes",
                              round(elapsed, 1), "core")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("File Transfer Utils", False, str(e),
                              round(elapsed, 1), "core")

    async def test_rest_api(self) -> TestResult:
        """HTTP GET /api/agents on the dashboard server."""
        start = time.perf_counter()
        try:
            url = f"http://localhost:{self.port}/api/agents"
            req = urllib.request.Request(url, method="GET")

            def _fetch():
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status, json.loads(resp.read().decode("utf-8"))

            status, body = await asyncio.to_thread(_fetch)
            elapsed = (time.perf_counter() - start) * 1000

            if status == 200 and isinstance(body, (dict, list)):
                return TestResult("REST API", True,
                                  f"GET /api/agents returned {status}",
                                  round(elapsed, 1), "network")
            else:
                return TestResult("REST API", False,
                                  f"Unexpected response: status={status}",
                                  round(elapsed, 1), "network")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("REST API", False, str(e),
                              round(elapsed, 1), "network")

    async def test_websocket(self) -> TestResult:
        """Open a WebSocket to the dashboard, receive initial message, close."""
        start = time.perf_counter()
        try:
            import websockets

            url = f"ws://localhost:{self.port}/ws/dashboard"

            async def _probe():
                async with websockets.connect(url, close_timeout=3) as ws:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    return json.loads(msg)

            data = await _probe()
            elapsed = (time.perf_counter() - start) * 1000

            msg_type = data.get("type", "unknown")
            return TestResult("WebSocket", True,
                              f"Connected, received '{msg_type}' message",
                              round(elapsed, 1), "network")

        except ImportError:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("WebSocket", False,
                              "websockets package not installed",
                              round(elapsed, 1), "network")
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("WebSocket", False, str(e),
                              round(elapsed, 1), "network")

    async def test_agent_connectivity(self) -> TestResult:
        """Check which registered agents have active WebSocket connections."""
        start = time.perf_counter()
        try:
            if self.registry is None:
                elapsed = (time.perf_counter() - start) * 1000
                return TestResult("Agent Connectivity", True,
                                  "No registry — skipped (standalone mode)",
                                  round(elapsed, 1), "network")

            all_agents = self.registry.list_all()
            # Filter out self-test agents
            real_agents = [a for a in all_agents if not a.agent_id.startswith("_selftest_")]

            if not real_agents:
                elapsed = (time.perf_counter() - start) * 1000
                return TestResult("Agent Connectivity", True,
                                  "No agents registered",
                                  round(elapsed, 1), "network")

            connected = []
            disconnected = []
            for agent in real_agents:
                if agent.agent_id in self.agent_websockets:
                    connected.append(agent.name)
                else:
                    disconnected.append(agent.name)

            elapsed = (time.perf_counter() - start) * 1000

            if disconnected:
                return TestResult("Agent Connectivity", False,
                                  f"{len(connected)} connected, "
                                  f"{len(disconnected)} missing WebSocket: "
                                  f"{', '.join(disconnected)}",
                                  round(elapsed, 1), "network")
            else:
                return TestResult("Agent Connectivity", True,
                                  f"All {len(connected)} agent(s) have active WebSocket connections",
                                  round(elapsed, 1), "network")

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult("Agent Connectivity", False, str(e),
                              round(elapsed, 1), "network")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    async def _test_sqlite_table(
        self,
        name: str,
        attr_name: str,
        table_name: str,
        instance: Any,
        category: str,
    ) -> TestResult:
        """Generic SQLite table probe — open connection, SELECT count(*)."""
        start = time.perf_counter()
        try:
            obj = instance
            if obj is None:
                elapsed = (time.perf_counter() - start) * 1000
                return TestResult(name, False,
                                  f"No {attr_name} instance provided",
                                  round(elapsed, 1), category)

            # Get the SQLite connection — different objects store it differently
            conn = getattr(obj, '_conn', None) or getattr(obj, '_db', None)
            if conn is None:
                elapsed = (time.perf_counter() - start) * 1000
                return TestResult(name, False,
                                  "Could not access database connection",
                                  round(elapsed, 1), category)

            cur = conn.cursor()
            cur.execute(f"SELECT count(*) FROM {table_name}")
            count = cur.fetchone()[0]

            elapsed = (time.perf_counter() - start) * 1000
            return TestResult(name, True,
                              f"{count} row(s) in {table_name}",
                              round(elapsed, 1), category)

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return TestResult(name, False, str(e),
                              round(elapsed, 1), category)

    # -------------------------------------------------------------------
    # Standalone CLI runner
    # -------------------------------------------------------------------

    @classmethod
    async def run_standalone(cls):
        """Run self-tests from the command line with colored output.

        Creates its own config, memories, router, etc.
        Skips network tests (no running server).
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

        print(f"\n{BOLD}CAM System Self-Test{RESET}")
        print("=" * 50)

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

        content_calendar = None
        try:
            from core.content_calendar import ContentCalendar
            db_path = "data/content_calendar.db"
            if config and hasattr(config, 'content_calendar'):
                db_path = getattr(config.content_calendar, 'db_path', db_path)
            content_calendar = ContentCalendar(db_path=db_path)
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create ContentCalendar: {e}{RESET}")

        research_store = None
        try:
            from core.research_store import ResearchStore
            db_path = "data/research.db"
            if config and hasattr(config, 'research'):
                db_path = getattr(config.research, 'db_path', db_path)
            research_store = ResearchStore(db_path=db_path)
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create ResearchStore: {e}{RESET}")

        security_audit_log = None
        try:
            from security.audit import SecurityAuditLog
            db_path = "data/security_audit.db"
            if config and hasattr(config, 'security'):
                db_path = getattr(config.security, 'audit_db_path', db_path)
            security_audit_log = SecurityAuditLog(db_path=db_path)
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not create SecurityAuditLog: {e}{RESET}")

        tester = cls(
            config=config,
            router=router,
            registry=registry,
            task_queue=task_queue,
            short_term_memory=short_term,
            working_memory=working,
            episodic_memory=episodic,
            long_term_memory=long_term,
            analytics=analytics,
            content_calendar=content_calendar,
            research_store=research_store,
            security_audit_log=security_audit_log,
        )

        # Run all tests — but define which are network tests to skip
        network_tests = {"test_rest_api", "test_websocket", "test_agent_connectivity"}
        all_tests = [
            tester.test_config,
            tester.test_db_analytics,
            tester.test_db_episodic,
            tester.test_db_content,
            tester.test_db_research,
            tester.test_db_security,
            tester.test_ollama,
            tester.test_model_query,
            tester.test_task_queue,
            tester.test_memory_short_term,
            tester.test_memory_working,
            tester.test_memory_episodic,
            tester.test_memory_long_term,
            tester.test_agent_registry,
            tester.test_file_transfer,
        ]

        results: list[TestResult] = []
        current_category = ""
        suite_start = time.perf_counter()

        for test_fn in all_tests:
            start = time.perf_counter()
            try:
                result = await test_fn()
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                result = TestResult(
                    name=test_fn.__name__.replace("test_", "").replace("_", " ").title(),
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

        # Print skipped network tests
        print(f"\n  {BOLD}NETWORK{RESET}")
        for test_name in network_tests:
            print(f"  [{YELLOW}SKIP{RESET}] {test_name.replace('test_', '').replace('_', ' ').title()} — "
                  f"requires running dashboard server")

        # Summary
        suite_duration = (time.perf_counter() - suite_start) * 1000
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        skipped = len(network_tests)

        print(f"\n{'=' * 50}")
        color = GREEN if failed == 0 else RED
        print(f"  {color}{BOLD}{passed}/{len(results)} passed{RESET}"
              f" {DIM}({skipped} skipped, {suite_duration:.0f}ms){RESET}")

        if failed > 0:
            print(f"\n  {RED}Failed tests:{RESET}")
            for r in results:
                if not r.passed:
                    print(f"    - {r.name}: {r.message}")

        print()
        sys.exit(0 if failed == 0 else 1)


# ---------------------------------------------------------------------------
# CLI entry point: python -m tests.self_test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(SelfTest.run_standalone())
