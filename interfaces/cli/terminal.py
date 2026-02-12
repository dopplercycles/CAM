"""
CAM CLI Interactive Terminal

A fast, lightweight REPL for interacting with CAM without the dashboard.
Natural language commands flow through the full OATI orchestrator loop,
with slash commands for system management. Rich library for formatted output.

Run with:
    cd ~/CAM
    python interfaces/cli/terminal.py

Or as a module:
    python -m interfaces.cli.terminal
"""

import asyncio
import logging
import sys
from functools import partial

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.agent_registry import AgentRegistry
from core.analytics import Analytics
from core.config import get_config
from core.content_calendar import ContentCalendar
from core.event_logger import EventLogger
from core.health_monitor import HealthMonitor
from core.memory import ShortTermMemory, WorkingMemory, LongTermMemory, EpisodicMemory
from core.orchestrator import Orchestrator
from core.persona import Persona
from core.research_store import ResearchStore
from core.task import TaskQueue, TaskComplexity
from agents.content_agent import ContentAgent
from agents.research_agent import ResearchAgent
from security.audit import SecurityAuditLog
from tests.self_test import SelfTest
from tools.content.tts_pipeline import TTSPipeline


# ---------------------------------------------------------------------------
# Logging — same pattern as server.py, but quieter for terminal use
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.cli")


# ---------------------------------------------------------------------------
# CAMTerminal
# ---------------------------------------------------------------------------

class CAMTerminal:
    """Interactive REPL for CAM — mirrors the dashboard's subsystem init.

    Provides a ``cam>`` prompt where George can type natural language
    (routed through the full orchestrator loop) or slash commands for
    system management.
    """

    def __init__(self):
        # ---- Configuration ------------------------------------------------
        self.config = get_config()
        self.console = Console()

        # ---- Shared subsystems (same init order as server.py) -------------
        self.registry = AgentRegistry(
            heartbeat_timeout=self.config.dashboard.heartbeat_timeout,
        )
        self.health_monitor = HealthMonitor(
            registry=self.registry,
            heartbeat_interval=self.config.health.heartbeat_interval,
        )
        self.event_logger = EventLogger(
            max_events=self.config.events.max_events,
        )
        self.analytics = Analytics(
            db_path=self.config.analytics.db_path,
        )

        # ---- Memory systems -----------------------------------------------
        self.short_term_memory = ShortTermMemory(
            max_messages=getattr(
                getattr(self.config, "memory", None),
                "short_term_max_messages", 200,
            ),
            summary_ratio=getattr(
                getattr(self.config, "memory", None),
                "short_term_summary_ratio", 0.5,
            ),
        )
        self.working_memory = WorkingMemory(
            persist_path=getattr(
                getattr(self.config, "memory", None),
                "working_memory_path", "data/tasks/working_memory.json",
            ),
        )
        self.episodic_memory = EpisodicMemory(
            db_path=getattr(
                getattr(self.config, "memory", None),
                "episodic_db_path", "data/memory/episodic.db",
            ),
            retention_days=getattr(
                getattr(self.config, "memory", None),
                "episodic_retention_days", 365,
            ),
        )
        self.long_term_memory = LongTermMemory(
            persist_directory=getattr(
                getattr(self.config, "memory", None),
                "long_term_persist_dir", "data/memory/chromadb",
            ),
            collection_name=getattr(
                getattr(self.config, "memory", None),
                "long_term_collection", "cam_long_term",
            ),
        )
        _seed_file = getattr(
            getattr(self.config, "memory", None),
            "long_term_seed_file", "CAM_BRAIN.md",
        )
        self.long_term_memory.seed_from_file(_seed_file)

        # ---- Persona, queues, logs ----------------------------------------
        self.persona = Persona()
        self.task_queue = TaskQueue()
        self.security_audit_log = SecurityAuditLog()
        self.content_calendar = ContentCalendar()
        self.research_store = ResearchStore()
        self.tts_pipeline = TTSPipeline(config=self.config)

        # ---- Orchestrator & agents (created in _startup) ------------------
        self.orchestrator: Orchestrator | None = None
        self.content_agent: ContentAgent | None = None
        self.research_agent: ResearchAgent | None = None

        # ---- Background task handle for the orchestrator loop -------------
        self._orchestrator_task: asyncio.Task | None = None

        # ---- Pending results: task_id → asyncio.Event ---------------------
        self._pending_results: dict[str, asyncio.Event] = {}

        # ---- Approval gate (non-blocking) ---------------------------------
        self._pending_approval: dict | None = None

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    async def run(self):
        """Start subsystems, run the REPL, then shut down cleanly."""
        await self._startup()
        try:
            await self._repl_loop()
        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            await self._shutdown()

    # -----------------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------------

    async def _startup(self):
        """Create the orchestrator with CLI-specific callbacks and start it."""

        # -- Callbacks for the orchestrator ---------------------------------

        def on_phase_change(task, phase, detail=""):
            """Print colored phase indicators to the terminal."""
            colors = {
                "OBSERVE": "cyan",
                "THINK": "yellow",
                "ACT": "blue",
                "ITERATE": "magenta",
                "COMPLETED": "green",
                "FAILED": "red",
                "BLOCKED": "red",
                "AWAITING_APPROVAL": "bright_yellow",
            }
            color = colors.get(phase, "white")
            self.console.print(
                f"  [{color}][{phase}][/{color}] {detail}" if detail else
                f"  [{color}][{phase}][/{color}]"
            )

        def on_task_update():
            """Check for completed/failed tasks and signal waiters."""
            for task in self.task_queue.list_all():
                if task.task_id in self._pending_results:
                    if task.status.value in ("completed", "failed"):
                        self._print_task_result(task)
                        event = self._pending_results.pop(task.task_id, None)
                        if event is not None:
                            event.set()

        def on_approval_request(task, perm_result):
            """Store approval info and print a Rich panel."""
            self._pending_approval = {
                "task_id": task.task_id,
                "short_id": task.short_id,
                "description": task.description,
                "action_type": perm_result.action_type,
                "risk_level": perm_result.risk_level,
                "reason": perm_result.reason,
            }
            panel = Panel(
                f"[bold]{perm_result.action_type}[/bold]\n"
                f"{task.description}\n\n"
                f"Risk: {perm_result.risk_level}  |  {perm_result.reason}\n\n"
                "[bold]Type y/yes to approve, n/no to reject[/bold]",
                title="Approval Required (Tier 2)",
                border_style="bright_yellow",
            )
            self.console.print(panel)

        def on_model_call(model, backend, tokens, latency_ms, cost_usd, task_short_id):
            """Record model call in analytics."""
            self.analytics.record_model_call(
                model=model,
                backend=backend,
                tokens=tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                task_short_id=task_short_id,
            )

        # -- Build the orchestrator -----------------------------------------

        self.orchestrator = Orchestrator(
            queue=self.task_queue,
            short_term_memory=self.short_term_memory,
            working_memory=self.working_memory,
            long_term_memory=self.long_term_memory,
            episodic_memory=self.episodic_memory,
            persona=self.persona,
            on_phase_change=on_phase_change,
            on_task_update=on_task_update,
            on_dispatch_to_agent=None,  # No remote agents in CLI mode
            on_model_call=on_model_call,
            on_approval_request=on_approval_request,
            audit_log=self.security_audit_log,
        )

        # -- Build local agents (same as server.py) -------------------------

        self.content_agent = ContentAgent(
            router=self.orchestrator.router,
            persona=self.persona,
            long_term_memory=self.long_term_memory,
            calendar=self.content_calendar,
            event_logger=self.event_logger,
            on_model_call=on_model_call,
            tts_pipeline=self.tts_pipeline,
        )

        self.research_agent = ResearchAgent(
            router=self.orchestrator.router,
            persona=self.persona,
            long_term_memory=self.long_term_memory,
            research_store=self.research_store,
            event_logger=self.event_logger,
            on_model_call=on_model_call,
        )

        # -- Start orchestrator loop as a background task -------------------

        self._orchestrator_task = asyncio.create_task(self.orchestrator.run())

        # -- Record session start in episodic memory ------------------------

        self.episodic_memory.record(
            "system",
            "CLI session started",
            context_tags=["cli", "session_start"],
        )

        # -- Print startup banner -------------------------------------------

        self._print_banner()

    # -----------------------------------------------------------------------
    # REPL loop
    # -----------------------------------------------------------------------

    async def _repl_loop(self):
        """Async input loop — reads from stdin via executor to stay non-blocking."""
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Non-blocking input via thread executor
                line = await loop.run_in_executor(None, partial(input, "cam> "))
            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /quit to exit.[/dim]")
                continue
            except EOFError:
                # Ctrl+D — clean exit
                break

            line = line.strip()
            if not line:
                continue

            # -- Check for pending approval first ---------------------------
            if self._pending_approval is not None:
                lower = line.lower()
                if lower in ("y", "yes"):
                    task_id = self._pending_approval["task_id"]
                    self.orchestrator.resolve_approval(task_id, True)
                    self.console.print("[green]Approved.[/green]")
                    self._pending_approval = None
                    continue
                elif lower in ("n", "no"):
                    task_id = self._pending_approval["task_id"]
                    self.orchestrator.resolve_approval(task_id, False)
                    self.console.print("[red]Rejected.[/red]")
                    self._pending_approval = None
                    continue

            # -- Record user input in episodic memory -----------------------
            self.episodic_memory.record(
                "user", line, context_tags=["cli"],
            )

            # -- Dispatch ---------------------------------------------------
            if line.startswith("/"):
                await self._dispatch_command(line)
            else:
                await self._submit_task(line)

    # -----------------------------------------------------------------------
    # Slash command dispatch
    # -----------------------------------------------------------------------

    async def _dispatch_command(self, line: str):
        """Route a slash command to the appropriate handler."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/status": self._cmd_status,
            "/agents": self._cmd_agents,
            "/task": self._cmd_task,
            "/memory": self._cmd_memory,
            "/history": self._cmd_history,
            "/kill": self._cmd_kill,
            "/config": self._cmd_config,
            "/test": self._cmd_test,
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler is None:
            self.console.print(f"[red]Unknown command: {cmd}[/red]  (try /help)")
            return

        await handler(arg)

    # -----------------------------------------------------------------------
    # Slash command implementations
    # -----------------------------------------------------------------------

    async def _cmd_status(self, _arg: str):
        """/status — Orchestrator status panel."""
        status = self.orchestrator.get_status()
        queue = status
        costs = status.get("session_costs", {})
        mem = status.get("memory", {})

        lines = []
        lines.append(f"[bold]Running:[/bold]  {'yes' if status['running'] else 'no'}")
        lines.append(
            f"[bold]Queue:[/bold]    {queue['pending']} pending, "
            f"{queue['running']} running, "
            f"{queue['completed']} completed, "
            f"{queue['failed']} failed"
        )
        lines.append(
            f"[bold]Costs:[/bold]    {costs.get('call_count', 0)} calls, "
            f"{costs.get('total_tokens', 0)} tokens, "
            f"${costs.get('total_cost_usd', 0):.4f}"
        )
        ltm_status = mem.get("long_term", {})
        ep_status = mem.get("episodic", {})
        lines.append(
            f"[bold]Memory:[/bold]   LTM {ltm_status.get('total_count', 0)} entries, "
            f"episodic {ep_status.get('total_count', 0)} episodes"
        )

        panel = Panel("\n".join(lines), title="CAM Status", border_style="cyan")
        self.console.print(panel)

    async def _cmd_agents(self, _arg: str):
        """/agents — Show all registered agents."""
        agents = self.registry.to_broadcast_list()
        if not agents:
            self.console.print("[dim]No agents registered.[/dim]")
            return

        table = Table(title="Agents")
        table.add_column("Name", style="bold")
        table.add_column("Status")
        table.add_column("IP")
        table.add_column("Capabilities")
        table.add_column("Last Heartbeat")

        for a in agents:
            status_style = {
                "online": "green",
                "offline": "red",
                "busy": "yellow",
            }.get(a["status"], "white")

            table.add_row(
                a["name"],
                f"[{status_style}]{a['status']}[/{status_style}]",
                a.get("ip_address", "?"),
                ", ".join(a.get("capabilities", [])) or "-",
                (a.get("last_heartbeat") or "-")[:19],
            )

        self.console.print(table)

    async def _cmd_task(self, arg: str):
        """/task <description> — Submit a task explicitly."""
        if not arg.strip():
            self.console.print("[red]Usage: /task <description>[/red]")
            return
        await self._submit_task(arg.strip())

    async def _cmd_memory(self, arg: str):
        """/memory <query> — Search long-term and episodic memory."""
        query = arg.strip()
        if not query:
            self.console.print("[red]Usage: /memory <query>[/red]")
            return

        # Long-term memory (semantic)
        ltm_results = self.long_term_memory.query(query, top_k=5)
        if ltm_results:
            table = Table(title="Long-Term Memory")
            table.add_column("Category", style="cyan")
            table.add_column("Score", justify="right")
            table.add_column("Content")

            for entry in ltm_results:
                table.add_row(
                    entry.category,
                    f"{entry.score:.2f}",
                    entry.content[:120] + ("..." if len(entry.content) > 120 else ""),
                )
            self.console.print(table)
        else:
            self.console.print("[dim]No long-term memory results.[/dim]")

        # Episodic memory (keyword)
        ep_results = self.episodic_memory.search(keyword=query, limit=10)
        if ep_results:
            table = Table(title="Episodic Memory")
            table.add_column("Time", style="dim")
            table.add_column("Who")
            table.add_column("Content")

            for ep in ep_results:
                table.add_row(
                    ep.timestamp[:19],
                    ep.participant,
                    ep.content[:120] + ("..." if len(ep.content) > 120 else ""),
                )
            self.console.print(table)
        else:
            self.console.print("[dim]No episodic memory results.[/dim]")

    async def _cmd_history(self, _arg: str):
        """/history — Show recent episodic memory entries."""
        episodes = self.episodic_memory.search(limit=20)
        if not episodes:
            self.console.print("[dim]No history yet.[/dim]")
            return

        table = Table(title="Recent History")
        table.add_column("Time", style="dim")
        table.add_column("Who")
        table.add_column("Content")

        for ep in episodes:
            table.add_row(
                ep.timestamp[:19],
                ep.participant,
                ep.content[:100] + ("..." if len(ep.content) > 100 else ""),
            )

        self.console.print(table)

    async def _cmd_kill(self, _arg: str):
        """/kill — Stop the orchestrator (requires confirmation)."""
        self.console.print(
            "[bold red]This will stop the orchestrator loop.[/bold red]\n"
            "Type CONFIRM to proceed:"
        )
        loop = asyncio.get_event_loop()
        try:
            confirm = await loop.run_in_executor(None, partial(input, "  > "))
        except (KeyboardInterrupt, EOFError):
            self.console.print("[dim]Cancelled.[/dim]")
            return

        if confirm.strip() == "CONFIRM":
            self.orchestrator.stop()
            self.console.print("[red]Orchestrator stopped.[/red]")
        else:
            self.console.print("[dim]Cancelled.[/dim]")

    async def _cmd_config(self, _arg: str):
        """/config — Show current configuration (masks sensitive values)."""
        data = self.config.to_dict()

        table = Table(title="Configuration")
        table.add_column("Section", style="bold")
        table.add_column("Key")
        table.add_column("Value")

        sensitive_substrings = ("key", "token", "secret", "password", "api_key")

        for section, values in sorted(data.items()):
            if isinstance(values, dict):
                for key, val in sorted(values.items()):
                    display_val = str(val)
                    # Mask sensitive values
                    if any(s in key.lower() for s in sensitive_substrings):
                        if val and str(val).strip():
                            display_val = "[set]"
                    # Truncate long nested dicts
                    if len(display_val) > 80:
                        display_val = display_val[:77] + "..."
                    table.add_row(section, key, display_val)
            else:
                table.add_row(section, "-", str(values))

        self.console.print(table)

    async def _cmd_test(self, _arg: str):
        """/test — Run the self-test suite."""
        self.console.print("[dim]Running self-tests...[/dim]")

        tester = SelfTest(
            config=self.config,
            event_logger=self.event_logger,
            router=self.orchestrator.router,
            registry=self.registry,
            task_queue=self.task_queue,
            short_term_memory=self.short_term_memory,
            working_memory=self.working_memory,
            episodic_memory=self.episodic_memory,
            long_term_memory=self.long_term_memory,
            analytics=self.analytics,
            content_calendar=self.content_calendar,
            research_store=self.research_store,
            security_audit_log=self.security_audit_log,
        )

        results = await tester.run_all()

        table = Table(title=f"Self-Test Results ({results['passed']}/{results['total']} passed)")
        table.add_column("Test", style="bold")
        table.add_column("Result")
        table.add_column("Message")
        table.add_column("Time", justify="right")

        for r in results["results"]:
            style = "green" if r["passed"] else "red"
            icon = "PASS" if r["passed"] else "FAIL"
            table.add_row(
                r["name"],
                f"[{style}]{icon}[/{style}]",
                r["message"][:80],
                f"{r['duration_ms']:.0f}ms",
            )

        self.console.print(table)
        self.console.print(
            f"[dim]Total: {results['duration_ms']:.0f}ms[/dim]"
        )

    async def _cmd_help(self, _arg: str):
        """/help — Show all available slash commands."""
        table = Table(title="Commands")
        table.add_column("Command", style="bold cyan")
        table.add_column("Description")

        commands = [
            ("/status", "Show orchestrator state, queue counts, costs, memory stats"),
            ("/agents", "List all registered agents with status and capabilities"),
            ("/task <desc>", "Submit a task explicitly (same as typing naturally)"),
            ("/memory <query>", "Search long-term and episodic memory"),
            ("/history", "Show the 20 most recent episodic memory entries"),
            ("/kill", "Stop the orchestrator (requires CONFIRM)"),
            ("/config", "Show current configuration (sensitive values masked)"),
            ("/test", "Run the system self-test suite"),
            ("/help", "Show this help table"),
            ("/quit, /exit", "Exit the CLI terminal"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)
        self.console.print(
            "[dim]Or just type a question — it goes through the full orchestrator loop.[/dim]"
        )

    async def _cmd_quit(self, _arg: str):
        """/quit or /exit — Trigger clean shutdown."""
        raise SystemExit

    # -----------------------------------------------------------------------
    # Task submission
    # -----------------------------------------------------------------------

    async def _submit_task(self, text: str):
        """Submit natural language to the orchestrator via the task queue."""
        task = self.task_queue.add_task(
            text,
            source="cli",
            complexity=TaskComplexity.AUTO,
        )

        # Record submission in episodic memory
        self.episodic_memory.record(
            "user",
            f"Task submitted: {text}",
            context_tags=["cli", "task_submit"],
            task_id=task.task_id,
        )

        # Set up a waiter so we know when the task finishes
        event = asyncio.Event()
        self._pending_results[task.task_id] = event

        # Wait with a spinner (5-minute timeout)
        try:
            with self.console.status("[bold cyan]Processing...[/bold cyan]"):
                await asyncio.wait_for(event.wait(), timeout=300.0)
        except asyncio.TimeoutError:
            self._pending_results.pop(task.task_id, None)
            self.console.print("[red]Task timed out after 5 minutes.[/red]")

    # -----------------------------------------------------------------------
    # Result display
    # -----------------------------------------------------------------------

    def _print_task_result(self, task):
        """Print a task result in a Rich panel."""
        if task.status.value == "completed":
            content = Markdown(task.result or "(no output)")
            panel = Panel(
                content,
                title=f"[green]Completed[/green] [{task.short_id}]",
                border_style="green",
            )
        else:
            panel = Panel(
                task.result or "(unknown error)",
                title=f"[red]Failed[/red] [{task.short_id}]",
                border_style="red",
            )
        self.console.print(panel)

    # -----------------------------------------------------------------------
    # Startup banner
    # -----------------------------------------------------------------------

    def _print_banner(self):
        """Print the startup banner with system info."""
        greeting = self.persona.get_greeting()
        status = self.orchestrator.get_status()
        ltm_count = status["memory"]["long_term"].get("total_count", 0)
        queue = status

        lines = [
            "[bold]CAM — Cycles Autonomous Manager[/bold]",
            "",
            f"[italic]{greeting}[/italic]",
            "",
            f"Orchestrator: {'running' if status['running'] else 'stopped'}  |  "
            f"Queue: {queue['pending']} pending  |  "
            f"LTM: {ltm_count} entries",
            "",
            "[dim]Type /help for commands, or just ask a question.[/dim]",
        ]

        panel = Panel(
            "\n".join(lines),
            border_style="bright_blue",
            padding=(1, 2),
        )
        self.console.print(panel)

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    async def _shutdown(self):
        """Stop the orchestrator and close all subsystems."""
        # 1. Stop orchestrator loop
        if self.orchestrator is not None:
            self.orchestrator.stop()
        if self._orchestrator_task is not None:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass

        # 2. Record session end
        self.episodic_memory.record(
            "system",
            "CLI session ended",
            context_tags=["cli", "session_end"],
        )

        # 3. Close subsystems with SQLite connections
        self.episodic_memory.close()
        self.analytics.close()
        self.content_calendar.close()
        self.research_store.close()
        self.security_audit_log.close()

        # 4. Goodbye
        self.console.print("[dim]Goodbye.[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    """Launch the CAM CLI terminal."""
    terminal = CAMTerminal()
    await terminal.run()


if __name__ == "__main__":
    asyncio.run(main())
