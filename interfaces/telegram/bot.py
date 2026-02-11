"""
CAM Telegram Bot Interface

Gives George mobile access to CAM from anywhere via Telegram.
Supports task submission, status checks, agent listing, and the kill switch.

All messages route through the full OATI loop and are recorded in
episodic memory. Auth is enforced on every handler — only allowed
chat IDs can interact.

Usage:
    bot = TelegramBot(token="...", allowed_chat_ids=[123456], ...)
    await bot.start()    # non-blocking
    await bot.stop()
"""

import asyncio
import logging
from typing import Callable, Awaitable

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger("cam.telegram")


class TelegramBot:
    """Telegram bot interface for CAM.

    Connects to the Telegram Bot API via long-polling. All shared state
    (task queue, registry, memory, etc.) is injected via the constructor
    to avoid circular imports.

    If token is empty, start() returns immediately and the bot is disabled.
    The dashboard shows a red dot; no crash.

    Args:
        token:              Bot token from @BotFather. Empty = disabled.
        allowed_chat_ids:   List of integer chat IDs authorized to use the bot.
        task_queue:         Shared TaskQueue instance.
        registry:           Shared AgentRegistry instance.
        health_monitor:     Shared HealthMonitor instance.
        episodic_memory:    Shared EpisodicMemory instance.
        event_logger:       Shared EventLogger instance.
        activate_kill_switch: Async callable that triggers the kill switch.
        on_status_change:   Async callback to broadcast status to dashboards.
        poll_interval:      Seconds between task result checks (default 0.5).
        task_timeout:       Seconds to wait for a task result (default 120).
    """

    def __init__(
        self,
        token: str,
        allowed_chat_ids: list[int],
        task_queue,
        registry,
        health_monitor,
        episodic_memory,
        event_logger,
        activate_kill_switch: Callable[[], Awaitable[None]],
        on_status_change: Callable[[dict], Awaitable[None]],
        poll_interval: float = 0.5,
        task_timeout: float = 120.0,
    ):
        self._token = token
        self._allowed_ids = set(allowed_chat_ids)
        self._task_queue = task_queue
        self._registry = registry
        self._health_monitor = health_monitor
        self._episodic_memory = episodic_memory
        self._event_logger = event_logger
        self._activate_kill_switch = activate_kill_switch
        self._on_status_change = on_status_change
        self._poll_interval = poll_interval
        self._task_timeout = task_timeout

        self._app: Application | None = None
        self._connected = False
        # Track pending kill switch confirmations by chat_id
        self._pending_kill_confirms: set[int] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Initialize and start polling. Non-blocking.

        If token is empty, logs a warning and returns (bot disabled).
        Uses initialize() + start() + start_polling() instead of
        run_polling() so we don't block the event loop.
        """
        if not self._token:
            logger.warning("Telegram bot disabled — no bot_token configured")
            return

        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("task", self._cmd_task))
        self._app.add_handler(CommandHandler("agents", self._cmd_agents))
        self._app.add_handler(CommandHandler("kill", self._cmd_kill))
        self._app.add_handler(CallbackQueryHandler(self._kill_callback, pattern="^kill_"))
        # Free-text messages → treat as task submission
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        self._connected = True
        logger.info("Telegram bot started (polling)")
        self._event_logger.info("system", "Telegram bot connected and polling")

        await self._on_status_change({
            "type": "telegram_status",
            "connected": True,
        })

    async def stop(self):
        """Gracefully stop the bot and clean up."""
        if self._app is None:
            return

        try:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:
            logger.warning("Error stopping Telegram bot: %s", e)

        self._connected = False
        logger.info("Telegram bot stopped")

        await self._on_status_change({
            "type": "telegram_status",
            "connected": False,
        })

    @property
    def is_connected(self) -> bool:
        """Whether the bot is actively polling Telegram."""
        return self._connected

    def get_status(self) -> dict:
        """Return a status dict for the dashboard."""
        return {
            "connected": self._connected,
            "allowed_chat_ids": list(self._allowed_ids),
        }

    # ------------------------------------------------------------------
    # Auth guard — Constitution: security on every handler
    # ------------------------------------------------------------------

    def _is_authorized(self, update: Update) -> bool:
        """Check if the message sender is in the allowed chat IDs list."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        return chat_id is not None and chat_id in self._allowed_ids

    async def _reject_unauthorized(self, update: Update):
        """Send rejection message and log the security event."""
        chat_id = update.effective_chat.id if update.effective_chat else "unknown"
        user = update.effective_user
        username = user.username if user else "unknown"
        logger.warning("Unauthorized Telegram access from chat_id=%s user=%s", chat_id, username)
        self._event_logger.warn(
            "security",
            f"Unauthorized Telegram access: chat_id={chat_id}, user={username}",
            chat_id=str(chat_id),
            username=username,
        )
        await update.message.reply_text(
            "Access denied. This bot is restricted to authorized users."
        )

    # ------------------------------------------------------------------
    # Episodic memory recording
    # ------------------------------------------------------------------

    def _record_incoming(self, text: str, chat_id: int, task_id: str | None = None):
        """Record an incoming Telegram message in episodic memory."""
        self._episodic_memory.record(
            participant="user",
            content=text,
            context_tags=["telegram", "incoming"],
            task_id=task_id,
            metadata={"chat_id": chat_id, "source": "telegram"},
        )

    def _record_outgoing(self, text: str, chat_id: int, task_id: str | None = None):
        """Record an outgoing Telegram reply in episodic memory."""
        self._episodic_memory.record(
            participant="assistant",
            content=text,
            context_tags=["telegram", "outgoing"],
            task_id=task_id,
            metadata={"chat_id": chat_id, "source": "telegram"},
        )

    # ------------------------------------------------------------------
    # Task result polling
    # ------------------------------------------------------------------

    async def _wait_for_task_result(self, task) -> str | None:
        """Poll a task's status until COMPLETED/FAILED or timeout.

        Uses asyncio.sleep between checks so other handlers still process.

        Returns:
            The task result string, or a timeout/failure message.
        """
        from core.task import TaskStatus

        elapsed = 0.0
        while elapsed < self._task_timeout:
            if task.status == TaskStatus.COMPLETED:
                return str(task.result) if task.result is not None else "(no output)"
            if task.status == TaskStatus.FAILED:
                return f"Task failed: {task.result or 'unknown error'}"
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

        return f"Task timed out after {self._task_timeout:.0f}s (still {task.status.value})"

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start — greeting and command list."""
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        text = (
            "CAM online. Like the camshaft, not the social media thing.\n\n"
            "Commands:\n"
            "/status — System overview\n"
            "/task <description> — Submit a task\n"
            "/agents — List connected agents\n"
            "/kill — Emergency kill switch\n"
            "/help — Show this list\n\n"
            "Or just send a message and I'll treat it as a task."
        )
        await update.message.reply_text(text)
        self._record_incoming("/start", update.effective_chat.id)
        self._record_outgoing(text, update.effective_chat.id)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help — same as /start."""
        await self._cmd_start(update, context)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status — system overview summary."""
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        chat_id = update.effective_chat.id
        self._record_incoming("/status", chat_id)

        # Gather data from shared singletons
        agents = self._registry.list_all()
        online = [a for a in agents if a.status in ("online", "busy")]
        health = self._health_monitor.to_broadcast_dict()
        queue = self._task_queue.get_status()

        lines = ["CAM Status Report\n"]

        # Agents
        lines.append(f"Agents: {len(online)}/{len(agents)} online")
        for a in agents:
            indicator = "+" if a.status == "online" else ("~" if a.status == "busy" else "-")
            lines.append(f"  [{indicator}] {a.name} ({a.status})")

        # Task queue
        lines.append(
            f"\nTasks: {queue['pending']} pending, {queue['running']} running, "
            f"{queue['completed']} completed, {queue['failed']} failed"
        )

        # Health summary
        if health:
            lines.append("\nHealth:")
            for agent_id, metrics in health.items():
                success_rate = metrics.get("success_rate")
                rate_str = f"{success_rate:.0f}%" if success_rate is not None else "n/a"
                lines.append(f"  {agent_id}: {metrics.get('status', '?')} (success: {rate_str})")

        text = "\n".join(lines)
        await update.message.reply_text(text)
        self._record_outgoing(text, chat_id)

    async def _cmd_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /task <description> — submit a task to the OATI queue."""
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        chat_id = update.effective_chat.id
        # Everything after "/task " is the description
        description = " ".join(context.args) if context.args else ""
        if not description:
            await update.message.reply_text("Usage: /task <what you need done>")
            return

        self._record_incoming(f"/task {description}", chat_id)

        task = self._task_queue.add_task(description=description, source="telegram")
        ack = f"Task {task.short_id} queued."
        await update.message.reply_text(ack)
        self._record_outgoing(ack, chat_id, task_id=task.short_id)

        self._event_logger.info(
            "task",
            f"Task {task.short_id} submitted via Telegram: {description[:80]}",
            task_id=task.short_id,
            source="telegram",
        )

        # Wait for result
        result_text = await self._wait_for_task_result(task)
        # Telegram message limit is 4096 chars
        if len(result_text) > 4000:
            result_text = result_text[:4000] + "\n... (truncated)"
        await update.message.reply_text(result_text)
        self._record_outgoing(result_text, chat_id, task_id=task.short_id)

    async def _cmd_agents(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /agents — list all known agents with details."""
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        chat_id = update.effective_chat.id
        self._record_incoming("/agents", chat_id)

        agents = self._registry.list_all()
        if not agents:
            text = "No agents registered."
            await update.message.reply_text(text)
            self._record_outgoing(text, chat_id)
            return

        health = self._health_monitor.to_broadcast_dict()
        lines = [f"Agents ({len(agents)})\n"]

        for a in agents:
            indicator = "+" if a.status == "online" else ("~" if a.status == "busy" else "-")
            lines.append(f"[{indicator}] {a.name}")
            lines.append(f"    Status: {a.status}")
            lines.append(f"    IP: {a.ip_address}")
            if a.capabilities:
                lines.append(f"    Capabilities: {', '.join(a.capabilities)}")
            # Add task stats from health metrics
            agent_health = health.get(a.agent_id, {})
            completed = agent_health.get("tasks_completed", 0)
            failed = agent_health.get("tasks_failed", 0)
            if completed or failed:
                lines.append(f"    Tasks: {completed} completed, {failed} failed")
            lines.append("")

        text = "\n".join(lines).rstrip()
        await update.message.reply_text(text)
        self._record_outgoing(text, chat_id)

    async def _cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /kill — show confirmation keyboard before activating kill switch."""
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        chat_id = update.effective_chat.id
        self._record_incoming("/kill", chat_id)
        self._pending_kill_confirms.add(chat_id)

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("YES — KILL ALL AGENTS", callback_data="kill_confirm"),
                InlineKeyboardButton("Cancel", callback_data="kill_cancel"),
            ]
        ])
        await update.message.reply_text(
            "KILL SWITCH\n\nThis will halt ALL autonomous action across all agents. Confirm?",
            reply_markup=keyboard,
        )

    async def _kill_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard response for kill switch confirmation."""
        query = update.callback_query
        await query.answer()

        chat_id = query.message.chat_id if query.message else None
        if chat_id is None or chat_id not in self._allowed_ids:
            return

        if query.data == "kill_confirm" and chat_id in self._pending_kill_confirms:
            self._pending_kill_confirms.discard(chat_id)
            await query.edit_message_text("KILL SWITCH ACTIVATED. All agents halting.")
            self._record_outgoing("KILL SWITCH ACTIVATED via Telegram", chat_id)
            self._event_logger.error(
                "system",
                "KILL SWITCH activated via Telegram",
                source="telegram",
                chat_id=str(chat_id),
            )
            await self._activate_kill_switch()

        elif query.data == "kill_cancel":
            self._pending_kill_confirms.discard(chat_id)
            await query.edit_message_text("Kill switch cancelled.")
            self._record_outgoing("Kill switch cancelled.", chat_id)

    # ------------------------------------------------------------------
    # Free-text handler — treat as task submission
    # ------------------------------------------------------------------

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle any non-command text message as a task submission."""
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        chat_id = update.effective_chat.id
        description = update.message.text.strip()
        if not description:
            return

        self._record_incoming(description, chat_id)

        task = self._task_queue.add_task(description=description, source="telegram")
        ack = f"Task {task.short_id} queued."
        await update.message.reply_text(ack)
        self._record_outgoing(ack, chat_id, task_id=task.short_id)

        self._event_logger.info(
            "task",
            f"Task {task.short_id} submitted via Telegram: {description[:80]}",
            task_id=task.short_id,
            source="telegram",
        )

        # Wait for result
        result_text = await self._wait_for_task_result(task)
        if len(result_text) > 4000:
            result_text = result_text[:4000] + "\n... (truncated)"
        await update.message.reply_text(result_text)
        self._record_outgoing(result_text, chat_id, task_id=task.short_id)
