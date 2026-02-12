"""
CAM Dashboard Server

FastAPI backend serving the agent status dashboard.
Agents connect via WebSocket to report their status.
The dashboard displays real-time agent status to George.

Run with:
    cd ~/CAM
    uvicorn interfaces.dashboard.server:app --host 0.0.0.0 --port 8080 --reload
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from core.agent_registry import AgentRegistry
from core.config import get_config
from core.event_logger import EventLogger
from core.backup import BackupManager
from core.message_bus import MessageBus
from core.deployer import Deployer
from core.file_transfer import FileTransferManager, chunk_file_data, compute_checksum
from core.health_monitor import HealthMonitor
from core.memory import ShortTermMemory, WorkingMemory, LongTermMemory, EpisodicMemory
from core.notifications import NotificationManager
from core.persona import Persona
from interfaces.telegram.bot import TelegramBot
from core.task import TaskQueue, TaskComplexity, TaskChain, Task, ChainStatus
from core.orchestrator import Orchestrator
from core.context_manager import ContextManager
from core.analytics import Analytics
from core.commands import CommandLibrary
from core.scheduler import Scheduler, ScheduleType
from core.content_calendar import ContentCalendar, ContentType, ContentStatus
from core.research_store import ResearchStore, ResearchStatus
from tools.doppler.scout import ScoutStore, DopplerScout, SearchCriteria, ListingStatus
from security.audit import SecurityAuditLog
from security.auth import SessionManager
from agents.content_agent import ContentAgent
from tools.content.tts_pipeline import TTSPipeline
from agents.research_agent import ResearchAgent
from agents.business_agent import BusinessAgent, BusinessStore
from tests.self_test import SelfTest


# ---------------------------------------------------------------------------
# Logging — CLAUDE.md says "comprehensive logging, if it happened there's a record"
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.dashboard")


# ---------------------------------------------------------------------------
# Configuration — loaded once at startup, hot-reloadable via dashboard
# ---------------------------------------------------------------------------

config = get_config()

# Path to static files directory (index.html lives here)
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Shared state
#
# The registry is the single source of truth for agent status.
# WebSocket connections and dashboard clients are server-specific.
# ---------------------------------------------------------------------------

# Agent registry — tracks all agents, shared with orchestrator later
# Note: heartbeat_timeout is read at construction — needs restart to change.
registry = AgentRegistry(heartbeat_timeout=config.dashboard.heartbeat_timeout)

# Health monitor — per-agent health metrics, 3-miss heartbeat policy
health_monitor = HealthMonitor(registry=registry, heartbeat_interval=config.health.heartbeat_interval)

# Event logger — centralized audit trail for all CAM activity
# Note: max_events is read at construction — needs restart to change.
event_logger = EventLogger(max_events=config.events.max_events)

# Message bus — inter-agent pub/sub communication
message_bus = MessageBus(
    max_messages=config.message_bus.max_messages,
    event_logger=event_logger,
)

# Analytics — SQLite-backed task history and model cost tracking
# Note: db_path is read at construction — needs restart to change.
analytics = Analytics(db_path=config.analytics.db_path)

# Command library — predefined commands for the dashboard command palette
command_library = CommandLibrary()

# Notification manager — evaluates events against configurable rules
notification_manager = NotificationManager(max_history=config.notifications.max_history)

# Memory systems — session context and persistent task state
short_term_memory = ShortTermMemory(
    max_messages=getattr(getattr(config, 'memory', None), 'short_term_max_messages', 200),
    summary_ratio=getattr(getattr(config, 'memory', None), 'short_term_summary_ratio', 0.5),
)
working_memory = WorkingMemory(
    persist_path=getattr(getattr(config, 'memory', None), 'working_memory_path',
                         'data/tasks/working_memory.json'),
)

# Persona — Cam's identity, voice, and behavioral traits (YAML-driven)
persona = Persona()

# Telegram bot — initialized in lifespan() after activate_kill_switch is defined
telegram_bot: TelegramBot | None = None

# Episodic memory — SQLite-backed conversation history
episodic_memory = EpisodicMemory(
    db_path=getattr(getattr(config, 'memory', None),
                    'episodic_db_path', 'data/memory/episodic.db'),
    retention_days=getattr(getattr(config, 'memory', None),
                           'episodic_retention_days', 365),
)

# Long-term memory — ChromaDB vector store for persistent knowledge
long_term_memory = LongTermMemory(
    persist_directory=getattr(getattr(config, 'memory', None),
                              'long_term_persist_dir', 'data/memory/chromadb'),
    collection_name=getattr(getattr(config, 'memory', None),
                            'long_term_collection', 'cam_long_term'),
)
_seed_file = getattr(getattr(config, 'memory', None),
                     'long_term_seed_file', 'CAM_BRAIN.md')
long_term_memory.seed_from_file(_seed_file)

# agent_id -> WebSocket (live connections, needed for commands + kill switch)
agent_websockets: dict[str, WebSocket] = {}

# Browser clients watching the dashboard (receive status pushes)
dashboard_clients: list[WebSocket] = []

# Task queue — shared between dashboard and orchestrator
task_queue = TaskQueue()

# Round-robin counter for fair agent dispatch
_dispatch_counter: int = 0

# Pending task dispatch responses — task_id → asyncio.Future
# Used by dispatch_to_agent() to wait for agent command_responses
pending_task_responses: dict[str, asyncio.Future] = {}

# Kill switch state — when True, all autonomous action is halted
kill_switch_active: bool = False

# Last self-test results cache — populated when dashboard runs self-test
last_self_test_results: dict | None = None

# Background task handle for the orchestrator loop
orchestrator_task: asyncio.Task | None = None

# File transfer manager — tracks active/completed file transfers
ft_manager: FileTransferManager | None = None  # initialized after broadcast_transfer_progress is defined


# ---------------------------------------------------------------------------
# Broadcast helper
# ---------------------------------------------------------------------------

async def broadcast_to_dashboards(message: dict):
    """Push a message to all connected dashboard browsers.

    Handles cleanup of any clients that have disconnected.
    Used by both agent status broadcasts and command responses.
    """
    disconnected = []
    for client in dashboard_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        dashboard_clients.remove(client)


async def broadcast_agent_status():
    """Push current agent status to all connected dashboard browsers.

    Reads from the registry — the single source of truth.
    """
    data = registry.to_broadcast_list()
    await broadcast_to_dashboards({"type": "agent_status", "agents": data})


async def broadcast_task_status():
    """Push current task queue state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "task_status",
        "tasks": task_queue.to_broadcast_list(),
        "counts": task_queue.get_status(),
    })


async def broadcast_health_status():
    """Push current agent health metrics to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "health_status",
        "health": health_monitor.to_broadcast_dict(),
    })


async def broadcast_event(event_dict: dict):
    """Push a single new event to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "event",
        "event": event_dict,
    })


async def broadcast_notification(notification_dict: dict):
    """Push a new notification to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "notification",
        "notification": notification_dict,
        "unread_count": notification_manager.get_unread_count(),
    })


# Wire notification manager's broadcast callback
notification_manager.set_broadcast_callback(broadcast_notification)


# Wire the event logger's broadcast callback — also evaluates notification rules
async def broadcast_event_and_evaluate(event_dict: dict):
    """Push event to dashboards and evaluate notification rules."""
    await broadcast_event(event_dict)
    notification_manager.evaluate_event(event_dict)

event_logger.set_broadcast_callback(broadcast_event_and_evaluate)


# Wire the message bus broadcast callback — pushes bus messages to dashboards
async def broadcast_bus_message(message_dict):
    """Push a bus message + stats to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "bus_message",
        "message": message_dict,
        "stats": message_bus.get_stats(),
    })

message_bus.set_broadcast_callback(broadcast_bus_message)


async def broadcast_analytics():
    """Push current analytics summary to all connected dashboard browsers."""
    await broadcast_to_dashboards({"type": "analytics", "data": analytics.get_summary()})


async def broadcast_chain_status():
    """Push current chain state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "chain_status",
        "chains": task_queue.chains_to_broadcast_list(),
    })


async def broadcast_transfer_progress(transfer):
    """Push file transfer progress to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "file_transfer_progress",
        **transfer.to_dict(),
    })


# Now initialize the file transfer manager with the broadcast callback
ft_manager = FileTransferManager(
    on_progress=broadcast_transfer_progress,
    history_size=config.file_transfer.history_size,
    max_active=config.file_transfer.max_active_transfers,
)

# Ensure server-side receive directory exists
Path(config.file_transfer.receive_dir).mkdir(parents=True, exist_ok=True)


async def broadcast_memory_status():
    """Push current memory system state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "memory_status",
        "short_term": short_term_memory.get_status(),
        "working": working_memory.get_status(),
        "long_term": long_term_memory.get_status(),
        "episodic": episodic_memory.get_status(),
    })


async def broadcast_schedule_status():
    """Push current schedule state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "schedule_status",
        "schedules": scheduler.to_broadcast_list(),
    })


async def broadcast_content_calendar_status():
    """Push current content calendar state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "content_calendar_status",
        "entries": content_calendar.to_broadcast_list(),
    })


async def broadcast_research_status():
    """Push current research results to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "research_status",
        "entries": research_store.to_broadcast_list(),
    })


# Content calendar — SQLite-backed content pipeline tracker
content_calendar = ContentCalendar(
    db_path=getattr(getattr(config, 'content_calendar', None),
                    'db_path', 'data/content_calendar.db'),
    on_change=broadcast_content_calendar_status,
)

# Research store — SQLite-backed research result storage
research_store = ResearchStore(
    db_path=getattr(getattr(config, 'research', None),
                    'db_path', 'data/research.db'),
    on_change=broadcast_research_status,
)


async def broadcast_scout_status():
    """Push current scout listings to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "scout_status",
        "entries": scout_store.to_broadcast_list(),
        "status": scout_store.get_status(),
    })


# Scout store — SQLite-backed motorcycle listing storage
scout_store = ScoutStore(
    db_path=getattr(getattr(config, 'scout', None),
                    'db_path', 'data/scout.db'),
    on_change=broadcast_scout_status,
)


async def broadcast_business_status():
    """Push current business data to all connected dashboard browsers."""
    data = business_store.to_broadcast_list()
    await broadcast_to_dashboards({
        "type": "business_status",
        **data,
        "status": business_store.get_status(),
    })


# Business store — SQLite-backed customer/appointment/invoice/inventory storage
_biz_cfg = getattr(config, 'business', None)
business_store = BusinessStore(
    db_path=getattr(_biz_cfg, 'db_path', 'data/business.db'),
    on_change=broadcast_business_status,
)
if getattr(_biz_cfg, 'seed_sample_data', True):
    business_store.seed_sample_data()


async def broadcast_security_audit_status():
    """Push current security audit state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "security_audit_status",
        "entries": security_audit_log.to_broadcast_list(),
        "status": security_audit_log.get_status(),
    })


# Security audit log — SQLite-backed persistent audit trail
security_audit_log = SecurityAuditLog(
    db_path=getattr(getattr(config, 'security', None),
                    'audit_db_path', 'data/security_audit.db'),
    on_change=broadcast_security_audit_status,
)

# Session manager — dashboard browser authentication
# When password_hash is empty, auth is disabled (open access, backward compatible)
session_manager = SessionManager(
    username=getattr(getattr(config, 'auth', None), 'username', 'george'),
    password_hash=getattr(getattr(config, 'auth', None), 'password_hash', ''),
    session_timeout=getattr(getattr(config, 'auth', None), 'session_timeout', 3600),
    max_login_attempts=getattr(getattr(config, 'auth', None), 'max_login_attempts', 5),
    lockout_duration=getattr(getattr(config, 'auth', None), 'lockout_duration', 300),
)


# Task scheduler — persists schedules to JSON, submits due tasks to the queue
scheduler = Scheduler(
    task_queue=task_queue,
    persist_path=config.scheduler.persist_file,
    check_interval=config.scheduler.check_interval,
    on_schedule_change=broadcast_schedule_status,
)

# Seed daily backup schedule if it doesn't already exist
if not any(s.name == "Daily Backup" for s in scheduler._schedules.values()):
    scheduler.add_schedule(
        name="Daily Backup",
        description="system_backup",
        schedule_type=ScheduleType.DAILY,
        run_at_time=config.backup.daily_backup_time,
    )


async def relay_file_to_agent(transfer, data: bytes, agent_ws):
    """Send file data to an agent in chunks over WebSocket.

    Launched as a background task from the upload endpoint or WS handler.
    Sends file_transfer_start, then iterates chunks with brief yields.
    The agent acks chunks asynchronously; acks update progress via the
    agent WS handler.
    """
    chunk_size = config.file_transfer.chunk_size
    chunks = chunk_file_data(data, chunk_size)

    try:
        # Tell the agent a transfer is starting
        await agent_ws.send_json({
            "type": "file_transfer_start",
            "transfer_id": transfer.transfer_id,
            "filename": transfer.filename,
            "file_size": transfer.file_size,
            "chunk_count": transfer.chunk_count,
            "dest_path": transfer.dest_path,
            "checksum": transfer.checksum,
        })

        # Stream chunks
        for i, b64_chunk in enumerate(chunks):
            await agent_ws.send_json({
                "type": "file_chunk",
                "transfer_id": transfer.transfer_id,
                "chunk_index": i,
                "chunk_count": transfer.chunk_count,
                "data": b64_chunk,
            })
            await asyncio.sleep(0)  # yield to event loop between chunks

    except Exception as e:
        logger.warning("Relay to agent failed for %s: %s", transfer.transfer_id, e)
        await ft_manager.fail_transfer(transfer.transfer_id, f"Relay failed: {e}")


# Agent deployer — orchestrates connector updates via existing file transfer + command systems
deployer = Deployer(
    ft_manager=ft_manager,
    event_logger=event_logger,
    registry=registry,
    agent_websockets=agent_websockets,
    relay_fn=relay_file_to_agent,
    config=config,
)

# Backup manager — creates/restores/rotates backup archives
backup_manager = BackupManager(config=config, event_logger=event_logger)


# ---------------------------------------------------------------------------
# Orchestrator callbacks — push OATI phase changes to dashboards in real time
# ---------------------------------------------------------------------------

async def on_task_phase_change(task, phase, detail):
    """Called by the orchestrator at each OATI phase boundary."""
    await broadcast_to_dashboards({
        "type": "task_phase",
        "task_id": task.task_id,
        "short_id": task.short_id,
        "phase": phase,
        "detail": detail,
    })
    severity = "error" if phase == "FAILED" else "info"
    getattr(event_logger, severity)(
        "task", f"[{phase}] {detail or task.short_id}",
        task_id=task.short_id, phase=phase,
    )

    # Record completed/failed tasks in analytics DB and push to dashboards
    if phase in ("COMPLETED", "FAILED"):
        analytics.record_task(task)
        await broadcast_analytics()
        await message_bus.publish("system", "task_updates", {
            "action": "phase_change", "task_id": task.short_id, "phase": phase,
        })

    # Push updated memory status to dashboards (memory changes during OATI)
    await broadcast_memory_status()


async def on_task_update():
    """Called by the orchestrator when any task's status changes."""
    await broadcast_task_status()


async def on_chain_update(chain):
    """Called by the orchestrator when a chain's status changes."""
    await broadcast_chain_status()
    status_str = chain.status.value if hasattr(chain.status, 'value') else str(chain.status)
    severity = "error" if status_str == "failed" else "info"
    getattr(event_logger, severity)(
        "chain",
        f"Chain '{chain.name}' ({chain.short_id}): {status_str} "
        f"— step {min(chain.current_step + 1, chain.total_steps)}/{chain.total_steps}",
        chain_id=chain.short_id, status=status_str,
    )


# ---------------------------------------------------------------------------
# Agent dispatch — sends tasks to remote agents, waits for response
# ---------------------------------------------------------------------------

# Dispatch timeout is read from config at call time (hot-reloadable)


async def dispatch_to_agent(task, plan):
    """Dispatch a task to an available remote agent.

    Routing logic:
    1. If task.assigned_agent is set and that agent is online → use it
    2. Otherwise → pick the first available agent from the registry
    3. If no agents available → return None (orchestrator falls back to model response)

    Sends the task description as a command via the agent's WebSocket,
    then waits for the agent's command_response with a matching task_id.

    Args:
        task: The Task being executed.
        plan: The plan dict from the THINK phase (unused for now, available for future).

    Returns:
        The agent's response string, or None if no agent was available.
    """
    # --- Local agents first — content agent handles content tasks without network overhead ---
    try:
        content_result = await content_agent.try_handle(task, plan)
        if content_result is not None:
            task.assigned_agent = "content_agent"
            event_logger.info("task", f"Content agent handled {task.short_id}",
                              task_id=task.short_id, agent="content_agent")
            await message_bus.publish("content_agent", "task_updates", {
                "action": "task_handled", "task_id": task.short_id,
            })
            return content_result
    except Exception as e:
        logger.warning("Content agent failed for task %s: %s", task.short_id, e)

    # --- Local agents: research agent handles research tasks ---
    try:
        research_result = await research_agent.try_handle(task, plan)
        if research_result is not None:
            task.assigned_agent = "research_agent"
            event_logger.info("task", f"Research agent handled {task.short_id}",
                              task_id=task.short_id, agent="research_agent")
            await message_bus.publish("research_agent", "task_updates", {
                "action": "task_handled", "task_id": task.short_id,
            })
            return research_result
    except Exception as e:
        logger.warning("Research agent failed for task %s: %s", task.short_id, e)

    # --- Local agents: business agent handles business tasks ---
    try:
        business_result = await business_agent.try_handle(task, plan)
        if business_result is not None:
            task.assigned_agent = "business_agent"
            event_logger.info("task", f"Business agent handled {task.short_id}",
                              task_id=task.short_id, agent="business_agent")
            await message_bus.publish("business_agent", "task_updates", {
                "action": "task_handled", "task_id": task.short_id,
            })
            return business_result
    except Exception as e:
        logger.warning("Business agent failed for task %s: %s", task.short_id, e)

    # --- System tasks: backup ---
    if task.description == "system_backup":
        task.assigned_agent = "backup_manager"
        result = backup_manager.backup()
        if result.get("ok"):
            return f"Backup complete: {result['filename']} ({result['file_count']} files)"
        return f"Backup failed: {result.get('error', 'unknown error')}"

    # --- Routing: find the target agent ---
    agent_id = None
    required_caps = getattr(task, "required_capabilities", []) or []

    if task.assigned_agent:
        # Check if the pre-assigned agent is online, connected, and capable
        agent_info = registry.get_by_id(task.assigned_agent)
        if agent_info and agent_info.status == "online" and task.assigned_agent in agent_websockets:
            if not required_caps or set(required_caps).issubset(set(agent_info.capabilities)):
                agent_id = task.assigned_agent
            else:
                logger.warning(
                    "Assigned agent '%s' lacks required capabilities %s (has %s) — trying others",
                    task.assigned_agent, required_caps, agent_info.capabilities,
                )

    if agent_id is None:
        # Pick a capable agent using round-robin so tasks are
        # distributed fairly across all available agents.
        global _dispatch_counter
        capable = registry.get_capable(required_caps)
        connected = [a for a in capable if a.agent_id in agent_websockets]
        if connected:
            idx = _dispatch_counter % len(connected)
            agent_id = connected[idx].agent_id
            _dispatch_counter += 1

    if agent_id is None:
        caps_msg = f" (required: {required_caps})" if required_caps else ""
        logger.info("No capable agents for task %s%s — falling back to model", task.short_id, caps_msg)
        event_logger.info("task", f"No capable agents for {task.short_id}{caps_msg}, using model fallback",
                          task_id=task.short_id)
        return None

    agent_info = registry.get_by_id(agent_id)
    agent_name = agent_info.name if agent_info else agent_id

    # --- Mark agent busy, set task assignment ---
    task.assigned_agent = agent_name
    if agent_info:
        agent_info.status = "busy"
    health_monitor.on_task_dispatched(agent_id)
    await broadcast_agent_status()

    logger.info(
        "Dispatching task %s to agent '%s' (%s)",
        task.short_id, agent_name, agent_id,
    )
    event_logger.info("task", f"Dispatching {task.short_id} to {agent_name}",
                      task_id=task.short_id, agent=agent_name)
    await broadcast_to_dashboards({
        "type": "task_phase",
        "task_id": task.task_id,
        "short_id": task.short_id,
        "phase": "ACT",
        "detail": f"Dispatching to {agent_name}",
    })

    # --- Send command and wait for response ---
    future = asyncio.get_event_loop().create_future()
    pending_task_responses[task.task_id] = future

    try:
        ws = agent_websockets[agent_id]
        await ws.send_json({
            "type": "command",
            "command": task.description,
            "task_id": task.task_id,
        })

        # Wait for the agent to respond (or timeout)
        dispatch_timeout = config.dashboard.agent_dispatch_timeout
        result = await asyncio.wait_for(future, timeout=dispatch_timeout)
        logger.info(
            "Agent '%s' responded to task %s: %.200s",
            agent_name, task.short_id, result,
        )
        health_monitor.on_task_completed(agent_id)
        event_logger.info("task", f"Agent {agent_name} completed {task.short_id}",
                          task_id=task.short_id, agent=agent_name)
        return result

    except asyncio.TimeoutError:
        logger.warning(
            "Agent '%s' timed out on task %s after %ds",
            agent_name, task.short_id, dispatch_timeout,
        )
        health_monitor.on_task_failed(agent_id)
        event_logger.warn("task", f"Agent {agent_name} timed out on {task.short_id}",
                          task_id=task.short_id, agent=agent_name,
                          timeout=dispatch_timeout)
        return f"[timeout] Agent '{agent_name}' did not respond within {dispatch_timeout}s"

    except Exception as e:
        logger.warning(
            "Dispatch to agent '%s' failed for task %s: %s",
            agent_name, task.short_id, e,
        )
        health_monitor.on_task_failed(agent_id)
        event_logger.error("task", f"Dispatch to {agent_name} failed: {e}",
                           task_id=task.short_id, agent=agent_name, error=str(e))
        return None

    finally:
        pending_task_responses.pop(task.task_id, None)
        # Restore agent to online
        if agent_info and agent_info.status == "busy":
            agent_info.status = "online"
        await broadcast_agent_status()


# ---------------------------------------------------------------------------
# Orchestrator instance — uses the shared task queue and callbacks
# ---------------------------------------------------------------------------

def on_model_call(model, backend, tokens, latency_ms, cost_usd, task_short_id):
    """Called by the orchestrator after each model router call."""
    event_logger.info(
        "model",
        f"Model call: {model} ({backend}) — {tokens} tokens, "
        f"{latency_ms:.0f}ms, ${cost_usd:.6f}",
        model=model, backend=backend, tokens=tokens,
        latency_ms=round(latency_ms, 1), cost_usd=round(cost_usd, 6),
        task_id=task_short_id,
    )
    analytics.record_model_call(
        model=model, backend=backend, tokens=tokens,
        latency_ms=latency_ms, cost_usd=cost_usd,
        task_short_id=task_short_id,
    )


async def on_approval_request(task, perm_result):
    """Called by the orchestrator when a Tier 2 task needs approval.

    Broadcasts the approval request to all connected dashboard browsers
    so George can approve or reject from the Security & Audit panel.
    """
    await broadcast_to_dashboards({
        "type": "approval_request",
        "task_id": task.task_id,
        "short_id": task.short_id,
        "description": task.description,
        "action_type": perm_result.action_type,
        "risk_level": perm_result.risk_level,
        "reason": perm_result.reason,
        "tier": perm_result.tier,
    })
    event_logger.warn(
        "security",
        f"Approval requested for {task.short_id}: {perm_result.action_type} — {task.description[:80]}",
        task_id=task.short_id, action_type=perm_result.action_type,
        risk_level=perm_result.risk_level,
    )


# Context window manager — shares memory instances with orchestrator
context_manager = ContextManager(
    short_term=short_term_memory,
    long_term=long_term_memory,
    episodic=episodic_memory,
    working=working_memory,
    persona=persona,
)

orchestrator = Orchestrator(
    queue=task_queue,
    short_term_memory=short_term_memory,
    working_memory=working_memory,
    long_term_memory=long_term_memory,
    episodic_memory=episodic_memory,
    persona=persona,
    on_phase_change=on_task_phase_change,
    on_task_update=on_task_update,
    on_dispatch_to_agent=dispatch_to_agent,
    on_model_call=on_model_call,
    on_chain_update=on_chain_update,
    on_approval_request=on_approval_request,
    audit_log=security_audit_log,
    context_manager=context_manager,
)

# TTS pipeline — Piper TTS with graceful fallback
tts_pipeline = TTSPipeline(config=config)

# Content agent — local in-process agent for content tasks
content_agent = ContentAgent(
    router=orchestrator.router,
    persona=persona,
    long_term_memory=long_term_memory,
    calendar=content_calendar,
    event_logger=event_logger,
    on_model_call=on_model_call,
    tts_pipeline=tts_pipeline,
)

# Research agent — local in-process agent for research tasks
research_agent = ResearchAgent(
    router=orchestrator.router,
    persona=persona,
    long_term_memory=long_term_memory,
    research_store=research_store,
    event_logger=event_logger,
    on_model_call=on_model_call,
)

# Business agent — local in-process agent for business tasks
business_agent = BusinessAgent(
    router=orchestrator.router,
    persona=persona,
    long_term_memory=long_term_memory,
    business_store=business_store,
    event_logger=event_logger,
    on_model_call=on_model_call,
)

# Doppler Scout — motorcycle listing scraper with deal scoring
_scout_cfg = getattr(config, 'scout', None)
_scout_criteria = SearchCriteria(
    makes=getattr(_scout_cfg, 'makes', ["honda", "yamaha", "suzuki", "kawasaki", "harley-davidson", "ducati"]),
    models=getattr(_scout_cfg, 'models', []),
    year_min=getattr(_scout_cfg, 'year_min', 0),
    year_max=getattr(_scout_cfg, 'year_max', 0),
    price_min=getattr(_scout_cfg, 'price_min', 0),
    price_max=getattr(_scout_cfg, 'price_max', 5000),
    location=getattr(_scout_cfg, 'location', 'Portland OR'),
    radius_miles=getattr(_scout_cfg, 'radius_miles', 50),
    keywords=getattr(_scout_cfg, 'keywords', []),
    exclude_keywords=getattr(_scout_cfg, 'exclude_keywords', ['parts only']),
)
doppler_scout = DopplerScout(
    store=scout_store,
    router=orchestrator.router,
    criteria=_scout_criteria,
    event_logger=event_logger,
    notification_manager=notification_manager,
    on_model_call=on_model_call,
)


# ---------------------------------------------------------------------------
# Kill switch — CAM Constitution: "Prominent, always visible. One click
# halts all autonomous action across all agents."
# ---------------------------------------------------------------------------

async def activate_kill_switch():
    """Send shutdown command to every connected agent.

    Per the CAM Constitution, the kill switch must:
    - Halt all autonomous action across all agents
    - Be always visible and one-click
    - Trigger graceful degradation — agents queue work and wait

    This sends a shutdown message to each agent's WebSocket and
    notifies all dashboard browsers of the new state.
    """
    global kill_switch_active
    kill_switch_active = True
    app.state.kill_switch["active"] = True
    logger.critical("KILL SWITCH ACTIVATED — shutting down all agents")
    event_logger.error("system", "KILL SWITCH ACTIVATED — all agents halting")

    # Stop the orchestrator loop
    orchestrator.stop()

    shutdown_msg = {"type": "shutdown", "reason": "kill_switch"}

    # Send shutdown to every connected agent
    disconnected = []
    for agent_id, ws in agent_websockets.items():
        try:
            await ws.send_json(shutdown_msg)
            logger.info("Shutdown sent to agent '%s'", agent_id)
        except Exception:
            disconnected.append(agent_id)

    # Clean up any that were already gone
    for agent_id in disconnected:
        del agent_websockets[agent_id]

    # Mark all agents offline in the registry
    for agent in registry.list_all():
        agent.status = "offline"

    # Notify all dashboard browsers
    await broadcast_agent_status()

    # Also send kill switch state to dashboards
    for client in dashboard_clients:
        try:
            await client.send_json({"type": "kill_switch", "active": True})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background session cleanup
# ---------------------------------------------------------------------------

async def cleanup_sessions():
    """Background task: purge expired sessions every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        session_manager.cleanup_expired()


# ---------------------------------------------------------------------------
# Background heartbeat checker
# ---------------------------------------------------------------------------

async def check_heartbeats():
    """Background task: check agent heartbeats and broadcast health.

    Uses the HealthMonitor's 3-missed-heartbeats policy as the primary
    offline detection. Also broadcasts health_status every cycle so
    the dashboard indicators stay current.
    """
    while True:
        await asyncio.sleep(config.dashboard.heartbeat_check_interval)
        newly_offline = health_monitor.check_heartbeats()
        if newly_offline:
            for aid in newly_offline:
                event_logger.warn("agent", f"Agent '{aid}' went offline (3 missed heartbeats)",
                                  agent_id=aid)
            await broadcast_agent_status()
        # Always broadcast health so dashboard indicators update
        await broadcast_health_status()


# ---------------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, clean up on shutdown."""
    global orchestrator_task, telegram_bot
    logger.info("CAM Dashboard starting up")
    event_logger.info("system", "CAM Dashboard starting up")
    heartbeat_task = asyncio.create_task(check_heartbeats())
    session_cleanup_task = asyncio.create_task(cleanup_sessions())
    orchestrator_task = asyncio.create_task(orchestrator.run())
    scheduler_task = asyncio.create_task(scheduler.run())
    logger.info("Orchestrator loop started as background task")
    logger.info("Scheduler loop started as background task")
    if session_manager.auth_enabled:
        logger.info("Dashboard authentication ENABLED (user: %s)", session_manager.username)
    else:
        logger.info("Dashboard authentication DISABLED (no password_hash configured)")

    # Telegram bot — instantiated here so activate_kill_switch is in scope
    telegram_bot = TelegramBot(
        token=getattr(getattr(config, 'telegram', None), 'bot_token', ''),
        allowed_chat_ids=getattr(getattr(config, 'telegram', None), 'allowed_chat_ids', []),
        task_queue=task_queue,
        registry=registry,
        health_monitor=health_monitor,
        episodic_memory=episodic_memory,
        event_logger=event_logger,
        activate_kill_switch=activate_kill_switch,
        on_status_change=broadcast_to_dashboards,
        poll_interval=getattr(getattr(config, 'telegram', None), 'poll_interval', 0.5),
        task_timeout=getattr(getattr(config, 'telegram', None), 'task_timeout', 120.0),
    )
    await telegram_bot.start()
    app.state.telegram_bot = telegram_bot

    yield

    # Shutdown — stop Telegram before orchestrator so in-flight polls end cleanly
    if telegram_bot:
        await telegram_bot.stop()
    orchestrator.stop()
    scheduler.stop()
    orchestrator_task.cancel()
    heartbeat_task.cancel()
    session_cleanup_task.cancel()
    scheduler_task.cancel()
    episodic_memory.close()
    content_calendar.close()
    research_store.close()
    scout_store.close()
    business_store.close()
    security_audit_log.close()
    analytics.close()
    logger.info("CAM Dashboard shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="CAM Dashboard", lifespan=lifespan)

# Expose shared state on app.state so the REST API router (and future
# routers) can access everything without circular imports.
app.state.registry = registry
app.state.health_monitor = health_monitor
app.state.event_logger = event_logger
app.state.analytics = analytics
app.state.task_queue = task_queue
app.state.scheduler = scheduler
app.state.agent_websockets = agent_websockets
app.state.config = config
app.state.short_term_memory = short_term_memory
app.state.working_memory = working_memory
app.state.long_term_memory = long_term_memory
app.state.episodic_memory = episodic_memory
app.state.content_calendar = content_calendar
app.state.content_agent = content_agent
app.state.tts_pipeline = tts_pipeline
app.state.research_store = research_store
app.state.research_agent = research_agent
app.state.scout_store = scout_store
app.state.doppler_scout = doppler_scout
app.state.business_store = business_store
app.state.business_agent = business_agent
app.state.security_audit_log = security_audit_log
app.state.session_manager = session_manager
app.state.deployer = deployer
app.state.backup_manager = backup_manager
app.state.message_bus = message_bus
app.state.kill_switch = {"active": False}       # mutable container
app.state.activate_kill_switch = activate_kill_switch

# Mount versioned REST API
from interfaces.api.routes import router as api_router
app.include_router(api_router)

# Mount static files (CSS, JS, images if any)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Authentication middleware
#
# When auth is enabled, all routes except the exempted prefixes require a
# valid cam_session cookie. Unauthenticated browser requests to non-API
# paths get the login page; AJAX/fetch requests get a 401 JSON response.
# ---------------------------------------------------------------------------

# Paths that bypass dashboard cookie auth
_AUTH_EXEMPT_PREFIXES = ("/auth/", "/api/v1/", "/ws/", "/static/")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Check cam_session cookie on dashboard routes when auth is enabled."""
    if not session_manager.auth_enabled:
        return await call_next(request)

    path = request.url.path

    # Exempt paths: auth endpoints, versioned API, agent WS, static files
    for prefix in _AUTH_EXEMPT_PREFIXES:
        if path.startswith(prefix):
            return await call_next(request)

    # The root path and /api/* (non-versioned dashboard endpoints) need auth
    token = request.cookies.get("cam_session")
    if session_manager.validate_session(token):
        return await call_next(request)

    # Not authenticated — for the root path, fall through to serve login.html
    # For API endpoints, return 401 JSON
    if path.startswith("/api/"):
        return JSONResponse(status_code=401, content={"error": "Not authenticated"})

    # For all other paths (including /), serve the login page
    return FileResponse(STATIC_DIR / "login.html")


# ---------------------------------------------------------------------------
# Auth routes — login, lock, status
# ---------------------------------------------------------------------------

@app.post("/auth/login")
async def auth_login(request: Request):
    """Authenticate and set a session cookie."""
    client_ip = request.client.host if request.client else "unknown"

    # Check lockout first
    remaining = session_manager.get_lockout_remaining(client_ip)
    if remaining > 0:
        event_logger.warn("auth", f"Login attempt from locked-out IP {client_ip}",
                          client_ip=client_ip, lockout_remaining=remaining)
        security_audit_log.log_action(
            action_type="login_attempt",
            actor=client_ip,
            target="dashboard",
            result="locked_out",
            risk_level="medium",
            tier=0,
        )
        return JSONResponse(
            status_code=429,
            content={"ok": False, "error": f"Too many attempts. Try again in {remaining}s."},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Invalid request"})

    username = body.get("username", "")
    password = body.get("password", "")

    token = session_manager.login(username, password, client_ip=client_ip)

    if token is None:
        event_logger.warn("auth", f"Failed login from {client_ip} (user: {username})",
                          client_ip=client_ip, username=username)
        security_audit_log.log_action(
            action_type="login_attempt",
            actor=client_ip,
            target="dashboard",
            result="failed",
            risk_level="medium",
            tier=0,
        )
        return JSONResponse(
            status_code=401,
            content={"ok": False, "error": "Invalid credentials."},
        )

    # Success
    event_logger.info("auth", f"Login successful from {client_ip} (user: {username})",
                      client_ip=client_ip, username=username)
    security_audit_log.log_action(
        action_type="login",
        actor=username,
        target="dashboard",
        result="success",
        risk_level="low",
        tier=0,
    )

    response = JSONResponse(content={"ok": True})
    response.set_cookie(
        key="cam_session",
        value=token,
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response


@app.post("/auth/lock")
async def auth_lock(request: Request):
    """Destroy the current session (lock screen)."""
    token = request.cookies.get("cam_session")
    session_manager.destroy_session(token)

    client_ip = request.client.host if request.client else "unknown"
    event_logger.info("auth", f"Session locked from {client_ip}", client_ip=client_ip)

    response = JSONResponse(content={"ok": True})
    response.delete_cookie("cam_session", path="/")
    return response


@app.get("/auth/status")
async def auth_status(request: Request):
    """Check if the current session is still valid."""
    if not session_manager.auth_enabled:
        return {"authenticated": True, "auth_enabled": False}

    token = request.cookies.get("cam_session")
    valid = session_manager.validate_session(token)
    return {"authenticated": valid, "auth_enabled": True}


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root(request: Request):
    """Serve the main dashboard page (or login page if auth is enabled and not authenticated)."""
    if session_manager.auth_enabled:
        token = request.cookies.get("cam_session")
        if not session_manager.validate_session(token):
            return FileResponse(STATIC_DIR / "login.html")
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/agents")
async def get_agents():
    """Return current status of all known agents.

    The browser fetches this on page load to get initial state,
    then switches to WebSocket for real-time updates.
    """
    return registry.to_broadcast_list()


@app.get("/api/tasks")
async def get_tasks():
    """Return current task queue state as JSON."""
    return {
        "tasks": task_queue.to_broadcast_list(),
        "counts": task_queue.get_status(),
    }


@app.get("/api/health")
async def get_health():
    """Return current agent health metrics as JSON."""
    return health_monitor.to_broadcast_dict()


@app.get("/api/analytics")
async def get_analytics():
    """Return aggregate analytics summary as JSON."""
    return analytics.get_summary()


@app.get("/api/events")
async def get_events(count: int = 200):
    """Return recent events from the event log."""
    return event_logger.get_recent(count)


@app.get("/api/memory")
async def get_memory():
    """Return current memory system status as JSON."""
    return {
        "short_term": short_term_memory.get_status(),
        "working": working_memory.get_status(),
        "long_term": long_term_memory.get_status(),
        "episodic": episodic_memory.get_status(),
    }


@app.get("/api/schedules")
async def get_schedules():
    """Return current scheduled tasks as JSON."""
    return {"schedules": scheduler.to_broadcast_list()}


@app.get("/api/content-calendar")
async def get_content_calendar():
    """Return current content calendar entries as JSON."""
    return {"entries": content_calendar.to_broadcast_list()}


@app.get("/api/research")
async def get_research():
    """Return current research results as JSON."""
    return {"entries": research_store.to_broadcast_list()}


@app.get("/api/scout")
async def get_scout():
    """Return current scout listings as JSON."""
    return {
        "entries": scout_store.to_broadcast_list(),
        "status": scout_store.get_status(),
    }


@app.post("/api/scout/scan")
async def trigger_scout_scan():
    """Trigger an immediate scout scan for motorcycle listings."""
    try:
        new_listings = await doppler_scout.scan()
        await broadcast_scout_status()
        return {
            "ok": True,
            "new_count": len(new_listings),
            "listings": [l.to_dict() for l in new_listings],
        }
    except Exception as e:
        logger.warning("Scout scan failed: %s", e)
        return {"ok": False, "error": str(e)}


@app.post("/api/scout/{entry_id}/status")
async def update_scout_status(entry_id: str, request: Request):
    """Update a scout listing's status."""
    body = await request.json()
    new_status = body.get("status", "")
    if new_status not in [s.value for s in ListingStatus]:
        return JSONResponse({"ok": False, "error": "Invalid status"}, status_code=400)
    updated = scout_store.update_entry(entry_id, status=new_status)
    if updated:
        await broadcast_scout_status()
        return {"ok": True, "entry": updated.to_dict()}
    return JSONResponse({"ok": False, "error": "Entry not found"}, status_code=404)


@app.post("/api/scout/{entry_id}/rescore")
async def rescore_scout_listing(entry_id: str):
    """Re-score a specific scout listing."""
    try:
        listing = await doppler_scout.score_single(entry_id)
        if listing:
            await broadcast_scout_status()
            return {"ok": True, "entry": listing.to_dict()}
        return JSONResponse({"ok": False, "error": "Entry not found"}, status_code=404)
    except Exception as e:
        logger.warning("Scout rescore failed: %s", e)
        return {"ok": False, "error": str(e)}


@app.get("/api/business")
async def get_business():
    """Return current business data as JSON."""
    return {
        **business_store.to_broadcast_list(),
        "status": business_store.get_status(),
    }


@app.get("/api/security-audit")
async def get_security_audit():
    """Return recent security audit entries as JSON."""
    return {
        "entries": security_audit_log.to_broadcast_list(),
        "status": security_audit_log.get_status(),
    }


@app.get("/api/deploy/status")
async def get_deploy_status():
    """Return deployer status: local version, agent versions, recent history."""
    return deployer.get_status()


@app.get("/api/backup/status")
async def get_backup_status():
    """Return backup system status: last backup, count, total size, list."""
    return backup_manager.get_status()


@app.get("/api/config")
async def get_config_endpoint():
    """Return current configuration as JSON."""
    return {"config": config.to_dict(), "last_loaded": config.last_loaded}


@app.get("/api/tts/status")
async def get_tts_status():
    """Return current TTS pipeline status as JSON."""
    return tts_pipeline.get_status()


@app.get("/api/tts/audio")
async def get_tts_audio_list():
    """Return list of generated audio files with metadata."""
    return {"audio": tts_pipeline.list_audio()}


@app.get("/api/tts/audio/{filename}")
async def get_tts_audio_file(filename: str):
    """Serve a WAV audio file for browser playback.

    Includes path traversal protection — filename must resolve
    to a path within the audio directory.
    """
    audio_dir = Path(tts_pipeline._audio_dir).resolve()
    file_path = (audio_dir / filename).resolve()

    # Path traversal protection
    if not str(file_path).startswith(str(audio_dir)):
        return JSONResponse(status_code=403, content={"error": "Access denied"})

    if not file_path.exists() or not file_path.suffix == ".wav":
        return JSONResponse(status_code=404, content={"error": "File not found"})

    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.post("/api/events/export")
async def export_events():
    """Export all events to a JSON file in data/logs/."""
    from datetime import datetime as dt
    filepath = Path("data/logs") / f"events_{dt.now().strftime('%Y%m%d_%H%M%S')}.json"
    num = event_logger.export_json(filepath)
    return {"ok": True, "filepath": str(filepath), "event_count": num}


@app.post("/api/file/upload")
async def upload_file(
    file: UploadFile = File(...),
    agent_id: str = Form(...),
    dest_path: str = Form(""),
):
    """Upload a file and relay it to an agent over WebSocket.

    Accepts multipart form data with:
        file      — the file to send
        agent_id  — target agent's ID
        dest_path — destination path on the agent (optional)
    """
    # Validate agent is online
    if agent_id not in agent_websockets:
        return {"ok": False, "error": "Agent is offline"}

    agent_info = registry.get_by_id(agent_id)
    agent_name = agent_info.name if agent_info else agent_id

    # Read file data
    data = await file.read()
    file_size = len(data)
    max_size = config.file_transfer.max_file_size

    if file_size > max_size:
        return {"ok": False, "error": f"File too large ({file_size} bytes, max {max_size})"}
    if file_size == 0:
        return {"ok": False, "error": "Empty file"}

    filename = file.filename or "upload"
    checksum = compute_checksum(data)
    chunk_size = config.file_transfer.chunk_size
    chunk_count = (file_size + chunk_size - 1) // chunk_size

    try:
        transfer = ft_manager.create_transfer(
            direction="to_agent",
            agent_id=agent_id,
            agent_name=agent_name,
            filename=filename,
            file_size=file_size,
            chunk_count=chunk_count,
            checksum=checksum,
            dest_path=dest_path,
        )
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    event_logger.info(
        "file_transfer",
        f"Uploading {filename} ({file_size} bytes) to {agent_name}",
        transfer_id=transfer.transfer_id, agent=agent_name, filename=filename,
    )

    # Launch relay as background task
    agent_ws = agent_websockets[agent_id]
    asyncio.create_task(relay_file_to_agent(transfer, data, agent_ws))

    return {
        "ok": True,
        "transfer_id": transfer.transfer_id,
        "filename": filename,
        "file_size": file_size,
        "chunk_count": chunk_count,
    }


# ---------------------------------------------------------------------------
# WebSocket: agents connect here to send heartbeats
# ---------------------------------------------------------------------------

@app.websocket("/ws/agent/{agent_id}")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for agents to connect and send heartbeats.

    Agents open a connection to /ws/agent/{their_id} and send JSON:
        {"type": "heartbeat", "name": "AgentName", "ip": "192.168.1.x",
         "capabilities": ["research", "content"]}

    On connect: agent is registered in the registry.
    On heartbeat: last_heartbeat timestamp updated.
    On disconnect: agent marked offline in the registry.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("Agent '%s' WebSocket opened from %s", agent_id, client_host)

    # Store the WebSocket so we can send commands back (e.g., kill switch)
    agent_websockets[agent_id] = websocket

    # Start health tracking for this connection session
    health_monitor.on_agent_connected(agent_id)

    try:
        while True:
            data = await websocket.receive_json()

            msg_type = data.get("type")

            if msg_type == "heartbeat":
                name = data.get("name", agent_id)
                ip = data.get("ip", client_host)
                capabilities = data.get("capabilities")

                if registry.get_by_id(agent_id) is None:
                    # First heartbeat — register in the registry
                    registry.register(
                        agent_id=agent_id,
                        name=name,
                        ip_address=ip,
                        capabilities=capabilities,
                    )
                    event_logger.info("agent", f"Agent '{name}' connected from {ip}",
                                      agent_id=agent_id, ip=ip)
                else:
                    # Subsequent heartbeat — update via registry
                    registry.update_heartbeat(
                        agent_id=agent_id,
                        ip_address=ip,
                        capabilities=capabilities,
                    )

                health_monitor.on_heartbeat(agent_id)
                await broadcast_agent_status()

            elif msg_type == "command_response":
                response_text = data.get("response", "")
                task_id = data.get("task_id")

                logger.info(
                    "Agent '%s' response (task_id=%s): %s",
                    agent_id, task_id, response_text,
                )

                # If this response is for a dispatched task, resolve the Future
                if task_id and task_id in pending_task_responses:
                    future = pending_task_responses[task_id]
                    if not future.done():
                        future.set_result(response_text)
                else:
                    # Manual command from dashboard — relay as before
                    await broadcast_to_dashboards({
                        "type": "command_response",
                        "agent_id": agent_id,
                        "command": data.get("command", ""),
                        "response": response_text,
                    })

            elif msg_type == "pong":
                ping_id = data.get("ping_id")
                if ping_id:
                    rtt_ms = health_monitor.resolve_pong(ping_id)
                    await broadcast_to_dashboards({
                        "type": "ping_result",
                        "agent_id": agent_id,
                        "rtt_ms": round(rtt_ms, 1) if rtt_ms is not None else None,
                        "error": None if rtt_ms is not None else "Unknown ping_id",
                    })

            # --- File transfer: agent acking our chunks ---
            elif msg_type == "file_chunk_ack":
                transfer_id = data.get("transfer_id", "")
                chunk_index = data.get("chunk_index", 0)
                ok = data.get("ok", True)

                if not ok:
                    error = data.get("error", "Agent rejected chunk")
                    await ft_manager.fail_transfer(transfer_id, error)
                else:
                    transfer = ft_manager.get_transfer(transfer_id)
                    if transfer:
                        await ft_manager.update_progress(transfer_id, chunk_index + 1)

            elif msg_type == "file_transfer_complete":
                transfer_id = data.get("transfer_id", "")
                ok = data.get("ok", True)

                if ok:
                    await ft_manager.complete_transfer(transfer_id)
                    event_logger.info(
                        "file_transfer",
                        f"Transfer {transfer_id} completed successfully",
                        transfer_id=transfer_id, agent_id=agent_id,
                    )
                else:
                    error = data.get("error", "Agent reported failure")
                    await ft_manager.fail_transfer(transfer_id, error)
                    event_logger.error(
                        "file_transfer",
                        f"Transfer {transfer_id} failed: {error}",
                        transfer_id=transfer_id, agent_id=agent_id,
                    )

            # --- File transfer: agent sending a file to us ---
            elif msg_type == "file_send_start":
                transfer_id = data.get("transfer_id", "")
                ok = data.get("ok", True)

                if not ok:
                    error = data.get("error", "Agent could not read file")
                    await ft_manager.fail_transfer(transfer_id, error)
                else:
                    filename = data.get("filename", "")
                    file_size = data.get("file_size", 0)
                    chunk_count = data.get("chunk_count", 0)
                    checksum = data.get("checksum", "")
                    agent_info = registry.get_by_id(agent_id)
                    agent_name = agent_info.name if agent_info else agent_id

                    transfer = ft_manager.get_transfer(transfer_id)
                    if transfer:
                        # Update the existing transfer with file metadata
                        transfer.filename = filename
                        transfer.file_size = file_size
                        transfer.chunk_count = chunk_count
                        transfer.checksum = checksum
                    else:
                        transfer = ft_manager.create_transfer(
                            direction="from_agent",
                            agent_id=agent_id,
                            agent_name=agent_name,
                            filename=filename,
                            file_size=file_size,
                            chunk_count=chunk_count,
                            checksum=checksum,
                        )
                    ft_manager.init_receive_buffer(transfer.transfer_id)

            elif msg_type == "file_send_chunk":
                transfer_id = data.get("transfer_id", "")
                chunk_index = data.get("chunk_index", 0)
                chunk_count = data.get("chunk_count", 0)
                b64_data = data.get("data", "")

                transfer = ft_manager.get_transfer(transfer_id)
                if not transfer:
                    continue

                try:
                    raw = base64.b64decode(b64_data)
                    ft_manager.store_chunk(transfer_id, chunk_index, raw)
                    await ft_manager.update_progress(transfer_id, chunk_index + 1)

                    # Check if all chunks received
                    if chunk_index + 1 >= chunk_count:
                        assembled = ft_manager.assemble_chunks(transfer_id, chunk_count)
                        if assembled is not None:
                            # Verify checksum
                            actual_checksum = compute_checksum(assembled)
                            if transfer.checksum and actual_checksum != transfer.checksum:
                                await ft_manager.fail_transfer(
                                    transfer_id,
                                    f"Checksum mismatch: expected {transfer.checksum}, got {actual_checksum}",
                                )
                                continue

                            # Write to receive directory
                            receive_dir = Path(config.file_transfer.receive_dir)
                            receive_dir.mkdir(parents=True, exist_ok=True)
                            dest = receive_dir / transfer.filename
                            dest.write_bytes(assembled)
                            transfer.dest_path = str(dest)
                            await ft_manager.complete_transfer(transfer_id)
                            event_logger.info(
                                "file_transfer",
                                f"Received {transfer.filename} from {agent_id} → {dest}",
                                transfer_id=transfer_id, agent_id=agent_id,
                                filename=transfer.filename,
                            )

                except Exception as e:
                    logger.warning("Error receiving chunk %d of %s: %s", chunk_index, transfer_id, e)
                    await ft_manager.fail_transfer(transfer_id, str(e))

            # --- Message Bus: remote agent publishing ---
            elif msg_type == "bus_publish":
                channel = data.get("channel", "")
                payload = data.get("payload", {})
                result = await message_bus.publish(agent_id, channel, payload)
                await websocket.send_json({
                    "type": "bus_publish_result",
                    "ok": result is not None,
                    "message_id": result.message_id if result else None,
                })

    except WebSocketDisconnect:
        logger.info("Agent '%s' disconnected", agent_id)
        event_logger.warn("agent", f"Agent '{agent_id}' disconnected", agent_id=agent_id)
        # Remove from live WebSocket tracking
        agent_websockets.pop(agent_id, None)
        # Accumulate uptime in health monitor before marking offline
        health_monitor.on_agent_disconnected(agent_id)
        # Mark offline in registry (keeps the record for dashboard display)
        registry.deregister(agent_id)
        await broadcast_agent_status()


# ---------------------------------------------------------------------------
# WebSocket: browser clients connect here to watch agent status
# ---------------------------------------------------------------------------

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard browser clients.

    Browsers connect here to receive real-time agent status updates.
    On connect, they immediately get the current state of all agents.
    After that, updates are pushed whenever agent status changes.

    When auth is enabled, the cam_session cookie must be valid or the
    connection is rejected with close code 4401.
    """
    # Validate session cookie before accepting the WebSocket
    if session_manager.auth_enabled:
        token = websocket.cookies.get("cam_session")
        if not session_manager.validate_session(token):
            await websocket.close(code=4401, reason="Not authenticated")
            return

    await websocket.accept()
    dashboard_clients.append(websocket)
    logger.info(
        "Dashboard client connected from %s",
        websocket.client.host if websocket.client else "unknown",
    )

    # Send current state right away so the page doesn't start blank
    await websocket.send_json({
        "type": "agent_status",
        "agents": registry.to_broadcast_list(),
    })

    # Also send current kill switch state
    await websocket.send_json({"type": "kill_switch", "active": kill_switch_active})

    # Send current task queue state
    await websocket.send_json({
        "type": "task_status",
        "tasks": task_queue.to_broadcast_list(),
        "counts": task_queue.get_status(),
    })

    # Send current health metrics
    await websocket.send_json({
        "type": "health_status",
        "health": health_monitor.to_broadcast_dict(),
    })

    # Send recent event log so the panel isn't empty on connect
    await websocket.send_json({
        "type": "event_log",
        "events": event_logger.get_recent(config.dashboard.default_event_count),
    })

    # Send current config so the settings panel renders immediately
    await websocket.send_json({
        "type": "config",
        "config": config.to_dict(),
        "last_loaded": config.last_loaded,
    })

    # Send current analytics summary
    await websocket.send_json({"type": "analytics", "data": analytics.get_summary()})

    # Send command library for the command palette
    await websocket.send_json({
        "type": "command_library",
        "commands": command_library.to_broadcast_list(),
    })

    # Send current chain state
    await websocket.send_json({
        "type": "chain_status",
        "chains": task_queue.chains_to_broadcast_list(),
    })

    # Send notification history so the bell icon shows correct state
    await websocket.send_json({
        "type": "notification_history",
        "notifications": notification_manager.get_recent(50),
        "unread_count": notification_manager.get_unread_count(),
    })

    # Send file transfer state (active + history)
    await websocket.send_json({
        "type": "file_transfer_history",
        "transfers": ft_manager.get_all_transfers(),
    })

    # Send deploy status (local version, agent versions, history)
    await websocket.send_json({
        "type": "deploy_status",
        **deployer.get_status(),
    })

    # Send backup status
    await websocket.send_json({
        "type": "backup_status",
        **backup_manager.get_status(),
    })

    # Send message bus state
    await websocket.send_json({
        "type": "bus_status",
        **message_bus.to_broadcast_dict(),
    })

    # Send current schedule state
    await websocket.send_json({
        "type": "schedule_status",
        "schedules": scheduler.to_broadcast_list(),
    })

    # Send current content calendar state
    await websocket.send_json({
        "type": "content_calendar_status",
        "entries": content_calendar.to_broadcast_list(),
    })

    # Send current research results
    await websocket.send_json({
        "type": "research_status",
        "entries": research_store.to_broadcast_list(),
    })

    # Send current scout listings
    await websocket.send_json({
        "type": "scout_status",
        "entries": scout_store.to_broadcast_list(),
        "status": scout_store.get_status(),
    })

    # Send current business data
    await websocket.send_json({
        "type": "business_status",
        **business_store.to_broadcast_list(),
        "status": business_store.get_status(),
    })

    # Send current security audit state
    await websocket.send_json({
        "type": "security_audit_status",
        "entries": security_audit_log.to_broadcast_list(),
        "status": security_audit_log.get_status(),
    })

    # Send current memory system status
    await websocket.send_json({
        "type": "memory_status",
        "short_term": short_term_memory.get_status(),
        "working": working_memory.get_status(),
        "long_term": long_term_memory.get_status(),
        "episodic": episodic_memory.get_status(),
    })

    # Send TTS pipeline status
    await websocket.send_json({
        "type": "tts_status",
        **tts_pipeline.get_status(),
    })

    # Send TTS audio list
    await websocket.send_json({
        "type": "tts_audio_list",
        "audio": tts_pipeline.list_audio(),
    })

    # Send TTS voice list
    await websocket.send_json({
        "type": "tts_voices",
        "voices": tts_pipeline.list_voices(),
    })

    # Send persona data for the persona preview panel
    await websocket.send_json({"type": "persona", "persona": persona.to_dict()})

    # Send Telegram bot connection status
    if telegram_bot:
        await websocket.send_json({
            "type": "telegram_status",
            "connected": telegram_bot.is_connected,
        })

    try:
        # Listen for dashboard commands (kill switch, task submit, agent controls)
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "kill_switch":
                await activate_kill_switch()

            elif msg.get("type") == "command":
                # Route a command to a specific agent
                target_id = msg.get("agent_id", "")
                command_text = msg.get("command", "")

                if target_id in agent_websockets:
                    try:
                        await agent_websockets[target_id].send_json({
                            "type": "command",
                            "command": command_text,
                        })
                        logger.info(
                            "Command sent to agent '%s': %s",
                            target_id, command_text,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to send command to agent '%s'",
                            target_id,
                        )
                        await broadcast_to_dashboards({
                            "type": "command_response",
                            "agent_id": target_id,
                            "command": command_text,
                            "response": "[error] Agent connection lost",
                        })
                else:
                    logger.warning(
                        "Command for unknown/offline agent '%s'",
                        target_id,
                    )
                    await broadcast_to_dashboards({
                        "type": "command_response",
                        "agent_id": target_id,
                        "command": command_text,
                        "response": "[error] Agent is offline",
                    })

            elif msg.get("type") == "task_submit":
                # Submit a new task to the queue
                description = (msg.get("description") or "").strip()
                complexity_str = (msg.get("complexity") or "auto").lower()
                raw_caps = msg.get("required_capabilities") or []
                required_capabilities = [c.strip() for c in raw_caps if isinstance(c, str) and c.strip()]

                if not description:
                    await websocket.send_json({
                        "type": "task_submitted",
                        "ok": False,
                        "error": "Description is required",
                    })
                    continue

                try:
                    complexity = TaskComplexity(complexity_str)
                except ValueError:
                    complexity = TaskComplexity.AUTO

                task = task_queue.add_task(
                    description=description,
                    source="dashboard",
                    complexity=complexity,
                    required_capabilities=required_capabilities,
                )
                logger.info(
                    "Task %s submitted from dashboard (caps=%s): %s",
                    task.short_id, required_capabilities or "none", description,
                )
                event_logger.info("task", f"Task {task.short_id} submitted: {description[:80]}",
                                  task_id=task.short_id, complexity=complexity.value,
                                  required_capabilities=required_capabilities or None,
                                  source="dashboard")

                await websocket.send_json({
                    "type": "task_submitted",
                    "ok": True,
                    "task_id": task.task_id,
                    "short_id": task.short_id,
                })
                await broadcast_task_status()

            elif msg.get("type") == "task_list":
                # Dashboard requesting a full task list refresh
                await websocket.send_json({
                    "type": "task_status",
                    "tasks": task_queue.to_broadcast_list(),
                    "counts": task_queue.get_status(),
                })

            elif msg.get("type") == "ping_agent":
                target_id = msg.get("agent_id", "")
                if target_id in agent_websockets:
                    try:
                        ping_msg = health_monitor.create_ping(target_id)
                        await agent_websockets[target_id].send_json(ping_msg)
                        logger.info("Ping sent to agent '%s'", target_id)
                    except Exception:
                        logger.warning("Failed to ping agent '%s'", target_id)
                        await broadcast_to_dashboards({
                            "type": "ping_result",
                            "agent_id": target_id,
                            "rtt_ms": None,
                            "error": "Failed to send ping",
                        })
                else:
                    await broadcast_to_dashboards({
                        "type": "ping_result",
                        "agent_id": target_id,
                        "rtt_ms": None,
                        "error": "Agent is offline",
                    })

            elif msg.get("type") == "command_execute":
                # Execute a predefined command from the command palette
                command_name = msg.get("command_name", "")
                target_id = msg.get("agent_id", "")
                params = msg.get("params", {})

                cmd = command_library.get(command_name)
                if not cmd:
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": False,
                        "error": f"Unknown command: {command_name}",
                    })
                    continue

                # Build command string from name + params
                param_parts = [command_name]
                for p in cmd.parameters:
                    value = params.get(p["name"])
                    if value is None and p.get("default") is not None:
                        value = p["default"]
                    if value is not None:
                        param_parts.append(f"{p['name']}={value}")
                command_text = " ".join(param_parts)

                if target_id not in agent_websockets:
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": False,
                        "error": "Agent is offline",
                        "agent_id": target_id,
                        "command_name": command_name,
                    })
                    event_logger.warn(
                        "command",
                        f"Command '{command_name}' failed: agent '{target_id}' offline",
                        command=command_name, agent_id=target_id,
                    )
                    continue

                try:
                    await agent_websockets[target_id].send_json({
                        "type": "command",
                        "command": command_text,
                    })
                    logger.info(
                        "Command palette: sent '%s' to agent '%s'",
                        command_text, target_id,
                    )
                    event_logger.info(
                        "command",
                        f"Executed '{command_name}' on agent '{target_id}'",
                        command=command_name, agent_id=target_id,
                        params=params or None,
                    )
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": True,
                        "agent_id": target_id,
                        "command_name": command_name,
                        "command_text": command_text,
                    })
                except Exception:
                    logger.warning(
                        "Failed to send command '%s' to agent '%s'",
                        command_name, target_id,
                    )
                    await websocket.send_json({
                        "type": "command_execute_result",
                        "ok": False,
                        "error": "Agent connection lost",
                        "agent_id": target_id,
                        "command_name": command_name,
                    })

            elif msg.get("type") == "command_list":
                # Dashboard requesting a command library refresh
                await websocket.send_json({
                    "type": "command_library",
                    "commands": command_library.to_broadcast_list(),
                })

            elif msg.get("type") == "config_reload":
                # Reload config from disk and broadcast to all dashboards
                changes = config.reload()
                event_logger.info(
                    "system",
                    f"Configuration reloaded ({len(changes)} change(s))",
                    changes=changes if changes else None,
                )
                await broadcast_to_dashboards({
                    "type": "config",
                    "config": config.to_dict(),
                    "last_loaded": config.last_loaded,
                })

            elif msg.get("type") == "persona_reload":
                # Reload persona config from disk and broadcast to all dashboards
                persona.reload()
                event_logger.info("system", "Persona configuration reloaded")
                await broadcast_to_dashboards({"type": "persona", "persona": persona.to_dict()})

            elif msg.get("type") == "chain_submit":
                # Submit a multi-step task chain
                chain_name = (msg.get("name") or "").strip()
                raw_steps = msg.get("steps") or []

                if not chain_name or not raw_steps:
                    await websocket.send_json({
                        "type": "chain_submitted",
                        "ok": False,
                        "error": "Chain name and at least one step are required",
                    })
                    continue

                # Build Task objects for each step
                step_tasks = []
                for step_def in raw_steps:
                    desc = (step_def.get("description") or "").strip()
                    if not desc:
                        continue
                    complexity_str = (step_def.get("complexity") or "auto").lower()
                    try:
                        complexity = TaskComplexity(complexity_str)
                    except ValueError:
                        complexity = TaskComplexity.AUTO
                    raw_caps = step_def.get("required_capabilities") or []
                    caps = [c.strip() for c in raw_caps if isinstance(c, str) and c.strip()]

                    step_tasks.append(Task(
                        description=desc,
                        source="chain",
                        complexity=complexity,
                        required_capabilities=caps,
                    ))

                if not step_tasks:
                    await websocket.send_json({
                        "type": "chain_submitted",
                        "ok": False,
                        "error": "No valid steps provided",
                    })
                    continue

                chain = TaskChain(
                    name=chain_name,
                    steps=step_tasks,
                    source="dashboard",
                )
                task_queue.add_chain(chain)

                logger.info(
                    "Chain %s submitted from dashboard (%d steps): %s",
                    chain.short_id, chain.total_steps, chain_name,
                )
                event_logger.info(
                    "chain",
                    f"Chain '{chain_name}' ({chain.short_id}) submitted: {chain.total_steps} steps",
                    chain_id=chain.short_id, steps=chain.total_steps,
                    source="dashboard",
                )

                await websocket.send_json({
                    "type": "chain_submitted",
                    "ok": True,
                    "chain_id": chain.chain_id,
                    "short_id": chain.short_id,
                })
                await broadcast_chain_status()
                await broadcast_task_status()

            elif msg.get("type") == "file_send":
                # Dashboard sending a small file (<1MB) via WebSocket
                target_id = msg.get("agent_id", "")
                filename = msg.get("filename", "")
                dest_path = msg.get("dest_path", "")
                b64_data = msg.get("data", "")

                if target_id not in agent_websockets:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": "Agent is offline",
                        "filename": filename,
                    })
                    continue

                try:
                    data = base64.b64decode(b64_data)
                except Exception:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": "Invalid base64 data",
                        "filename": filename,
                    })
                    continue

                file_size = len(data)
                max_size = config.file_transfer.max_file_size
                if file_size > max_size:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": f"File too large ({file_size} bytes, max {max_size})",
                        "filename": filename,
                    })
                    continue

                agent_info = registry.get_by_id(target_id)
                agent_name = agent_info.name if agent_info else target_id
                checksum = compute_checksum(data)
                chunk_size = config.file_transfer.chunk_size
                chunk_count = (file_size + chunk_size - 1) // chunk_size

                try:
                    transfer = ft_manager.create_transfer(
                        direction="to_agent",
                        agent_id=target_id,
                        agent_name=agent_name,
                        filename=filename,
                        file_size=file_size,
                        chunk_count=chunk_count,
                        checksum=checksum,
                        dest_path=dest_path,
                    )
                except ValueError as e:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": str(e),
                        "filename": filename,
                    })
                    continue

                event_logger.info(
                    "file_transfer",
                    f"Sending {filename} ({file_size} bytes) to {agent_name}",
                    transfer_id=transfer.transfer_id, agent=agent_name,
                )
                agent_ws = agent_websockets[target_id]
                asyncio.create_task(relay_file_to_agent(transfer, data, agent_ws))

            elif msg.get("type") == "file_request_from_agent":
                # Dashboard requesting a file from an agent
                target_id = msg.get("agent_id", "")
                file_path = msg.get("file_path", "")

                if target_id not in agent_websockets:
                    await websocket.send_json({
                        "type": "file_transfer_progress",
                        "status": "failed",
                        "error": "Agent is offline",
                        "filename": file_path,
                    })
                    continue

                agent_info = registry.get_by_id(target_id)
                agent_name = agent_info.name if agent_info else target_id

                # Create a placeholder transfer — metadata filled in by file_send_start
                transfer = ft_manager.create_transfer(
                    direction="from_agent",
                    agent_id=target_id,
                    agent_name=agent_name,
                    filename=os.path.basename(file_path) if file_path else "unknown",
                    file_size=0,
                    chunk_count=0,
                )

                event_logger.info(
                    "file_transfer",
                    f"Requesting {file_path} from {agent_name}",
                    transfer_id=transfer.transfer_id, agent=agent_name,
                )

                await agent_websockets[target_id].send_json({
                    "type": "file_request",
                    "transfer_id": transfer.transfer_id,
                    "file_path": file_path,
                })

            elif msg.get("type") == "file_transfer_cancel":
                transfer_id = msg.get("transfer_id", "")
                transfer = ft_manager.get_transfer(transfer_id)
                if transfer:
                    # Notify agent if it's a to_agent transfer
                    if transfer.agent_id in agent_websockets:
                        try:
                            await agent_websockets[transfer.agent_id].send_json({
                                "type": "file_transfer_cancel",
                                "transfer_id": transfer_id,
                                "reason": "Cancelled by user",
                            })
                        except Exception:
                            pass
                    await ft_manager.cancel_transfer(transfer_id)

            elif msg.get("type") == "deploy_check_version":
                # Check connector version on a single agent
                target_id = msg.get("agent_id", "")
                version_info = await deployer.check_version(target_id)
                local = deployer.get_local_version()
                match = (version_info.get("hash") == local["hash"]) if version_info.get("hash") else None
                await websocket.send_json({
                    "type": "deploy_version_result",
                    "agent_id": target_id,
                    "remote": version_info,
                    "local": local,
                    "match": match,
                })

            elif msg.get("type") == "deploy_agent":
                # Deploy connector to a single agent
                target_id = msg.get("agent_id", "")
                result = await deployer.deploy_connector(target_id)
                await broadcast_to_dashboards({
                    "type": "deploy_result",
                    **result.to_dict(),
                })
                # Push updated deploy status to all dashboards
                await broadcast_to_dashboards({
                    "type": "deploy_status",
                    **deployer.get_status(),
                })

            elif msg.get("type") == "deploy_all":
                # Deploy connector to all online agents sequentially
                await deployer.deploy_all(broadcast_fn=broadcast_to_dashboards)
                # Push final status
                await broadcast_to_dashboards({
                    "type": "deploy_status",
                    **deployer.get_status(),
                })

            elif msg.get("type") == "backup_now":
                # Run backup immediately
                result = backup_manager.backup()
                await broadcast_to_dashboards({
                    "type": "backup_result",
                    **result,
                })
                await broadcast_to_dashboards({
                    "type": "backup_status",
                    **backup_manager.get_status(),
                })

            elif msg.get("type") == "backup_restore":
                # Restore from a specific backup archive
                filename = msg.get("filename", "")
                result = backup_manager.restore(filename)
                await broadcast_to_dashboards({
                    "type": "backup_restore_result",
                    **result,
                })

            elif msg.get("type") == "schedule_create":
                sched_name = (msg.get("name") or "").strip()
                sched_desc = (msg.get("description") or "").strip()
                sched_type_str = (msg.get("schedule_type") or "interval").lower()
                complexity_str = (msg.get("complexity") or "auto").lower()
                interval_seconds = msg.get("interval_seconds", 300)
                run_at_time = msg.get("run_at_time", "09:00")
                raw_caps = msg.get("required_capabilities") or []
                caps = [c.strip() for c in raw_caps if isinstance(c, str) and c.strip()]

                if not sched_name or not sched_desc:
                    await websocket.send_json({
                        "type": "schedule_created",
                        "ok": False,
                        "error": "Name and description are required",
                    })
                    continue

                try:
                    complexity = TaskComplexity(complexity_str)
                except ValueError:
                    complexity = TaskComplexity.AUTO

                try:
                    sched_type = ScheduleType(sched_type_str)
                except ValueError:
                    sched_type = ScheduleType.INTERVAL

                try:
                    interval_seconds = int(interval_seconds)
                except (ValueError, TypeError):
                    interval_seconds = 300

                sched = scheduler.add_schedule(
                    name=sched_name,
                    description=sched_desc,
                    complexity=complexity,
                    schedule_type=sched_type,
                    interval_seconds=interval_seconds,
                    run_at_time=run_at_time,
                    required_capabilities=caps,
                )
                event_logger.info(
                    "scheduler",
                    f"Schedule '{sched_name}' ({sched.short_id}) created: {sched_type_str}",
                    schedule_id=sched.short_id, schedule_type=sched_type_str,
                )
                await websocket.send_json({
                    "type": "schedule_created",
                    "ok": True,
                    "schedule_id": sched.schedule_id,
                    "short_id": sched.short_id,
                })
                await broadcast_schedule_status()

            elif msg.get("type") == "schedule_update":
                schedule_id = msg.get("schedule_id", "")
                update_fields = {}
                for key in ("name", "description", "complexity", "schedule_type",
                            "interval_seconds", "run_at_time", "required_capabilities"):
                    if key in msg:
                        update_fields[key] = msg[key]

                result = scheduler.update_schedule(schedule_id, **update_fields)
                if result:
                    event_logger.info(
                        "scheduler",
                        f"Schedule '{result.name}' ({result.short_id}) updated",
                        schedule_id=result.short_id,
                    )
                    await websocket.send_json({"type": "schedule_updated", "ok": True})
                    await broadcast_schedule_status()
                else:
                    await websocket.send_json({
                        "type": "schedule_updated",
                        "ok": False,
                        "error": "Schedule not found",
                    })

            elif msg.get("type") == "schedule_delete":
                schedule_id = msg.get("schedule_id", "")
                removed = scheduler.remove_schedule(schedule_id)
                if removed:
                    event_logger.info(
                        "scheduler",
                        f"Schedule {schedule_id[:8]} deleted",
                        schedule_id=schedule_id[:8],
                    )
                    await websocket.send_json({"type": "schedule_deleted", "ok": True})
                    await broadcast_schedule_status()
                else:
                    await websocket.send_json({
                        "type": "schedule_deleted",
                        "ok": False,
                        "error": "Schedule not found",
                    })

            elif msg.get("type") == "schedule_toggle":
                schedule_id = msg.get("schedule_id", "")
                result = scheduler.toggle_enable(schedule_id)
                if result:
                    event_logger.info(
                        "scheduler",
                        f"Schedule '{result.name}' ({result.short_id}) {'enabled' if result.enabled else 'disabled'}",
                        schedule_id=result.short_id,
                        enabled=result.enabled,
                    )
                    await websocket.send_json({
                        "type": "schedule_toggled",
                        "ok": True,
                        "enabled": result.enabled,
                    })
                    await broadcast_schedule_status()
                else:
                    await websocket.send_json({
                        "type": "schedule_toggled",
                        "ok": False,
                        "error": "Schedule not found",
                    })

            elif msg.get("type") == "content_entry_create":
                title = (msg.get("title") or "").strip()
                description = (msg.get("description") or "").strip()
                content_type = (msg.get("content_type") or "general").lower()
                scheduled_date = msg.get("scheduled_date") or None

                if not title:
                    await websocket.send_json({
                        "type": "content_entry_created",
                        "ok": False,
                        "error": "Title is required",
                    })
                    continue

                entry = content_calendar.add_entry(
                    title=title,
                    content_type=content_type,
                    description=description,
                    scheduled_date=scheduled_date,
                )
                event_logger.info(
                    "content",
                    f"Content entry '{title}' ({entry.short_id}) created",
                    entry_id=entry.short_id, content_type=content_type,
                )
                await websocket.send_json({
                    "type": "content_entry_created",
                    "ok": True,
                    "entry_id": entry.entry_id,
                    "short_id": entry.short_id,
                })
                await broadcast_content_calendar_status()

            elif msg.get("type") == "content_entry_update":
                entry_id = msg.get("entry_id", "")
                update_fields = {}
                for key in ("title", "description", "content_type", "status",
                            "scheduled_date", "body"):
                    if key in msg:
                        update_fields[key] = msg[key]

                result = content_calendar.update_entry(entry_id, **update_fields)
                if result:
                    event_logger.info(
                        "content",
                        f"Content entry '{result.title}' ({result.short_id}) updated",
                        entry_id=result.short_id,
                    )
                    await websocket.send_json({"type": "content_entry_updated", "ok": True})
                    await broadcast_content_calendar_status()
                else:
                    await websocket.send_json({
                        "type": "content_entry_updated",
                        "ok": False,
                        "error": "Entry not found",
                    })

            elif msg.get("type") == "content_entry_delete":
                entry_id = msg.get("entry_id", "")
                removed = content_calendar.remove_entry(entry_id)
                if removed:
                    event_logger.info(
                        "content",
                        f"Content entry {entry_id[:8]} deleted",
                        entry_id=entry_id[:8],
                    )
                    await websocket.send_json({"type": "content_entry_deleted", "ok": True})
                    await broadcast_content_calendar_status()
                else:
                    await websocket.send_json({
                        "type": "content_entry_deleted",
                        "ok": False,
                        "error": "Entry not found",
                    })

            elif msg.get("type") == "research_result_delete":
                entry_id = msg.get("entry_id", "")
                removed = research_store.remove_entry(entry_id)
                if removed:
                    event_logger.info(
                        "research",
                        f"Research entry {entry_id[:8]} deleted",
                        entry_id=entry_id[:8],
                    )
                    await websocket.send_json({"type": "research_result_deleted", "ok": True})
                    await broadcast_research_status()
                else:
                    await websocket.send_json({
                        "type": "research_result_deleted",
                        "ok": False,
                        "error": "Entry not found",
                    })

            elif msg.get("type") == "research_search":
                keyword = (msg.get("keyword") or "").strip()
                if keyword:
                    results = research_store.search_by_query(keyword, limit=20)
                else:
                    results = research_store.list_all(limit=50)
                await websocket.send_json({
                    "type": "research_search_results",
                    "entries": [e.to_dict() for e in results],
                    "keyword": keyword,
                })

            elif msg.get("type") == "scout_scan":
                # Trigger an immediate scout scan
                try:
                    new_listings = await doppler_scout.scan()
                    await websocket.send_json({
                        "type": "scout_scan_result",
                        "ok": True,
                        "new_count": len(new_listings),
                    })
                    await broadcast_scout_status()
                except Exception as e:
                    await websocket.send_json({
                        "type": "scout_scan_result",
                        "ok": False,
                        "error": str(e),
                    })

            elif msg.get("type") == "scout_update_status":
                entry_id = msg.get("entry_id", "")
                new_status = msg.get("status", "")
                if new_status in [s.value for s in ListingStatus]:
                    updated = scout_store.update_entry(entry_id, status=new_status)
                    if updated:
                        event_logger.info(
                            "scout",
                            f"Scout listing {entry_id[:8]} marked {new_status}",
                            entry_id=entry_id[:8], status=new_status,
                        )
                        await broadcast_scout_status()

            elif msg.get("type") == "scout_delete":
                entry_id = msg.get("entry_id", "")
                removed = scout_store.remove_entry(entry_id)
                if removed:
                    event_logger.info(
                        "scout",
                        f"Scout listing {entry_id[:8]} deleted",
                        entry_id=entry_id[:8],
                    )
                    await websocket.send_json({"type": "scout_deleted", "ok": True})
                    await broadcast_scout_status()
                else:
                    await websocket.send_json({
                        "type": "scout_deleted",
                        "ok": False,
                        "error": "Entry not found",
                    })

            elif msg.get("type") == "scout_rescore":
                entry_id = msg.get("entry_id", "")
                try:
                    listing = await doppler_scout.score_single(entry_id)
                    if listing:
                        await websocket.send_json({
                            "type": "scout_rescored",
                            "ok": True,
                            "entry": listing.to_dict(),
                        })
                        await broadcast_scout_status()
                    else:
                        await websocket.send_json({
                            "type": "scout_rescored",
                            "ok": False,
                            "error": "Entry not found",
                        })
                except Exception as e:
                    await websocket.send_json({
                        "type": "scout_rescored",
                        "ok": False,
                        "error": str(e),
                    })

            # -----------------------------------------------------------
            # Business CRUD handlers
            # -----------------------------------------------------------
            elif msg.get("type") == "business_customer_create":
                cust = business_store.add_customer(
                    name=msg.get("name", ""),
                    phone=msg.get("phone", ""),
                    email=msg.get("email", ""),
                    bike_info=msg.get("bike_info", ""),
                    notes=msg.get("notes", ""),
                )
                event_logger.info(
                    "business",
                    f"Customer created: {cust.name} ({cust.short_id})",
                    customer_id=cust.short_id,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_customer_update":
                cust_id = msg.get("customer_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("name", "phone", "email", "bike_info", "notes")}
                updated = business_store.update_customer(cust_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Customer updated: {updated.name} ({updated.short_id})",
                        customer_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_customer_delete":
                cust_id = msg.get("customer_id", "")
                removed = business_store.remove_customer(cust_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Customer deleted: {cust_id[:8]}",
                        customer_id=cust_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_customer_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            elif msg.get("type") == "business_appointment_create":
                appt = business_store.add_appointment(
                    customer_id=msg.get("customer_id", ""),
                    customer_name=msg.get("customer_name", ""),
                    date=msg.get("date", ""),
                    time=msg.get("time", ""),
                    bike=msg.get("bike", ""),
                    service_type=msg.get("service_type", ""),
                    status=msg.get("status", "scheduled"),
                    notes=msg.get("notes", ""),
                    location=msg.get("location", ""),
                    estimated_cost=float(msg.get("estimated_cost", 0)),
                )
                event_logger.info(
                    "business",
                    f"Appointment created: {appt.customer_name} on {appt.date} ({appt.short_id})",
                    appointment_id=appt.short_id,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_appointment_update":
                appt_id = msg.get("appointment_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("customer_id", "customer_name", "date", "time",
                                    "bike", "service_type", "status", "notes",
                                    "location", "estimated_cost")}
                if "estimated_cost" in updates:
                    updates["estimated_cost"] = float(updates["estimated_cost"])
                updated = business_store.update_appointment(appt_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Appointment updated: {updated.short_id}",
                        appointment_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_appointment_delete":
                appt_id = msg.get("appointment_id", "")
                removed = business_store.remove_appointment(appt_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Appointment deleted: {appt_id[:8]}",
                        appointment_id=appt_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_appointment_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            elif msg.get("type") == "business_invoice_create":
                items = msg.get("items", [])
                inv = business_store.add_invoice(
                    customer_id=msg.get("customer_id", ""),
                    customer_name=msg.get("customer_name", ""),
                    date=msg.get("date", ""),
                    items=items,
                    labor_hours=float(msg.get("labor_hours", 0)),
                    labor_rate=float(msg.get("labor_rate",
                                             getattr(_biz_cfg, 'default_labor_rate', 75.0))),
                    status=msg.get("status", "draft"),
                    notes=msg.get("notes", ""),
                    appointment_id=msg.get("appointment_id", ""),
                    prefix=getattr(_biz_cfg, 'invoice_prefix', 'DC'),
                )
                event_logger.info(
                    "business",
                    f"Invoice created: {inv.invoice_number} — ${inv.total:.2f} ({inv.short_id})",
                    invoice_id=inv.short_id, invoice_number=inv.invoice_number,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_invoice_update":
                inv_id = msg.get("invoice_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("customer_id", "customer_name", "date", "items",
                                    "labor_hours", "labor_rate", "status", "notes",
                                    "appointment_id")}
                for float_key in ("labor_hours", "labor_rate"):
                    if float_key in updates:
                        updates[float_key] = float(updates[float_key])
                updated = business_store.update_invoice(inv_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Invoice updated: {updated.invoice_number} ({updated.short_id})",
                        invoice_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_invoice_delete":
                inv_id = msg.get("invoice_id", "")
                removed = business_store.remove_invoice(inv_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Invoice deleted: {inv_id[:8]}",
                        invoice_id=inv_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_invoice_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            elif msg.get("type") == "business_inventory_create":
                item = business_store.add_inventory_item(
                    name=msg.get("name", ""),
                    category=msg.get("category", ""),
                    quantity=int(msg.get("quantity", 0)),
                    cost=float(msg.get("cost", 0)),
                    location=msg.get("location", ""),
                    reorder_threshold=int(msg.get("reorder_threshold", 0)),
                    supplier=msg.get("supplier", ""),
                    part_number=msg.get("part_number", ""),
                    notes=msg.get("notes", ""),
                )
                event_logger.info(
                    "business",
                    f"Inventory item created: {item.name} ({item.short_id})",
                    item_id=item.short_id,
                )
                await broadcast_business_status()

            elif msg.get("type") == "business_inventory_update":
                item_id = msg.get("item_id", "")
                updates = {k: v for k, v in msg.items()
                           if k in ("name", "category", "quantity", "cost", "location",
                                    "reorder_threshold", "supplier", "part_number", "notes")}
                for int_key in ("quantity", "reorder_threshold"):
                    if int_key in updates:
                        updates[int_key] = int(updates[int_key])
                for float_key in ("cost",):
                    if float_key in updates:
                        updates[float_key] = float(updates[float_key])
                updated = business_store.update_inventory_item(item_id, **updates)
                if updated:
                    event_logger.info(
                        "business",
                        f"Inventory item updated: {updated.name} ({updated.short_id})",
                        item_id=updated.short_id,
                    )
                    await broadcast_business_status()

            elif msg.get("type") == "business_inventory_delete":
                item_id = msg.get("item_id", "")
                removed = business_store.remove_inventory_item(item_id)
                if removed:
                    event_logger.info(
                        "business",
                        f"Inventory item deleted: {item_id[:8]}",
                        item_id=item_id[:8],
                    )
                await websocket.send_json({
                    "type": "business_inventory_deleted",
                    "ok": removed,
                })
                if removed:
                    await broadcast_business_status()

            elif msg.get("type") == "approval_response":
                # Dashboard approving or rejecting a Tier 2 action
                task_id = msg.get("task_id", "")
                approved = msg.get("approved", False)
                orchestrator.resolve_approval(task_id, approved)
                action_str = "approved" if approved else "rejected"
                event_logger.info(
                    "security",
                    f"Tier 2 action {action_str} from dashboard (task {task_id[:8]})",
                    task_id=task_id[:8], action=action_str,
                )
                # Audit log update happens inside orchestrator.act()
                # Broadcast updated audit status after a brief delay
                await asyncio.sleep(0.2)
                await broadcast_security_audit_status()

            elif msg.get("type") == "security_audit_filter":
                # Dashboard requesting filtered audit entries
                risk_level = msg.get("risk_level") or None
                actor = msg.get("actor") or None
                result_filter = msg.get("result") or None
                tier = msg.get("tier")
                limit = min(int(msg.get("limit", 200)), 500)

                if tier is not None:
                    try:
                        tier = int(tier)
                    except (ValueError, TypeError):
                        tier = None

                entries = security_audit_log.filter_entries(
                    risk_level=risk_level,
                    actor=actor,
                    result=result_filter,
                    tier=tier,
                    limit=limit,
                )
                await websocket.send_json({
                    "type": "security_audit_filtered",
                    "entries": [e.to_dict() for e in entries],
                })

            elif msg.get("type") == "notification_dismiss":
                notif_id = msg.get("id", "")
                notification_manager.dismiss(notif_id)
                await broadcast_to_dashboards({
                    "type": "notification_count",
                    "unread_count": notification_manager.get_unread_count(),
                })

            elif msg.get("type") == "notification_dismiss_all":
                notification_manager.dismiss_all()
                await broadcast_to_dashboards({
                    "type": "notification_count",
                    "unread_count": notification_manager.get_unread_count(),
                })

            elif msg.get("type") == "run_self_test":
                # Run the system self-test suite
                global last_self_test_results
                self_test = SelfTest(
                    config=config,
                    event_logger=event_logger,
                    router=orchestrator.router,
                    registry=registry,
                    task_queue=task_queue,
                    short_term_memory=short_term_memory,
                    working_memory=working_memory,
                    episodic_memory=episodic_memory,
                    long_term_memory=long_term_memory,
                    analytics=analytics,
                    content_calendar=content_calendar,
                    research_store=research_store,
                    security_audit_log=security_audit_log,
                    agent_websockets=agent_websockets,
                    port=config.dashboard.port,
                )
                results = await self_test.run_all()
                last_self_test_results = results
                await broadcast_to_dashboards({
                    "type": "self_test_results",
                    **results,
                })

            elif msg.get("type") == "tts_synthesize":
                tts_text = (msg.get("text") or "").strip()
                tts_voice = msg.get("voice") or None
                if not tts_text:
                    await websocket.send_json({
                        "type": "tts_result",
                        "ok": False,
                        "error": "Text is required",
                    })
                    continue

                tts_result = await tts_pipeline.synthesize(
                    text=tts_text, voice=tts_voice,
                )
                result_data = {
                    "type": "tts_result",
                    "ok": tts_result.error is None,
                    **tts_result.to_dict(),
                }
                await broadcast_to_dashboards(result_data)
                if not tts_result.error:
                    event_logger.info(
                        "tts",
                        f"Synthesized audio: {tts_result.audio_path} ({tts_result.duration_secs:.1f}s)",
                        voice=tts_result.voice,
                    )

            elif msg.get("type") == "tts_queue":
                tts_items = msg.get("items") or []
                tts_results = await tts_pipeline.queue_synthesis(tts_items)
                await broadcast_to_dashboards({
                    "type": "tts_queue_result",
                    "results": [r.to_dict() for r in tts_results],
                })

            elif msg.get("type") == "tts_list_audio":
                await websocket.send_json({
                    "type": "tts_audio_list",
                    "audio": tts_pipeline.list_audio(),
                })

            elif msg.get("type") == "tts_list_voices":
                await websocket.send_json({
                    "type": "tts_voices",
                    "voices": tts_pipeline.list_voices(),
                })

            elif msg.get("type") == "tts_delete":
                tts_filename = msg.get("filename", "")
                tts_error = tts_pipeline.delete_audio(tts_filename)
                if tts_error:
                    await websocket.send_json({
                        "type": "tts_deleted",
                        "ok": False,
                        "error": tts_error,
                    })
                else:
                    event_logger.info("tts", f"Audio deleted: {tts_filename}")
                    await broadcast_to_dashboards({
                        "type": "tts_deleted",
                        "ok": True,
                        "filename": tts_filename,
                    })
                    # Broadcast updated audio list
                    await broadcast_to_dashboards({
                        "type": "tts_audio_list",
                        "audio": tts_pipeline.list_audio(),
                    })

            elif msg.get("type") == "tts_status":
                await websocket.send_json({
                    "type": "tts_status",
                    **tts_pipeline.get_status(),
                })

            elif msg.get("type") == "episodic_search":
                # Search episodic memory with optional filters
                keyword = msg.get("keyword", "")
                start_time = msg.get("start_time")
                end_time = msg.get("end_time")
                participant = msg.get("participant")
                task_id = msg.get("task_id")
                limit = min(int(msg.get("limit", 50)), 200)
                offset = int(msg.get("offset", 0))

                results = episodic_memory.search(
                    keyword=keyword or None,
                    start_time=start_time,
                    end_time=end_time,
                    participant=participant,
                    task_id=task_id,
                    limit=limit,
                    offset=offset,
                )
                await websocket.send_json({
                    "type": "episodic_search_results",
                    "episodes": [ep.to_dict() for ep in results],
                    "keyword": keyword,
                    "count": len(results),
                })

            # --- Message Bus: publish from dashboard ---
            elif msg.get("type") == "bus_publish":
                sender = msg.get("sender", "dashboard")
                channel = msg.get("channel", "")
                payload = msg.get("payload", {})
                result = await message_bus.publish(sender, channel, payload)
                await websocket.send_json({
                    "type": "bus_publish_result",
                    "ok": result is not None,
                    "message_id": result.message_id if result else None,
                })

            # --- Message Bus: fetch history ---
            elif msg.get("type") == "bus_history":
                channel = msg.get("channel")  # None = all channels
                count = min(int(msg.get("count", 50)), 200)
                await websocket.send_json({
                    "type": "bus_history_result",
                    "messages": message_bus.get_recent(channel=channel, count=count),
                    "stats": message_bus.get_stats(),
                    "matrix": message_bus.get_sender_channel_matrix(),
                })

    except WebSocketDisconnect:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)
        logger.info("Dashboard client disconnected")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.dashboard.host,
        port=config.dashboard.port,
        log_level=config.dashboard.log_level,
    )
