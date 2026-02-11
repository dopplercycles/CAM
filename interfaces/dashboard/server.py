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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.agent_registry import AgentRegistry
from core.config import get_config
from core.event_logger import EventLogger
from core.file_transfer import FileTransferManager, chunk_file_data, compute_checksum
from core.health_monitor import HealthMonitor
from core.memory import ShortTermMemory, WorkingMemory, LongTermMemory
from core.notifications import NotificationManager
from core.task import TaskQueue, TaskComplexity, TaskChain, Task, ChainStatus
from core.orchestrator import Orchestrator
from core.analytics import Analytics
from core.commands import CommandLibrary
from core.scheduler import Scheduler, ScheduleType


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
    })


async def broadcast_schedule_status():
    """Push current schedule state to all connected dashboard browsers."""
    await broadcast_to_dashboards({
        "type": "schedule_status",
        "schedules": scheduler.to_broadcast_list(),
    })


# Task scheduler — persists schedules to JSON, submits due tasks to the queue
scheduler = Scheduler(
    task_queue=task_queue,
    persist_path=config.scheduler.persist_file,
    check_interval=config.scheduler.check_interval,
    on_schedule_change=broadcast_schedule_status,
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


orchestrator = Orchestrator(
    queue=task_queue,
    short_term_memory=short_term_memory,
    working_memory=working_memory,
    long_term_memory=long_term_memory,
    on_phase_change=on_task_phase_change,
    on_task_update=on_task_update,
    on_dispatch_to_agent=dispatch_to_agent,
    on_model_call=on_model_call,
    on_chain_update=on_chain_update,
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
    global orchestrator_task
    logger.info("CAM Dashboard starting up")
    event_logger.info("system", "CAM Dashboard starting up")
    heartbeat_task = asyncio.create_task(check_heartbeats())
    orchestrator_task = asyncio.create_task(orchestrator.run())
    scheduler_task = asyncio.create_task(scheduler.run())
    logger.info("Orchestrator loop started as background task")
    logger.info("Scheduler loop started as background task")
    yield
    orchestrator.stop()
    scheduler.stop()
    orchestrator_task.cancel()
    heartbeat_task.cancel()
    scheduler_task.cancel()
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
app.state.kill_switch = {"active": False}       # mutable container
app.state.activate_kill_switch = activate_kill_switch

# Mount versioned REST API
from interfaces.api.routes import router as api_router
app.include_router(api_router)

# Mount static files (CSS, JS, images if any)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the main dashboard page."""
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
    }


@app.get("/api/schedules")
async def get_schedules():
    """Return current scheduled tasks as JSON."""
    return {"schedules": scheduler.to_broadcast_list()}


@app.get("/api/config")
async def get_config_endpoint():
    """Return current configuration as JSON."""
    return {"config": config.to_dict(), "last_loaded": config.last_loaded}


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
    """
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

    # Send current schedule state
    await websocket.send_json({
        "type": "schedule_status",
        "schedules": scheduler.to_broadcast_list(),
    })

    # Send current memory system status
    await websocket.send_json({
        "type": "memory_status",
        "short_term": short_term_memory.get_status(),
        "working": working_memory.get_status(),
        "long_term": long_term_memory.get_status(),
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
