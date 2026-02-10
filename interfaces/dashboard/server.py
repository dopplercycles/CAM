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
import json
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.agent_registry import AgentRegistry
from core.task import TaskQueue, TaskComplexity
from core.orchestrator import Orchestrator


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
# Constants
# ---------------------------------------------------------------------------

# How often to run the heartbeat check (seconds)
HEARTBEAT_CHECK_INTERVAL = 10

# How long before we consider an agent offline (seconds)
# Passed to the AgentRegistry. Set higher than the agent's heartbeat
# interval (default 30s) to allow for network jitter.
HEARTBEAT_TIMEOUT = 45

# Path to static files directory (index.html lives here)
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Shared state
#
# The registry is the single source of truth for agent status.
# WebSocket connections and dashboard clients are server-specific.
# ---------------------------------------------------------------------------

# Agent registry — tracks all agents, shared with orchestrator later
registry = AgentRegistry(heartbeat_timeout=HEARTBEAT_TIMEOUT)

# agent_id -> WebSocket (live connections, needed for commands + kill switch)
agent_websockets: dict[str, WebSocket] = {}

# Browser clients watching the dashboard (receive status pushes)
dashboard_clients: list[WebSocket] = []

# Task queue — shared between dashboard and orchestrator
task_queue = TaskQueue()

# Pending task dispatch responses — task_id → asyncio.Future
# Used by dispatch_to_agent() to wait for agent command_responses
pending_task_responses: dict[str, asyncio.Future] = {}

# Kill switch state — when True, all autonomous action is halted
kill_switch_active: bool = False

# Background task handle for the orchestrator loop
orchestrator_task: asyncio.Task | None = None


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


async def on_task_update():
    """Called by the orchestrator when any task's status changes."""
    await broadcast_task_status()


# ---------------------------------------------------------------------------
# Agent dispatch — sends tasks to remote agents, waits for response
# ---------------------------------------------------------------------------

# How long to wait for an agent to respond before timing out (seconds)
AGENT_DISPATCH_TIMEOUT = 120


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

    if task.assigned_agent:
        # Check if the pre-assigned agent is online and connected
        agent_info = registry.get_by_id(task.assigned_agent)
        if agent_info and agent_info.status == "online" and task.assigned_agent in agent_websockets:
            agent_id = task.assigned_agent

    if agent_id is None:
        # Pick the first available agent
        available = registry.get_available()
        for agent_info in available:
            if agent_info.agent_id in agent_websockets:
                agent_id = agent_info.agent_id
                break

    if agent_id is None:
        logger.info("No agents available for task %s — falling back to model", task.short_id)
        return None

    agent_info = registry.get_by_id(agent_id)
    agent_name = agent_info.name if agent_info else agent_id

    # --- Mark agent busy, set task assignment ---
    task.assigned_agent = agent_name
    if agent_info:
        agent_info.status = "busy"
    await broadcast_agent_status()

    logger.info(
        "Dispatching task %s to agent '%s' (%s)",
        task.short_id, agent_name, agent_id,
    )
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
        result = await asyncio.wait_for(future, timeout=AGENT_DISPATCH_TIMEOUT)
        logger.info(
            "Agent '%s' responded to task %s: %.200s",
            agent_name, task.short_id, result,
        )
        return result

    except asyncio.TimeoutError:
        logger.warning(
            "Agent '%s' timed out on task %s after %ds",
            agent_name, task.short_id, AGENT_DISPATCH_TIMEOUT,
        )
        return f"[timeout] Agent '{agent_name}' did not respond within {AGENT_DISPATCH_TIMEOUT}s"

    except Exception as e:
        logger.warning(
            "Dispatch to agent '%s' failed for task %s: %s",
            agent_name, task.short_id, e,
        )
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

orchestrator = Orchestrator(
    queue=task_queue,
    on_phase_change=on_task_phase_change,
    on_task_update=on_task_update,
    on_dispatch_to_agent=dispatch_to_agent,
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
    logger.critical("KILL SWITCH ACTIVATED — shutting down all agents")

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
    """Background task: mark agents offline if heartbeat times out.

    Delegates the actual check to the registry's heartbeat_check()
    method, which knows the timeout and handles the status changes.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL)
        timed_out = registry.heartbeat_check()
        if timed_out:
            await broadcast_agent_status()


# ---------------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, clean up on shutdown."""
    global orchestrator_task
    logger.info("CAM Dashboard starting up")
    heartbeat_task = asyncio.create_task(check_heartbeats())
    orchestrator_task = asyncio.create_task(orchestrator.run())
    logger.info("Orchestrator loop started as background task")
    yield
    orchestrator.stop()
    orchestrator_task.cancel()
    heartbeat_task.cancel()
    logger.info("CAM Dashboard shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="CAM Dashboard", lifespan=lifespan)

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
                else:
                    # Subsequent heartbeat — update via registry
                    registry.update_heartbeat(
                        agent_id=agent_id,
                        ip_address=ip,
                        capabilities=capabilities,
                    )

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

    except WebSocketDisconnect:
        logger.info("Agent '%s' disconnected", agent_id)
        # Remove from live WebSocket tracking
        agent_websockets.pop(agent_id, None)
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
                complexity_str = (msg.get("complexity") or "low").lower()

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
                    complexity = TaskComplexity.LOW

                task = task_queue.add_task(
                    description=description,
                    source="dashboard",
                    complexity=complexity,
                )
                logger.info(
                    "Task %s submitted from dashboard: %s",
                    task.short_id, description,
                )

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

    except WebSocketDisconnect:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)
        logger.info("Dashboard client disconnected")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
