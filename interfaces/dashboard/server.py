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
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


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

# How often agents should send heartbeats (seconds)
HEARTBEAT_INTERVAL = 10

# How long before we consider an agent offline (seconds)
# Set higher than the agent's heartbeat interval (default 30s) to
# allow for network jitter without false offline flickers.
HEARTBEAT_TIMEOUT = 45

# Path to static files directory (index.html lives here)
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class AgentStatus(BaseModel):
    """Tracks the current status of a connected agent.

    Each agent that connects via WebSocket gets one of these.
    The dashboard broadcasts the full list to browser clients
    whenever anything changes.
    """

    agent_id: str
    name: str
    ip_address: str
    status: str = "online"  # "online" or "offline"
    last_heartbeat: datetime | None = None
    connected_at: datetime | None = None


# ---------------------------------------------------------------------------
# In-memory state
#
# Ephemeral — lost on restart, rebuilt as agents reconnect.
# No database needed for a handful of agents on a local network.
# ---------------------------------------------------------------------------

# agent_id -> AgentStatus
connected_agents: dict[str, AgentStatus] = {}

# Browser clients watching the dashboard (receive status pushes)
dashboard_clients: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Broadcast helper
# ---------------------------------------------------------------------------

async def broadcast_agent_status():
    """Push current agent status to all connected dashboard browsers.

    Sends the full agent list each time — simple and correct for a
    small number of agents. Optimize later if needed.
    """
    data = [agent.model_dump(mode="json") for agent in connected_agents.values()]
    message = {"type": "agent_status", "agents": data}

    # Send to all dashboard clients, clean up any that disconnected
    disconnected = []
    for client in dashboard_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        dashboard_clients.remove(client)


# ---------------------------------------------------------------------------
# Background heartbeat checker
# ---------------------------------------------------------------------------

async def check_heartbeats():
    """Background task: mark agents offline if heartbeat times out.

    Runs every HEARTBEAT_INTERVAL seconds. If an agent hasn't sent
    a heartbeat within HEARTBEAT_TIMEOUT seconds, mark it offline
    and notify all dashboard browsers.

    This catches "dirty" disconnects where the WebSocket drops
    without a clean close frame (e.g., network cable pulled).
    """
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        now = datetime.now(timezone.utc)
        changed = False

        for agent in connected_agents.values():
            if agent.status == "online" and agent.last_heartbeat:
                elapsed = (now - agent.last_heartbeat).total_seconds()
                if elapsed > HEARTBEAT_TIMEOUT:
                    agent.status = "offline"
                    changed = True
                    logger.warning(
                        "Agent '%s' (%s) timed out after %.0fs",
                        agent.name, agent.agent_id, elapsed,
                    )

        if changed:
            await broadcast_agent_status()


# ---------------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, clean up on shutdown."""
    logger.info("CAM Dashboard starting up")
    heartbeat_task = asyncio.create_task(check_heartbeats())
    yield
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
    return list(connected_agents.values())


# ---------------------------------------------------------------------------
# WebSocket: agents connect here to send heartbeats
# ---------------------------------------------------------------------------

@app.websocket("/ws/agent/{agent_id}")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for agents to connect and send heartbeats.

    Agents open a connection to /ws/agent/{their_id} and send JSON:
        {"type": "heartbeat", "name": "AgentName", "ip": "192.168.1.x"}

    On connect: agent is registered and marked online.
    On heartbeat: last_heartbeat timestamp updated.
    On disconnect: agent marked offline immediately.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("Agent '%s' WebSocket opened from %s", agent_id, client_host)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "heartbeat":
                now = datetime.now(timezone.utc)

                if agent_id not in connected_agents:
                    # First heartbeat — register the agent
                    connected_agents[agent_id] = AgentStatus(
                        agent_id=agent_id,
                        name=data.get("name", agent_id),
                        ip_address=data.get("ip", client_host),
                        status="online",
                        last_heartbeat=now,
                        connected_at=now,
                    )
                    logger.info(
                        "Agent registered: %s (%s) at %s",
                        data.get("name", agent_id), agent_id,
                        data.get("ip", client_host),
                    )
                else:
                    # Subsequent heartbeat — update timestamp and status
                    agent = connected_agents[agent_id]
                    agent.last_heartbeat = now
                    agent.status = "online"
                    agent.ip_address = data.get("ip", agent.ip_address)

                await broadcast_agent_status()

    except WebSocketDisconnect:
        logger.info("Agent '%s' disconnected", agent_id)
        if agent_id in connected_agents:
            connected_agents[agent_id].status = "offline"
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
    data = [agent.model_dump(mode="json") for agent in connected_agents.values()]
    await websocket.send_json({"type": "agent_status", "agents": data})

    try:
        # Keep connection alive — listen for messages (future: dashboard commands)
        while True:
            await websocket.receive_text()
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
