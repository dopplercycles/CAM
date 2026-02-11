"""
CAM REST API — /api/v1/

Versioned REST endpoints for external tools, scripts, and the future mobile
app. Mounted on the existing FastAPI app; accesses shared state through
``request.app.state`` to avoid circular imports.

Auth: optional API key via ``X-API-Key`` header.  Set ``api.api_key`` in
config/settings.toml or the ``CAM_API_KEY`` env var.  Empty key = open access.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

logger = logging.getLogger("cam.api")

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(_api_key_header),
):
    """Check X-API-Key against the configured key.

    If the configured key is empty (default), auth is disabled and all
    requests are allowed through.  When a key is set, requests without
    a matching header receive 403.
    """
    configured_key: str = request.app.state.config.api.api_key
    if not configured_key:
        return  # open access
    if api_key != configured_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)],
)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class TaskSubmit(BaseModel):
    description: str
    complexity: str = "low"
    assigned_agent: Optional[str] = None
    required_capabilities: list[str] = []


class CommandSubmit(BaseModel):
    command: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# -- Agents ------------------------------------------------------------------

@router.get("/agents")
async def list_agents(request: Request):
    """Return all known agents."""
    registry = request.app.state.registry
    return {"agents": registry.to_broadcast_list()}


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str, request: Request):
    """Return a single agent by ID. 404 if not found."""
    registry = request.app.state.registry
    agent = registry.get_by_id(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {"agent": agent.to_dict()}


@router.post("/agents/{agent_id}/command")
async def send_command(agent_id: str, body: CommandSubmit, request: Request):
    """Send a fire-and-forget command to a connected agent.

    Returns immediately — use task submission for tracked work.
    """
    agent_websockets = request.app.state.agent_websockets
    registry = request.app.state.registry

    agent = registry.get_by_id(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    ws = agent_websockets.get(agent_id)
    if ws is None:
        return {"ok": False, "error": "Agent is offline"}

    try:
        await ws.send_json({"type": "command", "command": body.command})
        logger.info("API command sent to '%s': %s", agent_id, body.command)
        return {"ok": True, "agent_id": agent_id, "command": body.command}
    except Exception as e:
        logger.warning("API command to '%s' failed: %s", agent_id, e)
        return {"ok": False, "error": "Agent connection lost"}


# -- Tasks -------------------------------------------------------------------

@router.get("/tasks")
async def list_tasks(request: Request):
    """Return all tasks with status counts."""
    task_queue = request.app.state.task_queue
    return {
        "tasks": task_queue.to_broadcast_list(),
        "counts": task_queue.get_status(),
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    """Return a single task by full or short ID. 404 if not found."""
    task_queue = request.app.state.task_queue
    task = task_queue.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {"task": task.to_dict()}


@router.post("/tasks")
async def submit_task(body: TaskSubmit, request: Request):
    """Submit a new task to the queue."""
    from core.task import TaskComplexity

    task_queue = request.app.state.task_queue
    event_logger = request.app.state.event_logger

    description = body.description.strip()
    if not description:
        raise HTTPException(status_code=422, detail="Description is required")

    try:
        complexity = TaskComplexity(body.complexity.lower())
    except ValueError:
        complexity = TaskComplexity.LOW

    caps = [c.strip() for c in body.required_capabilities if c.strip()]

    task = task_queue.add_task(
        description=description,
        source="api",
        complexity=complexity,
        assigned_agent=body.assigned_agent,
        required_capabilities=caps or None,
    )

    logger.info("Task %s submitted via API: %s", task.short_id, description[:80])
    event_logger.info(
        "task",
        f"Task {task.short_id} submitted: {description[:80]}",
        task_id=task.short_id,
        complexity=complexity.value,
        required_capabilities=caps or None,
        source="api",
    )

    return {"ok": True, "task_id": task.task_id, "short_id": task.short_id}


# -- Analytics ---------------------------------------------------------------

@router.get("/analytics")
async def get_analytics(request: Request):
    """Return aggregate analytics summary."""
    analytics = request.app.state.analytics
    return {"analytics": analytics.get_summary()}


# -- Events ------------------------------------------------------------------

@router.get("/events")
async def get_events(request: Request, count: int = 200):
    """Return recent events from the event log."""
    event_logger = request.app.state.event_logger
    return {"events": event_logger.get_recent(count)}


# -- Health ------------------------------------------------------------------

@router.get("/health")
async def get_health(request: Request):
    """Return current agent health metrics."""
    health_monitor = request.app.state.health_monitor
    return {"health": health_monitor.to_broadcast_dict()}


# -- Kill Switch -------------------------------------------------------------

@router.get("/kill-switch")
async def get_kill_switch(request: Request):
    """Return whether the kill switch is currently active."""
    return {"active": request.app.state.kill_switch["active"]}


@router.post("/kill-switch")
async def activate_kill_switch_endpoint(request: Request):
    """Activate the kill switch — halts all autonomous action."""
    activate_fn = request.app.state.activate_kill_switch
    await activate_fn()
    logger.critical("Kill switch activated via REST API")
    return {"ok": True}


# -- Schedules ---------------------------------------------------------------

@router.get("/schedules")
async def list_schedules(request: Request):
    """Return all scheduled tasks."""
    scheduler = request.app.state.scheduler
    return {"schedules": scheduler.to_broadcast_list()}
