"""
CAM Agent Deployer

Orchestrates connector updates to remote agents using the existing
file transfer and command dispatch systems — no SSH/SCP needed.

Flow:
    1. Read local agents/connector.py, compute SHA-256 hash
    2. Push file to agent via WebSocket file transfer (ft_manager + relay_fn)
    3. Send restart_service command via WS command dispatch
    4. Wait for agent to reconnect
    5. Log result

Version tracking uses SHA-256 of file contents (first 12 hex chars for display).
"""

import asyncio
import hashlib
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from core.file_transfer import FileTransferManager, chunk_file_data, compute_checksum

logger = logging.getLogger("cam.deployer")

# Path to the connector source relative to the project root
CONNECTOR_REL_PATH = "agents/connector.py"


@dataclass
class DeployResult:
    """Outcome of a single connector deploy attempt."""

    agent_id: str
    agent_name: str
    ok: bool
    message: str
    old_version: str = ""       # short hash before deploy
    new_version: str = ""       # short hash after deploy
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "ok": self.ok,
            "message": self.message,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class Deployer:
    """Orchestrates connector deployments to remote agents.

    Uses the existing file transfer manager and relay function to push
    files, and sends commands via agent WebSockets to restart services.

    Args:
        ft_manager:        FileTransferManager for creating/tracking transfers.
        event_logger:      EventLogger for audit trail.
        registry:          AgentRegistry for looking up agent info.
        agent_websockets:  Dict of agent_id -> WebSocket (live connections).
        relay_fn:          The relay_file_to_agent async function.
        config:            CAM config object.
    """

    def __init__(
        self,
        ft_manager: FileTransferManager,
        event_logger,
        registry,
        agent_websockets: dict,
        relay_fn: Callable,
        config,
    ):
        self._ft = ft_manager
        self._events = event_logger
        self._registry = registry
        self._agent_ws = agent_websockets
        self._relay_fn = relay_fn
        self._config = config
        self._history: deque[DeployResult] = deque(maxlen=50)
        self._version_cache: dict[str, dict] = {}  # agent_id -> version info

        # Resolve connector path once
        self._connector_path = Path(CONNECTOR_REL_PATH).resolve()

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------

    def get_local_version(self) -> dict:
        """Read the local connector file and return its version info.

        Returns:
            {"hash": "full hex", "short": "first 12 chars", "size": bytes, "path": str}
        """
        try:
            data = self._connector_path.read_bytes()
            full_hash = hashlib.sha256(data).hexdigest()
            return {
                "hash": full_hash,
                "short": full_hash[:12],
                "size": len(data),
                "path": str(self._connector_path),
            }
        except FileNotFoundError:
            logger.error("Connector file not found: %s", self._connector_path)
            return {"hash": "", "short": "not found", "size": 0, "path": str(self._connector_path)}

    async def check_version(self, agent_id: str) -> dict:
        """Query an agent for its connector version via WS command.

        Sends the 'connector_version' command and awaits the response.
        Gracefully handles agents that don't support the command yet.

        Returns:
            {"hash": str, "short": str, "size": int, "path": str, "error": str|None}
        """
        ws = self._agent_ws.get(agent_id)
        if not ws:
            return {"hash": "", "short": "offline", "size": 0, "path": "", "error": "Agent offline"}

        try:
            # Send the command and wait for response via a Future
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            # We'll use a unique task_id to correlate the response
            task_id = f"deploy-version-{agent_id}-{os.urandom(4).hex()}"

            # Store the future where the agent WS handler can resolve it
            # We piggyback on the existing pending_task_responses dict
            from interfaces.dashboard.server import pending_task_responses
            pending_task_responses[task_id] = future

            await ws.send_json({
                "type": "command",
                "command": "connector_version",
                "task_id": task_id,
            })

            # Wait up to 10s for the response
            response_text = await asyncio.wait_for(future, timeout=10.0)

            # Parse the response — format: "hash=<hex> size=<n> path=<p>"
            info = self._parse_version_response(response_text)
            self._version_cache[agent_id] = info
            return info

        except asyncio.TimeoutError:
            result = {"hash": "", "short": "timeout", "size": 0, "path": "", "error": "Version check timed out"}
            self._version_cache[agent_id] = result
            return result
        except Exception as e:
            result = {"hash": "", "short": "unknown", "size": 0, "path": "", "error": str(e)}
            self._version_cache[agent_id] = result
            return result
        finally:
            # Always clean up
            from interfaces.dashboard.server import pending_task_responses
            pending_task_responses.pop(task_id, None)

    def _parse_version_response(self, text: str) -> dict:
        """Parse a connector_version response string into a version dict."""
        info = {"hash": "", "short": "unknown", "size": 0, "path": "", "error": None}

        if text.startswith("Unknown command"):
            info["short"] = "unsupported"
            info["error"] = "Agent connector too old (no connector_version command)"
            return info

        # Expected format: "hash=<hex> size=<n> path=<p>"
        for part in text.split():
            if part.startswith("hash="):
                h = part[5:]
                info["hash"] = h
                info["short"] = h[:12]
            elif part.startswith("size="):
                try:
                    info["size"] = int(part[5:])
                except ValueError:
                    pass
            elif part.startswith("path="):
                info["path"] = part[5:]

        return info

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    async def deploy_connector(self, agent_id: str) -> DeployResult:
        """Deploy the local connector.py to a single agent.

        Steps:
            1. Read local connector file
            2. Get remote version (graceful if unsupported)
            3. Push file via existing file transfer system
            4. Wait for transfer completion
            5. Send restart_service command
            6. Wait for agent to reconnect
            7. Log and return result
        """
        agent_info = self._registry.get_by_id(agent_id)
        agent_name = agent_info.name if agent_info else agent_id

        result = DeployResult(agent_id=agent_id, agent_name=agent_name, ok=False, message="")

        # Step 1: Read local connector
        local = self.get_local_version()
        if not local["hash"]:
            result.message = "Local connector file not found"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            return result

        result.new_version = local["short"]
        data = self._connector_path.read_bytes()

        # Step 2: Get remote version (optional — old connectors won't have it)
        ws = self._agent_ws.get(agent_id)
        if not ws:
            result.message = "Agent is offline"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            return result

        remote = await self.check_version(agent_id)
        result.old_version = remote.get("short", "unknown")
        remote_path = remote.get("path", "")

        # If versions match, skip deploy
        if remote.get("hash") and remote["hash"] == local["hash"]:
            result.ok = True
            result.message = "Already up to date"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            self._events.info(
                "deploy",
                f"Connector on {agent_name} already up to date ({local['short']})",
                agent_id=agent_id, version=local["short"],
            )
            return result

        # Step 3: Create file transfer
        dest_path = remote_path if remote_path else "~/connector.py"
        checksum = compute_checksum(data)
        chunk_size = self._config.file_transfer.chunk_size
        chunk_count = (len(data) + chunk_size - 1) // chunk_size

        try:
            transfer = self._ft.create_transfer(
                direction="to_agent",
                agent_id=agent_id,
                agent_name=agent_name,
                filename="connector.py",
                file_size=len(data),
                chunk_count=chunk_count,
                checksum=checksum,
                dest_path=dest_path,
            )
        except ValueError as e:
            result.message = f"Transfer creation failed: {e}"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            return result

        # Step 4: Push file and wait for completion
        logger.info("Deploy to %s: starting file transfer %s", agent_name, transfer.transfer_id)
        asyncio.create_task(self._relay_fn(transfer, data, ws))

        # Poll for transfer completion (timeout 30s)
        try:
            await self._wait_for_transfer(transfer.transfer_id, timeout=30.0)
        except TimeoutError:
            result.message = "File transfer timed out after 30s"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            return result

        # Check transfer result
        t = self._ft.get_transfer(transfer.transfer_id)
        if t and t.status == "failed":
            result.message = f"File transfer failed: {t.error}"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            return result

        # Step 5: Send restart command
        logger.info("Deploy to %s: restarting cam-agent service", agent_name)
        try:
            await ws.send_json({
                "type": "command",
                "command": "restart_service service_name=cam-agent",
            })
        except Exception as e:
            result.message = f"Restart command failed: {e}"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            return result

        # Step 6: Wait for agent to reconnect (it will disconnect on restart)
        logger.info("Deploy to %s: waiting for reconnect...", agent_name)
        try:
            await self._wait_for_reconnect(agent_id, timeout=30.0)
        except TimeoutError:
            # Agent didn't reconnect within timeout — might still be restarting
            result.ok = True  # file was transferred, restart was sent
            result.message = "Deployed but agent hasn't reconnected yet (may still be restarting)"
            result.completed_at = datetime.now(timezone.utc).isoformat()
            self._history.appendleft(result)
            self._events.warn(
                "deploy",
                f"Deployed connector to {agent_name} but reconnect timed out",
                agent_id=agent_id, old_version=result.old_version,
                new_version=result.new_version,
            )
            return result

        # Step 7: Success
        result.ok = True
        result.message = "Deployed successfully"
        result.completed_at = datetime.now(timezone.utc).isoformat()
        self._history.appendleft(result)
        self._events.info(
            "deploy",
            f"Deployed connector to {agent_name} ({result.old_version} -> {result.new_version})",
            agent_id=agent_id, old_version=result.old_version,
            new_version=result.new_version,
        )
        logger.info(
            "Deploy to %s complete: %s -> %s",
            agent_name, result.old_version, result.new_version,
        )
        return result

    async def deploy_all(self, broadcast_fn=None) -> list[DeployResult]:
        """Deploy connector to all online agents, one at a time.

        Sequential to avoid fleet-wide outage if something goes wrong.

        Args:
            broadcast_fn: Optional async callback(dict) to push progress updates.

        Returns:
            List of DeployResult for each agent attempted.
        """
        results = []
        online_agents = [
            a for a in self._registry.list_all()
            if a.status == "online" and a.agent_id in self._agent_ws
        ]

        if not online_agents:
            logger.info("Deploy all: no online agents")
            return results

        total = len(online_agents)
        for i, agent_info in enumerate(online_agents):
            if broadcast_fn:
                await broadcast_fn({
                    "type": "deploy_progress",
                    "agent_id": agent_info.agent_id,
                    "agent_name": agent_info.name,
                    "step": i + 1,
                    "total": total,
                    "status": "deploying",
                })

            result = await self.deploy_connector(agent_info.agent_id)
            results.append(result)

            if broadcast_fn:
                await broadcast_fn({
                    "type": "deploy_result",
                    **result.to_dict(),
                })

        return results

    # ------------------------------------------------------------------
    # History and status
    # ------------------------------------------------------------------

    def get_deploy_history(self) -> list[dict]:
        """Return the last 50 deploy results."""
        return [r.to_dict() for r in self._history]

    def get_status(self) -> dict:
        """Return local version, per-agent version cache, and recent history."""
        return {
            "local_version": self.get_local_version(),
            "agent_versions": dict(self._version_cache),
            "history": [r.to_dict() for r in list(self._history)[:10]],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _wait_for_transfer(self, transfer_id: str, timeout: float = 30.0):
        """Poll ft_manager until transfer completes or fails."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            t = self._ft.get_transfer(transfer_id)
            if t is None:
                # Transfer moved to history — check if it completed
                return
            if t.status in ("completed", "failed"):
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Transfer {transfer_id} did not complete within {timeout}s")

    async def _wait_for_reconnect(self, agent_id: str, timeout: float = 30.0):
        """Wait for an agent to come back online after restart.

        The agent will disconnect when its service restarts, then reconnect
        when the new connector starts up. We poll the agent_websockets dict
        and registry to detect reconnection.
        """
        # First, wait for the agent to go offline (disconnect)
        deadline = asyncio.get_event_loop().time() + timeout
        wait_disconnect_until = asyncio.get_event_loop().time() + 5.0

        while asyncio.get_event_loop().time() < wait_disconnect_until:
            if agent_id not in self._agent_ws:
                break
            await asyncio.sleep(0.5)

        # Now wait for it to come back
        while asyncio.get_event_loop().time() < deadline:
            if agent_id in self._agent_ws:
                agent = self._registry.get_by_id(agent_id)
                if agent and agent.status == "online":
                    return
            await asyncio.sleep(0.5)

        raise TimeoutError(f"Agent {agent_id} did not reconnect within {timeout}s")
