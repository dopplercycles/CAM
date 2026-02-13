"""
CAM Agent Registry

Single source of truth for all agents in the CAM network.

Tracks which agents are connected, what they can do, and whether
they're available for work. Used by both the dashboard (for display)
and the orchestrator (for task assignment).

Each agent has:
- Identity: name, ID, IP address
- Status: online, offline, or busy (working on a task)
- Capabilities: what the agent can do (e.g., "research", "content", "tts")
- Heartbeat tracking: last seen, heartbeat count, connection time

Usage:
    from core.agent_registry import AgentRegistry

    registry = AgentRegistry()
    registry.register("firehorseclawd", name="FireHorseClawd", ip="192.168.12.243")
    available = registry.get_available()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.registry")


# ---------------------------------------------------------------------------
# Agent info
# ---------------------------------------------------------------------------

@dataclass
class AgentInfo:
    """Everything CAM knows about a connected agent.

    This is the registry's view of an agent. The dashboard reads
    these for display, and the orchestrator reads them to decide
    which agent should handle a task.

    Attributes:
        agent_id:        Unique identifier (e.g., "firehorseclawd")
        name:            Human-readable display name
        ip_address:      Agent's local network IP
        status:          "online", "offline", or "busy"
        capabilities:    What this agent can do (e.g., ["research", "content"])
        last_heartbeat:  When we last heard from this agent
        connected_at:    When the agent first connected this session
        heartbeat_count: How many heartbeats received this session
    """
    agent_id: str
    name: str
    ip_address: str = "unknown"
    status: str = "online"
    capabilities: list[str] = field(default_factory=list)
    last_heartbeat: datetime | None = None
    connected_at: datetime | None = None
    heartbeat_count: int = 0
    model_override: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / dashboard broadcast.

        Uses ISO format for datetimes so the frontend can parse them.
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "ip_address": self.ip_address,
            "status": self.status,
            "capabilities": self.capabilities,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "heartbeat_count": self.heartbeat_count,
            "model_override": self.model_override,
        }


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Tracks all agents in the CAM network.

    The registry is the single source of truth for agent state.
    Multiple components share the same registry instance:

    - Dashboard server: registers/deregisters agents on WebSocket
      connect/disconnect, updates heartbeats, reads for broadcast.
    - Orchestrator: queries available agents for task assignment.
    - Future agent manager: spawns/kills agents, updates capabilities.

    All state is in-memory — rebuilt as agents reconnect after restart.
    """

    def __init__(self, heartbeat_timeout: float = 30.0):
        # agent_id -> AgentInfo
        self._agents: dict[str, AgentInfo] = {}

        # How many seconds without a heartbeat before marking offline
        self._heartbeat_timeout = heartbeat_timeout

        logger.info(
            "AgentRegistry initialized (heartbeat timeout: %.0fs)",
            heartbeat_timeout,
        )

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register(
        self,
        agent_id: str,
        name: str | None = None,
        ip_address: str = "unknown",
        capabilities: list[str] | None = None,
    ) -> AgentInfo:
        """Register a new agent or re-register an existing one.

        Called when an agent connects via WebSocket and sends its
        first heartbeat. If the agent was previously registered
        (e.g., reconnecting after a drop), its record is updated
        rather than duplicated.

        Args:
            agent_id:     Unique identifier for the agent.
            name:         Human-readable display name.
            ip_address:   Agent's local network IP.
            capabilities: List of things this agent can do.

        Returns:
            The AgentInfo record (new or updated).
        """
        now = datetime.now(timezone.utc)

        if agent_id in self._agents:
            # Re-registration — agent reconnected
            agent = self._agents[agent_id]
            agent.name = name or agent.name
            agent.ip_address = ip_address
            agent.status = "online"
            agent.capabilities = capabilities or agent.capabilities
            agent.last_heartbeat = now
            agent.connected_at = now
            agent.heartbeat_count = 1
            logger.info(
                "Agent re-registered: %s (%s) at %s",
                agent.name, agent_id, ip_address,
            )
        else:
            # New agent
            agent = AgentInfo(
                agent_id=agent_id,
                name=name or agent_id,
                ip_address=ip_address,
                status="online",
                capabilities=capabilities or [],
                last_heartbeat=now,
                connected_at=now,
                heartbeat_count=1,
            )
            self._agents[agent_id] = agent
            logger.info(
                "Agent registered: %s (%s) at %s, capabilities=%s",
                agent.name, agent_id, ip_address, agent.capabilities,
            )

        return agent

    def deregister(self, agent_id: str):
        """Mark an agent as offline.

        Called when an agent's WebSocket disconnects. We keep the
        record (don't delete it) so the dashboard can show it as
        offline rather than vanishing.

        Args:
            agent_id: The agent to mark offline.
        """
        if agent_id in self._agents:
            self._agents[agent_id].status = "offline"
            logger.info("Agent '%s' marked offline", agent_id)

    # -------------------------------------------------------------------
    # Heartbeat
    # -------------------------------------------------------------------

    def update_heartbeat(
        self,
        agent_id: str,
        ip_address: str | None = None,
        capabilities: list[str] | None = None,
    ) -> AgentInfo | None:
        """Record a heartbeat from an agent.

        Called each time an agent sends a heartbeat message. Updates
        the timestamp, increments the count, and optionally updates
        the IP and capabilities (in case they changed).

        Args:
            agent_id:     Which agent sent the heartbeat.
            ip_address:   Updated IP (if changed).
            capabilities: Updated capabilities (if changed).

        Returns:
            The updated AgentInfo, or None if the agent isn't registered.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return None

        agent.last_heartbeat = datetime.now(timezone.utc)
        agent.heartbeat_count += 1
        agent.status = "online"

        if ip_address:
            agent.ip_address = ip_address
        if capabilities is not None:
            agent.capabilities = capabilities

        return agent

    def heartbeat_check(self) -> list[str]:
        """Check all agents for heartbeat timeout.

        Marks any online agent as offline if it hasn't sent a
        heartbeat within the timeout period. Called periodically
        by the dashboard's background task.

        Returns:
            List of agent_ids that were marked offline.
        """
        now = datetime.now(timezone.utc)
        timed_out = []

        for agent in self._agents.values():
            if agent.status == "online" and agent.last_heartbeat:
                elapsed = (now - agent.last_heartbeat).total_seconds()
                if elapsed > self._heartbeat_timeout:
                    agent.status = "offline"
                    timed_out.append(agent.agent_id)
                    logger.warning(
                        "Agent '%s' (%s) timed out after %.0fs",
                        agent.name, agent.agent_id, elapsed,
                    )

        return timed_out

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def get_by_id(self, agent_id: str) -> AgentInfo | None:
        """Look up an agent by its ID."""
        return self._agents.get(agent_id)

    def get_by_name(self, name: str) -> AgentInfo | None:
        """Look up an agent by its display name (case-insensitive).

        Args:
            name: The agent's display name to search for.

        Returns:
            The AgentInfo if found, None otherwise.
        """
        name_lower = name.lower()
        for agent in self._agents.values():
            if agent.name.lower() == name_lower:
                return agent
        return None

    def get_available(self) -> list[AgentInfo]:
        """Return all agents that are online and not busy.

        Used by the orchestrator to find agents for task assignment.
        """
        return [
            a for a in self._agents.values()
            if a.status == "online"
        ]

    def get_capable(self, capabilities: list[str]) -> list[AgentInfo]:
        """Return online agents that have all the required capabilities.

        Args:
            capabilities: List of capability strings the agent must have.
                          If empty, returns all online agents (same as get_available).

        Returns:
            List of AgentInfo objects matching the requirements.
        """
        if not capabilities:
            return self.get_available()

        required = set(capabilities)
        return [
            a for a in self._agents.values()
            if a.status == "online" and required.issubset(set(a.capabilities))
        ]

    def list_all(self) -> list[AgentInfo]:
        """Return all known agents (any status)."""
        return list(self._agents.values())

    def list_online(self) -> list[AgentInfo]:
        """Return all agents currently online (including busy)."""
        return [a for a in self._agents.values() if a.status in ("online", "busy")]

    # -------------------------------------------------------------------
    # Serialization (for dashboard broadcast)
    # -------------------------------------------------------------------

    def to_broadcast_list(self) -> list[dict]:
        """Serialize all agents for dashboard WebSocket broadcast."""
        return [agent.to_dict() for agent in self._agents.values()]

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Summary of the registry state for the dashboard."""
        return {
            "total": len(self._agents),
            "online": len([a for a in self._agents.values() if a.status == "online"]),
            "offline": len([a for a in self._agents.values() if a.status == "offline"]),
            "busy": len([a for a in self._agents.values() if a.status == "busy"]),
        }

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        s = self.get_status()
        return (
            f"AgentRegistry(total={s['total']}, online={s['online']}, "
            f"busy={s['busy']}, offline={s['offline']})"
        )


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    registry = AgentRegistry(heartbeat_timeout=2.0)

    # Register some agents
    registry.register(
        "firehorseclawd",
        name="FireHorseClawd",
        ip_address="192.168.12.243",
        capabilities=["research", "content", "general"],
    )
    registry.register(
        "nova",
        name="Nova",
        ip_address="192.168.12.149",
        capabilities=["research", "agentic"],
    )

    print(f"Registry: {registry}")
    print(f"Status: {registry.get_status()}")

    # Query
    print(f"\nAvailable agents: {[a.name for a in registry.get_available()]}")
    print(f"By name 'nova': {registry.get_by_name('nova').name}")
    print(f"By ID 'firehorseclawd': {registry.get_by_id('firehorseclawd').name}")

    # Simulate heartbeat
    registry.update_heartbeat("firehorseclawd", ip_address="192.168.12.243")
    fh = registry.get_by_id("firehorseclawd")
    print(f"\nFireHorseClawd heartbeats: {fh.heartbeat_count}")

    # Simulate timeout
    print("\nWaiting for timeout (3s)...")
    time.sleep(3)
    timed_out = registry.heartbeat_check()
    print(f"Timed out: {timed_out}")
    print(f"Registry: {registry}")

    # Deregister
    registry.deregister("nova")
    print(f"\nAfter deregister nova: {registry}")

    # Broadcast format
    print("\nBroadcast data:")
    for entry in registry.to_broadcast_list():
        print(f"  {entry['name']}: status={entry['status']}, capabilities={entry['capabilities']}")
