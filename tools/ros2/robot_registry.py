"""
CAM Robot Registry — Robot Fleet State Tracking

Follows the same pattern as core/agent_registry.py but for physical robots
connected via ROS 2. Tracks robot identity, status, position, battery,
capabilities, and emergency stop state.

The registry is the single source of truth for robot fleet state.
Multiple components share the same instance:
- ROS 2 bridge: updates status from /robot/status topic callbacks
- Dashboard: reads for robot fleet panel display
- Tool executors: check robot availability before sending commands
- Kill switch: triggers emergency stop on all robots

Usage:
    from tools.ros2.robot_registry import RobotRegistry, RobotInfo

    registry = RobotRegistry(heartbeat_timeout=15)
    registry.register("p1", name="P1", robot_type="mobile_base",
                       capabilities=["navigation", "lidar"])
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("cam.ros2.registry")


# ---------------------------------------------------------------------------
# Robot info dataclass
# ---------------------------------------------------------------------------

@dataclass
class RobotInfo:
    """Everything CAM knows about a robot in the fleet.

    Attributes:
        robot_id:       Unique identifier (e.g., "p1", "p2")
        name:           Human-readable display name (e.g., "P1")
        robot_type:     Type of robot (e.g., "mobile_base", "arm", "drone")
        status:         "online", "offline", "busy", "e_stop"
        battery_pct:    Battery level 0-100, or None if unknown
        position:       {x, y, z, yaw} in map frame, or None
        current_task:   Description of what the robot is doing, or None
        topics:         Dict of configured ROS 2 topics for this robot
        services:       Dict of configured ROS 2 services
        capabilities:   What this robot can do (e.g., ["navigation", "lidar"])
        last_heartbeat: When we last received a status message
        e_stop_active:  Whether emergency stop is engaged
    """
    robot_id: str
    name: str
    robot_type: str = "mobile_base"
    status: str = "offline"
    battery_pct: float | None = None
    position: dict[str, float] | None = None
    current_task: str | None = None
    topics: dict[str, str] = field(default_factory=dict)
    services: dict[str, str] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    last_heartbeat: datetime | None = None
    e_stop_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON / dashboard broadcast."""
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "robot_type": self.robot_type,
            "status": self.status,
            "battery_pct": self.battery_pct,
            "position": self.position,
            "current_task": self.current_task,
            "capabilities": self.capabilities,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "e_stop_active": self.e_stop_active,
        }


# ---------------------------------------------------------------------------
# Robot Registry
# ---------------------------------------------------------------------------

class RobotRegistry:
    """Tracks all robots in the CAM fleet.

    Mirrors the AgentRegistry pattern: in-memory state rebuilt as
    robots come online. The ROS 2 bridge updates this registry from
    topic callbacks, and the dashboard reads it for display.

    Args:
        heartbeat_timeout: Seconds without a status message before
                           marking a robot offline.
    """

    def __init__(self, heartbeat_timeout: float = 15.0):
        self._robots: dict[str, RobotInfo] = {}
        self._heartbeat_timeout = heartbeat_timeout
        logger.info(
            "RobotRegistry initialized (heartbeat timeout: %.0fs)",
            heartbeat_timeout,
        )

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register(
        self,
        robot_id: str,
        name: str | None = None,
        robot_type: str = "mobile_base",
        capabilities: list[str] | None = None,
        topics: dict[str, str] | None = None,
        services: dict[str, str] | None = None,
    ) -> RobotInfo:
        """Register a robot from config or re-register on reconnect.

        Args:
            robot_id:     Unique identifier (e.g., "p1").
            name:         Display name (defaults to robot_id).
            robot_type:   Type of robot.
            capabilities: What the robot can do.
            topics:       Dict of topic names (status_topic, cmd_vel_topic, etc.)
            services:     Dict of service names.

        Returns:
            The RobotInfo record.
        """
        if robot_id in self._robots:
            robot = self._robots[robot_id]
            robot.name = name or robot.name
            robot.robot_type = robot_type
            robot.capabilities = capabilities or robot.capabilities
            if topics:
                robot.topics = topics
            if services:
                robot.services = services
            logger.info("Robot re-registered: %s (%s)", robot.name, robot_id)
        else:
            robot = RobotInfo(
                robot_id=robot_id,
                name=name or robot_id,
                robot_type=robot_type,
                status="offline",
                capabilities=capabilities or [],
                topics=topics or {},
                services=services or {},
            )
            self._robots[robot_id] = robot
            logger.info(
                "Robot registered: %s (%s), type=%s, capabilities=%s",
                robot.name, robot_id, robot_type, robot.capabilities,
            )

        return robot

    # -------------------------------------------------------------------
    # Status updates
    # -------------------------------------------------------------------

    def update_status(
        self,
        robot_id: str,
        battery_pct: float | None = None,
        position: dict[str, float] | None = None,
        current_task: str | None = None,
        status: str | None = None,
    ) -> RobotInfo | None:
        """Update a robot's status from a ROS 2 status message.

        Args:
            robot_id:     Which robot sent the status.
            battery_pct:  Battery level 0-100.
            position:     {x, y, z, yaw} in map frame.
            current_task: What the robot is doing.
            status:       Override status string.

        Returns:
            Updated RobotInfo, or None if robot not registered.
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return None

        robot.last_heartbeat = datetime.now(timezone.utc)

        # Don't override e_stop status with normal online status
        if robot.e_stop_active and status != "e_stop":
            pass  # Keep e_stop status until explicitly cleared
        elif status:
            robot.status = status
        elif robot.status == "offline":
            robot.status = "online"

        if battery_pct is not None:
            robot.battery_pct = battery_pct
        if position is not None:
            robot.position = position
        if current_task is not None:
            robot.current_task = current_task

        return robot

    def heartbeat_check(self) -> list[str]:
        """Check all robots for heartbeat timeout.

        Marks online robots as offline if no status received within
        the timeout period. Called periodically by the bridge's
        discovery loop.

        Returns:
            List of robot_ids that were marked offline.
        """
        now = datetime.now(timezone.utc)
        timed_out = []

        for robot in self._robots.values():
            if robot.status in ("online", "busy") and robot.last_heartbeat:
                elapsed = (now - robot.last_heartbeat).total_seconds()
                if elapsed > self._heartbeat_timeout:
                    robot.status = "offline"
                    timed_out.append(robot.robot_id)
                    logger.warning(
                        "Robot '%s' (%s) timed out after %.0fs",
                        robot.name, robot.robot_id, elapsed,
                    )

        return timed_out

    # -------------------------------------------------------------------
    # Emergency stop
    # -------------------------------------------------------------------

    def emergency_stop(self, robot_id: str | None = None):
        """Engage emergency stop on one or all robots.

        Args:
            robot_id: Specific robot to stop, or None for all.
        """
        if robot_id:
            robot = self._robots.get(robot_id)
            if robot:
                robot.status = "e_stop"
                robot.e_stop_active = True
                robot.current_task = None
                logger.critical("E-STOP engaged: %s (%s)", robot.name, robot_id)
        else:
            for robot in self._robots.values():
                robot.status = "e_stop"
                robot.e_stop_active = True
                robot.current_task = None
            logger.critical("E-STOP engaged: ALL ROBOTS")

    def clear_emergency_stop(self, robot_id: str | None = None):
        """Clear emergency stop on one or all robots.

        Robots return to offline status — they'll go online when
        the next status message arrives.

        Args:
            robot_id: Specific robot to clear, or None for all.
        """
        if robot_id:
            robot = self._robots.get(robot_id)
            if robot:
                robot.e_stop_active = False
                robot.status = "offline"
                logger.info("E-STOP cleared: %s (%s)", robot.name, robot_id)
        else:
            for robot in self._robots.values():
                robot.e_stop_active = False
                robot.status = "offline"
            logger.info("E-STOP cleared: ALL ROBOTS")

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def get_by_id(self, robot_id: str) -> RobotInfo | None:
        """Look up a robot by its ID."""
        return self._robots.get(robot_id)

    def get_available(self) -> list[RobotInfo]:
        """Return all robots that are online and not e-stopped."""
        return [
            r for r in self._robots.values()
            if r.status in ("online",)
        ]

    def get_capable(self, capabilities: list[str]) -> list[RobotInfo]:
        """Return online robots with all required capabilities.

        Args:
            capabilities: Required capability strings.

        Returns:
            List of matching RobotInfo objects.
        """
        if not capabilities:
            return self.get_available()

        required = set(capabilities)
        return [
            r for r in self._robots.values()
            if r.status == "online" and required.issubset(set(r.capabilities))
        ]

    def list_all(self) -> list[RobotInfo]:
        """Return all known robots (any status)."""
        return list(self._robots.values())

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_broadcast_list(self) -> list[dict]:
        """Serialize all robots for dashboard WebSocket broadcast."""
        return [robot.to_dict() for robot in self._robots.values()]

    def get_status(self) -> dict:
        """Summary of fleet state."""
        robots = list(self._robots.values())
        return {
            "total": len(robots),
            "online": len([r for r in robots if r.status == "online"]),
            "offline": len([r for r in robots if r.status == "offline"]),
            "busy": len([r for r in robots if r.status == "busy"]),
            "e_stop": len([r for r in robots if r.status == "e_stop"]),
        }

    def __len__(self) -> int:
        return len(self._robots)

    def __repr__(self) -> str:
        s = self.get_status()
        return (
            f"RobotRegistry(total={s['total']}, online={s['online']}, "
            f"busy={s['busy']}, e_stop={s['e_stop']}, offline={s['offline']})"
        )
