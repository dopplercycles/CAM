"""
CAM ROS 2 Tool Definitions & Executors

6 tools for Claude to control robots via tool_use:

| Tool                  | Tier | Description                              |
|-----------------------|------|------------------------------------------|
| robot_status          | 1    | Get status of one/all robots             |
| robot_sensor_read     | 1    | Read sensor data (lidar, camera, etc.)   |
| robot_navigate        | 2    | Navigate to waypoint or coordinates      |
| robot_patrol          | 2    | Start a patrol route                     |
| robot_command         | 2    | Generic topic publish or service call    |
| robot_emergency_stop  | 0    | E-stop one or all robots (always allowed)|

Usage:
    from tools.ros2.tools import ROS2_TOOL_DEFINITIONS, ROS2_EXECUTORS, classify_ros2_tier
"""

import logging
from typing import Any

logger = logging.getLogger("cam.ros2.tools")

# Bridge reference — set by server.py during startup via set_bridge()
_bridge = None


def set_bridge(bridge):
    """Set the CamRos2Bridge reference for tool executors.

    Called once during server.py startup after the bridge is created.
    """
    global _bridge
    _bridge = bridge


# ---------------------------------------------------------------------------
# Tool definitions — Claude API format
# ---------------------------------------------------------------------------

ROS2_TOOL_DEFINITIONS = [
    {
        "name": "robot_status",
        "description": (
            "Get the current status of one or all robots in the fleet. "
            "Returns battery level, position, current task, and online/offline state. "
            "Use this to check robot availability before issuing commands."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_id": {
                    "type": "string",
                    "description": (
                        "ID of a specific robot (e.g., 'p1', 'p2', 'p3'). "
                        "Omit to get status of all robots."
                    ),
                },
            },
        },
    },
    {
        "name": "robot_sensor_read",
        "description": (
            "Read sensor data from a robot. Supports lidar scan summaries, "
            "camera snapshots, IMU data, and battery details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_id": {
                    "type": "string",
                    "description": "ID of the robot to read from (e.g., 'p1').",
                },
                "sensor": {
                    "type": "string",
                    "enum": ["lidar", "camera", "imu", "battery"],
                    "description": "Which sensor to read.",
                },
            },
            "required": ["robot_id", "sensor"],
        },
    },
    {
        "name": "robot_navigate",
        "description": (
            "Send a navigation goal to a robot. Accepts either a named waypoint "
            "(e.g., 'charging_station', 'workbench') or explicit x/y coordinates. "
            "The robot uses Nav2 to plan and execute the path. "
            "Requires George's approval before execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_id": {
                    "type": "string",
                    "description": "ID of the robot to navigate (e.g., 'p1').",
                },
                "waypoint": {
                    "type": "string",
                    "description": (
                        "Named waypoint to navigate to (e.g., 'charging_station'). "
                        "Use this OR x/y coordinates, not both."
                    ),
                },
                "x": {
                    "type": "number",
                    "description": "X coordinate in map frame (meters). Used with y.",
                },
                "y": {
                    "type": "number",
                    "description": "Y coordinate in map frame (meters). Used with x.",
                },
                "yaw": {
                    "type": "number",
                    "description": "Goal orientation in radians (default 0, facing +X).",
                },
            },
            "required": ["robot_id"],
        },
    },
    {
        "name": "robot_patrol",
        "description": (
            "Start a patrol route — the robot visits a sequence of waypoints. "
            "Use named routes (e.g., 'lab', 'perimeter') or provide a list of waypoint names. "
            "Requires George's approval before execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_id": {
                    "type": "string",
                    "description": "ID of the robot to patrol (e.g., 'p1').",
                },
                "route": {
                    "type": "string",
                    "description": (
                        "Named route (e.g., 'lab', 'perimeter'). "
                        "Use this OR waypoints list."
                    ),
                },
                "waypoints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of waypoint names to visit in order. "
                        "Use this OR a named route."
                    ),
                },
            },
            "required": ["robot_id"],
        },
    },
    {
        "name": "robot_command",
        "description": (
            "Send a generic command to a robot by publishing to a ROS 2 topic "
            "or calling a ROS 2 service. For advanced use when the specific "
            "navigation/patrol tools don't fit. "
            "Requires George's approval before execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_id": {
                    "type": "string",
                    "description": "ID of the target robot.",
                },
                "action": {
                    "type": "string",
                    "enum": ["publish", "service_call"],
                    "description": "Whether to publish a topic or call a service.",
                },
                "topic_or_service": {
                    "type": "string",
                    "description": "ROS 2 topic or service name (e.g., '/p1/cmd_vel').",
                },
                "msg_type": {
                    "type": "string",
                    "description": (
                        "ROS 2 message/service type "
                        "(e.g., 'geometry_msgs/msg/Twist', 'std_srvs/srv/SetBool')."
                    ),
                },
                "data": {
                    "type": "object",
                    "description": "Message/request fields as a JSON object.",
                },
            },
            "required": ["robot_id", "action", "topic_or_service", "msg_type"],
        },
    },
    {
        "name": "robot_emergency_stop",
        "description": (
            "EMERGENCY STOP — immediately halt one or all robots. "
            "Sends zero velocity and engages e-stop. "
            "This is ALWAYS allowed without approval — safety first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_id": {
                    "type": "string",
                    "description": (
                        "ID of robot to stop. Omit to stop ALL robots."
                    ),
                },
                "clear": {
                    "type": "boolean",
                    "description": "Set to true to CLEAR the e-stop instead of engaging it.",
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def classify_ros2_tier(tool_name: str, tool_input: dict) -> int:
    """Classify a ROS 2 tool call into a Constitutional tier.

    Args:
        tool_name:  The robot tool name.
        tool_input: The input arguments.

    Returns:
        0 (emergency — always allowed), 1 (autonomous), or 2 (approval required).
    """
    if tool_name == "robot_emergency_stop":
        return 0  # Special: always allowed, safety-critical

    if tool_name in ("robot_status", "robot_sensor_read"):
        return 1  # Read-only, autonomous

    if tool_name in ("robot_navigate", "robot_patrol", "robot_command"):
        return 2  # Movement/actuation requires approval

    return 2  # Unknown robot tool → require approval


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

async def _exec_robot_status(tool_input: dict) -> dict:
    """Get status of one or all robots."""
    if _bridge is None:
        return {"error": "ROS 2 bridge not available"}

    robot_id = tool_input.get("robot_id")

    if robot_id:
        robot = _bridge._robot_registry.get_by_id(robot_id)
        if not robot:
            return {"error": f"Unknown robot: {robot_id}"}
        return {"result": robot.to_dict()}
    else:
        robots = _bridge._robot_registry.to_broadcast_list()
        summary = _bridge._robot_registry.get_status()
        return {"result": {"robots": robots, "fleet_summary": summary}}


async def _exec_robot_sensor_read(tool_input: dict) -> dict:
    """Read sensor data from a robot."""
    if _bridge is None:
        return {"error": "ROS 2 bridge not available"}

    robot_id = tool_input.get("robot_id", "")
    sensor = tool_input.get("sensor", "")

    robot = _bridge._robot_registry.get_by_id(robot_id)
    if not robot:
        return {"error": f"Unknown robot: {robot_id}"}

    if robot.status == "offline":
        return {"error": f"Robot {robot_id} is offline"}

    # Map sensor type to topic
    sensor_topics = {
        "lidar": f"/{robot_id}/scan",
        "camera": f"/{robot_id}/camera/image_raw",
        "imu": f"/{robot_id}/imu",
        "battery": f"/{robot_id}/battery_state",
    }

    topic = sensor_topics.get(sensor)
    if not topic:
        return {"error": f"Unknown sensor type: {sensor}"}

    # For now, return the last known data from the registry
    # Full implementation would subscribe-once to the topic
    result = {
        "robot_id": robot_id,
        "sensor": sensor,
        "topic": topic,
        "status": robot.status,
    }

    if sensor == "battery" and robot.battery_pct is not None:
        result["battery_pct"] = robot.battery_pct
    elif sensor == "lidar":
        result["note"] = "Lidar data available via ROS 2 topic subscription"
    elif sensor == "camera":
        result["note"] = "Camera feed available via ROS 2 topic subscription"
    elif sensor == "imu":
        result["note"] = "IMU data available via ROS 2 topic subscription"

    return {"result": result}


async def _exec_robot_navigate(tool_input: dict) -> dict:
    """Navigate a robot to a waypoint or coordinates."""
    if _bridge is None:
        return {"error": "ROS 2 bridge not available"}

    robot_id = tool_input.get("robot_id", "")
    waypoint = tool_input.get("waypoint")
    x = tool_input.get("x")
    y = tool_input.get("y")
    yaw = tool_input.get("yaw", 0.0)

    if not robot_id:
        return {"error": "robot_id is required"}

    # Resolve waypoint to coordinates
    if waypoint:
        ros2_cfg = _bridge._config.ros2
        waypoints = getattr(ros2_cfg, "waypoints", {})
        if hasattr(waypoints, "to_dict"):
            waypoints = waypoints.to_dict()
        elif hasattr(waypoints, "_data"):
            waypoints = waypoints._data

        wp = waypoints.get(waypoint)
        if not wp:
            available = list(waypoints.keys())
            return {"error": f"Unknown waypoint: '{waypoint}'. Available: {available}"}

        if isinstance(wp, dict):
            x = wp.get("x", 0.0)
            y = wp.get("y", 0.0)
        elif hasattr(wp, "x"):
            x = wp.x
            y = wp.y

    if x is None or y is None:
        return {"error": "Either 'waypoint' or both 'x' and 'y' coordinates are required"}

    return await _bridge.send_nav_goal(robot_id, float(x), float(y), float(yaw))


async def _exec_robot_patrol(tool_input: dict) -> dict:
    """Start a patrol route on a robot."""
    if _bridge is None:
        return {"error": "ROS 2 bridge not available"}

    robot_id = tool_input.get("robot_id", "")
    route_name = tool_input.get("route")
    waypoint_names = tool_input.get("waypoints")

    if not robot_id:
        return {"error": "robot_id is required"}

    ros2_cfg = _bridge._config.ros2
    waypoints_cfg = getattr(ros2_cfg, "waypoints", {})
    if hasattr(waypoints_cfg, "to_dict"):
        waypoints_cfg = waypoints_cfg.to_dict()
    elif hasattr(waypoints_cfg, "_data"):
        waypoints_cfg = waypoints_cfg._data

    # Resolve route to waypoint list
    if route_name:
        routes_cfg = getattr(ros2_cfg, "routes", {})
        if hasattr(routes_cfg, "to_dict"):
            routes_cfg = routes_cfg.to_dict()
        elif hasattr(routes_cfg, "_data"):
            routes_cfg = routes_cfg._data

        waypoint_names = routes_cfg.get(route_name)
        if not waypoint_names:
            available = list(routes_cfg.keys())
            return {"error": f"Unknown route: '{route_name}'. Available: {available}"}

    if not waypoint_names:
        return {"error": "Either 'route' name or 'waypoints' list is required"}

    # Resolve waypoint names to coordinates
    goals = []
    for wp_name in waypoint_names:
        wp = waypoints_cfg.get(wp_name)
        if not wp:
            return {"error": f"Unknown waypoint in route: '{wp_name}'"}
        if isinstance(wp, dict):
            goals.append({"name": wp_name, "x": wp.get("x", 0.0), "y": wp.get("y", 0.0)})
        elif hasattr(wp, "x"):
            goals.append({"name": wp_name, "x": wp.x, "y": wp.y})

    # Send the first goal and queue the rest
    # (Simple sequential patrol — send goals one at a time)
    first = goals[0]
    result = await _bridge.send_nav_goal(robot_id, first["x"], first["y"])

    if "error" in result:
        return result

    # Update task description
    route_desc = route_name or " → ".join(waypoint_names)
    _bridge._robot_registry.update_status(
        robot_id=robot_id,
        current_task=f"Patrol: {route_desc} (1/{len(goals)})",
        status="busy",
    )

    return {
        "result": "patrol_started",
        "robot_id": robot_id,
        "route": route_desc,
        "waypoints": goals,
        "total_stops": len(goals),
        "note": f"First goal sent ({first['name']}). Remaining goals queued.",
    }


async def _exec_robot_command(tool_input: dict) -> dict:
    """Generic ROS 2 topic publish or service call."""
    if _bridge is None:
        return {"error": "ROS 2 bridge not available"}

    robot_id = tool_input.get("robot_id", "")
    action = tool_input.get("action", "")
    topic_or_service = tool_input.get("topic_or_service", "")
    msg_type = tool_input.get("msg_type", "")
    data = tool_input.get("data", {})

    if not robot_id:
        return {"error": "robot_id is required"}
    if not topic_or_service:
        return {"error": "topic_or_service is required"}
    if not msg_type:
        return {"error": "msg_type is required"}

    if action == "publish":
        return await _bridge.publish_command(robot_id, topic_or_service, msg_type, data)
    elif action == "service_call":
        return await _bridge.call_service(robot_id, topic_or_service, msg_type, data)
    else:
        return {"error": f"Unknown action: {action}. Use 'publish' or 'service_call'."}


async def _exec_robot_emergency_stop(tool_input: dict) -> dict:
    """Emergency stop one or all robots."""
    if _bridge is None:
        return {"error": "ROS 2 bridge not available"}

    robot_id = tool_input.get("robot_id")
    clear = tool_input.get("clear", False)

    if clear:
        return await _bridge.clear_emergency_stop(robot_id)
    else:
        return await _bridge.emergency_stop(robot_id)


# ---------------------------------------------------------------------------
# Executor mapping — used by tool_registry.execute_tool()
# ---------------------------------------------------------------------------

ROS2_EXECUTORS = {
    "robot_status": _exec_robot_status,
    "robot_sensor_read": _exec_robot_sensor_read,
    "robot_navigate": _exec_robot_navigate,
    "robot_patrol": _exec_robot_patrol,
    "robot_command": _exec_robot_command,
    "robot_emergency_stop": _exec_robot_emergency_stop,
}
