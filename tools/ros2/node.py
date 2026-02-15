"""
CAM ROS 2 Bridge Node — CamRos2Bridge

Makes CAM a ROS 2 node so it can directly publish/subscribe to topics,
call services, and send navigation goals to the robot fleet.

Threading model:
- rclpy spins in a dedicated daemon thread (10Hz spin_once loop)
- ROS 2 callbacks use asyncio.run_coroutine_threadsafe() to push events
  to CAM's main asyncio event loop
- Publishing from asyncio is thread-safe (rclpy publishers allow it)

Usage:
    bridge = CamRos2Bridge(config, message_bus, event_logger, robot_registry)
    await bridge.start()
    ...
    await bridge.stop()
"""

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from tools.ros2.msg_bridge import ros2_msg_to_dict, dict_to_ros2_msg, get_msg_type
from tools.ros2.robot_registry import RobotRegistry

logger = logging.getLogger("cam.ros2.bridge")


class CamRos2Bridge:
    """ROS 2 bridge connecting CAM to the robot fleet.

    Manages the rclpy lifecycle: init, node creation, subscriptions,
    publishers, service clients, and action clients.

    Args:
        config:          CAM config object (config.ros2.* settings).
        message_bus:     CAM MessageBus for publishing robot_events.
        event_logger:    CAM EventLogger for audit trail.
        robot_registry:  RobotRegistry instance for fleet state.
    """

    def __init__(self, config, message_bus, event_logger, robot_registry: RobotRegistry):
        self._config = config
        self._message_bus = message_bus
        self._event_logger = event_logger
        self._robot_registry = robot_registry

        # ROS 2 state — initialized in start()
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._running = False

        # asyncio event loop reference — set in start()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Publishers cache: (robot_id, topic) → publisher
        self._publishers: dict[tuple[str, str], Any] = {}

        # Subscriptions list for cleanup
        self._subscriptions: list[Any] = []

        # Discovery task handle
        self._discovery_task: asyncio.Task | None = None

        # Dashboard broadcast callback (set by server.py)
        self._on_broadcast: Any = None

        # Status
        self._started_at: str | None = None

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def start(self):
        """Initialize rclpy, create node, set up subscriptions, start spin thread.

        Called from server.py lifespan startup.
        """
        self._loop = asyncio.get_running_loop()

        ros2_cfg = self._config.ros2
        node_name = getattr(ros2_cfg, "node_name", "cam_bridge")
        namespace = getattr(ros2_cfg, "namespace", "/cam")

        # Initialize rclpy
        try:
            rclpy.init()
        except RuntimeError:
            # Already initialized (e.g., in tests)
            pass

        # Create node
        self._node = Node(node_name, namespace=namespace)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Register robots from config and subscribe to their status topics
        robots_cfg = getattr(ros2_cfg, "robots", [])
        if isinstance(robots_cfg, list):
            for robot_cfg in robots_cfg:
                self._register_robot_from_config(robot_cfg)

        # Start spin thread
        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name="ros2-spin",
            daemon=True,
        )
        self._spin_thread.start()

        # Start periodic discovery/heartbeat check
        interval = getattr(ros2_cfg, "discovery_interval", 10)
        self._discovery_task = asyncio.create_task(self._discovery_loop(interval))

        self._started_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "ROS 2 bridge started (node: %s/%s, %d robots configured)",
            namespace, node_name, len(robots_cfg) if isinstance(robots_cfg, list) else 0,
        )
        self._event_logger.info(
            "ros2", f"ROS 2 bridge started (node: {namespace}/{node_name})"
        )

    async def stop(self):
        """Shut down rclpy and clean up. Called from server.py lifespan shutdown."""
        logger.info("ROS 2 bridge shutting down")

        # Stop discovery loop
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        # Stop spin thread
        self._running = False
        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)

        # Destroy node and shutdown rclpy
        if self._node:
            try:
                self._node.destroy_node()
            except Exception:
                pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

        logger.info("ROS 2 bridge shutdown complete")
        self._event_logger.info("ros2", "ROS 2 bridge shut down")

    def set_broadcast_callback(self, callback):
        """Set the async callback for broadcasting robot status to dashboards."""
        self._on_broadcast = callback

    # -------------------------------------------------------------------
    # Internal: spin loop (runs in daemon thread)
    # -------------------------------------------------------------------

    def _spin_loop(self):
        """Background thread function — spins the rclpy executor at ~10Hz."""
        logger.debug("ROS 2 spin thread started")
        while self._running:
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if self._running:
                    logger.debug("Spin loop exception: %s", e)
        logger.debug("ROS 2 spin thread exiting")

    # -------------------------------------------------------------------
    # Internal: robot registration from config
    # -------------------------------------------------------------------

    def _register_robot_from_config(self, robot_cfg):
        """Register a robot from config and subscribe to its status topic.

        Args:
            robot_cfg: Dict or ConfigSection with robot config fields.
        """
        # Handle both dict and ConfigSection
        if hasattr(robot_cfg, "to_dict"):
            robot_cfg = robot_cfg.to_dict()
        elif hasattr(robot_cfg, "_data"):
            robot_cfg = robot_cfg._data

        robot_id = robot_cfg.get("id", "")
        if not robot_id:
            return

        name = robot_cfg.get("name", robot_id)
        robot_type = robot_cfg.get("type", "mobile_base")
        capabilities = robot_cfg.get("capabilities", [])

        # Collect topic configuration
        topics = {}
        for key in ("status_topic", "cmd_vel_topic", "nav_action", "e_stop_topic"):
            val = robot_cfg.get(key)
            if val:
                topics[key] = val

        # Register in the robot registry
        self._robot_registry.register(
            robot_id=robot_id,
            name=name,
            robot_type=robot_type,
            capabilities=capabilities,
            topics=topics,
        )

        # Subscribe to status topic if configured
        status_topic = robot_cfg.get("status_topic")
        if status_topic:
            self._subscribe_to_status(robot_id, status_topic)

    def _subscribe_to_status(self, robot_id: str, topic: str):
        """Subscribe to a robot's status topic.

        Uses a generic String message type for flexibility — robots
        publish JSON status as a string. Real deployments may use
        custom message types.

        Args:
            robot_id: Which robot this subscription is for.
            topic:    The ROS 2 topic to subscribe to.
        """
        try:
            from std_msgs.msg import String
        except ImportError:
            logger.warning(
                "std_msgs not available — cannot subscribe to %s for %s",
                topic, robot_id,
            )
            return

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        def callback(msg, rid=robot_id):
            self._on_robot_status(msg, rid)

        sub = self._node.create_subscription(String, topic, callback, qos)
        self._subscriptions.append(sub)
        logger.debug("Subscribed to %s for robot %s", topic, robot_id)

    # -------------------------------------------------------------------
    # Internal: ROS 2 callbacks → asyncio bridge
    # -------------------------------------------------------------------

    def _on_robot_status(self, msg, robot_id: str):
        """Handle a robot status message (called from ROS 2 spin thread).

        Parses the JSON status, updates the registry, and bridges
        the event to the asyncio event loop.
        """
        try:
            # Try to parse as JSON string
            if hasattr(msg, "data"):
                data = json.loads(msg.data)
            else:
                data = ros2_msg_to_dict(msg)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Failed to parse status from %s: %s", robot_id, e)
            data = {}

        # Update registry
        self._robot_registry.update_status(
            robot_id=robot_id,
            battery_pct=data.get("battery_pct"),
            position=data.get("position"),
            current_task=data.get("current_task"),
            status=data.get("status"),
        )

        # Bridge to asyncio loop
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._async_on_robot_status(robot_id, data),
                self._loop,
            )

    async def _async_on_robot_status(self, robot_id: str, data: dict):
        """Async handler for robot status — publishes to message bus and dashboard."""
        # Publish to message bus
        await self._message_bus.publish(
            sender=f"robot:{robot_id}",
            channel="robot_events",
            payload={
                "event": "status_update",
                "robot_id": robot_id,
                **data,
            },
        )

        # Broadcast to dashboard
        if self._on_broadcast:
            try:
                await self._on_broadcast()
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Internal: discovery / heartbeat loop
    # -------------------------------------------------------------------

    async def _discovery_loop(self, interval: float):
        """Periodic loop: check robot heartbeats and discover new nodes."""
        while True:
            try:
                await asyncio.sleep(interval)

                # Check heartbeat timeouts
                timed_out = self._robot_registry.heartbeat_check()
                if timed_out:
                    for rid in timed_out:
                        self._event_logger.warn(
                            "ros2",
                            f"Robot '{rid}' timed out (no status message)",
                            robot_id=rid,
                        )
                    # Broadcast updated status
                    if self._on_broadcast:
                        try:
                            await self._on_broadcast()
                        except Exception:
                            pass

                # Future: discover_robots() via ros2 node list

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Discovery loop error: %s", e)

    # -------------------------------------------------------------------
    # Public: publish commands
    # -------------------------------------------------------------------

    async def publish_command(
        self,
        robot_id: str,
        topic: str,
        msg_type_str: str,
        data: dict,
    ) -> dict:
        """Publish a message to a robot's topic.

        Thread-safe — rclpy publishers allow publishing from any thread.

        Args:
            robot_id:     Target robot ID.
            topic:        ROS 2 topic name.
            msg_type_str: Message type like "geometry_msgs/msg/Twist".
            data:         Dict of field values.

        Returns:
            {"result": "published"} on success, {"error": ...} on failure.
        """
        robot = self._robot_registry.get_by_id(robot_id)
        if not robot:
            return {"error": f"Unknown robot: {robot_id}"}

        if robot.e_stop_active:
            return {"error": f"Robot {robot_id} is emergency stopped"}

        try:
            msg_type = get_msg_type(msg_type_str)
            msg = dict_to_ros2_msg(msg_type_str, data)

            # Get or create publisher
            pub_key = (robot_id, topic)
            if pub_key not in self._publishers:
                qos = QoSProfile(depth=10)
                self._publishers[pub_key] = self._node.create_publisher(
                    msg_type, topic, qos
                )

            self._publishers[pub_key].publish(msg)

            logger.info(
                "Published to %s (%s): %s",
                topic, msg_type_str, str(data)[:100],
            )
            self._event_logger.info(
                "ros2",
                f"Published {msg_type_str} to {topic} for {robot_id}",
                robot_id=robot_id, topic=topic,
            )

            return {"result": "published", "topic": topic, "robot_id": robot_id}

        except Exception as e:
            logger.error("Failed to publish to %s: %s", topic, e)
            return {"error": f"Publish failed: {e}"}

    async def call_service(
        self,
        robot_id: str,
        service_name: str,
        srv_type_str: str,
        request: dict,
    ) -> dict:
        """Call a ROS 2 service.

        Runs the service call in a thread to avoid blocking asyncio.

        Args:
            robot_id:     Target robot ID.
            service_name: ROS 2 service name.
            srv_type_str: Service type like "std_srvs/srv/SetBool".
            request:      Dict of request fields.

        Returns:
            Dict with service response fields, or {"error": ...}.
        """
        robot = self._robot_registry.get_by_id(robot_id)
        if not robot:
            return {"error": f"Unknown robot: {robot_id}"}

        if robot.e_stop_active:
            return {"error": f"Robot {robot_id} is emergency stopped"}

        try:
            srv_type = get_msg_type(srv_type_str)
            client = self._node.create_client(srv_type, service_name)

            if not client.wait_for_service(timeout_sec=5.0):
                return {"error": f"Service {service_name} not available (timeout)"}

            req = srv_type.Request()
            for key, value in request.items():
                if hasattr(req, key):
                    setattr(req, key, value)

            # Call service in a thread
            future = client.call_async(req)

            # Wait for result with timeout
            result = await asyncio.wait_for(
                asyncio.wrap_future(
                    asyncio.get_running_loop().run_in_executor(
                        None, lambda: rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)
                    )
                ),
                timeout=15.0,
            )

            if future.result() is not None:
                response = ros2_msg_to_dict(future.result())
                logger.info("Service %s responded: %s", service_name, str(response)[:100])
                return {"result": response}
            else:
                return {"error": f"Service {service_name} returned no result"}

        except asyncio.TimeoutError:
            return {"error": f"Service {service_name} timed out"}
        except Exception as e:
            logger.error("Service call to %s failed: %s", service_name, e)
            return {"error": f"Service call failed: {e}"}

    async def send_nav_goal(
        self,
        robot_id: str,
        x: float,
        y: float,
        yaw: float = 0.0,
    ) -> dict:
        """Send a navigation goal to a robot via Nav2 NavigateToPose action.

        Uses topic-based goal publishing as a simpler alternative to the
        full action client (which requires more complex threading).

        Args:
            robot_id: Target robot.
            x:        Goal X position in map frame.
            y:        Goal Y position in map frame.
            yaw:      Goal orientation (radians, default 0).

        Returns:
            {"result": "goal_sent"} on success, {"error": ...} on failure.
        """
        robot = self._robot_registry.get_by_id(robot_id)
        if not robot:
            return {"error": f"Unknown robot: {robot_id}"}

        if robot.e_stop_active:
            return {"error": f"Robot {robot_id} is emergency stopped"}

        if robot.status == "offline":
            return {"error": f"Robot {robot_id} is offline"}

        nav_action = robot.topics.get("nav_action", f"/{robot_id}/navigate_to_pose")

        # Publish a PoseStamped to the nav goal topic
        # Nav2 accepts goals on /{robot}/goal_pose as PoseStamped
        import math
        goal_topic = f"/{robot_id}/goal_pose"
        goal_data = {
            "header": {
                "frame_id": "map",
            },
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(yaw / 2.0),
                    "w": math.cos(yaw / 2.0),
                },
            },
        }

        result = await self.publish_command(
            robot_id=robot_id,
            topic=goal_topic,
            msg_type_str="geometry_msgs/msg/PoseStamped",
            data=goal_data,
        )

        if "error" in result:
            return result

        # Update robot status
        self._robot_registry.update_status(
            robot_id=robot_id,
            current_task=f"Navigating to ({x:.1f}, {y:.1f})",
            status="busy",
        )

        logger.info(
            "Nav goal sent to %s: (%.1f, %.1f, yaw=%.1f)",
            robot_id, x, y, yaw,
        )
        self._event_logger.info(
            "ros2",
            f"Navigation goal sent to {robot_id}: ({x:.1f}, {y:.1f})",
            robot_id=robot_id,
        )

        return {
            "result": "goal_sent",
            "robot_id": robot_id,
            "goal": {"x": x, "y": y, "yaw": yaw},
        }

    # -------------------------------------------------------------------
    # Emergency stop
    # -------------------------------------------------------------------

    async def emergency_stop(self, robot_id: str | None = None) -> dict:
        """Emergency stop one or all robots.

        Publishes a zero-velocity Twist to cmd_vel and an e-stop
        message to the e_stop topic. Updates registry.

        Args:
            robot_id: Specific robot, or None for all.

        Returns:
            {"result": "e_stop_engaged", ...}
        """
        targets = []
        if robot_id:
            robot = self._robot_registry.get_by_id(robot_id)
            if robot:
                targets.append(robot)
            else:
                return {"error": f"Unknown robot: {robot_id}"}
        else:
            targets = self._robot_registry.list_all()

        # Update registry first (safety-critical: stop state before publish)
        self._robot_registry.emergency_stop(robot_id)

        # Publish stop commands
        stopped = []
        for robot in targets:
            # Zero velocity
            cmd_vel = robot.topics.get("cmd_vel_topic", f"/{robot.robot_id}/cmd_vel")
            await self.publish_command(
                robot_id=robot.robot_id,
                topic=cmd_vel,
                msg_type_str="geometry_msgs/msg/Twist",
                data={"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                      "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
            )

            # E-stop signal
            e_stop_topic = robot.topics.get("e_stop_topic", f"/{robot.robot_id}/emergency_stop")
            try:
                from std_msgs.msg import Bool
                pub_key = (robot.robot_id, e_stop_topic)
                if pub_key not in self._publishers:
                    qos = QoSProfile(depth=10)
                    self._publishers[pub_key] = self._node.create_publisher(
                        Bool, e_stop_topic, qos
                    )
                e_stop_msg = Bool()
                e_stop_msg.data = True
                self._publishers[pub_key].publish(e_stop_msg)
            except ImportError:
                pass

            stopped.append(robot.robot_id)

        target_str = robot_id or "ALL"
        logger.critical("E-STOP sent to: %s", target_str)
        self._event_logger.error(
            "ros2",
            f"EMERGENCY STOP: {target_str}",
            robot_ids=stopped,
        )

        return {
            "result": "e_stop_engaged",
            "robots_stopped": stopped,
        }

    async def clear_emergency_stop(self, robot_id: str | None = None) -> dict:
        """Clear emergency stop on one or all robots.

        Args:
            robot_id: Specific robot, or None for all.

        Returns:
            {"result": "e_stop_cleared", ...}
        """
        self._robot_registry.clear_emergency_stop(robot_id)

        # Publish e-stop clear
        targets = []
        if robot_id:
            robot = self._robot_registry.get_by_id(robot_id)
            if robot:
                targets.append(robot)
        else:
            targets = self._robot_registry.list_all()

        cleared = []
        for robot in targets:
            e_stop_topic = robot.topics.get("e_stop_topic", f"/{robot.robot_id}/emergency_stop")
            try:
                from std_msgs.msg import Bool
                pub_key = (robot.robot_id, e_stop_topic)
                if pub_key not in self._publishers:
                    qos = QoSProfile(depth=10)
                    self._publishers[pub_key] = self._node.create_publisher(
                        Bool, e_stop_topic, qos
                    )
                msg = Bool()
                msg.data = False
                self._publishers[pub_key].publish(msg)
            except ImportError:
                pass
            cleared.append(robot.robot_id)

        target_str = robot_id or "ALL"
        logger.info("E-STOP cleared: %s", target_str)
        self._event_logger.info(
            "ros2",
            f"Emergency stop cleared: {target_str}",
            robot_ids=cleared,
        )

        return {
            "result": "e_stop_cleared",
            "robots_cleared": cleared,
        }

    # -------------------------------------------------------------------
    # Status / serialization
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return bridge status for dashboard display."""
        return {
            "running": self._running,
            "started_at": self._started_at,
            "node_name": self._node.get_name() if self._node else None,
            "namespace": self._node.get_namespace() if self._node else None,
            "fleet": self._robot_registry.get_status(),
        }

    def to_broadcast_dict(self) -> dict:
        """Full state for dashboard initial WS connect."""
        return {
            "bridge": self.get_status(),
            "robots": self._robot_registry.to_broadcast_list(),
        }
