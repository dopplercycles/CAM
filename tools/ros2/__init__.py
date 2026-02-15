"""
CAM ROS 2 Bridge — Conditional Import Guard

Checks for rclpy availability at import time. All downstream code checks
ROS2_AVAILABLE before touching any ROS 2 functionality.

When rclpy is not installed or ros2.enabled is false in config, CAM starts
normally without any robot tools — graceful degradation by design.
"""

import logging

logger = logging.getLogger("cam.ros2")

ROS2_AVAILABLE = False
try:
    import rclpy  # noqa: F401
    ROS2_AVAILABLE = True
    logger.info("rclpy found — ROS 2 bridge available")
except ImportError:
    logger.info("rclpy not found — ROS 2 bridge disabled (graceful degradation)")

if ROS2_AVAILABLE:
    from tools.ros2.node import CamRos2Bridge
    from tools.ros2.robot_registry import RobotRegistry
    from tools.ros2.tools import ROS2_TOOL_DEFINITIONS, ROS2_EXECUTORS, classify_ros2_tier
