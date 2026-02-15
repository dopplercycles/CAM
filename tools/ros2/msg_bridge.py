"""
CAM ROS 2 Message Bridge — ROS 2 ↔ JSON Serialization

Converts between ROS 2 message types and plain Python dicts so CAM's
asyncio world can work with robot data without importing message types
everywhere.

Usage:
    from tools.ros2.msg_bridge import ros2_msg_to_dict, dict_to_ros2_msg, get_msg_type

    # ROS 2 message → dict
    data = ros2_msg_to_dict(twist_msg)
    # {"linear": {"x": 1.0, "y": 0.0, "z": 0.0}, "angular": ...}

    # dict → ROS 2 message
    msg = dict_to_ros2_msg("geometry_msgs/msg/Twist", {"linear": {"x": 1.0}})

    # Dynamic type import
    Twist = get_msg_type("geometry_msgs/msg/Twist")
"""

import importlib
import logging
from typing import Any

logger = logging.getLogger("cam.ros2.msg_bridge")

# Cache resolved message types to avoid repeated imports
_msg_type_cache: dict[str, type] = {}


def get_msg_type(type_string: str) -> type:
    """Dynamically import a ROS 2 message/service type from a string.

    Args:
        type_string: ROS 2 type string like "geometry_msgs/msg/Twist"
                     or "std_srvs/srv/SetBool".

    Returns:
        The message/service class.

    Raises:
        ImportError: If the package or type cannot be found.
        ValueError: If the type string format is invalid.
    """
    if type_string in _msg_type_cache:
        return _msg_type_cache[type_string]

    parts = type_string.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid ROS 2 type string: '{type_string}'. "
            f"Expected format: 'package/msg_or_srv/TypeName'"
        )

    package, interface, name = parts
    module_path = f"{package}.{interface}"

    try:
        module = importlib.import_module(module_path)
        msg_type = getattr(module, name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot import ROS 2 type '{type_string}': {e}. "
            f"Is the package '{package}' installed?"
        ) from e

    _msg_type_cache[type_string] = msg_type
    return msg_type


def ros2_msg_to_dict(msg) -> dict[str, Any]:
    """Recursively convert a ROS 2 message to a plain Python dict.

    Handles:
    - Primitive fields (int, float, str, bool)
    - Nested messages (recursed)
    - Arrays/sequences (converted to lists)
    - Header timestamps (converted to float seconds)
    - bytes fields (converted to list of ints)

    Args:
        msg: A ROS 2 message instance.

    Returns:
        A JSON-serializable dict.
    """
    result = {}

    # get_fields_and_field_types() returns {"field_name": "type_string"}
    if not hasattr(msg, "get_fields_and_field_types"):
        # Not a ROS 2 message — return as-is (primitive)
        return msg

    for field_name, field_type in msg.get_fields_and_field_types().items():
        value = getattr(msg, field_name, None)
        result[field_name] = _convert_value(value, field_type)

    return result


def _convert_value(value: Any, field_type: str = "") -> Any:
    """Convert a single field value to a JSON-serializable type."""
    if value is None:
        return None

    # Primitive types
    if isinstance(value, (bool, int, float, str)):
        return value

    # bytes → list of ints
    if isinstance(value, (bytes, bytearray)):
        return list(value)

    # ROS 2 Time/Duration (builtin_interfaces)
    if hasattr(value, "sec") and hasattr(value, "nanosec"):
        return {
            "sec": value.sec,
            "nanosec": value.nanosec,
            "total_seconds": value.sec + value.nanosec / 1e9,
        }

    # Nested ROS 2 message
    if hasattr(value, "get_fields_and_field_types"):
        return ros2_msg_to_dict(value)

    # Array/sequence of messages or primitives
    if isinstance(value, (list, tuple)):
        return [_convert_value(item) for item in value]

    # numpy arrays (from sensor messages)
    if hasattr(value, "tolist"):
        return value.tolist()

    # Fallback — try str
    return str(value)


def dict_to_ros2_msg(msg_type_str: str, data: dict[str, Any]):
    """Create a ROS 2 message from a dict.

    Only populates fields present in the dict — missing fields keep
    their default values.

    Args:
        msg_type_str: ROS 2 type string like "geometry_msgs/msg/Twist".
        data:         Dict of field values to set.

    Returns:
        A populated ROS 2 message instance.
    """
    msg_type = get_msg_type(msg_type_str)
    msg = msg_type()

    if not data:
        return msg

    fields = msg.get_fields_and_field_types()

    for field_name, value in data.items():
        if field_name not in fields:
            logger.warning(
                "Skipping unknown field '%s' for type '%s'",
                field_name, msg_type_str,
            )
            continue

        field_type_str = fields[field_name]
        current = getattr(msg, field_name, None)

        # If the current field is a nested message and value is a dict, recurse
        if isinstance(value, dict) and hasattr(current, "get_fields_and_field_types"):
            nested_type = _infer_nested_type(field_type_str)
            if nested_type:
                setattr(msg, field_name, dict_to_ros2_msg(nested_type, value))
            else:
                # Can't determine nested type — set fields directly
                for k, v in value.items():
                    if hasattr(current, k):
                        setattr(current, k, v)
        else:
            try:
                setattr(msg, field_name, value)
            except (TypeError, AttributeError) as e:
                logger.warning(
                    "Failed to set field '%s' on '%s': %s",
                    field_name, msg_type_str, e,
                )

    return msg


def _infer_nested_type(field_type_str: str) -> str | None:
    """Try to infer the full ROS 2 type path from a field type string.

    Field type strings look like:
    - "geometry_msgs/Vector3" → "geometry_msgs/msg/Vector3"
    - "builtin_interfaces/Time" → "builtin_interfaces/msg/Time"
    - "float64" → None (primitive)
    - "sequence<float64>" → None (primitive array)

    Returns:
        Full type string for import, or None if it's a primitive.
    """
    # Skip primitives and sequences of primitives
    if "/" not in field_type_str:
        return None

    # Strip sequence<> wrapper if present
    clean = field_type_str
    if clean.startswith("sequence<"):
        clean = clean[9:].rstrip(">")

    parts = clean.split("/")
    if len(parts) == 2:
        # "geometry_msgs/Vector3" → "geometry_msgs/msg/Vector3"
        return f"{parts[0]}/msg/{parts[1]}"
    elif len(parts) == 3:
        return clean

    return None
