"""
CAM Command Library

Predefined commands that can be sent to agents via the dashboard
command palette. Each command has a name, description, optional
parameters, and a timeout.

Usage:
    library = CommandLibrary()
    cmd = library.get("status_report")
    all_commands = library.list_all()
"""

from dataclasses import dataclass, field


@dataclass
class Command:
    """A predefined command that can be sent to an agent.

    Attributes:
        name: Unique identifier for the command (e.g. "status_report").
        description: Human-readable description of what the command does.
        parameters: List of parameter dicts, each with:
            - name (str): Parameter name
            - description (str): What the parameter is for
            - required (bool): Whether the parameter must be provided
            - default (str | None): Default value if not provided
        timeout: How long to wait for a response (seconds).
    """
    name: str
    description: str
    parameters: list[dict] = field(default_factory=list)
    timeout: int = 30

    def to_dict(self) -> dict:
        """Serialize for sending to dashboard clients."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "timeout": self.timeout,
        }


class CommandLibrary:
    """In-memory registry of predefined commands.

    Instantiated once in the dashboard server and shared across
    all dashboard WebSocket connections. Commands are keyed by name.
    """

    def __init__(self):
        self._commands: dict[str, Command] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Pre-register the standard command set.

        Timeouts are read from config.commands.*; falls back to
        hardcoded values if config is unavailable.
        """
        # Read per-command timeouts from config
        try:
            from core.config import get_config
            cmd_timeouts = get_config().commands.to_dict()
        except Exception:
            cmd_timeouts = {}

        defaults = [
            Command(
                name="status_report",
                description="Request agent status and current activity",
                timeout=cmd_timeouts.get("status_report", 30),
            ),
            Command(
                name="system_info",
                description="Get system information (CPU, memory, disk, uptime)",
                timeout=cmd_timeouts.get("system_info", 30),
            ),
            Command(
                name="restart_service",
                description="Restart a named service on the agent host",
                parameters=[
                    {
                        "name": "service_name",
                        "description": "Name of the systemd service to restart",
                        "required": True,
                        "default": None,
                    },
                ],
                timeout=cmd_timeouts.get("restart_service", 60),
            ),
            Command(
                name="run_diagnostic",
                description="Run a diagnostic check on the agent",
                timeout=cmd_timeouts.get("run_diagnostic", 90),
            ),
            Command(
                name="capture_sensor_data",
                description="Capture current sensor readings",
                parameters=[
                    {
                        "name": "sensor_type",
                        "description": "Type of sensor to read (e.g. temperature, humidity)",
                        "required": False,
                        "default": "all",
                    },
                ],
                timeout=cmd_timeouts.get("capture_sensor_data", 45),
            ),
        ]
        for cmd in defaults:
            self._commands[cmd.name] = cmd

    def get(self, name: str) -> Command | None:
        """Look up a command by name. Returns None if not found."""
        return self._commands.get(name)

    def list_all(self) -> list[Command]:
        """Return all registered commands."""
        return list(self._commands.values())

    def to_broadcast_list(self) -> list[dict]:
        """Serialize all commands for sending to dashboard clients."""
        return [cmd.to_dict() for cmd in self._commands.values()]
