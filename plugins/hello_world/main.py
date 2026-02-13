"""
Hello World Plugin — demonstrates the CAM plugin architecture.

This is a minimal example showing how to:
  - Implement lifecycle hooks (startup, shutdown)
  - React to agent events (connect, disconnect)
  - React to task events (received, completed)
  - Register custom API routes
  - Use plugin configuration
"""

import logging

from core.plugins import Plugin

logger = logging.getLogger("cam.plugins.hello_world")


class HelloWorldPlugin(Plugin):
    """Example plugin that logs lifecycle events."""

    def on_startup(self) -> None:
        greeting = self.config.get("greeting", "Hello from the plugin system!")
        logger.info("[HelloWorld] Started — %s", greeting)

    def on_shutdown(self) -> None:
        logger.info("[HelloWorld] Shutting down")

    def on_task_received(self, task: dict) -> None:
        task_id = task.get("task_id", "?")[:8]
        logger.debug("[HelloWorld] Task received: %s", task_id)

    def on_task_completed(self, task: dict) -> None:
        task_id = task.get("task_id", "?")[:8]
        logger.debug("[HelloWorld] Task completed: %s", task_id)

    def on_agent_connected(self, agent_id: str) -> None:
        logger.info("[HelloWorld] Agent connected: %s", agent_id)

    def on_agent_disconnected(self, agent_id: str) -> None:
        logger.info("[HelloWorld] Agent disconnected: %s", agent_id)

    def register_routes(self, app) -> None:
        """Add a simple /api/plugins/hello endpoint."""

        @app.get("/api/plugins/hello")
        async def hello_plugin():
            greeting = self.config.get("greeting", "Hello from the plugin system!")
            return {
                "plugin": self.name,
                "plugin_id": self.plugin_id,
                "greeting": greeting,
            }

        logger.info("[HelloWorld] Registered route: /api/plugins/hello")
