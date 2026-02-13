"""
Plugin Architecture for CAM / Doppler Cycles.

Implements a plugin system for extending CAM without modifying core code.
Plugins live in a plugins/ directory and are auto-discovered on startup.
Each plugin has a manifest.yaml defining name, version, description,
dependencies, and required CAM version.

Plugin base class provides hooks:
  - on_startup()              — called when CAM boots
  - on_shutdown()             — called on graceful shutdown
  - on_task_received(task)    — called when a new task enters the queue
  - on_task_completed(task)   — called when a task finishes
  - on_agent_connected(id)    — called when an agent connects via WebSocket
  - on_agent_disconnected(id) — called when an agent disconnects
  - register_routes(app)      — add custom FastAPI endpoints

SQLite-backed state persistence for enable/disable status and config.
"""

import importlib
import importlib.util
import json
import logging
import sqlite3
import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("cam.plugins")

# Current CAM version for compatibility checks
CAM_VERSION = "0.1.0"

# Default plugins directory relative to project root
DEFAULT_PLUGINS_DIR = "plugins"


# ---------------------------------------------------------------------------
# Plugin Base Class
# ---------------------------------------------------------------------------

class Plugin(ABC):
    """Base class for all CAM plugins.

    Subclass this and implement any hooks your plugin needs.  At minimum,
    define the class and CAM will discover it via the manifest.yaml.

    Attributes:
        plugin_id:  Unique identifier (from manifest).
        name:       Human-readable name.
        config:     Plugin configuration dict (persisted across restarts).
        app_state:  Reference to FastAPI app.state for accessing managers.
    """

    def __init__(self):
        self.plugin_id: str = ""
        self.name: str = ""
        self.config: dict = {}
        self.app_state: Any = None

    # --- Lifecycle Hooks ---

    def on_startup(self) -> None:
        """Called when CAM boots and the plugin is enabled.

        Use this to initialize resources, start background tasks, etc.
        """
        pass

    def on_shutdown(self) -> None:
        """Called during graceful shutdown.

        Clean up resources, close connections, flush buffers.
        """
        pass

    # --- Task Hooks ---

    def on_task_received(self, task: dict) -> None:
        """Called when a new task is received by the orchestrator.

        Args:
            task: Dict with task_id, source, prompt, complexity, etc.
        """
        pass

    def on_task_completed(self, task: dict) -> None:
        """Called when a task completes (success or failure).

        Args:
            task: Dict with task_id, result, status, etc.
        """
        pass

    # --- Agent Hooks ---

    def on_agent_connected(self, agent_id: str) -> None:
        """Called when an agent connects via WebSocket.

        Args:
            agent_id: The unique identifier of the connecting agent.
        """
        pass

    def on_agent_disconnected(self, agent_id: str) -> None:
        """Called when an agent disconnects.

        Args:
            agent_id: The unique identifier of the disconnecting agent.
        """
        pass

    # --- Route Registration ---

    def register_routes(self, app: Any) -> None:
        """Register custom FastAPI routes for this plugin.

        Args:
            app: The FastAPI application instance.

        Example:
            @app.get("/api/plugins/my-plugin/data")
            async def get_data():
                return {"hello": "from my plugin"}
        """
        pass


# ---------------------------------------------------------------------------
# Manifest Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PluginManifest:
    """Parsed manifest.yaml for a plugin.

    Attributes:
        plugin_id:      Unique identifier (directory name).
        name:           Human-readable name.
        version:        Plugin version string.
        description:    What this plugin does.
        author:         Plugin author.
        entry_point:    Python module:class path (e.g. 'main:MyPlugin').
        dependencies:   List of required pip packages.
        cam_version:    Minimum required CAM version.
        config_schema:  Dict describing configurable fields.
    """
    plugin_id: str
    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    entry_point: str = "main:Plugin"
    dependencies: list[str] = field(default_factory=list)
    cam_version: str = "0.1.0"
    config_schema: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "cam_version": self.cam_version,
            "config_schema": self.config_schema,
        }

    @staticmethod
    def from_yaml(plugin_id: str, data: dict) -> "PluginManifest":
        return PluginManifest(
            plugin_id=plugin_id,
            name=data.get("name", plugin_id),
            version=str(data.get("version", "0.1.0")),
            description=data.get("description", ""),
            author=data.get("author", ""),
            entry_point=data.get("entry_point", "main:Plugin"),
            dependencies=data.get("dependencies", []),
            cam_version=str(data.get("cam_version", "0.1.0")),
            config_schema=data.get("config_schema", {}),
        )


# ---------------------------------------------------------------------------
# Plugin State (what we track in the DB per plugin)
# ---------------------------------------------------------------------------

@dataclass
class PluginState:
    """Runtime + persisted state for a plugin.

    Attributes:
        plugin_id:  Unique identifier.
        enabled:    Whether the plugin is enabled (persisted).
        loaded:     Whether the plugin instance is currently in memory.
        manifest:   Parsed manifest data.
        config:     Plugin configuration (persisted).
        instance:   The live Plugin object (None if not loaded).
        error:      Last error message, if any.
        loaded_at:  When the plugin was last loaded (ISO-8601).
    """
    plugin_id: str
    enabled: bool = False
    loaded: bool = False
    manifest: Optional[PluginManifest] = None
    config: dict = field(default_factory=dict)
    instance: Optional[Plugin] = None
    error: str = ""
    loaded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "plugin_id": self.plugin_id,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "error": self.error,
            "loaded_at": self.loaded_at,
            "config": self.config,
        }
        if self.manifest:
            d.update({
                "name": self.manifest.name,
                "version": self.manifest.version,
                "description": self.manifest.description,
                "author": self.manifest.author,
                "config_schema": self.manifest.config_schema,
            })
        else:
            d.update({"name": self.plugin_id, "version": "", "description": "",
                       "author": "", "config_schema": {}})
        return d


# ---------------------------------------------------------------------------
# Plugin Manager
# ---------------------------------------------------------------------------

class PluginManager:
    """Manages plugin discovery, loading, lifecycle, and state persistence.

    Auto-discovers plugins in the plugins/ directory on init.  Each
    subdirectory with a manifest.yaml is recognized as a plugin.  Enabled
    plugins are loaded (instantiated) on startup and receive lifecycle hooks.

    The enable/disable state and per-plugin config are persisted in SQLite
    so they survive restarts.
    """

    def __init__(
        self,
        plugins_dir: str = DEFAULT_PLUGINS_DIR,
        db_path: str = "data/plugins.db",
        *,
        on_change: Optional[Callable[[], Coroutine]] = None,
    ):
        self._plugins_dir = Path(plugins_dir)
        self._on_change = on_change
        self._plugins: dict[str, PluginState] = {}
        self._app = None  # Set later via set_app()

        # Ensure directories exist
        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        # Discover plugins from filesystem
        self._discover()

        logger.info("PluginManager initialized (dir=%s, db=%s, discovered=%d)",
                     self._plugins_dir, db_path, len(self._plugins))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS plugin_state (
                plugin_id  TEXT PRIMARY KEY,
                enabled    INTEGER DEFAULT 0,
                config     TEXT DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    def _fire_change(self):
        if self._on_change is None:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._on_change())
        except RuntimeError:
            pass

    def _load_persisted_state(self, plugin_id: str) -> tuple[bool, dict]:
        """Load enabled flag and config from DB for a plugin."""
        row = self._conn.execute(
            "SELECT enabled, config FROM plugin_state WHERE plugin_id = ?",
            (plugin_id,),
        ).fetchone()
        if row:
            enabled = bool(row["enabled"])
            try:
                config = json.loads(row["config"])
            except (json.JSONDecodeError, TypeError):
                config = {}
            return enabled, config
        return False, {}

    def _save_persisted_state(self, plugin_id: str, enabled: bool, config: dict):
        """Upsert enabled flag and config to DB."""
        now = self._now()
        config_json = json.dumps(config)
        self._conn.execute(
            """INSERT INTO plugin_state (plugin_id, enabled, config, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(plugin_id) DO UPDATE
               SET enabled = ?, config = ?, updated_at = ?""",
            (plugin_id, int(enabled), config_json, now,
             int(enabled), config_json, now),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self):
        """Scan the plugins directory for valid plugin folders.

        A valid plugin folder has a manifest.yaml file.
        """
        if not self._plugins_dir.is_dir():
            return

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not available — plugin discovery disabled")
            return

        for entry in sorted(self._plugins_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith((".", "_")):
                continue

            manifest_path = entry / "manifest.yaml"
            if not manifest_path.exists():
                # Also check manifest.yml
                manifest_path = entry / "manifest.yml"
                if not manifest_path.exists():
                    continue

            try:
                with open(manifest_path) as f:
                    data = yaml.safe_load(f) or {}

                manifest = PluginManifest.from_yaml(entry.name, data)
                enabled, config = self._load_persisted_state(entry.name)

                self._plugins[entry.name] = PluginState(
                    plugin_id=entry.name,
                    enabled=enabled,
                    loaded=False,
                    manifest=manifest,
                    config=config,
                )

                logger.info("Discovered plugin: %s v%s (%s)",
                            manifest.name, manifest.version,
                            "enabled" if enabled else "disabled")

            except Exception:
                logger.warning("Failed to parse manifest for plugin %s",
                               entry.name, exc_info=True)
                self._plugins[entry.name] = PluginState(
                    plugin_id=entry.name,
                    error="Failed to parse manifest.yaml",
                )

    # ------------------------------------------------------------------
    # Loading / Unloading
    # ------------------------------------------------------------------

    def set_app(self, app: Any):
        """Store a reference to the FastAPI app for route registration."""
        self._app = app

    def load_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Load (instantiate) a plugin from its entry point.

        The plugin module is imported and the specified class instantiated.
        If the plugin defines register_routes(), those are added to the app.
        """
        state = self._plugins.get(plugin_id)
        if not state or not state.manifest:
            logger.warning("Cannot load unknown plugin: %s", plugin_id)
            return None

        if state.loaded and state.instance:
            return state.instance

        manifest = state.manifest
        entry = manifest.entry_point  # e.g. "main:MyPlugin"

        try:
            module_name, class_name = entry.split(":", 1)
        except ValueError:
            module_name = entry
            class_name = "Plugin"

        plugin_path = self._plugins_dir / plugin_id / f"{module_name}.py"
        if not plugin_path.exists():
            err = f"Entry point file not found: {plugin_path}"
            state.error = err
            logger.warning(err)
            return None

        try:
            # Dynamic import from file path
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_id}.{module_name}",
                str(plugin_path),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the plugin class
            plugin_class = getattr(module, class_name, None)
            if plugin_class is None:
                err = f"Class '{class_name}' not found in {plugin_path}"
                state.error = err
                logger.warning(err)
                return None

            # Instantiate
            instance = plugin_class()
            instance.plugin_id = plugin_id
            instance.name = manifest.name
            instance.config = dict(state.config)
            if self._app:
                instance.app_state = getattr(self._app, "state", None)

            # Register routes if app available
            if self._app:
                try:
                    instance.register_routes(self._app)
                except Exception:
                    logger.warning("Plugin %s route registration failed",
                                   plugin_id, exc_info=True)

            state.instance = instance
            state.loaded = True
            state.loaded_at = self._now()
            state.error = ""

            logger.info("Plugin loaded: %s v%s", manifest.name, manifest.version)
            self._fire_change()
            return instance

        except Exception as e:
            state.error = str(e)
            logger.warning("Failed to load plugin %s: %s", plugin_id, e, exc_info=True)
            return None

    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin — call on_shutdown() and remove the instance."""
        state = self._plugins.get(plugin_id)
        if not state or not state.loaded:
            return False

        if state.instance:
            try:
                state.instance.on_shutdown()
            except Exception:
                logger.warning("Plugin %s shutdown error", plugin_id, exc_info=True)

        state.instance = None
        state.loaded = False
        state.loaded_at = ""
        logger.info("Plugin unloaded: %s", plugin_id)
        self._fire_change()
        return True

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin and load it if not already loaded."""
        state = self._plugins.get(plugin_id)
        if not state:
            return False

        state.enabled = True
        self._save_persisted_state(plugin_id, True, state.config)

        # Auto-load on enable
        if not state.loaded:
            self.load_plugin(plugin_id)

        # Call on_startup if just loaded
        if state.instance:
            try:
                state.instance.on_startup()
            except Exception:
                logger.warning("Plugin %s startup error", plugin_id, exc_info=True)

        logger.info("Plugin enabled: %s", plugin_id)
        self._fire_change()
        return True

    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin and unload it."""
        state = self._plugins.get(plugin_id)
        if not state:
            return False

        state.enabled = False
        self._save_persisted_state(plugin_id, False, state.config)

        # Unload on disable
        if state.loaded:
            self.unload_plugin(plugin_id)

        logger.info("Plugin disabled: %s", plugin_id)
        self._fire_change()
        return True

    def plugin_status(self, plugin_id: str) -> Optional[dict[str, Any]]:
        """Get the status dict for a specific plugin."""
        state = self._plugins.get(plugin_id)
        return state.to_dict() if state else None

    def update_config(self, plugin_id: str, config: dict) -> bool:
        """Update a plugin's configuration and persist it."""
        state = self._plugins.get(plugin_id)
        if not state:
            return False

        state.config.update(config)
        self._save_persisted_state(plugin_id, state.enabled, state.config)

        # Push config to live instance
        if state.instance:
            state.instance.config = dict(state.config)

        logger.info("Plugin config updated: %s", plugin_id)
        self._fire_change()
        return True

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_plugins(self) -> list[PluginState]:
        """List all discovered plugins."""
        return list(self._plugins.values())

    def get_enabled_plugins(self) -> list[Plugin]:
        """Get all enabled and loaded plugin instances."""
        return [
            s.instance for s in self._plugins.values()
            if s.enabled and s.loaded and s.instance
        ]

    # ------------------------------------------------------------------
    # Lifecycle Hooks (called by server.py)
    # ------------------------------------------------------------------

    def on_startup(self):
        """Load and start all enabled plugins.  Called during CAM boot."""
        for pid, state in self._plugins.items():
            if state.enabled:
                instance = self.load_plugin(pid)
                if instance:
                    try:
                        instance.on_startup()
                    except Exception:
                        logger.warning("Plugin %s startup error", pid, exc_info=True)

        loaded = sum(1 for s in self._plugins.values() if s.loaded)
        logger.info("Plugin startup complete: %d/%d loaded",
                     loaded, len(self._plugins))

    def on_shutdown(self):
        """Shut down all loaded plugins.  Called during CAM shutdown."""
        for pid, state in self._plugins.items():
            if state.loaded and state.instance:
                try:
                    state.instance.on_shutdown()
                except Exception:
                    logger.warning("Plugin %s shutdown error", pid, exc_info=True)
                state.loaded = False
                state.instance = None

        logger.info("Plugin shutdown complete")

    def on_task_received(self, task: dict):
        """Broadcast task_received to all enabled plugins."""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_task_received(task)
            except Exception:
                logger.debug("Plugin %s on_task_received error",
                             plugin.plugin_id, exc_info=True)

    def on_task_completed(self, task: dict):
        """Broadcast task_completed to all enabled plugins."""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_task_completed(task)
            except Exception:
                logger.debug("Plugin %s on_task_completed error",
                             plugin.plugin_id, exc_info=True)

    def on_agent_connected(self, agent_id: str):
        """Broadcast agent_connected to all enabled plugins."""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_agent_connected(agent_id)
            except Exception:
                logger.debug("Plugin %s on_agent_connected error",
                             plugin.plugin_id, exc_info=True)

    def on_agent_disconnected(self, agent_id: str):
        """Broadcast agent_disconnected to all enabled plugins."""
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_agent_disconnected(agent_id)
            except Exception:
                logger.debug("Plugin %s on_agent_disconnected error",
                             plugin.plugin_id, exc_info=True)

    # ------------------------------------------------------------------
    # Status & Broadcast
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard."""
        total = len(self._plugins)
        enabled = sum(1 for s in self._plugins.values() if s.enabled)
        loaded = sum(1 for s in self._plugins.values() if s.loaded)
        errors = sum(1 for s in self._plugins.values() if s.error)

        return {
            "total_plugins": total,
            "enabled": enabled,
            "loaded": loaded,
            "errors": errors,
        }

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Full state snapshot for WebSocket broadcast."""
        return {
            "plugins": [s.to_dict() for s in self._plugins.values()],
            "status": self.get_status(),
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
                logger.info("PluginManager closed")
            except Exception:
                pass
