"""
CAM Centralized Configuration

Loads settings from config/settings.toml, applies environment variable
overrides, and exposes a thread-safe singleton via get_config().

Zero new dependencies — uses stdlib tomllib (Python 3.12).

Usage:
    from core.config import get_config

    config = get_config()
    port = config.dashboard.port          # dot-access
    url  = config.models.ollama_url
    config.reload()                       # hot-reload from disk
"""

import logging
import os
import threading
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("cam.config")


# ---------------------------------------------------------------------------
# Hardcoded fallback defaults — system still starts if settings.toml is missing
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "dashboard": {
        "host": "0.0.0.0",
        "port": 8080,
        "log_level": "info",
        "heartbeat_check_interval": 10,
        "heartbeat_timeout": 120,
        "agent_dispatch_timeout": 120,
        "default_event_count": 200,
    },
    "orchestrator": {
        "poll_interval": 1.0,
    },
    "health": {
        "heartbeat_interval": 30.0,
        "missed_heartbeat_threshold": 3,
        "success_rate_threshold": 75.0,
    },
    "events": {
        "max_events": 1000,
    },
    "analytics": {
        "db_path": "data/analytics.db",
    },
    "models": {
        "ollama_url": "http://localhost:11434",
        "routing": {
            "simple": "glm-4.7-flash",
            "routine": "gpt-oss:20b",
            "agentic": "kimi-k2.5",
            "complex": "claude",
            "nuanced": "claude",
            "tier1": "phi4-mini:3.8b",
            "tier2": "gpt-oss:20b",
            "tier3": "glm-4.7-flash",
        },
        "costs": {
            "claude": {"input": 3.00, "output": 15.00},
            "kimi-k2.5": {"input": 0.50, "output": 1.50},
        },
    },
    "commands": {
        "status_report": 30,
        "system_info": 30,
        "restart_service": 60,
        "run_diagnostic": 90,
        "capture_sensor_data": 45,
    },
    "connector": {
        "heartbeat_interval": 30,
        "reconnect_delay": 3,
        "dashboard_host": "192.168.12.232",
        "dashboard_port": 8080,
        "systemctl_timeout": 30,
    },
    "notifications": {
        "enabled": True,
        "max_history": 200,
        "agent_disconnect": True,
        "task_failure": True,
        "kill_switch": True,
        "high_error_rate": True,
        "error_rate_threshold": 50.0,
        "error_rate_window": 20,
        "cost_threshold": True,
        "cost_threshold_usd": 1.00,
    },
    "file_transfer": {
        "chunk_size": 65536,
        "max_file_size": 52428800,
        "receive_dir": "data/transfers",
        "agent_receive_dir": "~/receive",
        "max_active_transfers": 5,
        "history_size": 100,
    },
    "memory": {
        "short_term_max_messages": 200,
        "short_term_summary_ratio": 0.5,
        "working_memory_path": "data/tasks/working_memory.json",
        "long_term_persist_dir": "data/memory/chromadb",
        "long_term_collection": "cam_long_term",
        "long_term_seed_file": "CAM_BRAIN.md",
        "episodic_db_path": "data/memory/episodic.db",
        "episodic_retention_days": 365,
    },
    "scheduler": {
        "check_interval": 30,
        "persist_file": "data/schedules.json",
    },
    "api": {
        "api_key": "",
    },
    "telegram": {
        "bot_token": "",
        "allowed_chat_ids": [],
        "poll_interval": 0.5,
        "task_timeout": 120.0,
    },
    "tts": {
        "audio_dir": "data/audio",
        "voices_dir": "data/audio/voices",
        "default_voice": "en_US-lessac-medium",
        "sample_rate": 22050,
    },
    "content_calendar": {
        "db_path": "data/content_calendar.db",
    },
    "research": {
        "db_path": "data/research.db",
        "max_pages_per_query": 5,
        "request_timeout": 10,
        "rate_limit_delay": 0.5,
    },
    "scout": {
        "db_path": "data/scout.db",
        "scan_interval": 3600,
        "score_threshold": 8,
        "makes": ["honda", "yamaha", "suzuki", "kawasaki", "harley-davidson", "ducati"],
        "models": [],
        "year_min": 0,
        "year_max": 0,
        "price_min": 0,
        "price_max": 5000,
        "location": "Portland OR",
        "radius_miles": 50,
        "keywords": [],
        "exclude_keywords": ["parts only"],
    },
    "business": {
        "db_path": "data/business.db",
        "seed_sample_data": True,
        "default_labor_rate": 75.0,
        "invoice_prefix": "DC",
    },
    "auth": {
        "username": "george",
        "password_hash": "",
        "session_timeout": 3600,
        "max_login_attempts": 5,
        "lockout_duration": 300,
    },
    "security": {
        "audit_db_path": "data/security_audit.db",
        "approval_timeout": 30,
        "log_tier1_actions": True,
    },
    "backup": {
        "backup_dir": "data/backups",
        "max_backups": 10,
        "daily_backup_time": "03:00",
    },
    "message_bus": {
        "max_messages": 500,
    },
    "webhooks": {
        "enabled": True,
        "db_path": "data/webhooks.db",
        "max_retries": 5,
        "retry_base_seconds": 10,
        "retry_max_seconds": 3600,
        "retry_check_interval": 15,
        "max_delivery_history": 500,
        "inbound_secret": "",
    },
    "knowledge": {
        "db_path": "data/knowledge_ingest.db",
        "inbox_dir": "data/knowledge/inbox",
        "processed_dir": "data/knowledge/processed",
        "scan_interval": 30,
        "max_file_size": 10485760,
        "chunk_target_size": 1000,
        "chunk_overlap": 100,
    },
    "context": {
        "rotation_threshold": 0.9,
        "ltm_top_k": 3,
        "ltm_min_score": 0.3,
        "episodic_recent_count": 5,
        "max_working_memory_tasks": 10,
        "token_estimate_divisor": 4,
        "limits": {
            "glm-4.7-flash": 128000,
            "gpt-oss:20b": 32000,
            "phi4-mini:3.8b": 8000,
            "claude": 200000,
            "kimi-k2.5": 128000,
        },
    },
}

# Environment variable overrides: CAM_<SECTION>_<KEY> → value
# Only flat (non-nested) keys are supported via env vars.
_ENV_OVERRIDES: list[tuple[str, str, type]] = [
    ("CAM_DASHBOARD_HOST",               "dashboard.host",               str),
    ("CAM_DASHBOARD_PORT",               "dashboard.port",               int),
    ("CAM_DASHBOARD_LOG_LEVEL",          "dashboard.log_level",          str),
    ("CAM_HEARTBEAT_CHECK_INTERVAL",     "dashboard.heartbeat_check_interval", int),
    ("CAM_HEARTBEAT_TIMEOUT",            "dashboard.heartbeat_timeout",  int),
    ("CAM_AGENT_DISPATCH_TIMEOUT",       "dashboard.agent_dispatch_timeout", int),
    ("CAM_ORCHESTRATOR_POLL_INTERVAL",   "orchestrator.poll_interval",   float),
    ("CAM_HEALTH_HEARTBEAT_INTERVAL",    "health.heartbeat_interval",    float),
    ("CAM_EVENTS_MAX_EVENTS",            "events.max_events",            int),
    ("CAM_ANALYTICS_DB_PATH",            "analytics.db_path",            str),
    ("CAM_OLLAMA_URL",                   "models.ollama_url",            str),
    ("CAM_API_KEY",                      "api.api_key",                  str),
    ("CAM_TELEGRAM_BOT_TOKEN",           "telegram.bot_token",           str),
    ("CAM_WEBHOOKS_ENABLED",             "webhooks.enabled",             lambda v: v.lower() in ("true", "1")),
]


# ---------------------------------------------------------------------------
# ConfigSection — dot-access wrapper for nested dicts
# ---------------------------------------------------------------------------

class ConfigSection:
    """Wraps a dict so values are accessible as attributes.

    Nested dicts become nested ConfigSections automatically.

        section = ConfigSection({"port": 8080, "nested": {"key": "val"}})
        section.port        # 8080
        section.nested.key  # "val"
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"Config has no key '{name}'. Available: {list(self._data.keys())}"
            )
        if isinstance(value, dict):
            return ConfigSection(value)
        return value

    def __repr__(self) -> str:
        return f"ConfigSection({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying dict (with nested dicts, not ConfigSections)."""
        return self._data


# ---------------------------------------------------------------------------
# CAMConfig — main config object
# ---------------------------------------------------------------------------

class CAMConfig:
    """Loads and manages CAM configuration.

    Reads config/settings.toml relative to the project root, merges
    with hardcoded defaults, applies environment variable overrides.

    Attributes are accessed via dot-notation through ConfigSection:
        config.dashboard.port
        config.models.routing.simple
    """

    def __init__(self, config_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._config_path = self._resolve_path(config_path)
        self._data: dict[str, Any] = {}
        self.last_loaded: str = ""
        self._load()

    @staticmethod
    def _resolve_path(config_path: str | Path | None) -> Path:
        """Resolve the config file path, defaulting to config/settings.toml."""
        if config_path is not None:
            return Path(config_path)
        # Walk up from this file (core/config.py) to find the project root
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "config" / "settings.toml"

    def _load(self):
        """Load config from TOML, merge with defaults, apply env overrides."""
        data = _deep_copy(_DEFAULTS)

        # Read TOML file
        if self._config_path.exists():
            try:
                with open(self._config_path, "rb") as f:
                    toml_data = tomllib.load(f)
                _deep_merge(data, toml_data)
                logger.info("Configuration loaded from %s", self._config_path)
            except Exception as e:
                logger.warning(
                    "Failed to read %s: %s — using fallback defaults",
                    self._config_path, e,
                )
        else:
            logger.warning(
                "Config file not found at %s — using fallback defaults",
                self._config_path,
            )

        # Apply environment variable overrides
        for env_var, dotpath, cast in _ENV_OVERRIDES:
            env_val = os.environ.get(env_var)
            if env_val is not None:
                try:
                    _set_nested(data, dotpath, cast(env_val))
                    logger.info("Env override: %s=%s", env_var, env_val)
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid env override %s=%s: %s", env_var, env_val, e)

        self._data = data
        self.last_loaded = datetime.now(timezone.utc).isoformat()

    def reload(self) -> dict[str, Any]:
        """Reload configuration from disk.

        Returns a dict of changed values for logging, e.g.:
            {"dashboard.port": {"old": 8080, "new": 9090}}

        Hot-reload scope: values read at runtime (poll_interval,
        dispatch_timeout, heartbeat thresholds) update immediately.
        Values read once at construction time (max_events, db_path)
        need a server restart to take effect.
        """
        with self._lock:
            old_data = _deep_copy(self._data)
            self._load()
            changes = _diff_dicts(old_data, self._data)
            if changes:
                logger.info("Configuration reloaded — %d change(s)", len(changes))
            else:
                logger.info("Configuration reloaded — no changes")
            return changes

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in ("last_loaded", "reload", "to_dict"):
            return super().__getattribute__(name)
        try:
            data = super().__getattribute__("_data")
        except AttributeError:
            raise AttributeError(name)
        try:
            value = data[name]
        except KeyError:
            raise AttributeError(
                f"Config has no section '{name}'. Available: {list(data.keys())}"
            )
        if isinstance(value, dict):
            return ConfigSection(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return the full config as a plain dict (JSON-serializable)."""
        return _deep_copy(self._data)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_instance: CAMConfig | None = None
_instance_lock = threading.Lock()


def get_config(config_path: str | Path | None = None) -> CAMConfig:
    """Return the global CAMConfig singleton.

    Thread-safe. The first call creates the instance; subsequent calls
    return the same object. Pass config_path only on first call to
    override the default location.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CAMConfig(config_path=config_path)
    return _instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_copy(d: dict) -> dict:
    """Simple deep copy for nested dicts of primitives."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _deep_merge(base: dict, override: dict):
    """Merge override into base in-place. Nested dicts are merged recursively."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _set_nested(data: dict, dotpath: str, value: Any):
    """Set a value in a nested dict using a dot-separated path.

    _set_nested(d, "dashboard.port", 9090)
    → d["dashboard"]["port"] = 9090
    """
    keys = dotpath.split(".")
    target = data
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _diff_dicts(old: dict, new: dict, prefix: str = "") -> dict[str, dict]:
    """Return a dict of changed values between two flat-ish dicts.

    Returns: {"dotpath": {"old": ..., "new": ...}}
    """
    changes = {}
    all_keys = set(old.keys()) | set(new.keys())
    for key in all_keys:
        dotpath = f"{prefix}.{key}" if prefix else key
        old_val = old.get(key)
        new_val = new.get(key)
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changes.update(_diff_dicts(old_val, new_val, dotpath))
        elif old_val != new_val:
            changes[dotpath] = {"old": old_val, "new": new_val}
    return changes
