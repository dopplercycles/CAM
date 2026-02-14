"""
CAM Access Control — Multi-User Role-Based Access Control

SQLite-backed user management with four roles (admin, operator, viewer, api_client).
Server-side permission enforcement for every WebSocket message type and dashboard panel.

Follows the SecurityAuditLog pattern: standalone module, no FastAPI dependency.

Usage:
    from core.access_control import AccessControl, Role, WS_PERMISSIONS

    ac = AccessControl(db_path="data/access_control.db")
    ac.ensure_default_admin("george", password_hash="$2b$12$...")

    # Permission checks
    if ac.check_ws_permission("george", "kill_switch"):
        ...  # allowed
    panels = ac.get_allowed_panels("george")
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger("cam.access_control")


# ---------------------------------------------------------------------------
# Role Enum & Hierarchy
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """User roles in order of increasing privilege."""
    ADMIN = "admin"           # George — full access
    OPERATOR = "operator"     # Employees — business ops, no system config
    VIEWER = "viewer"         # Read-only dashboard access
    API_CLIENT = "api_client" # Programmatic, scoped

ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.API_CLIENT: 1,
    Role.OPERATOR: 2,
    Role.ADMIN: 3,
}


# ---------------------------------------------------------------------------
# User Dataclass
# ---------------------------------------------------------------------------

@dataclass
class User:
    """A single CAM user account.

    Attributes:
        username:         Unique login name (primary key)
        display_name:     Friendly name for UI display
        role:             One of the Role enum values
        password_hash:    Bcrypt hash (never exposed via to_dict)
        telegram_chat_id: Optional Telegram chat ID for notifications
        api_scopes:       JSON list of allowed API scopes (for api_client role)
        is_active:        Whether the account is currently active
        created_at:       ISO timestamp of account creation
        updated_at:       ISO timestamp of last modification
        created_by:       Username of who created this account
    """
    username: str
    display_name: str
    role: Role = Role.VIEWER
    password_hash: str = ""
    telegram_chat_id: int | None = None
    api_scopes: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: str = ""
    updated_at: str = ""
    created_by: str = "system"

    def to_dict(self) -> dict:
        """Serialize to dict, excluding password_hash for safety."""
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role.value if isinstance(self.role, Role) else self.role,
            "telegram_chat_id": self.telegram_chat_id,
            "api_scopes": self.api_scopes,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "User":
        """Create a User from a SQLite row."""
        try:
            api_scopes = json.loads(row["api_scopes"]) if row["api_scopes"] else []
        except (json.JSONDecodeError, TypeError):
            api_scopes = []
        try:
            role = Role(row["role"])
        except ValueError:
            role = Role.VIEWER
        return cls(
            username=row["username"],
            display_name=row["display_name"],
            role=role,
            password_hash=row["password_hash"] or "",
            telegram_chat_id=row["telegram_chat_id"],
            api_scopes=api_scopes,
            is_active=bool(row["is_active"]),
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            created_by=row["created_by"] or "system",
        )


# ---------------------------------------------------------------------------
# Panel Visibility Maps
# ---------------------------------------------------------------------------

ADMIN_ONLY_PANELS: set[str] = {
    "cam-chat", "settings", "security", "self-test", "launch-readiness",
    "plugin-manager", "agent-deploy", "deploy", "backup",
    "performance", "context", "memory", "msgbus", "webhooks",
    "offline-mode", "persona", "training", "user-management",
}

OPERATOR_PANELS: set[str] = {
    "tasks", "chains", "swarms", "scheduler", "content", "pipeline",
    "youtube", "media-library", "social-media", "tts", "research",
    "business", "invoicing", "parts-inventory", "financial-dashboard",
    "route-planner", "highway20", "crm", "email-templates", "appointment-calendar",
    "service-records", "photo-docs", "ride-log", "market-monitor",
    "feedback", "warranty-recall", "maintenance-scheduler", "diagnostics",
    "reports", "knowledge-ingest", "scout", "market-analytics",
    "command-palette",
}

VIEWER_PANELS: set[str] = {
    "agents", "analytics", "event-log", "export", "file-transfer",
}

ROLE_PANELS: dict[Role, set[str]] = {
    Role.ADMIN: VIEWER_PANELS | OPERATOR_PANELS | ADMIN_ONLY_PANELS,
    Role.OPERATOR: VIEWER_PANELS | OPERATOR_PANELS,
    Role.VIEWER: VIEWER_PANELS,
    Role.API_CLIENT: set(),
}


# ---------------------------------------------------------------------------
# WebSocket Permission Map
# ---------------------------------------------------------------------------

# Maps every known WS message type to the minimum role required.
# Unknown types default to ADMIN (fail-safe).
WS_PERMISSIONS: dict[str, Role] = {
    # --- ADMIN only ---
    "kill_switch": Role.ADMIN,
    "config_reload": Role.ADMIN,
    "persona_reload": Role.ADMIN,
    "approval_response": Role.ADMIN,
    "deploy_start": Role.ADMIN,
    "deploy_status": Role.ADMIN,
    "deploy_cancel": Role.ADMIN,
    "backup_create": Role.ADMIN,
    "backup_restore": Role.ADMIN,
    "backup_delete": Role.ADMIN,
    "backup_list": Role.ADMIN,
    "plugin_install": Role.ADMIN,
    "plugin_uninstall": Role.ADMIN,
    "plugin_enable": Role.ADMIN,
    "plugin_disable": Role.ADMIN,
    "plugin_list": Role.ADMIN,
    "offline_force": Role.ADMIN,
    "offline_process": Role.ADMIN,
    "offline_clear": Role.ADMIN,
    "offline_remove": Role.ADMIN,
    "performance_apply_rec": Role.ADMIN,
    "performance_dismiss_rec": Role.ADMIN,
    "performance_update_threshold": Role.ADMIN,
    "run_self_test": Role.ADMIN,
    "run_launch_readiness": Role.ADMIN,
    "training_start": Role.ADMIN,
    "training_stop": Role.ADMIN,
    "training_status": Role.ADMIN,
    "webhook_register": Role.ADMIN,
    "webhook_update": Role.ADMIN,
    "webhook_delete": Role.ADMIN,
    "webhook_toggle": Role.ADMIN,
    "webhook_test": Role.ADMIN,
    "set_model": Role.ADMIN,
    "tool_approval_response": Role.ADMIN,
    "cam_chat_message": Role.ADMIN,
    "cam_chat_history": Role.ADMIN,
    "cam_voice_input": Role.ADMIN,
    "cam_tts_request": Role.ADMIN,
    "bus_publish": Role.ADMIN,
    "user_list": Role.ADMIN,
    "user_add": Role.ADMIN,
    "user_set_role": Role.ADMIN,
    "user_revoke": Role.ADMIN,
    "user_restore": Role.ADMIN,
    "user_delete": Role.ADMIN,

    # --- OPERATOR ---
    "command": Role.OPERATOR,
    "task_submit": Role.OPERATOR,
    "ping_agent": Role.OPERATOR,
    "command_execute": Role.OPERATOR,
    "chain_submit": Role.OPERATOR,
    "swarm_submit": Role.OPERATOR,
    "file_upload": Role.OPERATOR,
    "file_transfer_start": Role.OPERATOR,
    "file_chunk": Role.OPERATOR,
    "file_cancel": Role.OPERATOR,
    "schedule_add": Role.OPERATOR,
    "schedule_update": Role.OPERATOR,
    "schedule_delete": Role.OPERATOR,
    "schedule_toggle": Role.OPERATOR,
    "content_entry_add": Role.OPERATOR,
    "content_entry_update": Role.OPERATOR,
    "content_entry_delete": Role.OPERATOR,
    "pipeline_create": Role.OPERATOR,
    "pipeline_update": Role.OPERATOR,
    "pipeline_delete": Role.OPERATOR,
    "pipeline_transition": Role.OPERATOR,
    "yt_update": Role.OPERATOR,
    "yt_delete": Role.OPERATOR,
    "yt_schedule": Role.OPERATOR,
    "yt_publish": Role.OPERATOR,
    "media_upload": Role.OPERATOR,
    "media_delete": Role.OPERATOR,
    "media_update": Role.OPERATOR,
    "sm_post": Role.OPERATOR,
    "sm_schedule": Role.OPERATOR,
    "sm_delete": Role.OPERATOR,
    "sm_update": Role.OPERATOR,
    "email_send": Role.OPERATOR,
    "email_create": Role.OPERATOR,
    "email_update": Role.OPERATOR,
    "email_delete": Role.OPERATOR,
    "photo_upload": Role.OPERATOR,
    "photo_delete": Role.OPERATOR,
    "photo_update": Role.OPERATOR,
    "ride_add": Role.OPERATOR,
    "ride_update": Role.OPERATOR,
    "ride_delete": Role.OPERATOR,
    "market_watch": Role.OPERATOR,
    "market_unwatch": Role.OPERATOR,
    "market_refresh": Role.OPERATOR,
    "feedback_submit": Role.OPERATOR,
    "feedback_update": Role.OPERATOR,
    "feedback_delete": Role.OPERATOR,
    "warranty_add": Role.OPERATOR,
    "warranty_update": Role.OPERATOR,
    "warranty_delete": Role.OPERATOR,
    "maintenance_add": Role.OPERATOR,
    "maintenance_update": Role.OPERATOR,
    "maintenance_delete": Role.OPERATOR,
    "business_update": Role.OPERATOR,
    "service_add": Role.OPERATOR,
    "service_update": Role.OPERATOR,
    "service_delete": Role.OPERATOR,
    "diag_run": Role.OPERATOR,
    "diag_save": Role.OPERATOR,
    "crm_add": Role.OPERATOR,
    "crm_update": Role.OPERATOR,
    "crm_delete": Role.OPERATOR,
    "appt_schedule_add": Role.OPERATOR,
    "appt_schedule_update": Role.OPERATOR,
    "appt_schedule_delete": Role.OPERATOR,
    "invoicing_create": Role.OPERATOR,
    "invoicing_update": Role.OPERATOR,
    "invoicing_delete": Role.OPERATOR,
    "invoicing_send": Role.OPERATOR,
    "inventory_add": Role.OPERATOR,
    "inventory_update": Role.OPERATOR,
    "inventory_delete": Role.OPERATOR,
    "finances_add": Role.OPERATOR,
    "finances_update": Role.OPERATOR,
    "finances_delete": Role.OPERATOR,
    "route_plan": Role.OPERATOR,
    "route_save": Role.OPERATOR,
    "route_delete": Role.OPERATOR,
    "reports_generate": Role.OPERATOR,
    "reports_delete": Role.OPERATOR,
    "knowledge_scan_inbox": Role.OPERATOR,
    "scout_search": Role.OPERATOR,
    "scout_watch": Role.OPERATOR,
    "scout_unwatch": Role.OPERATOR,
    "tts_synthesize": Role.OPERATOR,
    "tts_queue": Role.OPERATOR,
    "tts_delete": Role.OPERATOR,
    "research_result_delete": Role.OPERATOR,
    "hwy20_add_segment": Role.OPERATOR,
    "hwy20_update_segment": Role.OPERATOR,
    "hwy20_delete_segment": Role.OPERATOR,
    "hwy20_mark_filmed": Role.OPERATOR,
    "hwy20_generate_shots": Role.OPERATOR,
    "hwy20_add_shot": Role.OPERATOR,
    "hwy20_update_shot": Role.OPERATOR,
    "hwy20_delete_shot": Role.OPERATOR,
    "hwy20_weather": Role.OPERATOR,
    "hwy20_scouting_note": Role.OPERATOR,

    # --- VIEWER ---
    "task_list": Role.VIEWER,
    "command_list": Role.VIEWER,
    "set_client_mode": Role.VIEWER,
    "security_audit_filter": Role.VIEWER,
    "notification_dismiss": Role.VIEWER,
    "notification_dismiss_all": Role.VIEWER,
    "tts_list": Role.VIEWER,
    "tts_status": Role.VIEWER,
    "export_csv": Role.VIEWER,
    "export_json": Role.VIEWER,
    "export_start": Role.VIEWER,
    "export_status": Role.VIEWER,
    "offline_get_status": Role.VIEWER,
    "performance_get_status": Role.VIEWER,
    "research_search": Role.VIEWER,
    "webhook_delivery_history": Role.VIEWER,
    "episodic_search": Role.VIEWER,
    "hwy20_search_notes": Role.VIEWER,
    "bus_history": Role.VIEWER,
}


# ---------------------------------------------------------------------------
# AccessControl Class
# ---------------------------------------------------------------------------

class AccessControl:
    """Multi-user role-based access control backed by SQLite.

    Manages user accounts, role assignments, and permission checks for
    WebSocket message types and dashboard panels.

    Args:
        db_path:    Path to SQLite database file.
        audit_log:  Optional SecurityAuditLog instance for logging changes.
        on_change:  Optional async callback invoked after user changes.
    """

    def __init__(
        self,
        db_path: str = "data/access_control.db",
        audit_log: Any = None,
        on_change: Callable[..., Coroutine] | None = None,
    ):
        self.db_path = db_path
        self.audit_log = audit_log
        self.on_change = on_change

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("AccessControl initialized (db=%s)", db_path)

    def _create_tables(self):
        """Create the users table if it doesn't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY,
                display_name    TEXT NOT NULL,
                role            TEXT NOT NULL DEFAULT 'viewer',
                password_hash   TEXT DEFAULT '',
                telegram_chat_id INTEGER,
                api_scopes      TEXT DEFAULT '[]',
                is_active       INTEGER DEFAULT 1,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                created_by      TEXT DEFAULT 'system'
            );
            CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
            CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
        """)
        self._conn.commit()

    def _now(self) -> str:
        """Current UTC timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat()

    def _audit(self, action: str, actor: str, target: str, **kwargs):
        """Log to audit trail if audit_log is available."""
        if self.audit_log:
            try:
                self.audit_log.log_action(
                    action_type=action,
                    actor=actor,
                    target=target,
                    result="executed",
                    risk_level="medium",
                    **kwargs,
                )
            except Exception as e:
                logger.warning("Failed to write audit entry: %s", e)

    # ----- User Management -----

    def add_user(
        self,
        username: str,
        display_name: str,
        role: str | Role = Role.VIEWER,
        password_hash: str = "",
        telegram_chat_id: int | None = None,
        api_scopes: list[str] | None = None,
        created_by: str = "system",
    ) -> User | None:
        """Create a new user account.

        Returns the created User, or None if username already exists.
        """
        if isinstance(role, str):
            try:
                role = Role(role)
            except ValueError:
                logger.warning("Invalid role '%s', defaulting to viewer", role)
                role = Role.VIEWER

        now = self._now()
        try:
            self._conn.execute(
                """INSERT INTO users
                   (username, display_name, role, password_hash, telegram_chat_id,
                    api_scopes, is_active, created_at, updated_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    username, display_name, role.value, password_hash,
                    telegram_chat_id, json.dumps(api_scopes or []),
                    now, now, created_by,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            logger.warning("User '%s' already exists", username)
            return None

        user = User(
            username=username,
            display_name=display_name,
            role=role,
            password_hash=password_hash,
            telegram_chat_id=telegram_chat_id,
            api_scopes=api_scopes or [],
            is_active=True,
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )
        self._audit("user_created", created_by, username, tier=2)
        logger.info("User '%s' created with role '%s' by '%s'", username, role.value, created_by)
        return user

    def get_user(self, username: str) -> User | None:
        """Fetch a single user by username. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return None
        return User.from_row(row)

    def set_role(self, username: str, role: str | Role, changed_by: str = "system") -> bool:
        """Change a user's role. Returns True on success."""
        if isinstance(role, str):
            try:
                role = Role(role)
            except ValueError:
                logger.warning("Invalid role '%s'", role)
                return False

        now = self._now()
        cursor = self._conn.execute(
            "UPDATE users SET role = ?, updated_at = ? WHERE username = ?",
            (role.value, now, username),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return False

        self._audit("user_role_changed", changed_by, username, tier=2)
        logger.info("User '%s' role changed to '%s' by '%s'", username, role.value, changed_by)
        return True

    def revoke_access(self, username: str, revoked_by: str = "system") -> bool:
        """Deactivate a user account. Returns True on success."""
        now = self._now()
        cursor = self._conn.execute(
            "UPDATE users SET is_active = 0, updated_at = ? WHERE username = ?",
            (now, username),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return False

        self._audit("user_revoked", revoked_by, username, tier=2)
        logger.info("User '%s' access revoked by '%s'", username, revoked_by)
        return True

    def restore_access(self, username: str, restored_by: str = "system") -> bool:
        """Reactivate a user account. Returns True on success."""
        now = self._now()
        cursor = self._conn.execute(
            "UPDATE users SET is_active = 1, updated_at = ? WHERE username = ?",
            (now, username),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return False

        self._audit("user_restored", restored_by, username, tier=2)
        logger.info("User '%s' access restored by '%s'", username, restored_by)
        return True

    def update_user(self, username: str, changed_by: str = "system", **fields) -> bool:
        """Update user fields. Accepts display_name, telegram_chat_id, api_scopes, password_hash.

        Returns True if user was found and updated.
        """
        allowed = {"display_name", "telegram_chat_id", "api_scopes", "password_hash"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False

        # Serialize api_scopes to JSON if present
        if "api_scopes" in updates and isinstance(updates["api_scopes"], list):
            updates["api_scopes"] = json.dumps(updates["api_scopes"])

        now = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [now, username]

        cursor = self._conn.execute(
            f"UPDATE users SET {set_clause}, updated_at = ? WHERE username = ?",
            values,
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return False

        self._audit("user_updated", changed_by, username, tier=2)
        logger.info("User '%s' updated by '%s' (fields: %s)", username, changed_by, list(updates.keys()))
        return True

    def delete_user(self, username: str, deleted_by: str = "system") -> bool:
        """Permanently remove a user. Returns True on success."""
        cursor = self._conn.execute(
            "DELETE FROM users WHERE username = ?", (username,)
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return False

        self._audit("user_deleted", deleted_by, username, tier=2)
        logger.info("User '%s' deleted by '%s'", username, deleted_by)
        return True

    def list_users(self, include_inactive: bool = False) -> list[User]:
        """List all users. Excludes inactive by default."""
        if include_inactive:
            rows = self._conn.execute(
                "SELECT * FROM users ORDER BY created_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM users WHERE is_active = 1 ORDER BY created_at"
            ).fetchall()
        return [User.from_row(r) for r in rows]

    # ----- Permission Checks -----

    def get_user_role(self, username: str) -> Role | None:
        """Get a user's role. Returns None if not found or inactive."""
        row = self._conn.execute(
            "SELECT role FROM users WHERE username = ? AND is_active = 1",
            (username,),
        ).fetchone()
        if not row:
            return None
        try:
            return Role(row["role"])
        except ValueError:
            return None

    def has_role(self, username: str, minimum_role: Role) -> bool:
        """Check if a user's role meets or exceeds the minimum.

        Uses the ROLE_HIERARCHY for comparison.
        """
        user_role = self.get_user_role(username)
        if user_role is None:
            return False
        return ROLE_HIERARCHY.get(user_role, -1) >= ROLE_HIERARCHY.get(minimum_role, 999)

    def check_ws_permission(self, username: str, msg_type: str) -> bool:
        """Check if a user is allowed to send a specific WS message type.

        Unknown message types require admin (fail-safe).
        """
        required_role = WS_PERMISSIONS.get(msg_type, Role.ADMIN)
        return self.has_role(username, required_role)

    def check_panel_access(self, username: str, panel_id: str) -> bool:
        """Check if a user can see a specific dashboard panel."""
        user_role = self.get_user_role(username)
        if user_role is None:
            return False
        allowed = ROLE_PANELS.get(user_role, set())
        return panel_id in allowed

    def get_allowed_panels(self, username: str) -> list[str]:
        """Get the list of panel IDs a user is allowed to see."""
        user_role = self.get_user_role(username)
        if user_role is None:
            return []
        return sorted(ROLE_PANELS.get(user_role, set()))

    def get_role_info(self, username: str) -> dict:
        """Get role info suitable for sending to the client.

        Returns dict with username, display_name, role, allowed_panels, is_admin.
        """
        user = self.get_user(username)
        if not user or not user.is_active:
            return {
                "username": username,
                "display_name": username,
                "role": "viewer",
                "allowed_panels": sorted(ROLE_PANELS.get(Role.VIEWER, set())),
                "is_admin": False,
            }
        return {
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role.value if isinstance(user.role, Role) else user.role,
            "allowed_panels": self.get_allowed_panels(username),
            "is_admin": user.role == Role.ADMIN,
        }

    # ----- Backward Compatibility -----

    def ensure_default_admin(self, username: str = "george", password_hash: str = "") -> User | None:
        """Create the default admin user if no users exist yet.

        Called at startup to ensure backward compatibility. If the DB
        already has users, this is a no-op.

        Returns the created User, or None if users already exist.
        """
        count = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count > 0:
            return None

        user = self.add_user(
            username=username,
            display_name=username.title(),
            role=Role.ADMIN,
            password_hash=password_hash,
            created_by="system",
        )
        if user:
            logger.info("Default admin user '%s' created", username)
        return user

    # ----- Dashboard Broadcast -----

    def to_broadcast_list(self) -> list[dict]:
        """Return all users (active and inactive) as dicts for dashboard display.

        Excludes password_hash for safety.
        """
        rows = self._conn.execute(
            "SELECT * FROM users ORDER BY created_at"
        ).fetchall()
        return [User.from_row(r).to_dict() for r in rows]

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("AccessControl database closed")
