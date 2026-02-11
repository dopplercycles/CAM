"""
CAM Dashboard Authentication — Session Manager

Cookie-based server-side sessions with bcrypt password hashing.
Single-user design (George only). When password_hash is empty in config,
auth is disabled entirely (backward compatible).

No FastAPI dependency — pure Python module. The dashboard server imports
this and wires it into middleware and routes.

Usage:
    from security.auth import SessionManager

    sm = SessionManager(
        username="george",
        password_hash="$2b$12$...",
        session_timeout=3600,
        max_login_attempts=5,
        lockout_duration=300,
    )

    # Verify login
    token = sm.login("george", "password123", client_ip="192.168.1.5")
    # token is a string on success, None on failure

    # Check session
    if sm.validate_session(token):
        ...  # authenticated

    # Lock / logout
    sm.destroy_session(token)
"""

import logging
import secrets
import time
from dataclasses import dataclass, field

import bcrypt

logger = logging.getLogger("cam.auth")


# ---------------------------------------------------------------------------
# Standalone helper — used by set_password.py CLI
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt. Returns the hash string."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """A single authenticated browser session."""
    token: str
    username: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def touch(self):
        """Update last activity timestamp (sliding timeout)."""
        self.last_activity = time.time()


# ---------------------------------------------------------------------------
# Rate limit tracker
# ---------------------------------------------------------------------------

@dataclass
class _LoginAttempts:
    """Track failed login attempts per IP for rate limiting."""
    count: int = 0
    first_attempt: float = field(default_factory=time.time)
    locked_until: float = 0.0


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages dashboard authentication sessions.

    When password_hash is empty, auth_enabled is False and all checks
    are bypassed — the dashboard behaves as before (open access).

    Args:
        username: The single allowed username.
        password_hash: Bcrypt hash of the password. Empty = auth disabled.
        session_timeout: Seconds of inactivity before session expires.
        max_login_attempts: Failed attempts before lockout.
        lockout_duration: Seconds of lockout after max failures.
    """

    def __init__(
        self,
        username: str = "george",
        password_hash: str = "",
        session_timeout: int = 3600,
        max_login_attempts: int = 5,
        lockout_duration: int = 300,
    ):
        self.username = username
        self.password_hash = password_hash
        self.session_timeout = session_timeout
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = lockout_duration

        # Active sessions: token → Session
        self._sessions: dict[str, Session] = {}

        # Rate limiting: IP → _LoginAttempts
        self._attempts: dict[str, _LoginAttempts] = {}

    @property
    def auth_enabled(self) -> bool:
        """True if a password hash is configured (auth is active)."""
        return bool(self.password_hash)

    def _check_password(self, plain: str) -> bool:
        """Verify a plaintext password against the stored bcrypt hash."""
        if not self.password_hash:
            return False
        try:
            return bcrypt.checkpw(
                plain.encode("utf-8"),
                self.password_hash.encode("utf-8"),
            )
        except Exception as e:
            logger.warning("Password check error: %s", e)
            return False

    def _is_locked_out(self, client_ip: str) -> tuple[bool, int]:
        """Check if an IP is currently locked out.

        Returns:
            (is_locked, seconds_remaining)
        """
        attempts = self._attempts.get(client_ip)
        if not attempts:
            return False, 0
        if attempts.locked_until > time.time():
            remaining = int(attempts.locked_until - time.time()) + 1
            return True, remaining
        return False, 0

    def login(self, username: str, password: str, client_ip: str = "unknown") -> str | None:
        """Attempt to log in. Returns a session token on success, None on failure.

        Handles rate limiting: after max_login_attempts failures from the
        same IP, further attempts are rejected for lockout_duration seconds.

        Args:
            username: Submitted username.
            password: Submitted plaintext password.
            client_ip: Client IP for rate limiting.

        Returns:
            Session token string on success, None on failure.
        """
        # Check lockout
        locked, remaining = self._is_locked_out(client_ip)
        if locked:
            logger.warning("Login attempt from locked-out IP %s (%ds remaining)", client_ip, remaining)
            return None

        # Verify credentials
        if username != self.username or not self._check_password(password):
            # Record failed attempt
            if client_ip not in self._attempts:
                self._attempts[client_ip] = _LoginAttempts()
            attempts = self._attempts[client_ip]
            attempts.count += 1
            logger.warning(
                "Failed login attempt from %s (attempt %d/%d)",
                client_ip, attempts.count, self.max_login_attempts,
            )
            if attempts.count >= self.max_login_attempts:
                attempts.locked_until = time.time() + self.lockout_duration
                logger.warning(
                    "IP %s locked out for %ds after %d failed attempts",
                    client_ip, self.lockout_duration, attempts.count,
                )
            return None

        # Success — clear any failed attempts for this IP
        self._attempts.pop(client_ip, None)

        # Create session
        token = secrets.token_urlsafe(32)
        self._sessions[token] = Session(token=token, username=username)
        logger.info("Login successful for '%s' from %s", username, client_ip)
        return token

    def validate_session(self, token: str | None) -> bool:
        """Check if a session token is valid and not expired.

        Touches the session (sliding timeout) if valid.

        Returns:
            True if the session is valid.
        """
        if not token:
            return False
        session = self._sessions.get(token)
        if not session:
            return False
        if time.time() - session.last_activity > self.session_timeout:
            # Expired — clean up
            del self._sessions[token]
            logger.info("Session expired for '%s'", session.username)
            return False
        session.touch()
        return True

    def destroy_session(self, token: str | None):
        """Destroy a session (logout / lock)."""
        if token and token in self._sessions:
            session = self._sessions.pop(token)
            logger.info("Session destroyed for '%s'", session.username)

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns the number removed."""
        now = time.time()
        expired = [
            tok for tok, sess in self._sessions.items()
            if now - sess.last_activity > self.session_timeout
        ]
        for tok in expired:
            del self._sessions[tok]
        if expired:
            logger.info("Cleaned up %d expired session(s)", len(expired))

        # Also clean up old rate limit entries (older than lockout_duration)
        stale_ips = [
            ip for ip, att in self._attempts.items()
            if att.locked_until and att.locked_until < now
        ]
        for ip in stale_ips:
            del self._attempts[ip]

        return len(expired)

    def get_lockout_remaining(self, client_ip: str) -> int:
        """Return seconds remaining in lockout for an IP, or 0 if not locked."""
        _, remaining = self._is_locked_out(client_ip)
        return remaining

    @property
    def active_session_count(self) -> int:
        """Number of currently active (non-expired) sessions."""
        return len(self._sessions)
