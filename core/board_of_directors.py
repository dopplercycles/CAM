"""
AI Board of Directors

Multi-model "boardroom" where 7 AI board members from different providers
discuss questions simultaneously. Each member has a distinct role, persona,
and backing model.

Usage:
    from core.board_of_directors import BoardOfDirectors

    board = BoardOfDirectors()
    session_id = board.create_session("Doppler Cycles strategy meeting")
    await board.ask_board(session_id, "Should we expand into electric bikes?",
                          on_member_response=my_callback)
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

# Anthropic SDK — optional
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore
    _HAS_ANTHROPIC = False

# OpenAI SDK — used for DeepSeek, OpenAI, xAI (OpenAI-compatible APIs)
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore
    _HAS_OPENAI = False

# httpx — used for Google Gemini REST API
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore
    _HAS_HTTPX = False

logger = logging.getLogger("cam.board_of_directors")


# ---------------------------------------------------------------------------
# Board member definitions
# ---------------------------------------------------------------------------

BOARD_MEMBERS = {
    "cam": {
        "name": "Cam",
        "role": "Chief Facilitator",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "color": "#e74c3c",
        "persona": (
            "You are Cam, the Chief Facilitator of Doppler Cycles' AI Board of Directors. "
            "You synthesize perspectives, keep discussion focused, and ensure actionable outcomes. "
            "You know the business intimately — mobile motorcycle diagnostics in Portland, OR. "
            "Keep responses concise and practical. Speak like a trusted shop foreman."
        ),
    },
    "claude": {
        "name": "Claude",
        "role": "Strategic Advisor",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "color": "#d4a574",
        "persona": (
            "You are Claude, Strategic Advisor on Doppler Cycles' AI Board. "
            "You focus on long-term strategy, risk assessment, and competitive positioning. "
            "Think 2-5 years ahead. Weigh trade-offs carefully. Be direct about risks."
        ),
    },
    "claude_code": {
        "name": "Claude Code",
        "role": "Technical Lead",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "color": "#9b59b6",
        "persona": (
            "You are Claude Code, Technical Lead on Doppler Cycles' AI Board. "
            "You evaluate technical feasibility, implementation costs, and architecture decisions. "
            "Think about what can actually be built, maintained, and scaled by a solo operator. "
            "Be practical — George is bootstrapping, so every tech choice must earn its keep."
        ),
    },
    "deepseek": {
        "name": "DeepSeek",
        "role": "R&D Analyst",
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "color": "#2ecc71",
        "persona": (
            "You are DeepSeek, R&D Analyst on Doppler Cycles' AI Board. "
            "You research emerging technologies, industry trends, and innovation opportunities. "
            "Think deeply about what's coming next in motorcycle diagnostics, EVs, and shop tech. "
            "Back up insights with reasoning. Be the board's futurist."
        ),
    },
    "grok": {
        "name": "Grok",
        "role": "Market Intelligence",
        "provider": "xai",
        "model": "grok-3",
        "color": "#3498db",
        "persona": (
            "You are Grok, Market Intelligence officer on Doppler Cycles' AI Board. "
            "You track market trends, competitor moves, and consumer sentiment. "
            "Be bold and direct. Cut through noise. Identify opportunities others miss. "
            "Think like a market analyst who rides motorcycles."
        ),
    },
    "chatgpt": {
        "name": "ChatGPT",
        "role": "Customer Strategy",
        "provider": "openai",
        "model": "gpt-4o",
        "color": "#1abc9c",
        "persona": (
            "You are ChatGPT, Customer Strategy lead on Doppler Cycles' AI Board. "
            "You focus on customer experience, marketing, content strategy, and brand building. "
            "Think about how to attract and retain customers for a mobile diagnostics business. "
            "Be creative but grounded in what a solo operator can execute."
        ),
    },
    "gemini": {
        "name": "Gemini",
        "role": "Data & Analytics",
        "provider": "google",
        "model": "gemini-2.0-flash",
        "color": "#f39c12",
        "persona": (
            "You are Gemini, Data & Analytics lead on Doppler Cycles' AI Board. "
            "You focus on metrics, data-driven decisions, KPIs, and operational efficiency. "
            "Quantify everything you can. Suggest what to measure and how to use data. "
            "Be precise and numbers-oriented."
        ),
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BoardMessage:
    """A single message in a board session."""
    member_id: str          # "cam", "user", etc.
    member_name: str
    role: str
    text: str
    color: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = 0.0


@dataclass
class BoardSession:
    """A board discussion session."""
    session_id: str
    title: str
    briefing: str = ""
    messages: list[BoardMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    active_members: list[str] = field(default_factory=lambda: list(BOARD_MEMBERS.keys()))


# ---------------------------------------------------------------------------
# BoardOfDirectors class
# ---------------------------------------------------------------------------

class BoardOfDirectors:
    """Manages board sessions and coordinates multi-model API calls.

    API keys for Anthropic and DeepSeek come from environment variables.
    Keys for OpenAI, xAI, and Google are set at runtime via the dashboard
    (stored in memory only, never persisted to disk).
    """

    def __init__(self):
        self._sessions: dict[str, BoardSession] = {}
        # Runtime API keys for providers not in .env
        self._runtime_keys: dict[str, str] = {}  # provider -> api_key

    # -- Session management -------------------------------------------------

    def create_session(self, title: str = "Board Meeting") -> str:
        """Create a new board session and return its ID."""
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = BoardSession(
            session_id=session_id,
            title=title,
        )
        logger.info("Board session created: %s (%s)", session_id, title)
        return session_id

    def get_session(self, session_id: str) -> BoardSession | None:
        return self._sessions.get(session_id)

    def set_briefing(self, session_id: str, briefing: str) -> bool:
        """Set the briefing context for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.briefing = briefing
        logger.info("Board briefing set for session %s (%d chars)", session_id, len(briefing))
        return True

    def set_active_members(self, session_id: str, member_ids: list[str]) -> bool:
        """Set which members are active for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.active_members = [m for m in member_ids if m in BOARD_MEMBERS]
        return True

    def set_api_key(self, provider: str, key: str):
        """Store a runtime API key for a provider (memory only)."""
        self._runtime_keys[provider] = key
        logger.info("Runtime API key set for provider: %s", provider)

    def get_api_keys_status(self) -> dict[str, bool]:
        """Return which providers have keys configured."""
        return {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "openai": bool(self._runtime_keys.get("openai")),
            "xai": bool(self._runtime_keys.get("xai")),
            "google": bool(self._runtime_keys.get("google")),
        }

    # -- Main board query ---------------------------------------------------

    async def ask_board(
        self,
        session_id: str,
        question: str,
        on_member_response: Callable[[str, BoardMessage], Awaitable[None]] | None = None,
    ) -> list[BoardMessage]:
        """Ask all active board members a question concurrently.

        Args:
            session_id: The board session ID.
            question: The question to ask.
            on_member_response: Async callback(session_id, message) called
                                as each member responds.

        Returns:
            List of BoardMessage responses.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Unknown session: {session_id}")

        # Record user question
        user_msg = BoardMessage(
            member_id="user",
            member_name="George",
            role="Owner",
            text=question,
            color="#ffffff",
        )
        session.messages.append(user_msg)

        # Build tasks for active members
        tasks = []
        for member_id in session.active_members:
            member = BOARD_MEMBERS[member_id]
            tasks.append(self._query_member(session, member_id, member, question, on_member_response))

        # Fan out — all members respond concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for r in results:
            if isinstance(r, BoardMessage):
                responses.append(r)
            elif isinstance(r, Exception):
                logger.error("Board member query failed: %s", r)

        return responses

    async def _query_member(
        self,
        session: BoardSession,
        member_id: str,
        member: dict,
        question: str,
        on_member_response: Callable | None,
    ) -> BoardMessage:
        """Query a single board member and return their response."""
        provider = member["provider"]

        # Build context with briefing + conversation history
        context_parts = []
        if session.briefing:
            context_parts.append(f"BRIEFING:\n{session.briefing}\n")
        # Include last 10 messages for context
        recent = session.messages[-10:]
        if recent:
            history = "\n".join(
                f"{m.member_name} ({m.role}): {m.text}" for m in recent
            )
            context_parts.append(f"RECENT DISCUSSION:\n{history}\n")
        context_parts.append(f"CURRENT QUESTION: {question}")
        full_prompt = "\n\n".join(context_parts)

        system_prompt = (
            f"{member['persona']}\n\n"
            f"You are in a board meeting with 6 other AI advisors. "
            f"Keep your response focused and under 300 words. "
            f"Address the question directly from your role's perspective."
        )

        start = time.perf_counter()
        try:
            if provider == "anthropic":
                text = await self._call_anthropic(full_prompt, member["model"], system_prompt)
            elif provider == "deepseek":
                text = await self._call_deepseek(full_prompt, member["model"], system_prompt)
            elif provider == "openai":
                text = await self._call_openai(full_prompt, member["model"], system_prompt)
            elif provider == "xai":
                text = await self._call_xai(full_prompt, member["model"], system_prompt)
            elif provider == "google":
                text = await self._call_google(full_prompt, member["model"], system_prompt)
            else:
                text = f"[{member['name']}: Unknown provider '{provider}']"
        except Exception as e:
            logger.error("Board member %s failed: %s", member_id, e)
            text = f"[{member['name']}: Error — {e}]"

        elapsed_ms = (time.perf_counter() - start) * 1000

        msg = BoardMessage(
            member_id=member_id,
            member_name=member["name"],
            role=member["role"],
            text=text,
            color=member["color"],
            latency_ms=round(elapsed_ms, 1),
        )
        session.messages.append(msg)

        # Fire callback so server can broadcast as each member finishes
        if on_member_response:
            await on_member_response(session.session_id, msg)

        return msg

    # -- Provider API callers -----------------------------------------------

    async def _call_anthropic(self, prompt: str, model: str, system_prompt: str) -> str:
        """Call Anthropic Claude API."""
        if not _HAS_ANTHROPIC:
            return "[Claude: anthropic SDK not installed]"

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "[Claude: API key not configured]"

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text.strip()

    async def _call_deepseek(self, prompt: str, model: str, system_prompt: str) -> str:
        """Call DeepSeek API (OpenAI-compatible)."""
        if not _HAS_OPENAI:
            return "[DeepSeek: openai SDK not installed]"

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return "[DeepSeek: API key not configured]"

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        messages = []
        # deepseek-reasoner doesn't support system messages — prepend to user
        if model == "deepseek-reasoner":
            messages.append({"role": "user", "content": f"{system_prompt}\n\n{prompt}"})
        else:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
        )
        return (response.choices[0].message.content or "").strip()

    async def _call_openai(self, prompt: str, model: str, system_prompt: str) -> str:
        """Call OpenAI API (for ChatGPT)."""
        if not _HAS_OPENAI:
            return "[ChatGPT: openai SDK not installed]"

        api_key = self._runtime_keys.get("openai", "")
        if not api_key:
            return "[ChatGPT: API key not configured]"

        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
        return (response.choices[0].message.content or "").strip()

    async def _call_xai(self, prompt: str, model: str, system_prompt: str) -> str:
        """Call xAI Grok API (OpenAI-compatible)."""
        if not _HAS_OPENAI:
            return "[Grok: openai SDK not installed]"

        api_key = self._runtime_keys.get("xai", "")
        if not api_key:
            return "[Grok: API key not configured]"

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
        return (response.choices[0].message.content or "").strip()

    async def _call_google(self, prompt: str, model: str, system_prompt: str) -> str:
        """Call Google Gemini API via REST."""
        if not _HAS_HTTPX:
            return "[Gemini: httpx not installed]"

        api_key = self._runtime_keys.get("google", "")
        if not api_key:
            return "[Gemini: API key not configured]"

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract text from Gemini response
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            return f"[Gemini: Unexpected response format: {data}]"

    # -- Export -------------------------------------------------------------

    def export_session_markdown(self, session_id: str) -> str | None:
        """Export a board session as a Markdown document."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        lines = [
            f"# AI Board of Directors — {session.title}",
            f"*Session: {session.session_id} | Created: {session.created_at}*",
            "",
        ]

        if session.briefing:
            lines.extend([
                "## Briefing",
                session.briefing,
                "",
            ])

        lines.append("## Discussion")
        lines.append("")

        for msg in session.messages:
            if msg.member_id == "user":
                lines.append(f"### George (Owner)")
            else:
                lines.append(f"### {msg.member_name} — {msg.role}")
            lines.append(f"*{msg.timestamp}*")
            if msg.latency_ms > 0:
                lines.append(f"*Response time: {msg.latency_ms:.0f}ms*")
            lines.append("")
            lines.append(msg.text)
            lines.append("")

        lines.extend([
            "---",
            f"*Exported from CAM AI Board of Directors*",
        ])

        return "\n".join(lines)
