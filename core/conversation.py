"""
CAM Conversation Manager

Handles direct George-to-Cam conversations through the dashboard "Talk to Cam"
panel. Every conversation goes through Claude Opus (task_complexity="boss") —
this is George talking to his operations manager, not a delegated subtask.

The ConversationManager:
    - Classifies intent (conversation, status query, command, question)
    - Builds system prompts with live system state for context
    - Manages multi-turn conversation history via episodic memory
    - Routes through the model router at the "boss" tier
    - Records all messages for history persistence

Usage:
    from core.conversation import ConversationManager

    cm = ConversationManager(
        model_router=router, persona=persona,
        episodic_memory=episodic, short_term_memory=stm,
    )
    response = await cm.chat("How are we doing today?")
    print(response.text)
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("cam.conversation")

# Tool-use imports — Claude API tool definitions and executor
from tools.tool_registry import TOOL_DEFINITIONS, classify_tool_tier, execute_tool

# Maximum tool-use rounds before forcing a text response
_MAX_TOOL_ROUNDS = 10


# ---------------------------------------------------------------------------
# Chat response dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """Response from a Cam conversation turn.

    Attributes:
        text:          Cam's response text
        model_used:    Which model generated this (e.g. "claude-opus-4-6")
        input_tokens:  Prompt tokens consumed
        output_tokens: Response tokens generated
        cost:          Estimated USD cost for this turn
        intent:        Classified intent (conversation, status_query, command, question)
        timestamp:     ISO timestamp of the response
    """
    text: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cost: float
    intent: str
    timestamp: str
    tool_calls_made: int = 0


# ---------------------------------------------------------------------------
# Model switch detection — checked before other intents (zero API cost)
# ---------------------------------------------------------------------------

# Patterns that signal a model switch request
_MODEL_SWITCH_PATTERNS = [
    r"\bswitch\b.*\bto\b",
    r"\buse\b.*\bmodels?\b",
    r"\buse\b.*\b(local|ollama|kimi|opus|sonnet|claude|glm|flash|moonshot|gpt.oss)\b",
    r"\bchange\b.*\bmodels?\b",
    r"\bput\b.*\bon\b",
    r"\bswitch\b.*\bover\b",
    r"\bgo\b.*\bback\b.*\bto\b",
    r"\bswitch\b.*\bback\b",
]

# Model name aliases → full model IDs
_MODEL_ALIASES: dict[str, str] = {
    # Claude models
    "opus": "claude-opus-4-6",
    "claude opus": "claude-opus-4-6",
    "claude-opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-5-20250929",
    "claude sonnet": "claude-sonnet-4-5-20250929",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    # Kimi / Moonshot
    "kimi": "kimi-k2.5",
    "kimi k2": "kimi-k2.5",
    "kimi k2.5": "kimi-k2.5",
    "moonshot": "kimi-k2.5",
    # Local models
    "local": "glm-4.7-flash",
    "local model": "glm-4.7-flash",
    "local models": "glm-4.7-flash",
    "glm": "glm-4.7-flash",
    "glm-4": "glm-4.7-flash",
    "flash": "glm-4.7-flash",
    "gpt-oss": "gpt-oss:20b",
    "gpt oss": "gpt-oss:20b",
}

# Patterns for self-targeting ("switch yourself to...", "use sonnet")
_SELF_PATTERNS = [
    r"\byourself\b", r"\byour model\b", r"\byou\b",
    r"\bcam\b", r"\bcam'?s\b",
]


def _resolve_model(message: str) -> str | None:
    """Try to extract a model name from a message.

    Scans the message for known model aliases and returns the full
    model ID. Returns None if no model name is found.
    """
    lower = message.lower()
    # Check longest aliases first to avoid partial matches
    # (e.g., "claude opus" before "opus")
    for alias in sorted(_MODEL_ALIASES, key=len, reverse=True):
        if alias in lower:
            return _MODEL_ALIASES[alias]
    return None


def _is_model_switch(message: str) -> bool:
    """Check if a message looks like a model switch command."""
    lower = message.lower().strip()
    # Must match a switch pattern AND contain a recognized model name
    has_pattern = any(re.search(p, lower) for p in _MODEL_SWITCH_PATTERNS)
    has_model = _resolve_model(message) is not None
    return has_pattern and has_model


# ---------------------------------------------------------------------------
# Intent classification patterns
# ---------------------------------------------------------------------------

_STATUS_PATTERNS = [
    r"\bhow are we\b", r"\bwhat'?s running\b", r"\bagents? online\b",
    r"\bstatus\b", r"\bsystem state\b", r"\bhow'?s (everything|the system|it going)\b",
    r"\bwhat'?s (up|happening|going on)\b", r"\bactive tasks?\b",
    r"\bqueue\b", r"\bhealth\b", r"\buptime\b", r"\bcost so far\b",
]

_COMMAND_PATTERNS = [
    r"\bschedule\b", r"\bpublish\b", r"\bsend\b", r"\bcreate\b",
    r"\bdelete\b", r"\bcancel\b", r"\bstart\b", r"\bstop\b",
    r"\brun\b", r"\bbuild\b", r"\bdeploy\b", r"\bbackup\b",
]

_QUESTION_PATTERNS = [
    r"^(what|when|where|who|why|how|which|can|could|should|would|is|are|do|does|did)\b",
    r"\?$",
]


def _classify_intent(message: str) -> str:
    """Classify a user message into an intent category.

    Uses keyword/pattern matching — no AI call needed. Fast, free,
    deterministic. Model switch is checked first (highest priority).

    Returns one of: "model_switch", "status_query", "command", "question", "conversation"
    """
    lower = message.lower().strip()

    # Model switch — highest priority, zero cost
    if _is_model_switch(lower):
        return "model_switch"

    # Status queries
    for pattern in _STATUS_PATTERNS:
        if re.search(pattern, lower):
            return "status_query"

    # Commands
    for pattern in _COMMAND_PATTERNS:
        if re.search(pattern, lower):
            return "command"

    # Questions
    for pattern in _QUESTION_PATTERNS:
        if re.search(pattern, lower):
            return "question"

    return "conversation"


# ---------------------------------------------------------------------------
# Conversation Manager
# ---------------------------------------------------------------------------

class ConversationManager:
    """Manages George-to-Cam conversations through the dashboard.

    Routes all conversations through the model router at "boss" complexity
    (Claude Opus). Builds context from persona + live system state + recent
    chat history. Persists messages in episodic memory.

    Args:
        model_router:      ModelRouter instance for API calls
        persona:           Persona instance for Cam's identity
        episodic_memory:   EpisodicMemory for chat persistence
        short_term_memory: ShortTermMemory for session context
        context_manager:   ContextManager (optional, for status)
        orchestrator:      Orchestrator (optional, for active tasks)
        registry:          AgentRegistry (optional, for agent status)
        finance_tracker:   FinanceTracker (optional, for financials)
        task_queue:        TaskQueue (optional, for pending tasks)
        analytics:         Analytics (optional, for cost tracking)
    """

    def __init__(
        self,
        model_router,
        persona,
        episodic_memory,
        short_term_memory,
        context_manager=None,
        orchestrator=None,
        registry=None,
        finance_tracker=None,
        task_queue=None,
        analytics=None,
        reference_doc_path: str = "",
    ):
        self._router = model_router
        self._persona = persona
        self._episodic = episodic_memory
        self._stm = short_term_memory
        self._context_manager = context_manager
        self._orchestrator = orchestrator
        self._registry = registry
        self._finance_tracker = finance_tracker
        self._task_queue = task_queue
        self._analytics = analytics
        self._reference_doc_path = reference_doc_path

        # Tool-use state: pending approval futures and event callback
        self._pending_tool_approvals: dict[str, asyncio.Future] = {}
        self._on_tool_event = None  # async callback set by server.py

        logger.info("ConversationManager initialized (reference_doc=%s)",
                     reference_doc_path or "none")

    # -------------------------------------------------------------------
    # Tool approval resolution (called from dashboard WS handler)
    # -------------------------------------------------------------------

    def resolve_tool_approval(self, approval_id: str, approved: bool):
        """Resolve a pending Tier 2 tool approval from the dashboard.

        Args:
            approval_id: UUID of the pending approval.
            approved:    True to approve, False to reject.
        """
        future = self._pending_tool_approvals.get(approval_id)
        if future is not None and not future.done():
            future.set_result(approved)
            logger.info("Tool approval resolved: %s → %s", approval_id[:8],
                        "approved" if approved else "rejected")

    # -------------------------------------------------------------------
    # Main chat method
    # -------------------------------------------------------------------

    async def chat(self, message: str, participant: str = "george") -> ChatResponse:
        """Process a message from George and return Cam's response.

        Steps:
            1. Classify intent (model_switch / conversation / status_query / command / question)
            2. If model_switch, handle directly (zero API cost)
            3. Build system prompt with persona + system state context
            4. Get recent chat history from episodic memory
            5. Route through model_router with task_complexity="boss"
            6. Record both user msg + Cam response in episodic memory
            7. Return ChatResponse

        Args:
            message:     The user's message text
            participant: Who is talking (default "george")

        Returns:
            ChatResponse with Cam's response and metadata
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        intent = _classify_intent(message)

        logger.info(
            "Chat from %s (intent=%s): %.80s%s",
            participant, intent, message,
            "..." if len(message) > 80 else "",
        )

        # 1. Record user message in episodic memory
        self._episodic.record(
            participant=participant,
            content=message,
            context_tags=["chat", "george_cam"],
            metadata={"intent": intent, "source": "dashboard"},
        )

        # 2. Model switch — handle directly, no API call
        if intent == "model_switch":
            switch_result = self._handle_model_switch(message, timestamp)
            if switch_result is not None:
                # Record in episodic memory + STM
                self._episodic.record(
                    participant="cam",
                    content=switch_result.text,
                    context_tags=["chat", "george_cam", "model_switch"],
                    metadata={
                        "intent": "model_switch",
                        "model": "direct",
                        "cost": 0.0,
                        "source": "dashboard",
                    },
                )
                self._stm.add("user", message, {"source": "chat"})
                self._stm.add("assistant", switch_result.text, {
                    "source": "chat", "model": "direct",
                })
                return switch_result

        # 3. Build system prompt
        system_prompt = await self._build_system_prompt(intent)

        # 4. Build message list with recent chat history
        messages = await self._build_messages(message)

        # 5. Route to Claude Opus via "boss" complexity tier (with tools)
        response = await self._router.route(
            prompt=message,
            task_complexity="boss",
            system_prompt=system_prompt,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        # 5b. Tool-use loop — execute tools until Claude finishes or limit hit
        total_input_tokens = response.prompt_tokens
        total_output_tokens = response.response_tokens
        total_cost = response.cost_usd
        tool_rounds = 0

        while response.stop_reason == "tool_use" and tool_rounds < _MAX_TOOL_ROUNDS:
            tool_rounds += 1
            tool_results = []

            for call in (response.tool_calls or []):
                tool_name = call["name"]
                tool_input = call["input"]
                tool_id = call["id"]
                tier = classify_tool_tier(tool_name, tool_input)

                logger.info("Tool call: %s (tier=%d, id=%s)", tool_name, tier, tool_id[:8])

                # Notify dashboard of tool call
                if self._on_tool_event:
                    await self._on_tool_event({
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tier": tier,
                        "status": "running" if tier == 1 else "awaiting_approval",
                    })

                if tier == 3:
                    # Blocked — Constitutional Tier 3
                    result = {"error": f"Blocked: {tool_name} on this input is prohibited by the constitution."}
                    if self._on_tool_event:
                        await self._on_tool_event({"tool_id": tool_id, "status": "blocked"})
                    logger.warning("Tool blocked (tier 3): %s", tool_name)

                elif tier == 2:
                    # Approval required — Tier 2
                    approval_id = str(uuid.uuid4())
                    if self._on_tool_event:
                        await self._on_tool_event({
                            "tool_id": tool_id,
                            "status": "awaiting_approval",
                            "approval_id": approval_id,
                        })

                    future: asyncio.Future = asyncio.get_event_loop().create_future()
                    self._pending_tool_approvals[approval_id] = future

                    try:
                        approved = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        approved = False
                        logger.info("Tool approval timed out: %s (%s)", tool_name, approval_id[:8])
                    finally:
                        self._pending_tool_approvals.pop(approval_id, None)

                    if approved:
                        result = await execute_tool(tool_name, tool_input)
                        status = "completed"
                    else:
                        result = {"error": "George declined this action."}
                        status = "rejected"

                    if self._on_tool_event:
                        await self._on_tool_event({"tool_id": tool_id, "status": status})
                    logger.info("Tool %s: %s (%s)", status, tool_name, approval_id[:8])

                else:
                    # Tier 1 — execute autonomously
                    result = await execute_tool(tool_name, tool_input)
                    if self._on_tool_event:
                        await self._on_tool_event({"tool_id": tool_id, "status": "completed"})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                })

            # Build messages for next Claude call — assistant message with
            # tool_use blocks, then user message with tool_results
            assistant_content = []
            if response.raw_content:
                for block in response.raw_content:
                    if hasattr(block, "type"):
                        if block.type == "text" and block.text:
                            assistant_content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            assistant_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            # Call Claude again with tool results
            response = await self._router.route(
                prompt=message,
                task_complexity="boss",
                system_prompt=system_prompt,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            total_input_tokens += response.prompt_tokens
            total_output_tokens += response.response_tokens
            total_cost += response.cost_usd

        if tool_rounds > 0:
            logger.info("Tool loop completed: %d rounds", tool_rounds)

        # 6. Record Cam's response in episodic memory
        self._episodic.record(
            participant="cam",
            content=response.text,
            context_tags=["chat", "george_cam"],
            metadata={
                "intent": intent,
                "model": response.model,
                "cost": total_cost,
                "tool_rounds": tool_rounds,
                "source": "dashboard",
            },
        )

        # 7. Also add to short-term memory for session context
        self._stm.add("user", message, {"source": "chat"})
        self._stm.add("assistant", response.text, {"source": "chat", "model": response.model})

        result = ChatResponse(
            text=response.text,
            model_used=response.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost=total_cost,
            intent=intent,
            timestamp=timestamp,
            tool_calls_made=tool_rounds,
        )

        logger.info(
            "Chat response: model=%s, tokens=%d+%d, cost=$%.6f, tools=%d",
            result.model_used, result.input_tokens, result.output_tokens,
            result.cost, result.tool_calls_made,
        )

        return result

    # -------------------------------------------------------------------
    # Model switch handler
    # -------------------------------------------------------------------

    def _handle_model_switch(self, message: str, timestamp: str) -> ChatResponse | None:
        """Handle a model switch command directly — zero API cost.

        Parses the target model and who to switch (Cam or a specific agent).
        Mutates the router's state directly. Returns None if the command
        can't be parsed, allowing fallthrough to normal Claude chat.

        Args:
            message:   The user's message text.
            timestamp: ISO timestamp for the response.

        Returns:
            ChatResponse confirming the switch, or None to fall through.
        """
        lower = message.lower()
        model = _resolve_model(message)
        if model is None:
            return None  # Fall through to Claude

        # Determine target: specific agent or Cam (self)
        target_agent_id = None
        target_name = None

        # Check for agent names in the message (match against registry)
        if self._registry:
            try:
                for agent in self._registry.list_all():
                    # Match by name or ID (case-insensitive)
                    if agent.name.lower() in lower or agent.agent_id.lower() in lower:
                        target_agent_id = agent.agent_id
                        target_name = agent.name
                        break
            except Exception:
                pass

        if target_agent_id:
            # Switch a specific agent's model
            self._router.set_agent_model(target_agent_id, model)

            # Update agent_info for dashboard display
            if self._registry:
                try:
                    agent_info = self._registry.get_by_id(target_agent_id)
                    if agent_info:
                        agent_info.model_override = model
                except Exception:
                    pass

            response_text = (
                f"Done. {target_name} is now set to {model}. "
                f"This is runtime-only — it'll revert to defaults on restart."
            )
            logger.info("Model switch: agent %s → %s", target_agent_id, model)
        else:
            # Switch Cam's own model (the boss tier)
            old_model = self._router._models.get("boss", "unknown")
            self._router._models["boss"] = model
            response_text = (
                f"Switched from {old_model} to {model}. "
                f"I'll use {model} for our conversations now. "
                f"Runtime-only — restarts revert to defaults."
            )
            logger.info("Model switch: boss tier %s → %s", old_model, model)

        return ChatResponse(
            text=response_text,
            model_used="direct",
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            intent="model_switch",
            timestamp=timestamp,
        )

    # -------------------------------------------------------------------
    # System prompt builder
    # -------------------------------------------------------------------

    async def _build_system_prompt(self, intent: str) -> str:
        """Build the system prompt with persona + reference doc + live system state.

        The system prompt gives Cam context about who it is, complete
        operational knowledge from the reference doc, and the current
        state of the system, so responses are informed.

        Args:
            intent: The classified intent of the user message.

        Returns:
            Complete system prompt string.
        """
        # Start with persona base prompt
        parts = [self._persona.build_system_prompt()]

        # Add conversation-specific instructions
        parts.append(
            "You are having a direct conversation with George, your boss and "
            "the owner of Doppler Cycles. Be natural, helpful, and concise. "
            "Use your personality — you're Cam, the AI operations manager. "
            "Don't be stiff or overly formal. George is a friend and colleague."
        )

        # Inject reference doc (auto-reloads on file change)
        if self._reference_doc_path:
            from core.persona import load_reference_doc
            ref_doc = load_reference_doc(self._reference_doc_path)
            if ref_doc:
                parts.append(f"## System Reference\n{ref_doc}")

        # For status queries, inject live system state
        if intent in ("status_query", "command", "question"):
            state = await self._build_system_state()
            if state:
                parts.append(f"Current system state:\n{state}")

        return "\n\n".join(parts)

    # -------------------------------------------------------------------
    # System state builder
    # -------------------------------------------------------------------

    async def _build_system_state(self) -> str:
        """Gather live data from subsystems for context injection.

        Collects agent status, active tasks, financial summary, etc.
        Returns a formatted text block. Failures are silently skipped
        — partial state is better than no state.
        """
        lines = []

        # Agent registry status
        try:
            if self._registry:
                agents = self._registry.get_all_agents()
                online = [a for a in agents.values() if a.get("status") == "online"]
                lines.append(f"Agents: {len(online)}/{len(agents)} online")
                for aid, info in agents.items():
                    status = info.get("status", "unknown")
                    task = info.get("current_task", "idle")
                    lines.append(f"  - {aid}: {status} ({task})")
        except Exception:
            pass

        # Task queue status
        try:
            if self._task_queue:
                status = self._task_queue.get_status()
                lines.append(
                    f"Tasks: {status.get('pending', 0)} pending, "
                    f"{status.get('active', 0)} active, "
                    f"{status.get('completed', 0)} completed"
                )
        except Exception:
            pass

        # Model cost tracking and current assignments
        try:
            if self._router:
                costs = self._router.get_session_costs()
                lines.append(
                    f"Session model costs: ${costs['total_cost_usd']:.4f} "
                    f"({costs['call_count']} calls, {costs['total_tokens']} tokens)"
                )
                # Current model assignments (so Claude knows what's active)
                boss_model = self._router._models.get("boss", "unknown")
                lines.append(f"Cam's current model (boss tier): {boss_model}")
                overrides = self._router._agent_model_overrides
                if overrides:
                    for aid, m in overrides.items():
                        lines.append(f"  Agent override: {aid} → {m}")
        except Exception:
            pass

        # Finance tracker summary
        try:
            if self._finance_tracker:
                summary = self._finance_tracker.get_summary()
                if summary:
                    lines.append(f"Finances: {summary}")
        except Exception:
            pass

        return "\n".join(lines) if lines else ""

    # -------------------------------------------------------------------
    # Message list builder
    # -------------------------------------------------------------------

    async def _build_messages(self, current_message: str) -> list[dict]:
        """Build the messages list with recent chat history.

        Fetches recent chat episodes from episodic memory and formats
        them as a multi-turn conversation for the Claude API.

        Args:
            current_message: The current user message.

        Returns:
            List of message dicts with role and content keys.
        """
        messages = []

        # Get recent chat history from episodic memory
        try:
            recent = self._episodic.search(
                keyword=None,
                participant=None,
                limit=20,
            )
            # Filter to chat messages only and reverse to chronological order
            chat_episodes = [
                ep for ep in reversed(recent)
                if "chat" in (ep.context_tags or [])
            ]

            # Take the last 10 turns (20 messages) to keep context reasonable
            chat_episodes = chat_episodes[-20:]

            for ep in chat_episodes:
                if ep.participant in ("george", "user"):
                    role = "user"
                elif ep.participant in ("cam", "assistant"):
                    role = "assistant"
                else:
                    continue
                messages.append({"role": role, "content": ep.content})
        except Exception:
            logger.debug("Failed to load chat history", exc_info=True)

        # Add the current message
        messages.append({"role": "user", "content": current_message})

        # Ensure the message list starts with a user message (Claude API requirement)
        while messages and messages[0].get("role") != "user":
            messages.pop(0)

        # Ensure no consecutive same-role messages (merge them)
        cleaned: list[dict] = []
        for msg in messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += "\n" + msg["content"]
            else:
                cleaned.append(msg)

        return cleaned

    # -------------------------------------------------------------------
    # Chat history retrieval
    # -------------------------------------------------------------------

    async def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent chat messages for the dashboard.

        Fetches from episodic memory filtered to chat-tagged episodes.

        Args:
            limit: Maximum messages to return.

        Returns:
            List of message dicts with participant, content, timestamp, metadata.
        """
        try:
            recent = self._episodic.search(keyword=None, limit=limit * 2)
            chat_episodes = [
                ep for ep in recent
                if "chat" in (ep.context_tags or [])
            ][:limit]

            return [
                {
                    "participant": ep.participant,
                    "content": ep.content,
                    "timestamp": ep.timestamp,
                    "metadata": ep.metadata,
                }
                for ep in reversed(chat_episodes)  # chronological order
            ]
        except Exception:
            logger.error("Failed to get chat history", exc_info=True)
            return []
