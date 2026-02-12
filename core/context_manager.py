"""
CAM Context Window Manager

Tracks token usage across model calls, assembles optimal context from
all memory systems, and handles context rotation when approaching the
model's limit.

Per CLAUDE.md: when an agent approaches 90% context capacity, CAM should
summarize, archive, and restart with a fresh context. This module
implements that protocol plus a reusable build_context() that assembles
context from short-term, long-term, episodic, and working memory.

Usage:
    from core.context_manager import ContextManager

    cm = ContextManager(
        short_term=stm, long_term=ltm,
        episodic=episodic, working=working,
        persona=persona,
    )

    # Build context for a task
    ctx = cm.build_context(task, model="glm-4.7-flash")
    response = await router.route(prompt=ctx.prompt, system_prompt=ctx.system_prompt)
    cm.record_usage(response)

    # Check if rotation is needed
    if cm.should_rotate():
        await cm.rotate(router=router)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from core.config import get_config

logger = logging.getLogger("cam.context_manager")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContextBlock:
    """A labeled chunk of context with estimated token count and trim priority.

    Attributes:
        label:    Human-readable label (e.g. "ltm_result_1", "episodic_recent")
        content:  The text content of this block
        tokens:   Estimated token count for this block
        priority: Trim priority — 0=never trim, 1=trim under pressure, 2=trim first
    """
    label: str
    content: str
    tokens: int
    priority: int = 1


@dataclass
class BuiltContext:
    """Assembled context ready for the model router.

    Attributes:
        system_prompt:  The persona system prompt (sent separately)
        prompt:         The assembled user prompt with all context blocks
        blocks:         The individual ContextBlocks that were included
        total_tokens:   Estimated total tokens (system + prompt)
        model:          Target model name
        context_limit:  Token limit for this model
        usage_pct:      Percentage of context limit used
    """
    system_prompt: str
    prompt: str
    blocks: list[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    model: str = ""
    context_limit: int = 0
    usage_pct: float = 0.0


# ---------------------------------------------------------------------------
# Default context limits per model
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_LIMITS = {
    "glm-4.7-flash": 128_000,
    "gpt-oss:20b": 32_000,
    "phi4-mini:3.8b": 8_000,
    "claude": 200_000,
    "kimi-k2.5": 128_000,
}


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """Manages context window usage, assembly, and rotation.

    Tracks token usage across model calls, assembles optimal context
    from all memory systems for each task, and handles context rotation
    when session usage approaches the model's context limit.

    Args:
        short_term:   ShortTermMemory instance (session conversation buffer)
        long_term:    LongTermMemory instance (ChromaDB vector store)
        episodic:     EpisodicMemory instance (SQLite conversation history)
        working:      WorkingMemory instance (persistent task state)
        persona:      Persona instance (system prompt builder)
        on_rotation:  Optional callback fired after rotation — signature: (summary: str) -> None
    """

    def __init__(
        self,
        short_term,
        long_term,
        episodic,
        working,
        persona,
        on_rotation: Callable[[str], Any] | None = None,
    ):
        self._stm = short_term
        self._ltm = long_term
        self._episodic = episodic
        self._working = working
        self._persona = persona
        self._on_rotation = on_rotation

        # Load config with fallback defaults
        self._cfg = self._load_config()

        # Session token tracking
        self._session_prompt_tokens: int = 0
        self._session_response_tokens: int = 0
        self._session_total_tokens: int = 0
        self._current_model: str = ""
        self._rotation_count: int = 0
        self._last_rotation_summary: str = ""
        self._last_rotation_at: str | None = None
        self._session_start: str = datetime.now(timezone.utc).isoformat()

        logger.info(
            "ContextManager initialized (rotation_threshold=%.0f%%, "
            "ltm_top_k=%d, episodic_recent=%d)",
            self._cfg["rotation_threshold"] * 100,
            self._cfg["ltm_top_k"],
            self._cfg["episodic_recent_count"],
        )

    # -------------------------------------------------------------------
    # Config loading
    # -------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Load context config from CAMConfig, with fallback defaults."""
        defaults = {
            "rotation_threshold": 0.9,
            "ltm_top_k": 3,
            "ltm_min_score": 0.3,
            "episodic_recent_count": 5,
            "max_working_memory_tasks": 10,
            "token_estimate_divisor": 4,
            "limits": dict(_DEFAULT_CONTEXT_LIMITS),
        }
        try:
            cfg = get_config()
            ctx = cfg.context.to_dict()
            # Merge — config values override defaults
            for k, v in ctx.items():
                if k == "limits" and isinstance(v, dict):
                    defaults["limits"].update(v)
                else:
                    defaults[k] = v
        except (AttributeError, Exception):
            logger.debug("No context config found, using defaults")
        return defaults

    # -------------------------------------------------------------------
    # Token estimation
    # -------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using a character-based heuristic.

        Uses len(text) // divisor (default 4). Same heuristic as persona.py.
        Good enough for budgeting — not meant for billing accuracy.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0
        divisor = self._cfg.get("token_estimate_divisor", 4)
        return len(text) // divisor

    # -------------------------------------------------------------------
    # Context limits
    # -------------------------------------------------------------------

    def get_context_limit(self, model: str | None = None) -> int:
        """Look up the context window size for a model.

        Args:
            model: Model name (e.g. "glm-4.7-flash"). If None, uses
                   current session model or falls back to 8000.

        Returns:
            Maximum token count for this model's context window.
        """
        model = model or self._current_model
        limits = self._cfg.get("limits", _DEFAULT_CONTEXT_LIMITS)
        return limits.get(model, 8000)

    # -------------------------------------------------------------------
    # Usage tracking
    # -------------------------------------------------------------------

    def record_usage(self, response) -> None:
        """Accumulate token counts from a model response.

        Args:
            response: A ModelResponse (or any object with prompt_tokens,
                      response_tokens, total_tokens, and model attributes).
        """
        self._session_prompt_tokens += getattr(response, "prompt_tokens", 0)
        self._session_response_tokens += getattr(response, "response_tokens", 0)
        self._session_total_tokens += getattr(response, "total_tokens", 0)

        model = getattr(response, "model", "")
        if model:
            self._current_model = model

        logger.debug(
            "Usage recorded: +%d tokens (session total: %d, model: %s)",
            getattr(response, "total_tokens", 0),
            self._session_total_tokens,
            self._current_model,
        )

    # -------------------------------------------------------------------
    # Rotation check
    # -------------------------------------------------------------------

    def should_rotate(self, model: str | None = None) -> bool:
        """Check if session tokens have reached the rotation threshold.

        Returns True when session_total_tokens >= threshold % of the
        model's context limit.

        Args:
            model: Override model for limit lookup. Defaults to current.

        Returns:
            True if rotation is recommended.
        """
        limit = self.get_context_limit(model)
        threshold = self._cfg.get("rotation_threshold", 0.9)
        return self._session_total_tokens >= (limit * threshold)

    # -------------------------------------------------------------------
    # Context rotation
    # -------------------------------------------------------------------

    async def rotate(self, router=None, reason: str = "approaching context limit") -> str:
        """Perform a context rotation — archive, summarize, and reset.

        Steps:
            1. Capture current STM context
            2. Archive to episodic memory with context_rotation tags
            3. Generate summary (model-based if router available, extractive fallback)
            4. Clear STM, load summary as starting context
            5. Reset token counters, increment rotation_count
            6. Fire on_rotation callback

        Args:
            router:  ModelRouter instance for model-based summarization.
                     If None, uses extractive summary (key points only).
            reason:  Human-readable reason for the rotation.

        Returns:
            The handoff summary string.
        """
        logger.info(
            "Context rotation #%d starting (reason: %s, session_tokens: %d)",
            self._rotation_count + 1, reason, self._session_total_tokens,
        )

        # 1. Capture current STM context
        stm_context = self._stm.get_context()
        stm_text = "\n".join(
            f"[{m['role']}] {m['content']}" for m in stm_context
        )

        # 2. Archive to episodic memory
        try:
            self._episodic.record(
                "system",
                f"Context rotation #{self._rotation_count + 1}: {reason}\n\n"
                f"Session context at rotation:\n{stm_text[:2000]}",
                context_tags=["context_rotation", "session_archive"],
                metadata={
                    "rotation_number": self._rotation_count + 1,
                    "session_tokens": self._session_total_tokens,
                    "reason": reason,
                },
            )
        except Exception:
            logger.debug("Failed to archive rotation to episodic memory", exc_info=True)

        # 3. Generate summary
        summary = ""
        if router is not None and stm_text:
            try:
                summary_prompt = (
                    "Summarize the following conversation context into a concise "
                    "handoff summary. Focus on: active tasks, key decisions made, "
                    "important context that should carry forward. Be brief.\n\n"
                    f"{stm_text[:4000]}"
                )
                response = await router.route(
                    prompt=summary_prompt,
                    task_complexity="simple",
                    system_prompt="You are a context summarizer. Be concise and factual.",
                )
                summary = response.text
                logger.info("Model-based rotation summary generated (%d chars)", len(summary))
            except Exception:
                logger.debug("Model summary failed, falling back to extractive", exc_info=True)

        # Extractive fallback — take the last few messages as summary
        if not summary:
            recent = stm_context[-5:] if stm_context else []
            summary_lines = [
                f"- [{m['role']}] {m['content'][:150]}" for m in recent
            ]
            summary = (
                f"[Context rotation #{self._rotation_count + 1} — "
                f"extractive summary]\n" + "\n".join(summary_lines)
            )

        # 4. Clear STM and load summary
        self._stm.clear()
        self._stm.add("summary", f"[Previous session context]\n{summary}", {
            "rotation_number": self._rotation_count + 1,
            "type": "rotation_handoff",
        })

        # 5. Reset counters
        self._session_prompt_tokens = 0
        self._session_response_tokens = 0
        self._session_total_tokens = 0
        self._rotation_count += 1
        self._last_rotation_summary = summary
        self._last_rotation_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Context rotation #%d complete (summary: %d chars)",
            self._rotation_count, len(summary),
        )

        # 6. Fire callback
        if self._on_rotation is not None:
            try:
                self._on_rotation(summary)
            except Exception:
                logger.debug("on_rotation callback error (non-fatal)", exc_info=True)

        return summary

    # -------------------------------------------------------------------
    # Context building
    # -------------------------------------------------------------------

    def build_context(self, task, model: str | None = None) -> BuiltContext:
        """Assemble optimal context from all memory systems for a task.

        Builds a prompt with prioritized context blocks:
            Priority 0 (never trim): Task description
            Priority 1 (trim under pressure): LTM results, working memory tasks
            Priority 2 (trim first): Recent episodic history, rotation summary

        The system prompt is built separately via persona.build_system_prompt()
        and passed as a separate field (not part of the token budget).

        Budget = context_limit - system_prompt_tokens - 25% response headroom.

        Args:
            task: A Task object (must have .description) or a string description.
            model: Target model name for limit lookup. Uses current if None.

        Returns:
            A BuiltContext with assembled prompt and metadata.
        """
        # Get task description — support both Task objects and raw strings
        if isinstance(task, str):
            task_description = task
        else:
            task_description = getattr(task, "description", str(task))

        model = model or self._current_model or "glm-4.7-flash"
        context_limit = self.get_context_limit(model)

        # Build system prompt (separate from user context)
        system_prompt = self._persona.build_system_prompt()
        system_tokens = self.estimate_tokens(system_prompt)

        # Budget: total limit minus system prompt minus 25% response headroom
        response_headroom = int(context_limit * 0.25)
        budget = context_limit - system_tokens - response_headroom
        budget = max(budget, 500)  # floor — always allow at least something

        blocks: list[ContextBlock] = []

        # --- Priority 0: Task description (never trimmed) ---
        task_block = ContextBlock(
            label="task",
            content=task_description,
            tokens=self.estimate_tokens(task_description),
            priority=0,
        )
        blocks.append(task_block)

        # --- Priority 2: Rotation summary (if available, trim first) ---
        if self._last_rotation_summary:
            rotation_block = ContextBlock(
                label="rotation_summary",
                content=f"Previous session context:\n{self._last_rotation_summary}",
                tokens=self.estimate_tokens(self._last_rotation_summary),
                priority=2,
            )
            blocks.append(rotation_block)

        # --- Priority 2: Recent episodic history ---
        try:
            episodic_count = self._cfg.get("episodic_recent_count", 5)
            episodes = self._episodic.get_recent(count=episodic_count)
            if episodes:
                ep_lines = []
                for ep in reversed(episodes):  # oldest first
                    content = getattr(ep, "content", str(ep))
                    participant = getattr(ep, "participant", "?")
                    ep_lines.append(f"[{participant}] {content[:200]}")
                ep_text = "Recent conversation history:\n" + "\n".join(ep_lines)
                blocks.append(ContextBlock(
                    label="episodic_recent",
                    content=ep_text,
                    tokens=self.estimate_tokens(ep_text),
                    priority=2,
                ))
        except Exception:
            logger.debug("Episodic retrieval failed (non-fatal)", exc_info=True)

        # --- Priority 1: Long-term memory results ---
        try:
            ltm_top_k = self._cfg.get("ltm_top_k", 3)
            ltm_min_score = self._cfg.get("ltm_min_score", 0.3)
            ltm_results = self._ltm.query(task_description, top_k=ltm_top_k)
            relevant = [r for r in ltm_results if getattr(r, "score", 0) > ltm_min_score]
            if relevant:
                ltm_lines = []
                for r in relevant:
                    category = getattr(r, "category", "general")
                    content = getattr(r, "content", str(r))
                    ltm_lines.append(f"- [{category}] {content}")
                ltm_text = "Relevant knowledge from memory:\n" + "\n".join(ltm_lines)
                blocks.append(ContextBlock(
                    label="ltm_results",
                    content=ltm_text,
                    tokens=self.estimate_tokens(ltm_text),
                    priority=1,
                ))
        except Exception:
            logger.debug("LTM retrieval failed (non-fatal)", exc_info=True)

        # --- Priority 1: Working memory active tasks ---
        try:
            max_tasks = self._cfg.get("max_working_memory_tasks", 10)
            active = self._working.get_all_active()
            if active:
                wm_lines = []
                for tid, state in list(active.items())[:max_tasks]:
                    desc = state.get("description", "")[:100]
                    phase = state.get("phase", "?")
                    wm_lines.append(f"- [{phase}] {desc}")
                wm_text = "Active tasks in working memory:\n" + "\n".join(wm_lines)
                blocks.append(ContextBlock(
                    label="working_memory",
                    content=wm_text,
                    tokens=self.estimate_tokens(wm_text),
                    priority=1,
                ))
        except Exception:
            logger.debug("Working memory retrieval failed (non-fatal)", exc_info=True)

        # --- Fit blocks to budget ---
        blocks = self._fit_to_budget(blocks, budget)

        # --- Assemble final prompt ---
        prompt_parts = []
        for block in blocks:
            if block.label == "task":
                prompt_parts.insert(0, block.content)  # task always first
            else:
                prompt_parts.append(block.content)
        prompt = "\n\n".join(prompt_parts)

        total_tokens = system_tokens + self.estimate_tokens(prompt)
        usage_pct = round((total_tokens / context_limit) * 100, 1) if context_limit > 0 else 0.0

        logger.info(
            "Context built for model=%s: %d blocks, ~%d tokens (%.1f%% of %d limit)",
            model, len(blocks), total_tokens, usage_pct, context_limit,
        )

        return BuiltContext(
            system_prompt=system_prompt,
            prompt=prompt,
            blocks=blocks,
            total_tokens=total_tokens,
            model=model,
            context_limit=context_limit,
            usage_pct=usage_pct,
        )

    def _fit_to_budget(self, blocks: list[ContextBlock], budget: int) -> list[ContextBlock]:
        """Trim lowest-priority, largest blocks to fit within the token budget.

        Priority 0 blocks are never trimmed. Among equal priorities,
        the largest block gets trimmed first.

        Args:
            blocks:  List of ContextBlocks to fit.
            budget:  Maximum total tokens allowed.

        Returns:
            Filtered list of blocks that fit within budget.
        """
        total = sum(b.tokens for b in blocks)
        if total <= budget:
            return blocks

        # Sort trimmable blocks by priority (highest first = trim first),
        # then by size (largest first within same priority)
        trimmable = sorted(
            [b for b in blocks if b.priority > 0],
            key=lambda b: (-b.priority, -b.tokens),
        )
        kept = [b for b in blocks if b.priority == 0]
        kept_tokens = sum(b.tokens for b in kept)

        # Add back trimmable blocks in reverse order (lowest priority last)
        # until we'd exceed budget
        remaining = []
        for block in reversed(trimmable):
            if kept_tokens + block.tokens <= budget:
                remaining.append(block)
                kept_tokens += block.tokens

        # Preserve original order
        block_order = {id(b): i for i, b in enumerate(blocks)}
        result = kept + remaining
        result.sort(key=lambda b: block_order.get(id(b), 999))

        trimmed_count = len(blocks) - len(result)
        if trimmed_count > 0:
            logger.info(
                "Context trimmed: %d block(s) removed to fit budget (%d tokens)",
                trimmed_count, budget,
            )

        return result

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of context manager state for the dashboard.

        Returns:
            Dict with current model, limits, usage, rotation info.
        """
        limit = self.get_context_limit()
        usage_pct = (
            round((self._session_total_tokens / limit) * 100, 1)
            if limit > 0 else 0.0
        )
        threshold = self._cfg.get("rotation_threshold", 0.9)
        return {
            "current_model": self._current_model,
            "context_limit": limit,
            "session_prompt_tokens": self._session_prompt_tokens,
            "session_response_tokens": self._session_response_tokens,
            "session_total_tokens": self._session_total_tokens,
            "usage_pct": usage_pct,
            "rotation_threshold_pct": round(threshold * 100, 1),
            "rotation_count": self._rotation_count,
            "should_rotate": self.should_rotate(),
            "session_start": self._session_start,
            "last_rotation_at": self._last_rotation_at,
        }

    # -------------------------------------------------------------------
    # Session reset
    # -------------------------------------------------------------------

    def reset_session(self) -> None:
        """Zero all counters for a fresh session.

        Called manually or after a full system restart. Does NOT clear
        memory systems — only resets the context manager's own tracking.
        """
        self._session_prompt_tokens = 0
        self._session_response_tokens = 0
        self._session_total_tokens = 0
        self._current_model = ""
        self._rotation_count = 0
        self._last_rotation_summary = ""
        self._last_rotation_at = None
        self._session_start = datetime.now(timezone.utc).isoformat()
        logger.info("Context manager session reset")
