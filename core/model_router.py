"""
CAM Model Router

Routes prompts to the right model based on task complexity.
Model-agnostic design — easy to swap models as the landscape evolves.

Routing table (from CLAUDE.md):

    | Complexity    | Model                   | Reason                          |
    |---------------|-------------------------|---------------------------------|
    | simple        | Local Ollama (glm-4.7-flash) | Fast, free                 |
    | routine       | Local Ollama (gpt-oss:20b)   | Good enough for drafts     |
    | agentic       | Kimi K2.5 (Moonshot API)     | Best cost/perf for agents  |
    | complex       | Claude API                   | Multi-step reasoning       |
    | nuanced       | Claude API                   | Quality matters            |

Cost tracking is built in — every call logged with model, token count,
estimated cost, and latency.

Usage:
    from core.model_router import ModelRouter

    router = ModelRouter()
    response = await router.route("What year was the Honda CB750 introduced?")
    print(response.text)
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Anthropic SDK — optional dependency, graceful skip if not installed
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore
    _HAS_ANTHROPIC = False

# OpenAI SDK — used for Moonshot/Kimi K2.5 (OpenAI-compatible API)
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore
    _HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.model_router")


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    """Standardized response from any model backend.

    Every model call — local or API — returns one of these so the
    rest of CAM doesn't need to know which model was used.

    Attributes:
        text:            The model's response text
        model:           Which model generated this (e.g., "glm-4.7-flash")
        backend:         Where it ran ("ollama", "claude", "moonshot")
        prompt_tokens:   Number of tokens in the prompt
        response_tokens: Number of tokens in the response
        total_tokens:    prompt_tokens + response_tokens
        latency_ms:      How long the call took in milliseconds
        cost_usd:        Estimated cost in USD (0.0 for local models)
        timestamp:       When the call was made
    """
    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

# Module-level fallbacks — used if config is unavailable
_DEFAULT_MODELS = {
    "simple": "glm-4.7-flash",
    "routine": "gpt-oss:20b",
    "agentic": "kimi-k2.5",
    "complex": "claude",
    "nuanced": "claude",
    # George talking to Cam — ALWAYS uses the best model, no cost optimization
    "boss": "claude-opus-4-6",
    # Tier-based routing (from task classifier)
    "tier1": "glm-4.7-flash",    # Small/fast — placeholder until tiny models installed
    "tier2": "gpt-oss:20b",      # Medium — language understanding, multi-sentence
    "tier3": "glm-4.7-flash",    # Large — reasoning, judgment, creativity
}

_API_COSTS = {
    "claude": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "kimi-k2.5": {"input": 0.50, "output": 1.50},
}

_OLLAMA_URL = "http://localhost:11434"


class ModelRouter:
    """Routes prompts to the appropriate model based on task complexity.

    The router is the single point of contact for all model calls in CAM.
    It handles:
    - Model selection based on complexity tier
    - Calling the right backend (Ollama local, Claude API, Moonshot API)
    - Token counting and cost tracking
    - Logging every call for the audit trail

    All calls are async-safe. Local Ollama calls use asyncio.to_thread()
    to avoid blocking the event loop (stdlib urllib is synchronous).
    """

    def __init__(self, ollama_url: str | None = None):
        # Read from config, fall back to module-level defaults
        try:
            from core.config import get_config
            cfg = get_config()
            self._ollama_url = ollama_url or cfg.models.ollama_url
            self._models = cfg.models.routing.to_dict()
            self._api_costs = cfg.models.costs.to_dict()
        except Exception:
            self._ollama_url = ollama_url or _OLLAMA_URL
            self._models = dict(_DEFAULT_MODELS)
            self._api_costs = {k: dict(v) for k, v in _API_COSTS.items()}

        # Per-agent model overrides — runtime only, not persisted
        # Maps agent_id → model name (e.g., "firehorseclawd" → "kimi-k2.5")
        self._agent_model_overrides: dict[str, str] = {}

        # Running cost totals for the session
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._call_count: int = 0

        # Full call log — every call recorded for audit
        self._call_log: list[ModelResponse] = []

        logger.info("ModelRouter initialized (Ollama at %s)", self._ollama_url)

    # -------------------------------------------------------------------
    # Main routing method
    # -------------------------------------------------------------------

    async def route(
        self,
        prompt: str,
        task_complexity: str = "simple",
        model_override: str | None = None,
        system_prompt: str | None = None,
        messages: list[dict] | None = None,
    ) -> ModelResponse:
        """Route a prompt to the appropriate model.

        This is the main entry point. Call this instead of hitting
        model APIs directly — it handles selection, cost tracking,
        and logging automatically.

        Args:
            prompt:          The user/task prompt to send.
            task_complexity: One of "simple", "routine", "agentic",
                             "complex", "nuanced", "boss". Determines
                             which model handles the request.
                             "boss" = George talking to Cam, always
                             uses the best model (Claude Opus).
            model_override:  Force a specific model (bypasses routing).
            system_prompt:   Optional system prompt for the model.
            messages:        Optional pre-built message list for
                             multi-turn conversations. If provided,
                             used instead of wrapping prompt in a
                             single user message.

        Returns:
            A ModelResponse with the text and metadata.
        """
        # Determine which model to use
        model = model_override or self._models.get(task_complexity, "glm-4.7-flash")

        logger.info(
            "Routing prompt (complexity=%s, model=%s): %.80s%s",
            task_complexity, model,
            prompt, "..." if len(prompt) > 80 else "",
        )

        # Route to the right backend — Claude API models go to _call_claude
        if model.startswith("claude"):
            response = await self._call_claude(prompt, model, system_prompt, messages)
        elif model in ("kimi-k2.5",):
            response = await self._call_moonshot(prompt, model, system_prompt)
        else:
            # Everything else goes to Ollama (local models)
            response = await self._call_ollama(prompt, model, system_prompt)

        # Track costs
        self._total_cost += response.cost_usd
        self._total_tokens += response.total_tokens
        self._call_count += 1
        self._call_log.append(response)

        logger.info(
            "Response from %s (%s): %d tokens, %.1fms, $%.6f — %.80s%s",
            response.model, response.backend,
            response.total_tokens, response.latency_ms, response.cost_usd,
            response.text, "..." if len(response.text) > 80 else "",
        )

        # --- Override Rule 3: Auto-retry on Tier 1 error ---
        # If a tier1 call fails (response contains [error]), retry at tier2.
        if task_complexity == "tier1" and "[error]" in response.text:
            logger.warning(
                "Tier 1 model returned error — retrying at tier2 (Rule 3: auto-retry)"
            )
            return await self.route(
                prompt=prompt,
                task_complexity="tier2",
                system_prompt=system_prompt,
            )

        return response

    # -------------------------------------------------------------------
    # Ollama (local models)
    # -------------------------------------------------------------------

    async def _call_ollama(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Call a local Ollama model.

        Uses the /api/generate endpoint. Runs in a thread to avoid
        blocking the async event loop (urllib is synchronous).

        Args:
            prompt:        The prompt text.
            model:         Ollama model name (e.g., "glm-4.7-flash").
            system_prompt: Optional system message.

        Returns:
            ModelResponse with the result and token counts.
        """
        # Build the request — Ollama expects JSON POST to /api/generate
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Get the full response at once
        }
        if system_prompt:
            payload["system"] = system_prompt

        # Run the blocking HTTP call in a thread
        response = await asyncio.to_thread(
            self._ollama_request, payload
        )

        return response

    def _ollama_request(self, payload: dict) -> ModelResponse:
        """Synchronous Ollama HTTP request (runs in thread).

        Separated out so asyncio.to_thread() can call it cleanly.
        """
        url = f"{self._ollama_url}/api/generate"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.error("Ollama request failed: %s", e)
            return ModelResponse(
                text=f"[error] Ollama unavailable: {e}",
                model=payload.get("model", "unknown"),
                backend="ollama",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract token counts from Ollama's response
        # Ollama returns these in the response JSON when stream=False
        prompt_tokens = body.get("prompt_eval_count", 0) or 0
        response_tokens = body.get("eval_count", 0) or 0

        return ModelResponse(
            text=body.get("response", "").strip(),
            model=payload.get("model", "unknown"),
            backend="ollama",
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=prompt_tokens + response_tokens,
            latency_ms=round(elapsed_ms, 1),
            cost_usd=0.0,  # Local models are free
        )

    # -------------------------------------------------------------------
    # Claude API (placeholder)
    # -------------------------------------------------------------------

    async def _call_claude(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        messages: list[dict] | None = None,
    ) -> ModelResponse:
        """Call the Claude API for complex/nuanced/boss-tier tasks.

        Uses the Anthropic Python SDK (async client). Requires the
        ANTHROPIC_API_KEY environment variable to be set. Falls back
        to a helpful error message if the SDK isn't installed or the
        key is missing.

        Args:
            prompt:        The prompt text (used if messages is None).
            model:         Claude model name (e.g. "claude-opus-4-6").
            system_prompt: Optional system message.
            messages:      Optional pre-built message list for multi-turn.

        Returns:
            ModelResponse with the result and token/cost tracking.
        """
        # Check SDK availability
        if not _HAS_ANTHROPIC:
            logger.warning("anthropic SDK not installed — returning error")
            return ModelResponse(
                text="[error] The anthropic Python package is not installed. "
                     "Run: .venv/bin/pip install anthropic",
                model=model,
                backend="claude",
            )

        # Check API key
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — returning error")
            return ModelResponse(
                text="[error] ANTHROPIC_API_KEY environment variable is not set. "
                     "Set it before starting the server to enable Claude API calls.",
                model=model,
                backend="claude",
            )

        # Build messages list
        if messages:
            msgs = messages
        else:
            msgs = [{"role": "user", "content": prompt}]

        start = time.perf_counter()
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 4096,
                "messages": msgs,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            logger.error("Claude API auth error: %s", e)
            return ModelResponse(
                text=f"[error] Claude API authentication failed. Check your ANTHROPIC_API_KEY.",
                model=model,
                backend="claude",
            )
        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return ModelResponse(
                text=f"[error] Claude API call failed: {e}",
                model=model,
                backend="claude",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract text from response content blocks
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Token counts from the API response
        prompt_tokens = response.usage.input_tokens
        response_tokens = response.usage.output_tokens
        total_tokens = prompt_tokens + response_tokens

        # Calculate cost using per-model pricing
        costs = self._api_costs.get(model, self._api_costs.get("claude", {"input": 3.0, "output": 15.0}))
        cost_usd = (
            (prompt_tokens / 1_000_000) * costs["input"]
            + (response_tokens / 1_000_000) * costs["output"]
        )

        return ModelResponse(
            text=text.strip(),
            model=model,
            backend="claude",
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            latency_ms=round(elapsed_ms, 1),
            cost_usd=round(cost_usd, 6),
        )

    # -------------------------------------------------------------------
    # Moonshot / Kimi K2.5 API (placeholder)
    # -------------------------------------------------------------------

    async def _call_moonshot(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Call the Moonshot API for agentic workflow tasks.

        Uses the OpenAI-compatible API at api.moonshot.cn. Requires
        the MOONSHOT_API_KEY environment variable to be set.

        Args:
            prompt:        The prompt text.
            model:         Model name (e.g. "kimi-k2.5").
            system_prompt: Optional system message.

        Returns:
            ModelResponse with the result and token/cost tracking.
        """
        if not _HAS_OPENAI:
            logger.warning("openai SDK not installed — returning error")
            return ModelResponse(
                text="[error] The openai Python package is not installed. "
                     "Run: .venv/bin/pip install openai",
                model=model,
                backend="moonshot",
            )

        api_key = os.environ.get("MOONSHOT_API_KEY", "")
        if not api_key:
            logger.warning("MOONSHOT_API_KEY not set — returning error")
            return ModelResponse(
                text="[error] MOONSHOT_API_KEY environment variable is not set.",
                model=model,
                backend="moonshot",
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        try:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1",
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error("Moonshot API call failed: %s", e)
            return ModelResponse(
                text=f"[error] Moonshot API call failed: {e}",
                model=model,
                backend="moonshot",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content or ""
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        response_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        total_tokens = prompt_tokens + response_tokens

        costs = self._api_costs.get(model, self._api_costs.get("kimi-k2.5", {"input": 0.5, "output": 1.5}))
        cost_usd = (
            (prompt_tokens / 1_000_000) * costs["input"]
            + (response_tokens / 1_000_000) * costs["output"]
        )

        return ModelResponse(
            text=text.strip(),
            model=model,
            backend="moonshot",
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            latency_ms=round(elapsed_ms, 1),
            cost_usd=round(cost_usd, 6),
        )

    # -------------------------------------------------------------------
    # Per-agent model overrides (runtime switching)
    # -------------------------------------------------------------------

    def set_agent_model(self, agent_id: str, model: str):
        """Set a runtime model override for a specific agent.

        This override is runtime-only — it won't survive a server restart.
        The agent will use this model instead of the default routing.

        Args:
            agent_id: The agent to override (e.g., "firehorseclawd").
            model:    Full model name (e.g., "kimi-k2.5", "glm-4.7-flash").
        """
        self._agent_model_overrides[agent_id] = model
        logger.info("Model override set: %s → %s", agent_id, model)

    def get_agent_model(self, agent_id: str) -> str | None:
        """Get the current model override for an agent, if any."""
        return self._agent_model_overrides.get(agent_id)

    def clear_agent_model(self, agent_id: str):
        """Remove an agent's model override, reverting to default routing."""
        removed = self._agent_model_overrides.pop(agent_id, None)
        if removed:
            logger.info("Model override cleared: %s (was %s)", agent_id, removed)

    def get_model_assignments(self) -> dict:
        """Return current model routing table and any agent overrides.

        Used by the dashboard to display which model each component is using.
        """
        return {
            "routing": dict(self._models),
            "agent_overrides": dict(self._agent_model_overrides),
        }

    # -------------------------------------------------------------------
    # Cost tracking
    # -------------------------------------------------------------------

    def get_session_costs(self) -> dict:
        """Return cost summary for the current session.

        Useful for the dashboard cost tracker panel.
        """
        return {
            "total_cost_usd": round(self._total_cost, 6),
            "total_tokens": self._total_tokens,
            "call_count": self._call_count,
            "by_backend": self._costs_by_backend(),
        }

    def _costs_by_backend(self) -> dict[str, dict]:
        """Break down costs by backend (ollama, claude, moonshot)."""
        breakdown: dict[str, dict[str, Any]] = {}
        for resp in self._call_log:
            if resp.backend not in breakdown:
                breakdown[resp.backend] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                }
            entry = breakdown[resp.backend]
            entry["calls"] += 1
            entry["tokens"] += resp.total_tokens
            entry["cost_usd"] = round(entry["cost_usd"] + resp.cost_usd, 6)
        return breakdown

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    async def list_local_models(self) -> list[str]:
        """Fetch the list of models available in Ollama.

        Useful for health checks and the dashboard model selector.
        """
        try:
            result = await asyncio.to_thread(self._ollama_list_models)
            return result
        except Exception as e:
            logger.error("Failed to list Ollama models: %s", e)
            return []

    def _ollama_list_models(self) -> list[str]:
        """Synchronous Ollama model list (runs in thread)."""
        url = f"{self._ollama_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in body.get("models", [])]


# ---------------------------------------------------------------------------
# Direct execution — test with a real Ollama query
# ---------------------------------------------------------------------------

async def _test():
    """Test the model router with a real local Ollama call."""
    router = ModelRouter()

    # Show available models
    models = await router.list_local_models()
    print(f"Available Ollama models: {len(models)}")
    for m in models:
        print(f"  - {m}")

    # Test a simple query (routes to glm-4.7-flash)
    print("\n--- Simple query (glm-4.7-flash) ---")
    resp = await router.route(
        "What year was the Honda CB750 first introduced? Reply in one sentence.",
        task_complexity="simple",
    )
    print(f"Model:    {resp.model}")
    print(f"Response: {resp.text}")
    print(f"Tokens:   {resp.total_tokens} (prompt={resp.prompt_tokens}, response={resp.response_tokens})")
    print(f"Latency:  {resp.latency_ms}ms")
    print(f"Cost:     ${resp.cost_usd}")

    # Test Claude placeholder (should return placeholder message)
    print("\n--- Complex query (claude placeholder) ---")
    resp2 = await router.route(
        "Analyze the market trends for vintage Japanese motorcycles.",
        task_complexity="complex",
    )
    print(f"Model:    {resp2.model}")
    print(f"Response: {resp2.text}")

    # Session cost summary
    print(f"\n--- Session costs ---")
    costs = router.get_session_costs()
    print(f"Total calls:  {costs['call_count']}")
    print(f"Total tokens: {costs['total_tokens']}")
    print(f"Total cost:   ${costs['total_cost_usd']}")
    for backend, data in costs["by_backend"].items():
        print(f"  {backend}: {data['calls']} calls, {data['tokens']} tokens, ${data['cost_usd']}")


if __name__ == "__main__":
    asyncio.run(_test())
