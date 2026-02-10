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
import time
import urllib.request
import urllib.error
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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

# Default models for each complexity tier (matches CLAUDE.md routing table)
DEFAULT_MODELS = {
    "simple": "glm-4.7-flash",     # Fast, free, handles routine queries
    "routine": "gpt-oss:20b",      # Good enough for drafts and summaries
    "agentic": "kimi-k2.5",        # Best cost/performance for agent tasks
    "complex": "claude",            # Multi-step reasoning, quality matters
    "nuanced": "claude",            # Nuanced content, debugging, planning
}

# Approximate cost per million tokens for API models
# Local models are free. Updated as pricing changes.
API_COSTS = {
    "claude": {"input": 3.00, "output": 15.00},       # Claude Sonnet-class pricing
    "kimi-k2.5": {"input": 0.50, "output": 1.50},     # Moonshot pricing estimate
}

# Ollama API base URL
OLLAMA_URL = "http://localhost:11434"


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

    def __init__(self, ollama_url: str = OLLAMA_URL):
        self._ollama_url = ollama_url

        # Running cost totals for the session
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._call_count: int = 0

        # Full call log — every call recorded for audit
        self._call_log: list[ModelResponse] = []

        logger.info("ModelRouter initialized (Ollama at %s)", ollama_url)

    # -------------------------------------------------------------------
    # Main routing method
    # -------------------------------------------------------------------

    async def route(
        self,
        prompt: str,
        task_complexity: str = "simple",
        model_override: str | None = None,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Route a prompt to the appropriate model.

        This is the main entry point. Call this instead of hitting
        model APIs directly — it handles selection, cost tracking,
        and logging automatically.

        Args:
            prompt:          The user/task prompt to send.
            task_complexity: One of "simple", "routine", "agentic",
                             "complex", "nuanced". Determines which
                             model handles the request.
            model_override:  Force a specific model (bypasses routing).
            system_prompt:   Optional system prompt for the model.

        Returns:
            A ModelResponse with the text and metadata.
        """
        # Determine which model to use
        model = model_override or DEFAULT_MODELS.get(task_complexity, "glm-4.7-flash")

        logger.info(
            "Routing prompt (complexity=%s, model=%s): %.80s%s",
            task_complexity, model,
            prompt, "..." if len(prompt) > 80 else "",
        )

        # Route to the right backend
        if model in ("claude",):
            response = await self._call_claude(prompt, model, system_prompt)
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
    ) -> ModelResponse:
        """Call the Claude API for complex/nuanced tasks.

        PLACEHOLDER — not yet implemented. Will use the Anthropic
        Python SDK when we add it as a dependency. For now, returns
        an error message so the rest of the system can handle it.

        Future implementation:
            from anthropic import Anthropic
            client = Anthropic(api_key=config.claude_api_key)
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
        """
        logger.warning("Claude API not yet configured — returning placeholder")

        # Estimate what it would cost so cost tracking stays honest
        # Rough estimate: ~4 chars per token
        est_prompt_tokens = len(prompt) // 4
        costs = API_COSTS.get("claude", {"input": 0, "output": 0})
        est_cost = (est_prompt_tokens / 1_000_000) * costs["input"]

        return ModelResponse(
            text="[placeholder] Claude API not yet configured. "
                 "Install anthropic SDK and add API key to config.",
            model="claude",
            backend="claude",
            prompt_tokens=est_prompt_tokens,
            response_tokens=0,
            total_tokens=est_prompt_tokens,
            cost_usd=est_cost,
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

        PLACEHOLDER — not yet implemented. Moonshot uses an
        OpenAI-compatible API format.

        Future implementation:
            import openai
            client = openai.OpenAI(
                api_key=config.moonshot_api_key,
                base_url="https://api.moonshot.cn/v1",
            )
            response = client.chat.completions.create(
                model="kimi-k2.5",
                messages=[{"role": "user", "content": prompt}],
            )
        """
        logger.warning("Moonshot API not yet configured — returning placeholder")

        est_prompt_tokens = len(prompt) // 4
        costs = API_COSTS.get("kimi-k2.5", {"input": 0, "output": 0})
        est_cost = (est_prompt_tokens / 1_000_000) * costs["input"]

        return ModelResponse(
            text="[placeholder] Moonshot API not yet configured. "
                 "Add API key to config.",
            model="kimi-k2.5",
            backend="moonshot",
            prompt_tokens=est_prompt_tokens,
            response_tokens=0,
            total_tokens=est_prompt_tokens,
            cost_usd=est_cost,
        )

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
