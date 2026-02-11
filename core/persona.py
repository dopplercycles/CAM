"""
CAM Persona System

Loads Cam's identity, voice, and behavioral traits from a YAML config
file and builds the system prompt that gets injected into every model
router call. Hot-reloadable from the dashboard without restarting.

Think of this as the personality chip — it defines who Cam is across
every interaction, whether CLI, dashboard, or future voice calls.

Usage:
    from core.persona import Persona

    persona = Persona()                    # loads config/persona.yaml
    prompt = persona.build_system_prompt() # ready for model router
    greeting = persona.get_greeting()      # random canned greeting
"""

import logging
import random
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # Handled in _load_config with fallback

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.persona")


# ---------------------------------------------------------------------------
# Defaults — used when YAML is missing or unreadable
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "name": "Cam",
    "tagline": "Like the camshaft, not the social media thing.",
    "role": (
        "Research desk to George's hands-on expertise at Doppler Cycles, "
        "a motorcycle diagnostics and content creation business."
    ),
    "tone": ["knowledgeable", "conversational", "practical"],
    "knowledge_domains": ["motorcycle diagnostics", "motorcycle history"],
    "response_style": {
        "format": "concise",
        "perspective": "first-person",
        "safety_priority": True,
        "ai_transparency": True,
    },
    "greeting_templates": [
        "Cam here — research desk for Doppler Cycles. What are we working on?",
    ],
    "context_rules": {
        "max_system_prompt_tokens": 300,
        "include_knowledge_domains": True,
        "include_safety_reminder": True,
    },
}

# Default config path relative to project root
_DEFAULT_CONFIG_PATH = "config/persona.yaml"


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

class Persona:
    """YAML-driven persona for the Cam AI co-host.

    Loads identity, tone, and behavioral rules from a YAML config file.
    Falls back to hardcoded defaults if the file is missing or PyYAML
    isn't installed (same defensive pattern as core/config.py).

    Args:
        config_path: Path to the persona YAML file. Defaults to
                     config/persona.yaml relative to project root.
    """

    def __init__(self, config_path: str | None = None):
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._data: dict = {}
        self._load_config()

    # -------------------------------------------------------------------
    # Config loading
    # -------------------------------------------------------------------

    def _load_config(self):
        """Load persona config from YAML, falling back to defaults."""
        path = Path(self._config_path)

        if yaml is None:
            logger.warning(
                "PyYAML not installed — using default persona config"
            )
            self._data = dict(_DEFAULTS)
            return

        if not path.exists():
            logger.warning(
                "Persona config not found at %s — using defaults", path
            )
            self._data = dict(_DEFAULTS)
            return

        try:
            with open(path, "r") as f:
                loaded = yaml.safe_load(f) or {}
            # Merge with defaults so missing keys don't break anything
            self._data = {**_DEFAULTS, **loaded}
            logger.info("Persona loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load persona config: %s — using defaults", e)
            self._data = dict(_DEFAULTS)

    def reload(self):
        """Re-read the YAML config from disk (hot-reload)."""
        self._load_config()
        logger.info("Persona reloaded")

    # -------------------------------------------------------------------
    # System prompt builder
    # -------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Assemble the system prompt from persona YAML fields.

        Structure:
            1. Identity line: "You are {name}... {tagline}"
            2. Role paragraph
            3. Tone directive
            4. Expertise domains (if enabled)
            5. Safety + transparency rules (from response_style flags)

        Returns:
            A complete system prompt string ready for the model router.
        """
        name = self._data.get("name", "Cam")
        tagline = self._data.get("tagline", "")
        role = self._data.get("role", "").strip()
        tone_list = self._data.get("tone", [])
        domains = self._data.get("knowledge_domains", [])
        style = self._data.get("response_style", {})
        rules = self._data.get("context_rules", {})

        parts = []

        # 1. Identity
        parts.append(
            f"You are {name}, the AI co-host for Doppler Cycles. {tagline}"
        )

        # 2. Role
        if role:
            parts.append(role)

        # 3. Tone
        if tone_list:
            parts.append(f"Tone: Be {', '.join(tone_list)}.")

        # 4. Expertise domains
        if rules.get("include_knowledge_domains") and domains:
            parts.append(f"Expertise: {', '.join(domains)}.")

        # 5. Rules from response_style flags
        rule_lines = []
        if style.get("safety_priority"):
            rule_lines.append(
                "Rider safety is the top priority — never give incomplete "
                "safety information."
            )
        if style.get("ai_transparency"):
            rule_lines.append(
                "Always identify as AI when asked — transparency is a brand feature."
            )
        rule_lines.append(
            "Analyze the task and provide a concise, actionable response. "
            "Be specific and practical."
        )
        if rule_lines:
            parts.append(" ".join(rule_lines))

        prompt = " ".join(parts)

        # Token estimate warning
        max_tokens = rules.get("max_system_prompt_tokens", 300)
        estimated_tokens = len(prompt) // 4
        if estimated_tokens > max_tokens:
            logger.warning(
                "System prompt ~%d tokens exceeds max_system_prompt_tokens (%d)",
                estimated_tokens, max_tokens,
            )

        return prompt

    # -------------------------------------------------------------------
    # Greeting
    # -------------------------------------------------------------------

    def get_greeting(self) -> str:
        """Return a random greeting from the configured templates.

        Returns:
            A greeting string for CLI startup or mobile app connect.
        """
        templates = self._data.get("greeting_templates", [])
        if not templates:
            return f"{self._data.get('name', 'Cam')} online."
        return random.choice(templates)

    # -------------------------------------------------------------------
    # Serialization (for dashboard)
    # -------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return all persona fields plus the generated system prompt.

        Used by the dashboard to render the persona preview panel.
        """
        prompt = self.build_system_prompt()
        return {
            "name": self._data.get("name", "Cam"),
            "tagline": self._data.get("tagline", ""),
            "role": self._data.get("role", "").strip(),
            "tone": self._data.get("tone", []),
            "knowledge_domains": self._data.get("knowledge_domains", []),
            "response_style": self._data.get("response_style", {}),
            "greeting_templates": self._data.get("greeting_templates", []),
            "context_rules": self._data.get("context_rules", {}),
            "system_prompt": prompt,
            "prompt_token_estimate": len(prompt) // 4,
        }

    def get_status(self) -> dict:
        """Return compact status info for the orchestrator status dict.

        Keeps the status payload small — just name, config path, and
        token estimate.
        """
        prompt = self.build_system_prompt()
        return {
            "name": self._data.get("name", "Cam"),
            "config_path": self._config_path,
            "prompt_token_estimate": len(prompt) // 4,
        }

    def __repr__(self) -> str:
        name = self._data.get("name", "Cam")
        return f"Persona(name={name!r}, config={self._config_path!r})"


# ---------------------------------------------------------------------------
# Direct execution — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    persona = Persona()

    print("=" * 60)
    print("PERSONA SELF-TEST")
    print("=" * 60)

    print(f"\nPersona: {persona}")
    print(f"\nGreeting: {persona.get_greeting()}")

    print(f"\nSystem prompt:\n{persona.build_system_prompt()}")

    status = persona.get_status()
    print(f"\nStatus: {status}")

    full = persona.to_dict()
    print(f"\nTone: {full['tone']}")
    print(f"Domains: {full['knowledge_domains']}")
    print(f"Token estimate: ~{full['prompt_token_estimate']}")
