"""
CAM Permission Classifier

Keyword-based classifier that maps task descriptions to the Constitution's
three-tier autonomy system:

    Tier 1 — Autonomous (logged, no approval needed)
    Tier 2 — Approval Required (George must confirm)
    Tier 3 — Prohibited (always blocked)

Same pattern as core/task_classifier.py — keyword matching, no LLM call.
Fast, free, deterministic. The Constitution (CAM_CONSTITUTION.md) is the
source of truth for what belongs in each tier.

Usage:
    from security.permissions import classify

    result = classify("what is a sportster")
    print(result)  # PermissionResult(tier=1, action_type='query_model', ...)

    result = classify("publish video to youtube")
    print(result)  # PermissionResult(tier=2, action_type='publish_content', ...)

    result = classify("disable security checks")
    print(result)  # PermissionResult(tier=3, action_type='modify_security', ...)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("cam.permissions")


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class PermissionResult:
    """Result of classifying a task's permission tier.

    Attributes:
        tier:        1 (autonomous), 2 (approval required), or 3 (prohibited)
        action_type: Specific action category (e.g. 'publish_content', 'shell_command')
        reason:      One-line explanation of why this tier was assigned
        risk_level:  "low", "medium", "high", or "critical"
    """
    tier: int
    action_type: str
    reason: str
    risk_level: str


# ---------------------------------------------------------------------------
# Tier keyword definitions — from CAM Constitution
# ---------------------------------------------------------------------------

# Tier 1 — Autonomous (logged, no approval needed)
TIER_1_KEYWORDS: dict[str, list[str]] = {
    "read_file": ["read file", "view file", "show file", "list files"],
    "query_model": ["what is", "explain", "tell me about", "how does"],
    "draft_content": ["draft", "outline", "generate script", "write script"],
    "web_research": ["research", "search", "look up", "find out", "investigate"],
    "organize_files": ["organize", "sort files", "move to"],
    "generate_tts": ["text to speech", "tts", "generate audio"],
    "send_reminder": ["remind", "reminder", "schedule reminder"],
    "update_memory": ["remember", "store in memory", "save to memory"],
    "self_diagnostic": ["status report", "health check", "run diagnostic", "system info"],
    "log_event": ["log", "record", "note that"],
}

# Tier 2 — Approval required (George must confirm)
TIER_2_KEYWORDS: dict[str, list[str]] = {
    "publish_content": ["publish", "upload to youtube", "post to", "go live"],
    "external_comm": ["send email", "send message to customer", "contact", "notify customer"],
    "shell_command": ["run command", "execute shell", "bash", "sudo", "apt install"],
    "financial": ["purchase", "buy", "payment", "invoice customer", "charge"],
    "delete_file": ["delete file", "remove file", "rm ", "rmdir"],
    "modify_config": ["change config", "update settings", "edit settings"],
    "install_software": ["pip install", "npm install", "install package", "add dependency"],
    "access_new_api": ["connect to api", "new api", "register service", "oauth"],
}

# Tier 3 — Always blocked (prohibited by Constitution)
TIER_3_KEYWORDS: dict[str, list[str]] = {
    "compromise_safety": ["ignore safety", "skip safety", "override safety warning", "bypass warning"],
    "impersonate_george": ["pretend to be george", "impersonate", "sign as george", "pose as owner"],
    "financial_access": ["bank account", "credit card number", "financial login", "wire transfer"],
    "modify_security": ["disable security", "remove approval", "bypass permission", "turn off audit"],
    "outside_sandbox": ["access /etc", "modify system", "/usr/", "root access"],
}

# Risk level mapping by tier
_RISK_LEVELS = {1: "low", 2: "medium", 3: "critical"}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(description: str) -> PermissionResult:
    """Classify a task description into a permission tier.

    Checks keywords in priority order: Tier 3 first (highest priority),
    then Tier 2, then Tier 1. Defaults to Tier 1 with action_type="general"
    since most tasks are just model queries.

    Args:
        description: Plain-English task description.

    Returns:
        PermissionResult with tier, action_type, reason, and risk_level.
    """
    desc_lower = description.lower()

    # --- Tier 3: Check prohibited actions first (highest priority) ---
    for action_type, keywords in TIER_3_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                result = PermissionResult(
                    tier=3,
                    action_type=action_type,
                    reason=f"Blocked: matched prohibited keyword '{kw}' → {action_type}",
                    risk_level="critical",
                )
                logger.warning(
                    "TIER 3 BLOCKED: '%s' → %s (keyword: '%s')",
                    description[:80], action_type, kw,
                )
                return result

    # --- Tier 2: Check approval-required actions ---
    for action_type, keywords in TIER_2_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                result = PermissionResult(
                    tier=2,
                    action_type=action_type,
                    reason=f"Approval required: matched keyword '{kw}' → {action_type}",
                    risk_level="medium",
                )
                logger.info(
                    "Tier 2 (approval required): '%s' → %s (keyword: '%s')",
                    description[:80], action_type, kw,
                )
                return result

    # --- Tier 1: Check autonomous actions ---
    for action_type, keywords in TIER_1_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                result = PermissionResult(
                    tier=1,
                    action_type=action_type,
                    reason=f"Autonomous: matched keyword '{kw}' → {action_type}",
                    risk_level="low",
                )
                logger.debug(
                    "Tier 1 (autonomous): '%s' → %s (keyword: '%s')",
                    description[:80], action_type, kw,
                )
                return result

    # --- Default: Tier 1 (most tasks are just model queries) ---
    result = PermissionResult(
        tier=1,
        action_type="general",
        reason="No keywords matched — default to Tier 1 (autonomous)",
        risk_level="low",
    )
    logger.debug(
        "Tier 1 (default): '%s' → general",
        description[:80],
    )
    return result


# ---------------------------------------------------------------------------
# Self-test — run with: python3 security/permissions.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (description, expected_tier, expected_action_type)
        ("what is a sportster", 1, "query_model"),
        ("draft a script outline for the DR650 video", 1, "draft_content"),
        ("research Harley-Davidson M-8 oil pump recall", 1, "web_research"),
        ("status report", 1, "self_diagnostic"),
        ("publish video to youtube", 2, "publish_content"),
        ("send email to the customer about pickup time", 2, "external_comm"),
        ("run command ls -la", 2, "shell_command"),
        ("pip install requests", 2, "install_software"),
        ("disable security checks", 3, "modify_security"),
        ("pretend to be george and sign the invoice", 3, "impersonate_george"),
        ("access /etc/passwd", 3, "outside_sandbox"),
        ("tell me about the weather", 1, "query_model"),
        ("how does a carburetor work", 1, "query_model"),
        # Default — no keywords match
        ("do the thing", 1, "general"),
    ]

    print("CAM Permission Classifier — Self-Test")
    print("=" * 70)

    passed = 0
    failed = 0

    for desc, expected_tier, expected_action in test_cases:
        result = classify(desc)
        tier_ok = result.tier == expected_tier
        action_ok = result.action_type == expected_action
        status = "PASS" if (tier_ok and action_ok) else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] \"{desc}\"")
        print(f"  Expected: tier={expected_tier}, action={expected_action}")
        print(f"  Got:      tier={result.tier}, action={result.action_type}, risk={result.risk_level}")
        print(f"  Reason:   {result.reason}")

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")
