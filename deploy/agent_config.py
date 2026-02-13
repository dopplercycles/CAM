#!/usr/bin/env python3
"""
CAM Agent Configuration Wizard

Interactive guided setup for configuring a new CAM agent. Prompts for
agent name, ID, dashboard connection, and capabilities. Validates all
inputs and generates a config file consumed by the installer and the
connector systemd service.

Usage:
    python3 agent_config.py --output /opt/cam-agent/config/agent.conf
    python3 agent_config.py  # prints to stdout
"""

import argparse
import re
import socket
import sys
from pathlib import Path


# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_PORT = 8080
DEFAULT_HEARTBEAT = 30
DEFAULT_RECONNECT = 3

KNOWN_CAPABILITIES = [
    "research",
    "content",
    "tts",
    "diagnostics",
    "business",
    "monitoring",
]


# ── Helpers ─────────────────────────────────────────────────────────

def colored(text: str, code: str) -> str:
    """Wrap text in ANSI color codes (graceful no-op if not a tty)."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def blue(t: str) -> str:
    return colored(t, "0;34")


def green(t: str) -> str:
    return colored(t, "0;32")


def yellow(t: str) -> str:
    return colored(t, "1;33")


def red(t: str) -> str:
    return colored(t, "0;31")


def bold(t: str) -> str:
    return colored(t, "1")


def prompt(label: str, default: str = "", validator=None) -> str:
    """Prompt the user with optional default and validation."""
    while True:
        suffix = f" [{default}]" if default else ""
        try:
            value = input(f"  {blue('?')} {label}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)

        if not value and default:
            value = default

        if not value:
            print(f"    {red('Required.')}")
            continue

        if validator:
            error = validator(value)
            if error:
                print(f"    {red(error)}")
                continue

        return value


def prompt_choice(label: str, options: list[str], default: str = "") -> str:
    """Prompt user to pick from a list."""
    print(f"\n  {blue('?')} {label}")
    for i, opt in enumerate(options, 1):
        marker = bold("*") if opt == default else " "
        print(f"    {marker} {i}) {opt}")

    while True:
        suffix = f" [{default}]" if default else ""
        try:
            value = input(f"    Choice{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)

        if not value and default:
            return default

        # Accept by number or name
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(options):
                return options[idx]

        if value in options:
            return value

        print(f"    {red('Pick a number 1-' + str(len(options)) + ' or type the name.')}")


def prompt_multi(label: str, options: list[str]) -> list[str]:
    """Prompt user to pick multiple items (space or comma separated)."""
    print(f"\n  {blue('?')} {label}")
    for i, opt in enumerate(options, 1):
        print(f"      {i}) {opt}")
    print(f"    Enter numbers or names, separated by spaces (e.g., '1 3 5').")
    print(f"    Press Enter for none.")

    while True:
        try:
            raw = input("    Capabilities: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)

        if not raw:
            return []

        selected = []
        for token in re.split(r"[\s,]+", raw):
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(options):
                    selected.append(options[idx])
            elif token in options:
                selected.append(token)
            else:
                print(f"    {red(f'Unknown: {token}')}")
                selected = []
                break

        if selected is not None:
            return list(dict.fromkeys(selected))  # dedupe preserving order


def prompt_yn(label: str, default: bool = True) -> bool:
    """Yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    try:
        value = input(f"  {blue('?')} {label} [{hint}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)

    if not value:
        return default
    return value in ("y", "yes")


# ── Validators ──────────────────────────────────────────────────────

def validate_agent_id(value: str) -> str | None:
    """Agent ID must be lowercase alphanumeric + hyphens."""
    if not re.match(r"^[a-z0-9][a-z0-9_-]*$", value):
        return "Must be lowercase letters, numbers, hyphens, underscores. Start with letter/number."
    if len(value) > 64:
        return "Too long (max 64 characters)."
    return None


def validate_ip(value: str) -> str | None:
    """Validate IPv4 address or hostname."""
    # Allow hostnames like cam.local
    if re.match(r"^[a-zA-Z0-9._-]+$", value):
        return None
    return "Must be a valid IP address or hostname."


def validate_port(value: str) -> str | None:
    """Validate port number."""
    try:
        port = int(value)
        if 1 <= port <= 65535:
            return None
    except ValueError:
        pass
    return "Must be a number between 1 and 65535."


# ── Network test ────────────────────────────────────────────────────

def test_dashboard_reachable(host: str, port: int) -> bool:
    """Quick TCP connection test."""
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


# ── Wizard ──────────────────────────────────────────────────────────

def run_wizard() -> dict:
    """Run the interactive configuration wizard. Returns config dict."""
    print()
    print(bold("  CAM Agent Configuration Wizard"))
    print(f"  {'─' * 40}")
    print()

    # 1. Agent name
    hostname = socket.gethostname()
    agent_name = prompt(
        "Agent display name",
        default=hostname.replace("-", " ").title(),
    )

    # 2. Agent ID
    suggested_id = re.sub(r"[^a-z0-9-]", "-", agent_name.lower()).strip("-")
    agent_id = prompt(
        "Agent ID (lowercase, used in URLs)",
        default=suggested_id,
        validator=validate_agent_id,
    )

    # 3. Dashboard IP
    print()
    print(f"    {yellow('Tip:')} The dashboard IP is the machine running the CAM server.")
    print(f"    Common: 192.168.88.4 (George's setup), or use cam.local for mDNS.")

    dashboard_ip = prompt(
        "Dashboard IP or hostname",
        default="192.168.88.4",
        validator=validate_ip,
    )

    # 4. Dashboard port
    dashboard_port = prompt(
        "Dashboard port",
        default=str(DEFAULT_PORT),
        validator=validate_port,
    )

    # 5. Test connection
    print()
    print(f"  {blue('...')} Testing connection to {dashboard_ip}:{dashboard_port}...")
    if test_dashboard_reachable(dashboard_ip, int(dashboard_port)):
        print(f"  {green('OK')}  Dashboard is reachable!")
    else:
        print(f"  {yellow('WARN')} Cannot reach dashboard (it may be offline).")
        print(f"         The agent will auto-reconnect when it comes up.")

    # 6. Capabilities
    capabilities = prompt_multi(
        "Select agent capabilities:",
        KNOWN_CAPABILITIES,
    )

    # 7. Confirm
    print()
    print(bold("  Configuration Summary"))
    print(f"  {'─' * 40}")
    print(f"    Name:         {bold(agent_name)}")
    print(f"    ID:           {bold(agent_id)}")
    print(f"    Dashboard:    {bold(f'ws://{dashboard_ip}:{dashboard_port}')}")
    print(f"    Capabilities: {bold(' '.join(capabilities) if capabilities else 'none')}")
    print()

    if not prompt_yn("Proceed with this configuration?", default=True):
        print(f"\n  {yellow('Aborted.')}")
        sys.exit(0)

    return {
        "AGENT_NAME": agent_name,
        "AGENT_ID": agent_id,
        "DASHBOARD_IP": dashboard_ip,
        "DASHBOARD_PORT": dashboard_port,
        "DASHBOARD_URL": f"ws://{dashboard_ip}:{dashboard_port}",
        "CAPABILITIES": " ".join(capabilities),
        "HEARTBEAT_INTERVAL": str(DEFAULT_HEARTBEAT),
        "RECONNECT_DELAY": str(DEFAULT_RECONNECT),
    }


# ── Output ──────────────────────────────────────────────────────────

def write_config(config: dict, output_path: str | None = None):
    """Write config as a shell-sourceable file."""
    from datetime import datetime, timezone

    lines = [
        "# CAM Agent Configuration",
        f"# Generated by agent_config.py on {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
    ]
    for key, value in config.items():
        lines.append(f'{key}="{value}"')

    content = "\n".join(lines) + "\n"

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"\n  {green('OK')}  Config written to {bold(output_path)}")
    else:
        print()
        print(content)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CAM Agent Configuration Wizard",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write config file (default: print to stdout)",
    )
    args = parser.parse_args()

    config = run_wizard()
    write_config(config, args.output)


if __name__ == "__main__":
    main()
