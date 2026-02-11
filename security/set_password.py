"""
CAM Dashboard Password Setup

Run once to set (or change) the dashboard login password.
Prompts for a password, generates a bcrypt hash, and writes it
to the [auth] section of config/settings.toml.

Usage:
    cd ~/CAM
    python -m security.set_password

After setting the password, restart the dashboard server.
"""

import getpass
import re
import sys
from pathlib import Path

from security.auth import hash_password


def main():
    """Prompt for a password and write the bcrypt hash to settings.toml."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"

    print("CAM Dashboard — Set Login Password")
    print("=" * 40)
    print()

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Prompt for password (twice for confirmation)
    password = getpass.getpass("New password: ")
    if not password:
        print("Error: Password cannot be empty.")
        sys.exit(1)

    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Error: Passwords do not match.")
        sys.exit(1)

    # Generate bcrypt hash
    pw_hash = hash_password(password)
    print(f"\nGenerated hash: {pw_hash}")

    # Read current config
    content = config_path.read_text()

    # Check if [auth] section already exists
    if re.search(r'^\[auth\]', content, re.MULTILINE):
        # Update existing password_hash line
        content = re.sub(
            r'^(password_hash\s*=\s*).*$',
            f'password_hash = "{pw_hash}"',
            content,
            flags=re.MULTILINE,
        )
    else:
        # Append [auth] section
        auth_section = (
            f'\n[auth]\n'
            f'username = "george"\n'
            f'password_hash = "{pw_hash}"\n'
            f'session_timeout = 3600\n'
            f'max_login_attempts = 5\n'
            f'lockout_duration = 300\n'
        )
        content += auth_section

    config_path.write_text(content)
    print(f"\nPassword hash written to {config_path}")
    print("Restart the dashboard server to activate authentication.")


if __name__ == "__main__":
    main()
