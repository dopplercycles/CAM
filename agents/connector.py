#!/usr/bin/env python3
"""
CAM Agent Connector

Lightweight script that runs on a remote machine (e.g., Raspberry Pi)
and connects to the CAM dashboard via WebSocket, sending periodic
heartbeats so the dashboard knows this agent is alive.

Auto-detects the machine's hostname and local IP address.

Usage:
    python3 connector.py --dashboard ws://192.168.1.100:8080

    # Override the auto-detected name or ID:
    python3 connector.py --dashboard ws://192.168.1.100:8080 --name "FireHorseClawd" --id firehorseclawd

Dependencies:
    pip install websockets

Deployment to Raspberry Pi:
    See the deploy instructions at the bottom of this file, or run:
    python3 connector.py --help
"""

import argparse
import asyncio
import json
import logging
import socket
import sys

try:
    import websockets
except ImportError:
    print("Missing dependency: websockets")
    print("Install with: pip install websockets")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cam.connector")


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def get_hostname() -> str:
    """Return this machine's hostname."""
    return socket.gethostname()


def get_local_ip() -> str:
    """Return this machine's local IP address.

    Opens a UDP socket to a public DNS address (doesn't actually send
    anything) to determine which local interface would be used for
    outbound traffic. This works even without internet access as long
    as a default route exists.
    """
    try:
        # Connect to a known external address to find our local IP.
        # No data is actually sent — this just lets the OS pick the
        # right interface and source address.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        # Fallback: no network route available
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Heartbeat loop
# ---------------------------------------------------------------------------

async def send_heartbeats(ws, agent_name: str, interval: int):
    """Send periodic heartbeats to the dashboard.

    Runs as a concurrent task alongside the message listener.
    """
    while True:
        ip = get_local_ip()
        heartbeat = json.dumps({
            "type": "heartbeat",
            "name": agent_name,
            "ip": ip,
        })
        await ws.send(heartbeat)
        logger.info("Heartbeat sent (ip=%s)", ip)
        await asyncio.sleep(interval)


async def listen_for_commands(ws):
    """Listen for incoming commands from the dashboard.

    Handles:
        - {"type": "shutdown", "reason": "..."} — graceful shutdown
          Triggered by the Kill Switch on the dashboard.
          Per the CAM Constitution: "One click halts all autonomous
          action across all agents. Graceful degradation kicks in."

        - {"type": "command", "command": "..."} — command from George
          Logged and acknowledged. Actual execution comes later when
          the tool/security framework is built.

    Returns True if the agent should shut down, False if just disconnected.
    """
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")

        if msg_type == "shutdown":
            reason = msg.get("reason", "unknown")
            logger.critical("SHUTDOWN received from dashboard (reason: %s)", reason)
            return True

        elif msg_type == "command":
            command_text = msg.get("command", "")
            logger.info("Command received: %s", command_text)

            # Acknowledge receipt — actual execution comes later
            # when the orchestrator and tool framework are built
            response = json.dumps({
                "type": "command_response",
                "command": command_text,
                "response": f"Acknowledged: {command_text}",
            })
            await ws.send(response)
            logger.info("Acknowledgment sent for: %s", command_text)

    # Connection closed without shutdown command
    return False


async def heartbeat_loop(dashboard_url: str, agent_id: str, agent_name: str,
                         interval: int):
    """Connect to the dashboard, send heartbeats, and listen for commands.

    Runs two concurrent tasks:
    1. Sending periodic heartbeats so the dashboard knows we're alive
    2. Listening for incoming commands (e.g., kill switch shutdown)

    If the connection drops, waits a few seconds and retries.
    If a shutdown command is received, exits cleanly.

    Args:
        dashboard_url: WebSocket URL of the dashboard, e.g. ws://192.168.1.100:8080
        agent_id:      Unique ID for this agent (used in the URL path)
        agent_name:    Human-readable display name shown on the dashboard
        interval:      Seconds between heartbeats
    """
    ws_url = f"{dashboard_url}/ws/agent/{agent_id}"
    reconnect_delay = 3  # seconds to wait before retrying after disconnect

    while True:
        try:
            logger.info("Connecting to %s", ws_url)
            async with websockets.connect(ws_url) as ws:
                logger.info("Connected to dashboard")

                # Run heartbeats and command listener concurrently.
                # If either finishes, cancel the other.
                heartbeat_task = asyncio.create_task(
                    send_heartbeats(ws, agent_name, interval)
                )
                listener_task = asyncio.create_task(
                    listen_for_commands(ws)
                )

                # Wait for whichever finishes first
                done, pending = await asyncio.wait(
                    [heartbeat_task, listener_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel the other task
                for task in pending:
                    task.cancel()

                # Check if we got a shutdown command
                for task in done:
                    if task == listener_task and task.result() is True:
                        logger.info("Kill switch shutdown — exiting")
                        return

        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            logger.warning("Connection lost: %s", e)
            logger.info("Reconnecting in %ds...", reconnect_delay)
            await asyncio.sleep(reconnect_delay)

        except asyncio.CancelledError:
            logger.info("Shutting down")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CAM Agent Connector — sends heartbeats to the CAM dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DEPLOY_INSTRUCTIONS,
    )
    parser.add_argument(
        "--dashboard", "-d",
        required=True,
        help="Dashboard WebSocket URL (e.g., ws://192.168.1.100:8080)",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Agent ID (default: hostname, lowercased)",
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Agent display name (default: hostname)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Heartbeat interval in seconds (default: 30)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Deployment instructions (shown with --help)
# ---------------------------------------------------------------------------

DEPLOY_INSTRUCTIONS = """
─────────────────────────────────────────────────────────────
  Deploying to a Raspberry Pi (Ubuntu 24.04)
─────────────────────────────────────────────────────────────

1. Copy this script to the Pi:

    scp agents/connector.py pi@<pi-ip>:~/connector.py

2. SSH into the Pi and install the dependency:

    ssh pi@<pi-ip>
    sudo apt update && sudo apt install python3-websockets -y

    Or, if you prefer pip:
    pip install websockets

3. Test it manually:

    python3 ~/connector.py --dashboard ws://<dashboard-ip>:8080

    You should see heartbeat logs, and the agent should appear
    on the dashboard at http://<dashboard-ip>:8080

4. Set it up as a systemd service so it starts on boot:

    sudo tee /etc/systemd/system/cam-agent.service << 'UNIT'
    [Unit]
    Description=CAM Agent Connector
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    User=pi
    ExecStart=/usr/bin/python3 /home/pi/connector.py --dashboard ws://<dashboard-ip>:8080
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    UNIT

    sudo systemctl daemon-reload
    sudo systemctl enable cam-agent
    sudo systemctl start cam-agent

5. Check status:

    sudo systemctl status cam-agent
    journalctl -u cam-agent -f

6. To customize the agent name and ID:

    ExecStart=/usr/bin/python3 /home/pi/connector.py \\
        --dashboard ws://<dashboard-ip>:8080 \\
        --name "FireHorseClawd" \\
        --id firehorseclawd

─────────────────────────────────────────────────────────────
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    hostname = get_hostname()
    agent_id = args.id or hostname.lower().replace(" ", "-")
    agent_name = args.name or hostname

    logger.info("Agent ID:   %s", agent_id)
    logger.info("Agent Name: %s", agent_name)
    logger.info("Dashboard:  %s", args.dashboard)
    logger.info("Interval:   %ds", args.interval)

    try:
        asyncio.run(heartbeat_loop(
            dashboard_url=args.dashboard,
            agent_id=agent_id,
            agent_name=agent_name,
            interval=args.interval,
        ))
    except KeyboardInterrupt:
        logger.info("Stopped by user")


if __name__ == "__main__":
    main()
