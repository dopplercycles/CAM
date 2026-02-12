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
import base64
import glob
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import socket
import sys
import time

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
# Command parsing and handlers
# ---------------------------------------------------------------------------

# Track when the connector started so we can report agent uptime
_CONNECTOR_START = time.monotonic()

# Module-level settings — set from CLI args (which may come from config)
_SYSTEMCTL_TIMEOUT: int = 30
_DASHBOARD_HOST: str = "192.168.12.232"
_DASHBOARD_PORT: int = 8080
_RECEIVE_DIR: str = os.path.expanduser("~/receive")

# In-progress incoming file transfers: transfer_id → {metadata, chunks: {index: bytes}}
_pending_receives: dict = {}


def parse_command(text: str) -> tuple[str, dict[str, str]]:
    """Parse a command string into (command_name, params_dict).

    Format: "command_name key=value key2=value2 ..."
    Values may be quoted: key="some value with spaces"

    Returns:
        Tuple of (command_name, dict_of_params).
        If the text is empty, returns ("", {}).
    """
    text = text.strip()
    if not text:
        return ("", {})

    parts = text.split()
    name = parts[0]
    params: dict[str, str] = {}

    # Rejoin remaining parts in case of quoted values, then parse key=value
    rest = text[len(name):].strip()
    if rest:
        # Simple state machine: split on spaces but respect quotes
        current_key = None
        current_val_parts: list[str] = []
        for token in rest.split():
            if "=" in token and current_key is None:
                eq_idx = token.index("=")
                current_key = token[:eq_idx]
                val_part = token[eq_idx + 1:]
                if val_part.startswith('"') and not val_part.endswith('"'):
                    current_val_parts = [val_part[1:]]
                    continue
                # Strip surrounding quotes
                params[current_key] = val_part.strip('"')
                current_key = None
            elif current_key is not None:
                # Continuing a quoted value
                if token.endswith('"'):
                    current_val_parts.append(token[:-1])
                    params[current_key] = " ".join(current_val_parts)
                    current_key = None
                    current_val_parts = []
                else:
                    current_val_parts.append(token)

    return (name, params)


async def handle_status_report() -> str:
    """Return agent name, uptime, Python version, cwd, asyncio task count."""
    try:
        uptime_secs = time.monotonic() - _CONNECTOR_START
        hours, remainder = divmod(int(uptime_secs), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        task_count = len(asyncio.all_tasks())
        return (
            f"Status Report\n"
            f"  Hostname:       {socket.gethostname()}\n"
            f"  Python:         {platform.python_version()}\n"
            f"  Platform:       {platform.platform()}\n"
            f"  CWD:            {os.getcwd()}\n"
            f"  Agent uptime:   {uptime_str}\n"
            f"  Asyncio tasks:  {task_count}"
        )
    except Exception as e:
        return f"Error generating status report: {e}"


async def handle_system_info() -> str:
    """Return CPU, memory, disk, and system uptime info."""
    try:
        lines = ["System Info"]

        # CPU count and load averages
        cpu_count = os.cpu_count() or "unknown"
        try:
            load1, load5, load15 = os.getloadavg()
            lines.append(f"  CPUs:           {cpu_count}")
            lines.append(f"  Load avg:       {load1:.2f} / {load5:.2f} / {load15:.2f}")
        except OSError:
            lines.append(f"  CPUs:           {cpu_count}")
            lines.append("  Load avg:       not available")

        # Memory from /proc/meminfo
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        meminfo[key] = int(parts[1])  # value in kB
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
            used_kb = total_kb - avail_kb
            used_pct = (used_kb / total_kb * 100) if total_kb > 0 else 0
            lines.append(
                f"  Memory:         {total_kb // 1024} MB total, "
                f"{avail_kb // 1024} MB available, "
                f"{used_pct:.1f}% used"
            )
        except (OSError, ValueError):
            lines.append("  Memory:         not available")

        # Disk usage
        try:
            usage = shutil.disk_usage("/")
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            used_pct = usage.used / usage.total * 100
            lines.append(
                f"  Disk (/):       {total_gb:.1f} GB total, "
                f"{used_gb:.1f} GB used, "
                f"{free_gb:.1f} GB free ({used_pct:.1f}% used)"
            )
        except OSError:
            lines.append("  Disk (/):       not available")

        # System uptime from /proc/uptime
        try:
            with open("/proc/uptime", "r") as f:
                uptime_secs = float(f.read().split()[0])
            days, remainder = divmod(int(uptime_secs), 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)
            lines.append(f"  System uptime:  {days}d {hours}h {minutes}m")
        except (OSError, ValueError):
            lines.append("  System uptime:  not available")

        return "\n".join(lines)
    except Exception as e:
        return f"Error generating system info: {e}"


async def handle_restart_service(params: dict[str, str]) -> str:
    """Restart a systemd service. Requires 'service_name' param."""
    service_name = params.get("service_name", "").strip()
    if not service_name:
        return "Error: 'service_name' parameter is required.\nUsage: restart_service service_name=cam-agent"

    # Basic validation: only allow alphanumeric, hyphens, underscores, dots, @
    if not re.match(r'^[a-zA-Z0-9._@-]+$', service_name):
        return f"Error: invalid service name '{service_name}'"

    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "systemctl", "restart", service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SYSTEMCTL_TIMEOUT)

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        if proc.returncode == 0:
            result = f"Service '{service_name}' restarted successfully (exit code 0)."
        else:
            result = f"Service '{service_name}' restart failed (exit code {proc.returncode})."

        if stdout_text:
            result += f"\nstdout: {stdout_text}"
        if stderr_text:
            result += f"\nstderr: {stderr_text}"

        return result
    except asyncio.TimeoutError:
        return f"Error: restart of '{service_name}' timed out after {_SYSTEMCTL_TIMEOUT}s"
    except Exception as e:
        return f"Error restarting service '{service_name}': {e}"


async def handle_run_diagnostic() -> str:
    """Run a suite of health checks and return a diagnostic report."""
    try:
        lines = ["Diagnostic Report"]
        warnings = []

        # Python version
        lines.append(f"  Python:         {platform.python_version()}")

        # websockets version
        try:
            import websockets
            lines.append(f"  websockets:     {websockets.__version__}")
        except AttributeError:
            lines.append("  websockets:     installed (version unknown)")

        # Disk space warning if >90% used
        try:
            usage = shutil.disk_usage("/")
            used_pct = usage.used / usage.total * 100
            lines.append(f"  Disk usage:     {used_pct:.1f}%")
            if used_pct > 90:
                warnings.append(f"  WARN: Disk usage is {used_pct:.1f}% (>90%)")
        except OSError:
            lines.append("  Disk usage:     not available")

        # Memory warning if available <10%
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo: dict[str, int] = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        meminfo[key] = int(parts[1])
            total = meminfo.get("MemTotal", 0)
            avail = meminfo.get("MemAvailable", 0)
            avail_pct = (avail / total * 100) if total > 0 else 0
            lines.append(f"  Memory avail:   {avail_pct:.1f}%")
            if avail_pct < 10:
                warnings.append(f"  WARN: Available memory is {avail_pct:.1f}% (<10%)")
        except (OSError, ValueError):
            lines.append("  Memory avail:   not available")

        # Network connectivity: can we resolve the dashboard host?
        try:
            # Try to resolve the main machine IP (dashboard)
            socket.getaddrinfo(_DASHBOARD_HOST, _DASHBOARD_PORT, socket.AF_INET, socket.SOCK_STREAM)
            lines.append("  Network:        dashboard host reachable")
        except socket.gaierror:
            warnings.append("  WARN: Cannot resolve dashboard host")
            lines.append("  Network:        dashboard host NOT reachable")

        # Agent uptime
        uptime_secs = time.monotonic() - _CONNECTOR_START
        hours, remainder = divmod(int(uptime_secs), 3600)
        minutes, seconds = divmod(remainder, 60)
        lines.append(f"  Agent uptime:   {hours}h {minutes}m {seconds}s")

        if warnings:
            lines.append("\nWarnings:")
            lines.extend(warnings)
        else:
            lines.append("\n  All checks passed.")

        return "\n".join(lines)
    except Exception as e:
        return f"Error running diagnostic: {e}"


async def handle_capture_sensor_data(params: dict[str, str]) -> str:
    """Read CPU temperature from thermal zones. Optional 'sensor_type' filter."""
    try:
        sensor_type = params.get("sensor_type", "").strip().lower()
        thermal_paths = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))

        if not thermal_paths:
            return "Sensor Data\n  No thermal zones found on this system."

        lines = ["Sensor Data"]
        found_any = False

        for temp_path in thermal_paths:
            zone_dir = os.path.dirname(temp_path)
            zone_name = os.path.basename(zone_dir)

            # Read the sensor type for this zone
            type_path = os.path.join(zone_dir, "type")
            try:
                with open(type_path, "r") as f:
                    zone_type = f.read().strip()
            except OSError:
                zone_type = "unknown"

            # If sensor_type filter is set, skip non-matching zones
            if sensor_type and sensor_type not in zone_type.lower():
                continue

            # Read temperature (millidegrees Celsius)
            try:
                with open(temp_path, "r") as f:
                    temp_milli = int(f.read().strip())
                temp_c = temp_milli / 1000.0
                lines.append(f"  {zone_name} ({zone_type}): {temp_c:.1f}°C")
                found_any = True
            except (OSError, ValueError):
                lines.append(f"  {zone_name} ({zone_type}): read error")
                found_any = True

        if not found_any and sensor_type:
            lines.append(f"  No sensors matching '{sensor_type}' found.")

        return "\n".join(lines)
    except Exception as e:
        return f"Error capturing sensor data: {e}"


async def handle_connector_version() -> str:
    """Return this connector file's SHA-256 hash for version tracking.

    The deployer on the dashboard uses this to check whether an agent
    is running the latest connector before deciding to push an update.
    """
    try:
        connector_path = os.path.abspath(__file__)
        with open(connector_path, "rb") as f:
            data = f.read()
        file_hash = hashlib.sha256(data).hexdigest()
        return f"hash={file_hash} size={len(data)} path={connector_path}"
    except Exception as e:
        return f"Error reading connector version: {e}"


# Map command names to their async handler functions.
# Handlers that need params take a dict; those that don't take no args.
COMMAND_HANDLERS = {
    "status_report":       lambda params: handle_status_report(),
    "system_info":         lambda params: handle_system_info(),
    "restart_service":     lambda params: handle_restart_service(params),
    "run_diagnostic":      lambda params: handle_run_diagnostic(),
    "capture_sensor_data": lambda params: handle_capture_sensor_data(params),
    "connector_version":   lambda params: handle_connector_version(),
}


# ---------------------------------------------------------------------------
# Heartbeat loop
# ---------------------------------------------------------------------------

async def send_heartbeats(ws, agent_name: str, interval: int,
                          capabilities: list[str] | None = None):
    """Send periodic heartbeats to the dashboard.

    Runs as a concurrent task alongside the message listener.
    """
    while True:
        ip = get_local_ip()
        heartbeat = {
            "type": "heartbeat",
            "name": agent_name,
            "ip": ip,
        }
        if capabilities:
            heartbeat["capabilities"] = capabilities
        await ws.send(json.dumps(heartbeat))
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

        elif msg_type == "ping":
            ping_id = msg.get("ping_id")
            await ws.send(json.dumps({"type": "pong", "ping_id": ping_id}))

        elif msg_type == "file_transfer_start":
            # Server wants to send us a file — set up receive buffer
            transfer_id = msg.get("transfer_id", "")
            filename = msg.get("filename", "")
            file_size = msg.get("file_size", 0)
            chunk_count = msg.get("chunk_count", 0)
            dest_path = msg.get("dest_path", "")
            checksum = msg.get("checksum", "")

            # Resolve destination path
            if dest_path:
                dest = os.path.expanduser(dest_path)
            else:
                dest = os.path.join(_RECEIVE_DIR, filename)

            # Ensure parent directory exists
            dest_dir = os.path.dirname(dest)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

            _pending_receives[transfer_id] = {
                "filename": filename,
                "file_size": file_size,
                "chunk_count": chunk_count,
                "checksum": checksum,
                "dest_path": dest,
                "chunks": {},
            }
            logger.info(
                "File transfer %s started: %s (%d bytes, %d chunks) → %s",
                transfer_id, filename, file_size, chunk_count, dest,
            )

        elif msg_type == "file_chunk":
            # Receive a chunk of file data
            transfer_id = msg.get("transfer_id", "")
            chunk_index = msg.get("chunk_index", 0)
            chunk_count = msg.get("chunk_count", 0)
            b64_data = msg.get("data", "")

            recv = _pending_receives.get(transfer_id)
            if not recv:
                await ws.send(json.dumps({
                    "type": "file_chunk_ack",
                    "transfer_id": transfer_id,
                    "chunk_index": chunk_index,
                    "ok": False,
                    "error": "Unknown transfer_id",
                }))
                continue

            try:
                raw = base64.b64decode(b64_data)
                recv["chunks"][chunk_index] = raw

                await ws.send(json.dumps({
                    "type": "file_chunk_ack",
                    "transfer_id": transfer_id,
                    "chunk_index": chunk_index,
                    "ok": True,
                }))

                # Check if all chunks received
                if len(recv["chunks"]) >= recv["chunk_count"]:
                    # Assemble file
                    parts = []
                    for i in range(recv["chunk_count"]):
                        parts.append(recv["chunks"][i])
                    file_data = b"".join(parts)

                    # Verify checksum
                    expected = recv["checksum"]
                    actual = "sha256:" + hashlib.sha256(file_data).hexdigest()
                    if expected and actual != expected:
                        logger.error(
                            "Checksum mismatch for %s: expected %s, got %s",
                            transfer_id, expected, actual,
                        )
                        await ws.send(json.dumps({
                            "type": "file_transfer_complete",
                            "transfer_id": transfer_id,
                            "ok": False,
                            "error": f"Checksum mismatch: expected {expected}, got {actual}",
                        }))
                        _pending_receives.pop(transfer_id, None)
                        continue

                    # Write file to disk
                    with open(recv["dest_path"], "wb") as f:
                        f.write(file_data)

                    logger.info(
                        "File transfer %s complete: %s written (%d bytes)",
                        transfer_id, recv["dest_path"], len(file_data),
                    )
                    await ws.send(json.dumps({
                        "type": "file_transfer_complete",
                        "transfer_id": transfer_id,
                        "ok": True,
                        "checksum": actual,
                        "final_path": recv["dest_path"],
                    }))
                    _pending_receives.pop(transfer_id, None)

            except Exception as e:
                logger.exception("Error processing chunk %d of %s", chunk_index, transfer_id)
                await ws.send(json.dumps({
                    "type": "file_chunk_ack",
                    "transfer_id": transfer_id,
                    "chunk_index": chunk_index,
                    "ok": False,
                    "error": str(e),
                }))

        elif msg_type == "file_transfer_cancel":
            transfer_id = msg.get("transfer_id", "")
            _pending_receives.pop(transfer_id, None)
            logger.info("File transfer %s cancelled by server", transfer_id)

        elif msg_type == "file_request":
            # Server is asking us to send a file back
            transfer_id = msg.get("transfer_id", "")
            file_path = msg.get("file_path", "")

            try:
                full_path = os.path.expanduser(file_path)
                if not os.path.isfile(full_path):
                    await ws.send(json.dumps({
                        "type": "file_send_start",
                        "transfer_id": transfer_id,
                        "ok": False,
                        "error": f"File not found: {full_path}",
                    }))
                    continue

                with open(full_path, "rb") as f:
                    file_data = f.read()

                filename = os.path.basename(full_path)
                file_size = len(file_data)
                checksum = "sha256:" + hashlib.sha256(file_data).hexdigest()

                # Chunk the data at 64KB
                chunk_size = 65536
                chunks = []
                for offset in range(0, len(file_data), chunk_size):
                    raw_chunk = file_data[offset:offset + chunk_size]
                    chunks.append(base64.b64encode(raw_chunk).decode("ascii"))

                chunk_count = len(chunks)

                await ws.send(json.dumps({
                    "type": "file_send_start",
                    "transfer_id": transfer_id,
                    "ok": True,
                    "filename": filename,
                    "file_size": file_size,
                    "chunk_count": chunk_count,
                    "checksum": checksum,
                }))

                # Send chunks
                for i, b64_chunk in enumerate(chunks):
                    await ws.send(json.dumps({
                        "type": "file_send_chunk",
                        "transfer_id": transfer_id,
                        "chunk_index": i,
                        "chunk_count": chunk_count,
                        "data": b64_chunk,
                    }))
                    await asyncio.sleep(0)  # yield to event loop

                logger.info(
                    "File %s sent: %s (%d bytes, %d chunks)",
                    transfer_id, filename, file_size, chunk_count,
                )

            except Exception as e:
                logger.exception("Error sending file for %s", transfer_id)
                await ws.send(json.dumps({
                    "type": "file_send_start",
                    "transfer_id": transfer_id,
                    "ok": False,
                    "error": str(e),
                }))

        elif msg_type == "command":
            command_text = msg.get("command", "")
            task_id = msg.get("task_id")
            logger.info("Command received: %s (task_id=%s)", command_text, task_id)

            # Parse command and dispatch to handler
            cmd_name, cmd_params = parse_command(command_text)
            handler = COMMAND_HANDLERS.get(cmd_name)

            if handler:
                try:
                    response_text = await handler(cmd_params)
                except Exception as e:
                    response_text = f"Error executing '{cmd_name}': {e}"
                    logger.exception("Handler error for %s", cmd_name)
            else:
                response_text = f"Unknown command: {cmd_name}"

            resp = {
                "type": "command_response",
                "command": command_text,
                "response": response_text,
            }
            # Echo task_id back so the orchestrator can correlate responses
            if task_id is not None:
                resp["task_id"] = task_id

            await ws.send(json.dumps(resp))
            logger.info("Response sent for: %s", cmd_name)

    # Connection closed without shutdown command
    return False


async def heartbeat_loop(dashboard_url: str, agent_id: str, agent_name: str,
                         interval: int, capabilities: list[str] | None = None,
                         reconnect_delay: int = 3):
    """Connect to the dashboard, send heartbeats, and listen for commands.

    Runs two concurrent tasks:
    1. Sending periodic heartbeats so the dashboard knows we're alive
    2. Listening for incoming commands (e.g., kill switch shutdown)

    If the connection drops, waits a few seconds and retries.
    If a shutdown command is received, exits cleanly.

    Args:
        dashboard_url:   WebSocket URL of the dashboard, e.g. ws://192.168.1.100:8080
        agent_id:        Unique ID for this agent (used in the URL path)
        agent_name:      Human-readable display name shown on the dashboard
        interval:        Seconds between heartbeats
        capabilities:    List of things this agent can do (e.g., ["research", "content"])
        reconnect_delay: Seconds to wait before retrying after disconnect
    """
    ws_url = f"{dashboard_url}/ws/agent/{agent_id}"

    while True:
        try:
            logger.info("Connecting to %s", ws_url)
            async with websockets.connect(ws_url) as ws:
                logger.info("Connected to dashboard")

                # Run heartbeats and command listener concurrently.
                # If either finishes, cancel the other.
                heartbeat_task = asyncio.create_task(
                    send_heartbeats(ws, agent_name, interval, capabilities)
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
    parser.add_argument(
        "--capabilities", "-c",
        nargs="+",
        default=None,
        help="List of capabilities (e.g., --capabilities research content tts)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to CAM settings.toml (optional — connector works without it)",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=int,
        default=None,
        help="Seconds to wait before reconnecting after disconnect (default: 3)",
    )
    parser.add_argument(
        "--systemctl-timeout",
        type=int,
        default=None,
        help="Timeout for systemctl restart commands in seconds (default: 30)",
    )
    parser.add_argument(
        "--receive-dir",
        default=None,
        help="Directory for received files (default: ~/receive)",
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
    global _SYSTEMCTL_TIMEOUT, _DASHBOARD_HOST, _DASHBOARD_PORT, _RECEIVE_DIR

    args = parse_args()

    # Load config file if provided (optional — connector works without it)
    connector_cfg = {}
    if args.config:
        try:
            import tomllib
            with open(args.config, "rb") as f:
                toml_data = tomllib.load(f)
            connector_cfg = toml_data.get("connector", {})
            logger.info("Loaded config from %s", args.config)
        except Exception as e:
            logger.warning("Failed to load config %s: %s — using defaults", args.config, e)

    hostname = get_hostname()
    agent_id = args.id or hostname.lower().replace(" ", "-")
    agent_name = args.name or hostname
    capabilities = args.capabilities

    # Resolve settings: CLI args > config file > hardcoded defaults
    interval = args.interval  # CLI default is 30
    reconnect_delay = (
        args.reconnect_delay
        if args.reconnect_delay is not None
        else connector_cfg.get("reconnect_delay", 3)
    )
    _SYSTEMCTL_TIMEOUT = (
        args.systemctl_timeout
        if args.systemctl_timeout is not None
        else connector_cfg.get("systemctl_timeout", 30)
    )
    _DASHBOARD_HOST = connector_cfg.get("dashboard_host", "192.168.12.232")
    _DASHBOARD_PORT = connector_cfg.get("dashboard_port", 8080)
    _RECEIVE_DIR = os.path.expanduser(
        args.receive_dir
        if args.receive_dir is not None
        else connector_cfg.get("agent_receive_dir", "~/receive")
    )
    os.makedirs(_RECEIVE_DIR, exist_ok=True)

    logger.info("Agent ID:      %s", agent_id)
    logger.info("Agent Name:    %s", agent_name)
    logger.info("Dashboard:     %s", args.dashboard)
    logger.info("Interval:      %ds", interval)
    logger.info("Reconnect:     %ds", reconnect_delay)
    logger.info("Systemctl TO:  %ds", _SYSTEMCTL_TIMEOUT)
    logger.info("Receive dir:   %s", _RECEIVE_DIR)
    if capabilities:
        logger.info("Capabilities:  %s", capabilities)

    try:
        asyncio.run(heartbeat_loop(
            dashboard_url=args.dashboard,
            agent_id=agent_id,
            agent_name=agent_name,
            interval=interval,
            capabilities=capabilities,
            reconnect_delay=reconnect_delay,
        ))
    except KeyboardInterrupt:
        logger.info("Stopped by user")


if __name__ == "__main__":
    main()
