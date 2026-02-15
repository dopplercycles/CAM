"""
CAM Tool Registry — Claude API Tool-Use Definitions & Execution

Defines the tools Cam can use during conversation, classifies each by
Constitutional tier, and provides execution functions.

Tool Tiers (from CAM_CONSTITUTION.md):
    Tier 1 — Autonomous: read_file (safe paths), list_directory, system_status,
             web_search, web_fetch
    Tier 2 — Approval Required: run_command, write_file (code/config paths)
    Tier 1 — Autonomous writes: data/notes/, data/research/, data/content/,
             data/reports/, /tmp/
    Tier 3 — Blocked: read_file on sensitive paths, destructive commands

Usage:
    from tools.tool_registry import TOOL_DEFINITIONS, classify_tool_tier, execute_tool

    tier = classify_tool_tier("read_file", {"path": "/etc/shadow"})
    # → 3 (blocked)

    result = await execute_tool("system_status", {"detail": "summary"})
    # → {"result": "CPU: 12%, Memory: 4.2/8GB, ..."}
"""

import asyncio
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger("cam.tool_registry")


# ---------------------------------------------------------------------------
# Security boundaries
# ---------------------------------------------------------------------------

# Paths that are always Tier 3 (blocked) for read operations
BLOCKED_PATHS = {
    "/etc/shadow", "/etc/passwd", "/etc/sudoers",
    "/root/", "/var/log/auth",
}

# Path prefixes that are blocked for reads
BLOCKED_PATH_PREFIXES = (
    "/etc/", "/usr/", "/var/", "/root/",
    os.path.expanduser("~/.ssh/"),
    os.path.expanduser("~/.gnupg/"),
    os.path.expanduser("~/.aws/"),
    os.path.expanduser("~/.config/"),
)

# Autonomous write directories — Tier 1, Cam writes freely (working data)
AUTONOMOUS_WRITE_DIRS = (
    os.path.expanduser("~/CAM/data/notes/"),
    os.path.expanduser("~/CAM/data/research/"),
    os.path.expanduser("~/CAM/data/content/"),
    os.path.expanduser("~/CAM/data/reports/"),
    "/tmp/",
)

# Allowed directories for write_file (Tier 2 — requires approval for code/config)
ALLOWED_WRITE_DIRS = (
    os.path.expanduser("~/CAM/"),
    "/tmp/cam-",
)

# Dangerous command patterns — blocked even with approval
BLOCKED_COMMANDS = (
    "rm -rf /",
    "mkfs",
    "dd if=",
    ":(){",
    "> /dev/sd",
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
)


# ---------------------------------------------------------------------------
# Tool definitions — Claude API format
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "run_command",
        "description": (
            "Execute a shell command on the server. Use this for system "
            "administration tasks, checking processes, package management, etc. "
            "Always requires George's approval before execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 120).",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. Works for text files within "
            "the CAM project directory and other safe locations. "
            "Sensitive system files (e.g., /etc/, ~/.ssh/) are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return (default 500).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Autonomous (Tier 1) for working data "
            "directories: data/notes/, data/research/, data/content/, "
            "data/reports/, and /tmp/. Requires approval (Tier 2) for code "
            "and config files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path for the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List the contents of a directory, showing files and subdirectories "
            "with basic metadata (size, modification time)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the directory to list.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "system_status",
        "description": (
            "Get current system status including CPU load, memory usage, "
            "disk space, and uptime. No subprocess calls — reads from /proc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "detail": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "description": "Level of detail: 'summary' or 'full'.",
                },
            },
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. Returns titles, URLs, and "
            "snippets for the top results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch a web page and extract its text content. Useful for "
            "reading articles, documentation, or product pages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def _classify_read_tier(path: str) -> int:
    """Classify a read_file call's tier based on the target path.

    Returns 1 for safe paths, 3 for blocked paths.
    """
    try:
        resolved = str(Path(path).expanduser().resolve())
    except (ValueError, OSError):
        return 3  # Can't resolve → block

    # Check exact blocked paths
    if resolved in BLOCKED_PATHS:
        return 3

    # Check blocked prefixes
    for prefix in BLOCKED_PATH_PREFIXES:
        if resolved.startswith(prefix):
            return 3

    return 1


def _classify_write_tier(path: str) -> int:
    """Classify a write_file call's tier based on the target path.

    Returns 1 for autonomous working dirs (data/notes, data/research, etc.),
    2 for other allowed paths (code, config — needs approval), 3 for blocked.
    """
    try:
        resolved = str(Path(path).expanduser().resolve())
    except (ValueError, OSError):
        return 3

    # Tier 1 — autonomous working directories (Cam writes freely)
    for auto_dir in AUTONOMOUS_WRITE_DIRS:
        if resolved.startswith(auto_dir):
            return 1

    # Tier 2 — allowed but needs approval (code, config, other project files)
    for allowed in ALLOWED_WRITE_DIRS:
        if resolved.startswith(allowed):
            return 2

    return 3  # Outside allowed dirs → blocked


def _is_command_blocked(command: str) -> bool:
    """Check if a shell command matches a blocked pattern."""
    lower = command.lower().strip()
    for pattern in BLOCKED_COMMANDS:
        if pattern in lower:
            return True
    return False


def classify_tool_tier(tool_name: str, tool_input: dict) -> int:
    """Classify a tool call into a Constitutional tier.

    Args:
        tool_name:  The tool being called.
        tool_input: The input arguments.

    Returns:
        1 (autonomous), 2 (approval required), or 3 (blocked).
    """
    if tool_name == "run_command":
        command = tool_input.get("command", "")
        if _is_command_blocked(command):
            return 3
        return 2  # Always requires approval

    if tool_name == "read_file":
        return _classify_read_tier(tool_input.get("path", ""))

    if tool_name == "write_file":
        return _classify_write_tier(tool_input.get("path", ""))

    if tool_name == "list_directory":
        return 1

    if tool_name == "system_status":
        return 1

    if tool_name == "web_search":
        return 1

    if tool_name == "web_fetch":
        return 1

    # Unknown tool → block
    return 3


# ---------------------------------------------------------------------------
# Tool execution functions
# ---------------------------------------------------------------------------

async def execute_tool(name: str, tool_input: dict) -> dict:
    """Execute a tool and return the result.

    Args:
        name:       Tool name from TOOL_DEFINITIONS.
        tool_input: The input arguments from Claude's tool_use block.

    Returns:
        Dict with "result" key on success, or "error" key on failure.
    """
    executors = {
        "run_command": _exec_run_command,
        "read_file": _exec_read_file,
        "write_file": _exec_write_file,
        "list_directory": _exec_list_directory,
        "system_status": _exec_system_status,
        "web_search": _exec_web_search,
        "web_fetch": _exec_web_fetch,
    }

    executor = executors.get(name)
    if executor is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        return await executor(tool_input)
    except Exception as e:
        logger.error("Tool execution failed (%s): %s", name, e, exc_info=True)
        return {"error": f"Tool execution failed: {e}"}


# --- Individual executors ---

async def _exec_run_command(tool_input: dict) -> dict:
    """Execute a shell command with timeout and output capping."""
    command = tool_input.get("command", "").strip()
    if not command:
        return {"error": "No command provided."}

    timeout = min(int(tool_input.get("timeout", 30)), 120)

    logger.info("Executing command: %s (timeout=%ds)", command, timeout)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return {"error": f"Command timed out after {timeout}s."}
    except Exception as e:
        return {"error": f"Failed to execute command: {e}"}

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    # Cap output at 10000 chars
    max_chars = 10000
    if len(stdout_text) > max_chars:
        stdout_text = stdout_text[:max_chars] + f"\n... (truncated, {len(stdout_text)} chars total)"
    if len(stderr_text) > max_chars:
        stderr_text = stderr_text[:max_chars] + f"\n... (truncated, {len(stderr_text)} chars total)"

    result = {
        "exit_code": proc.returncode,
        "stdout": stdout_text,
    }
    if stderr_text.strip():
        result["stderr"] = stderr_text

    return {"result": result}


async def _exec_read_file(tool_input: dict) -> dict:
    """Read a file's contents with line limit."""
    path = tool_input.get("path", "")
    max_lines = min(int(tool_input.get("max_lines", 500)), 2000)

    if not path:
        return {"error": "No path provided."}

    try:
        resolved = Path(path).expanduser().resolve()
    except (ValueError, OSError) as e:
        return {"error": f"Invalid path: {e}"}

    if not resolved.exists():
        return {"error": f"File not found: {resolved}"}

    if not resolved.is_file():
        return {"error": f"Not a file: {resolved}"}

    # Read in a thread to avoid blocking
    def _read():
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
            return lines

    try:
        lines = await asyncio.to_thread(_read)
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    # Get total line count for info
    total_lines = len(lines)
    content = "".join(lines)

    # Cap total content
    if len(content) > 50000:
        content = content[:50000] + "\n... (truncated)"

    result = {
        "path": str(resolved),
        "content": content,
        "lines_read": total_lines,
    }
    if total_lines >= max_lines:
        result["note"] = f"Output capped at {max_lines} lines. File may be longer."

    return {"result": result}


async def _exec_write_file(tool_input: dict) -> dict:
    """Write content to a file within allowed directories."""
    path = tool_input.get("path", "")
    content = tool_input.get("content", "")

    if not path:
        return {"error": "No path provided."}

    try:
        resolved = Path(path).expanduser().resolve()
    except (ValueError, OSError) as e:
        return {"error": f"Invalid path: {e}"}

    # Create parent directories if needed
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"error": f"Cannot create directory: {e}"}

    def _write():
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        return {"error": f"Failed to write file: {e}"}

    logger.info("File written: %s (%d bytes)", resolved, len(content))
    return {"result": {"path": str(resolved), "bytes_written": len(content)}}


async def _exec_list_directory(tool_input: dict) -> dict:
    """List directory contents with basic metadata."""
    path = tool_input.get("path", "")
    if not path:
        return {"error": "No path provided."}

    try:
        resolved = Path(path).expanduser().resolve()
    except (ValueError, OSError) as e:
        return {"error": f"Invalid path: {e}"}

    if not resolved.exists():
        return {"error": f"Directory not found: {resolved}"}

    if not resolved.is_dir():
        return {"error": f"Not a directory: {resolved}"}

    def _list():
        entries = []
        try:
            for entry in sorted(resolved.iterdir()):
                try:
                    stat = entry.stat()
                    entries.append({
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": stat.st_size if entry.is_file() else None,
                    })
                except OSError:
                    entries.append({"name": entry.name, "type": "unknown"})
        except PermissionError:
            return None
        return entries

    entries = await asyncio.to_thread(_list)
    if entries is None:
        return {"error": f"Permission denied: {resolved}"}

    # Cap at 200 entries
    total = len(entries)
    if total > 200:
        entries = entries[:200]

    result = {"path": str(resolved), "entries": entries, "total": total}
    if total > 200:
        result["note"] = f"Showing first 200 of {total} entries."

    return {"result": result}


async def _exec_system_status(tool_input: dict) -> dict:
    """Get system status from /proc and os — no dependencies needed."""
    detail = tool_input.get("detail", "summary")
    info = {}

    # CPU load from /proc/loadavg
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().strip().split()
            info["load_avg"] = {
                "1min": float(parts[0]),
                "5min": float(parts[1]),
                "15min": float(parts[2]),
            }
    except (FileNotFoundError, IndexError, ValueError):
        info["load_avg"] = "unavailable"

    # Memory from /proc/meminfo
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    # Parse "12345 kB" → int in kB
                    val = val.strip().split()[0]
                    meminfo[key.strip()] = int(val)

        total_mb = meminfo.get("MemTotal", 0) / 1024
        available_mb = meminfo.get("MemAvailable", 0) / 1024
        used_mb = total_mb - available_mb
        info["memory"] = {
            "total_mb": round(total_mb, 1),
            "used_mb": round(used_mb, 1),
            "available_mb": round(available_mb, 1),
            "percent_used": round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0,
        }
    except (FileNotFoundError, KeyError, ValueError):
        info["memory"] = "unavailable"

    # Disk usage from os.statvfs
    try:
        stat = os.statvfs("/")
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        used_gb = total_gb - free_gb
        info["disk"] = {
            "total_gb": round(total_gb, 1),
            "used_gb": round(used_gb, 1),
            "free_gb": round(free_gb, 1),
            "percent_used": round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0,
        }
    except OSError:
        info["disk"] = "unavailable"

    # Uptime from /proc/uptime
    try:
        with open("/proc/uptime", "r") as f:
            uptime_secs = float(f.read().strip().split()[0])
            days = int(uptime_secs // 86400)
            hours = int((uptime_secs % 86400) // 3600)
            mins = int((uptime_secs % 3600) // 60)
            info["uptime"] = f"{days}d {hours}h {mins}m"
    except (FileNotFoundError, ValueError):
        info["uptime"] = "unavailable"

    # Platform info (always included)
    info["hostname"] = platform.node()
    info["platform"] = f"{platform.system()} {platform.release()}"

    if detail == "full":
        # CPU count
        try:
            info["cpu_count"] = os.cpu_count()
        except Exception:
            pass

        # Home directory disk
        try:
            home = os.path.expanduser("~")
            stat = os.statvfs(home)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            info["home_disk"] = {
                "path": home,
                "total_gb": round(total_gb, 1),
                "free_gb": round(free_gb, 1),
            }
        except OSError:
            pass

    return {"result": info}


async def _exec_web_search(tool_input: dict) -> dict:
    """Search the web via DuckDuckGo using the existing WebTool."""
    query = tool_input.get("query", "").strip()
    if not query:
        return {"error": "No search query provided."}

    max_results = min(int(tool_input.get("max_results", 5)), 10)

    try:
        from tools.web import WebTool
        web = WebTool()
        results = web.search(query, max_results=max_results)

        return {"result": [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            }
            for r in results
        ]}
    except Exception as e:
        return {"error": f"Web search failed: {e}"}


async def _exec_web_fetch(tool_input: dict) -> dict:
    """Fetch a web page and extract text using the existing WebTool."""
    url = tool_input.get("url", "").strip()
    if not url:
        return {"error": "No URL provided."}

    try:
        from tools.web import WebTool
        web = WebTool()
        page = web.fetch_page(url)

        text = page.text or ""
        # Cap at 15000 chars
        if len(text) > 15000:
            text = text[:15000] + "\n... (truncated)"

        return {"result": {
            "url": page.url,
            "title": page.title,
            "text": text,
            "status_code": page.status_code,
        }}
    except Exception as e:
        return {"error": f"Web fetch failed: {e}"}
