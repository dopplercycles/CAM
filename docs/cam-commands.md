# CAM Quick Command Reference

## Service Management

```bash
# Start / Stop / Restart
sudo systemctl start cam
sudo systemctl stop cam
sudo systemctl restart cam

# Check status
systemctl status cam

# Follow live logs
journalctl -u cam -f

# Logs from today only
journalctl -u cam --since today

# Enable/disable auto-start on boot
sudo systemctl enable cam
sudo systemctl disable cam
```

## Starting Without systemd

```bash
cd ~/CAM

# Foreground (logs to terminal, Ctrl+C to stop)
./start.sh

# Background (logs to data/logs/dashboard.log)
./start.sh --bg

# Stop background instance
./stop.sh
```

## CLI Terminal

```bash
cd ~/CAM
python -m interfaces.cli.terminal
```

Once inside the CLI:

| Command          | What it does                                    |
|------------------|-------------------------------------------------|
| `/status`        | Show system state, queue counts, costs, memory  |
| `/agents`        | List all agents with status and capabilities    |
| `/task <text>`   | Submit a task to CAM                            |
| `/memory <text>` | Search long-term and episodic memory            |
| `/history`       | Show 20 most recent memory entries              |
| `/config`        | Show current config (secrets masked)            |
| `/test`          | Run the self-test suite                         |
| `/kill`          | Emergency stop the orchestrator (requires CONFIRM) |
| `/help`          | Show all available commands                     |
| `/quit`          | Exit the CLI                                    |

Or just type natural language and CAM will process it as a task.

## Dashboard

Open in browser: `http://192.168.12.232:8080`

## Checking Agent Status (from terminal)

```bash
# Quick check - are agents connected?
curl -s http://localhost:8080/api/agents | python3 -m json.tool

# System health
curl -s http://localhost:8080/api/health | python3 -m json.tool

# Recent events
curl -s http://localhost:8080/api/events | python3 -m json.tool

# Current tasks
curl -s http://localhost:8080/api/tasks | python3 -m json.tool

# Analytics summary
curl -s http://localhost:8080/api/analytics | python3 -m json.tool
```

## Remote Agent Management

```bash
# FireHorseClawd (Pi 5)
ssh firehorse@192.168.12.243
sudo systemctl status cam-agent
sudo systemctl restart cam-agent
journalctl -u cam-agent -f

# Nova (N150) - no systemd, manual start
ssh george@192.168.12.149
# (start connector manually on Nova)
```

## REST API (v1)

```bash
# List agents
curl -s http://localhost:8080/api/v1/agents | python3 -m json.tool

# Get specific agent
curl -s http://localhost:8080/api/v1/agents/firehorseclawd | python3 -m json.tool

# Send command to agent
curl -s -X POST http://localhost:8080/api/v1/agents/nova/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status"}'

# Submit a task
curl -s -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "Research Yamaha MT-07 common issues"}'

# Kill switch (emergency stop)
curl -s -X POST http://localhost:8080/api/v1/kill-switch
```

## Validation & Testing

```bash
cd ~/CAM

# Full v1.0 validation suite
python v1_validation.py

# Single category
python v1_validation.py --category core

# Push results to dashboard
python v1_validation.py --dashboard

# Categories: core, infrastructure, business, content, intelligence, operations
```

## Agent Deployment

```bash
# Install agent on a new machine (interactive)
sudo ./deploy/install_agent.sh

# Non-interactive install
sudo ./deploy/install_agent.sh --agent-name Nova --agent-id nova \
  --dashboard-ip 192.168.12.232 --capabilities research content

# Prepare USB bundle for offline install
./deploy/install_agent.sh --prepare-usb /media/usb/cam-agent
```
