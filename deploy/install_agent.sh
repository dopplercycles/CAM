#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# CAM Agent Installer
# Deploys a CAM agent connector on Ubuntu, Debian, or Raspbian.
#
# Usage:
#   Standalone:   ./install_agent.sh
#   One-liner:    curl -fsSL https://raw.githubusercontent.com/dopplercycles/cam/main/deploy/install_agent.sh | bash
#   USB mode:     ./install_agent.sh --usb /media/usb
#   Prepare USB:  ./install_agent.sh --prepare-usb /media/usb
#   Non-interactive: ./install_agent.sh --agent-name Nova --agent-id nova \
#                     --dashboard-ip 192.168.88.4 --capabilities research content
#
# Doppler Cycles — https://github.com/dopplercycles/cam
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────

CAM_VERSION="0.1.0"
CONNECTOR_URL="https://raw.githubusercontent.com/dopplercycles/cam/main/agents/connector.py"
CONFIG_WIZARD_URL="https://raw.githubusercontent.com/dopplercycles/cam/main/deploy/agent_config.py"
INSTALL_DIR="/opt/cam-agent"
SERVICE_NAME="cam-agent"
SERVICE_USER=""          # detected or prompted
VENV_DIR="${INSTALL_DIR}/venv"

# CLI overrides (for non-interactive mode)
ARG_AGENT_NAME=""
ARG_AGENT_ID=""
ARG_DASHBOARD_IP=""
ARG_DASHBOARD_PORT="8080"
ARG_CAPABILITIES=""
ARG_USB_PATH=""
ARG_PREPARE_USB=""
ARG_QUIET=false

# ── Colors ──────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ─────────────────────────────────────────────────────────

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
fatal() { err "$*"; exit 1; }

banner() {
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║       CAM Agent Installer v${CAM_VERSION}            ║${NC}"
    echo -e "${CYAN}${BOLD}║       Doppler Cycles                        ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════╝${NC}"
    echo ""
}

need_root() {
    if [[ $EUID -ne 0 ]]; then
        fatal "This installer must be run as root (try: sudo $0)"
    fi
}

# ── Parse arguments ─────────────────────────────────────────────────

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --agent-name)       ARG_AGENT_NAME="$2";      shift 2 ;;
            --agent-id)         ARG_AGENT_ID="$2";         shift 2 ;;
            --dashboard-ip)     ARG_DASHBOARD_IP="$2";     shift 2 ;;
            --dashboard-port)   ARG_DASHBOARD_PORT="$2";   shift 2 ;;
            --capabilities)     shift; ARG_CAPABILITIES=""
                                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                                    ARG_CAPABILITIES="${ARG_CAPABILITIES} $1"; shift
                                done ;;
            --usb)              ARG_USB_PATH="$2";         shift 2 ;;
            --prepare-usb)      ARG_PREPARE_USB="$2";      shift 2 ;;
            --quiet|-q)         ARG_QUIET=true;            shift ;;
            --help|-h)          show_help; exit 0 ;;
            *)                  warn "Unknown option: $1";  shift ;;
        esac
    done
}

show_help() {
    cat <<'HELP'
CAM Agent Installer

USAGE:
    sudo ./install_agent.sh [OPTIONS]

OPTIONS:
    --agent-name NAME       Agent display name (e.g., "FireHorseClawd")
    --agent-id ID           Agent identifier (e.g., "firehorseclawd")
    --dashboard-ip IP       Dashboard server IP (e.g., "192.168.88.4")
    --dashboard-port PORT   Dashboard port (default: 8080)
    --capabilities CAP...   Space-separated capabilities (e.g., research content tts)
    --usb PATH              Install from USB bundle at PATH (offline mode)
    --prepare-usb PATH      Prepare a USB drive for offline deployment
    --quiet, -q             Suppress interactive prompts (use defaults)
    --help, -h              Show this help

EXAMPLES:
    # Interactive install
    sudo ./install_agent.sh

    # Non-interactive install
    sudo ./install_agent.sh --agent-name Nova --agent-id nova \
        --dashboard-ip 192.168.88.4 --capabilities research content

    # Install from USB (no internet required)
    sudo ./install_agent.sh --usb /media/usb/cam-agent

    # Prepare a USB drive for offline installs
    sudo ./install_agent.sh --prepare-usb /media/usb/cam-agent
HELP
}

# ── OS Detection ────────────────────────────────────────────────────

detect_os() {
    info "Detecting operating system..."

    if [[ ! -f /etc/os-release ]]; then
        fatal "Cannot detect OS — /etc/os-release not found"
    fi

    # shellcheck disable=SC1091
    source /etc/os-release

    OS_ID="${ID:-unknown}"
    OS_VERSION="${VERSION_ID:-unknown}"
    OS_NAME="${PRETTY_NAME:-unknown}"
    OS_ARCH=$(uname -m)

    # Normalize to family
    case "$OS_ID" in
        ubuntu|pop)             OS_FAMILY="ubuntu" ;;
        debian)                 OS_FAMILY="debian" ;;
        raspbian)               OS_FAMILY="raspbian" ;;
        *)
            if [[ -f /etc/rpi-issue ]] || grep -qi raspberry /proc/cpuinfo 2>/dev/null; then
                OS_FAMILY="raspbian"
            elif [[ "$ID_LIKE" == *debian* ]]; then
                OS_FAMILY="debian"
            else
                fatal "Unsupported OS: ${OS_NAME}. Supported: Ubuntu, Debian, Raspbian"
            fi
            ;;
    esac

    ok "Detected: ${OS_NAME} (${OS_FAMILY}/${OS_ARCH})"
}

# ── Dependency Installation ─────────────────────────────────────────

install_dependencies() {
    info "Installing system dependencies..."

    # Update package lists
    apt-get update -qq

    # Core packages
    local packages=(
        python3
        python3-pip
        python3-venv
        avahi-daemon
        avahi-utils
        libnss-mdns
        curl
    )

    # Raspbian may need these
    if [[ "$OS_FAMILY" == "raspbian" ]]; then
        packages+=(python3-dev)
    fi

    apt-get install -y -qq "${packages[@]}"

    # Ensure avahi is running
    systemctl enable avahi-daemon 2>/dev/null || true
    systemctl start avahi-daemon 2>/dev/null || true

    ok "System dependencies installed"
}

install_dependencies_usb() {
    local usb_path="$1"
    info "Installing from USB bundle at ${usb_path}..."

    if [[ ! -d "${usb_path}" ]]; then
        fatal "USB bundle not found at ${usb_path}"
    fi

    # Install .deb packages if bundled
    if [[ -d "${usb_path}/debs" ]]; then
        info "Installing bundled .deb packages..."
        dpkg -i "${usb_path}/debs/"*.deb 2>/dev/null || true
        apt-get install -f -y -qq 2>/dev/null || true
        ok "Bundled packages installed"
    fi

    ok "USB dependencies processed"
}

# ── Directory Setup ─────────────────────────────────────────────────

setup_directories() {
    info "Setting up ${INSTALL_DIR}..."

    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${INSTALL_DIR}/config"
    mkdir -p "${INSTALL_DIR}/logs"
    mkdir -p "${INSTALL_DIR}/receive"

    ok "Directory structure created"
}

# ── Detect service user ────────────────────────────────────────────

detect_user() {
    # Try to find the right user to run the agent as
    if [[ -n "$SUDO_USER" && "$SUDO_USER" != "root" ]]; then
        SERVICE_USER="$SUDO_USER"
    elif id -u pi &>/dev/null; then
        SERVICE_USER="pi"
    elif id -u george &>/dev/null; then
        SERVICE_USER="george"
    else
        SERVICE_USER="$(logname 2>/dev/null || echo root)"
    fi
    info "Service user: ${SERVICE_USER}"
}

# ── Python Environment ──────────────────────────────────────────────

setup_python() {
    info "Setting up Python virtual environment..."

    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet --upgrade pip

    if [[ -n "$ARG_USB_PATH" && -d "${ARG_USB_PATH}/wheels" ]]; then
        # Install from bundled wheels (offline)
        "${VENV_DIR}/bin/pip" install --quiet --no-index \
            --find-links "${ARG_USB_PATH}/wheels" websockets
        ok "Python packages installed from USB"
    else
        # Install from PyPI
        "${VENV_DIR}/bin/pip" install --quiet websockets
        ok "Python packages installed from PyPI"
    fi
}

# ── Connector Deployment ────────────────────────────────────────────

deploy_connector() {
    info "Deploying connector..."

    local connector_dest="${INSTALL_DIR}/connector.py"

    if [[ -n "$ARG_USB_PATH" && -f "${ARG_USB_PATH}/connector.py" ]]; then
        # Copy from USB
        cp "${ARG_USB_PATH}/connector.py" "${connector_dest}"
        ok "Connector copied from USB"
    elif [[ -f "$(dirname "$0")/../agents/connector.py" ]]; then
        # Copy from local repo
        cp "$(dirname "$0")/../agents/connector.py" "${connector_dest}"
        ok "Connector copied from local repo"
    else
        # Download from GitHub
        info "Downloading connector from GitHub..."
        curl -fsSL "${CONNECTOR_URL}" -o "${connector_dest}"
        ok "Connector downloaded"
    fi

    chmod 644 "${connector_dest}"
}

deploy_config_wizard() {
    local wizard_dest="${INSTALL_DIR}/agent_config.py"

    if [[ -n "$ARG_USB_PATH" && -f "${ARG_USB_PATH}/agent_config.py" ]]; then
        cp "${ARG_USB_PATH}/agent_config.py" "${wizard_dest}"
    elif [[ -f "$(dirname "$0")/agent_config.py" ]]; then
        cp "$(dirname "$0")/agent_config.py" "${wizard_dest}"
    else
        curl -fsSL "${CONFIG_WIZARD_URL}" -o "${wizard_dest}"
    fi

    chmod 755 "${wizard_dest}"
}

# ── Configuration ───────────────────────────────────────────────────

run_config() {
    info "Running configuration..."

    local config_file="${INSTALL_DIR}/config/agent.conf"
    local wizard="${INSTALL_DIR}/agent_config.py"

    if [[ -n "$ARG_AGENT_NAME" && -n "$ARG_AGENT_ID" && -n "$ARG_DASHBOARD_IP" ]]; then
        # Non-interactive mode — write config directly
        info "Non-interactive mode — using provided values"
        cat > "${config_file}" <<CONF
# CAM Agent Configuration
# Generated by install_agent.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ')

AGENT_NAME="${ARG_AGENT_NAME}"
AGENT_ID="${ARG_AGENT_ID}"
DASHBOARD_IP="${ARG_DASHBOARD_IP}"
DASHBOARD_PORT="${ARG_DASHBOARD_PORT}"
DASHBOARD_URL="ws://${ARG_DASHBOARD_IP}:${ARG_DASHBOARD_PORT}"
CAPABILITIES="${ARG_CAPABILITIES## }"
HEARTBEAT_INTERVAL="30"
RECONNECT_DELAY="3"
CONF
        ok "Configuration written to ${config_file}"
    else
        # Interactive mode — run the wizard
        deploy_config_wizard
        "${VENV_DIR}/bin/python3" "${wizard}" --output "${config_file}"
    fi

    # Source the config for use in this script
    # shellcheck disable=SC1090
    source "${config_file}"
}

# ── mDNS Hostname ───────────────────────────────────────────────────

configure_mdns() {
    info "Configuring mDNS hostname..."

    # Set hostname to agent ID if it's a fresh install
    local target_hostname="${AGENT_ID}"

    # Only change if current hostname is generic
    local current_hostname
    current_hostname=$(hostname)
    if [[ "$current_hostname" == "raspberrypi" || "$current_hostname" == "localhost" || "$current_hostname" == "debian" ]]; then
        info "Updating hostname: ${current_hostname} -> ${target_hostname}"
        hostnamectl set-hostname "${target_hostname}" 2>/dev/null || \
            echo "${target_hostname}" > /etc/hostname

        # Update /etc/hosts
        sed -i "s/127\.0\.1\.1.*/127.0.1.1\t${target_hostname}/" /etc/hosts 2>/dev/null || true
        if ! grep -q "127.0.1.1" /etc/hosts; then
            echo "127.0.1.1	${target_hostname}" >> /etc/hosts
        fi

        ok "Hostname set to ${target_hostname} (${target_hostname}.local via mDNS)"
    else
        info "Keeping existing hostname: ${current_hostname} (${current_hostname}.local)"
    fi

    # Ensure Avahi advertises the SSH service (useful for discovery)
    local avahi_ssh="/etc/avahi/services/ssh.service"
    if [[ ! -f "${avahi_ssh}" ]]; then
        cat > "${avahi_ssh}" <<'AVAHI'
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">%h SSH</name>
  <service>
    <type>_ssh._tcp</type>
    <port>22</port>
  </service>
</service-group>
AVAHI
        info "Avahi SSH service file created"
    fi

    # Advertise CAM agent service via mDNS
    local avahi_cam="/etc/avahi/services/cam-agent.service"
    cat > "${avahi_cam}" <<AVAHI
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">%h CAM Agent</name>
  <service>
    <type>_cam-agent._tcp</type>
    <port>0</port>
    <txt-record>agent_id=${AGENT_ID}</txt-record>
    <txt-record>version=${CAM_VERSION}</txt-record>
  </service>
</service-group>
AVAHI
    ok "mDNS CAM agent service advertised"

    # Restart avahi to pick up changes
    systemctl restart avahi-daemon 2>/dev/null || true
}

# ── Systemd Service ─────────────────────────────────────────────────

create_service() {
    info "Creating systemd service..."

    local service_file="/etc/systemd/system/${SERVICE_NAME}.service"
    local connector="${INSTALL_DIR}/connector.py"
    local python="${VENV_DIR}/bin/python3"

    # Build capabilities argument
    local caps_arg=""
    if [[ -n "${CAPABILITIES:-}" ]]; then
        caps_arg="--capabilities ${CAPABILITIES}"
    fi

    cat > "${service_file}" <<SERVICE
[Unit]
Description=CAM Agent Connector (${AGENT_NAME})
Documentation=https://github.com/dopplercycles/cam
After=network-online.target avahi-daemon.service
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${python} ${connector} \\
    --dashboard ${DASHBOARD_URL} \\
    --name "${AGENT_NAME}" \\
    --id ${AGENT_ID} \\
    --receive-dir ${INSTALL_DIR}/receive \\
    ${caps_arg}
Restart=always
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cam-agent

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=${INSTALL_DIR}

[Install]
WantedBy=multi-user.target
SERVICE

    # Fix ownership
    chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

    # Reload and enable
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}"

    ok "Systemd service created: ${SERVICE_NAME}"
}

# ── Connection Test ─────────────────────────────────────────────────

test_connection() {
    info "Testing connection to dashboard..."

    local dashboard_host="${DASHBOARD_IP}"
    local dashboard_port="${DASHBOARD_PORT}"

    # Test TCP connectivity
    if timeout 5 bash -c "echo > /dev/tcp/${dashboard_host}/${dashboard_port}" 2>/dev/null; then
        ok "Dashboard reachable at ${dashboard_host}:${dashboard_port}"
    else
        warn "Cannot reach dashboard at ${dashboard_host}:${dashboard_port}"
        warn "The agent will auto-reconnect when the dashboard is available."
        return 0
    fi

    # Test mDNS resolution (cam.local)
    if avahi-resolve-host-name cam.local &>/dev/null; then
        ok "mDNS: cam.local resolves successfully"
    else
        info "mDNS: cam.local not found (dashboard may not have Avahi configured)"
    fi

    # Quick WebSocket test via the connector
    info "Starting agent for connection test (5 seconds)..."
    local test_result
    if timeout 8 "${VENV_DIR}/bin/python3" "${INSTALL_DIR}/connector.py" \
        --dashboard "${DASHBOARD_URL}" \
        --name "${AGENT_NAME}" \
        --id "${AGENT_ID}" \
        --interval 2 2>&1 | head -5 | grep -qi "connected\|heartbeat\|websocket"; then
        ok "WebSocket connection successful"
    else
        warn "Could not verify WebSocket connection (dashboard may be offline)"
        info "The service will keep retrying automatically."
    fi
}

# ── Start Service ───────────────────────────────────────────────────

start_service() {
    info "Starting ${SERVICE_NAME}..."
    systemctl start "${SERVICE_NAME}"

    # Wait a moment and check status
    sleep 2
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        ok "Service ${SERVICE_NAME} is running"
    else
        warn "Service may still be starting. Check: journalctl -u ${SERVICE_NAME} -f"
    fi
}

# ── USB Preparation ─────────────────────────────────────────────────

prepare_usb() {
    local usb_path="$1"
    info "Preparing USB deployment bundle at ${usb_path}..."

    mkdir -p "${usb_path}"
    mkdir -p "${usb_path}/wheels"

    # Copy connector
    local connector_src
    if [[ -f "$(dirname "$0")/../agents/connector.py" ]]; then
        connector_src="$(dirname "$0")/../agents/connector.py"
    else
        info "Downloading connector..."
        connector_src="/tmp/cam_connector.py"
        curl -fsSL "${CONNECTOR_URL}" -o "${connector_src}"
    fi
    cp "${connector_src}" "${usb_path}/connector.py"

    # Copy config wizard
    local wizard_src
    if [[ -f "$(dirname "$0")/agent_config.py" ]]; then
        wizard_src="$(dirname "$0")/agent_config.py"
    else
        info "Downloading config wizard..."
        wizard_src="/tmp/cam_agent_config.py"
        curl -fsSL "${CONFIG_WIZARD_URL}" -o "${wizard_src}"
    fi
    cp "${wizard_src}" "${usb_path}/agent_config.py"

    # Copy this installer
    cp "$0" "${usb_path}/install_agent.sh"
    chmod +x "${usb_path}/install_agent.sh"

    # Download wheels for offline pip install
    info "Downloading Python wheels for offline install..."
    local tmpvenv="/tmp/cam_usb_venv"
    python3 -m venv "${tmpvenv}"
    "${tmpvenv}/bin/pip" download --dest "${usb_path}/wheels" websockets
    rm -rf "${tmpvenv}"

    # Download system .debs (best-effort, platform-specific)
    info "Bundling system packages..."
    mkdir -p "${usb_path}/debs"
    apt-get download python3 python3-pip python3-venv \
        avahi-daemon avahi-utils libnss-mdns 2>/dev/null || \
        warn "Could not bundle all .deb packages (install target may need internet for apt)"

    # Move any downloaded .debs
    mv ./*.deb "${usb_path}/debs/" 2>/dev/null || true

    # Create README
    cat > "${usb_path}/README.txt" <<'README'
CAM Agent USB Deployment Bundle
===============================

To install on a target machine:

    sudo ./install_agent.sh --usb /path/to/this/directory

Or mount the USB and run:

    cd /media/usb/cam-agent
    sudo ./install_agent.sh --usb .

The bundle includes:
  - connector.py      — The CAM agent connector
  - agent_config.py   — Configuration wizard
  - install_agent.sh  — This installer
  - wheels/           — Python packages for offline install
  - debs/             — System packages (platform-specific)

Doppler Cycles — https://github.com/dopplercycles/cam
README

    ok "USB bundle prepared at ${usb_path}"
    echo ""
    info "Contents:"
    ls -la "${usb_path}/"
    echo ""
    info "To deploy: plug USB into target machine and run:"
    echo -e "  ${BOLD}sudo ${usb_path}/install_agent.sh --usb ${usb_path}${NC}"
}

# ── Summary ─────────────────────────────────────────────────────────

show_summary() {
    echo ""
    echo -e "${GREEN}${BOLD}════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  CAM Agent Installation Complete${NC}"
    echo -e "${GREEN}${BOLD}════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Agent Name:     ${BOLD}${AGENT_NAME}${NC}"
    echo -e "  Agent ID:       ${BOLD}${AGENT_ID}${NC}"
    echo -e "  Dashboard:      ${BOLD}${DASHBOARD_URL}${NC}"
    echo -e "  Capabilities:   ${BOLD}${CAPABILITIES:-none}${NC}"
    echo -e "  Install Dir:    ${BOLD}${INSTALL_DIR}${NC}"
    echo -e "  Service:        ${BOLD}${SERVICE_NAME}${NC}"
    echo -e "  Service User:   ${BOLD}${SERVICE_USER}${NC}"
    echo -e "  mDNS:           ${BOLD}$(hostname).local${NC}"
    echo ""
    echo -e "  ${CYAN}Useful commands:${NC}"
    echo -e "    journalctl -u ${SERVICE_NAME} -f     ${BLUE}# Follow agent logs${NC}"
    echo -e "    systemctl status ${SERVICE_NAME}      ${BLUE}# Check service status${NC}"
    echo -e "    systemctl restart ${SERVICE_NAME}     ${BLUE}# Restart agent${NC}"
    echo -e "    systemctl stop ${SERVICE_NAME}        ${BLUE}# Stop agent${NC}"
    echo ""
    echo -e "  Config: ${BOLD}${INSTALL_DIR}/config/agent.conf${NC}"
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────

main() {
    parse_args "$@"

    # Handle USB preparation (doesn't need root for bundle creation)
    if [[ -n "$ARG_PREPARE_USB" ]]; then
        banner
        prepare_usb "$ARG_PREPARE_USB"
        exit 0
    fi

    banner
    need_root
    detect_os
    detect_user

    # Install dependencies
    if [[ -n "$ARG_USB_PATH" ]]; then
        install_dependencies_usb "$ARG_USB_PATH"
        install_dependencies  # Still need apt for anything not bundled
    else
        install_dependencies
    fi

    setup_directories
    setup_python
    deploy_connector

    # Configuration
    run_config

    # System setup
    configure_mdns
    create_service

    # Test and start
    test_connection
    start_service

    show_summary
}

main "$@"
