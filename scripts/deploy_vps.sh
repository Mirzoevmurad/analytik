#!/usr/bin/env bash
# Idempotent deploy of analytik-bot to VPS as a systemd service.
# Requires: Ubuntu/Debian, root, ANALYTIK_BOT_TOKEN and GROQ_API_KEY in environment.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Mirzoevmurad/analytik.git}"
BRANCH="${BRANCH:-init}"
APP_USER="${APP_USER:-analytik}"
APP_DIR="${APP_DIR:-/opt/analytik-bot}"
ENV_FILE="${ENV_FILE:-/etc/analytik-bot.env}"
SERVICE_NAME="${SERVICE_NAME:-analytik-bot}"

log() { echo -e "\033[1m[analytik]\033[0m $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root (sudo)"

# Save user-provided env vars (they take priority over env file).
declare -A USER_ENV
for v in ANALYTIK_BOT_TOKEN GROQ_API_KEY ALLOWED_USER_IDS \
         STT_MODEL LLM_MODEL DEFAULT_LANG MAX_AUDIO_MB \
         MAX_CONTEXT_MESSAGES VOICE_RESPONSES; do
    if [[ -n "${!v:-}" ]]; then
        USER_ENV[$v]="${!v}"
    fi
done

# Source existing env file if present.
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Restore user-provided vars (priority).
for v in "${!USER_ENV[@]}"; do
    export "$v"="${USER_ENV[$v]}"
done

[[ -n "${ANALYTIK_BOT_TOKEN:-}" ]] || die "ANALYTIK_BOT_TOKEN not set (pass in env or prepare $ENV_FILE)"
[[ -n "${GROQ_API_KEY:-}" ]]      || die "GROQ_API_KEY not set (pass in env or prepare $ENV_FILE)"
[[ -n "${ALLOWED_USER_IDS:-}" ]]   || die "ALLOWED_USER_IDS not set (pass in env or prepare $ENV_FILE)"

log "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git ffmpeg ca-certificates

if ! id "$APP_USER" &>/dev/null; then
    log "Creating system user $APP_USER..."
    useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

if [[ -d "$APP_DIR/.git" ]]; then
    log "Updating repo in $APP_DIR..."
    sudo -u "$APP_USER" git -C "$APP_DIR" remote set-url origin "$REPO_URL"
    sudo -u "$APP_USER" git -C "$APP_DIR" fetch --depth 1 origin "$BRANCH"
    sudo -u "$APP_USER" git -C "$APP_DIR" reset --hard FETCH_HEAD
else
    log "Cloning $REPO_URL -> $APP_DIR..."
    mkdir -p "$APP_DIR"
    chown "$APP_USER:$APP_USER" "$APP_DIR"
    sudo -u "$APP_USER" git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

log "Installing Python dependencies in venv..."
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade -q pip wheel
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

log "Writing $ENV_FILE (chmod 600)..."
cat >"$ENV_FILE" <<ENV
ANALYTIK_BOT_TOKEN=${ANALYTIK_BOT_TOKEN}
GROQ_API_KEY=${GROQ_API_KEY}
ALLOWED_USER_IDS=${ALLOWED_USER_IDS}
DB_PATH=${APP_DIR}/data/analytik.sqlite
STT_MODEL=${STT_MODEL:-whisper-large-v3-turbo}
LLM_MODEL=${LLM_MODEL:-llama-3.3-70b-versatile}
DEFAULT_LANG=${DEFAULT_LANG:-auto}
MAX_AUDIO_MB=${MAX_AUDIO_MB:-25}
MAX_CONTEXT_MESSAGES=${MAX_CONTEXT_MESSAGES:-20}
VOICE_RESPONSES=${VOICE_RESPONSES:-false}
ENV
chmod 600 "$ENV_FILE"
chown root:"$APP_USER" "$ENV_FILE"

log "Creating systemd unit /etc/systemd/system/${SERVICE_NAME}.service..."
cat >"/etc/systemd/system/${SERVICE_NAME}.service" <<UNIT
[Unit]
Description=Analytik Telegram bot (Groq LLM data analyst)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=HOME=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/bot.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=false
PrivateTmp=true

[Install]
WantedBy=multi-user.target
UNIT

mkdir -p "$APP_DIR/data"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/data"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}" >/dev/null
systemctl restart "${SERVICE_NAME}"

sleep 3
if systemctl is-active --quiet "${SERVICE_NAME}"; then
    log "Service ${SERVICE_NAME} is running. Send /start to your bot."
else
    log "ERROR: service failed to start. Logs:"
    journalctl -u "${SERVICE_NAME}" -n 30 --no-pager
    exit 1
fi

echo
echo "============================================================"
log "Done."
echo "Status:    systemctl status ${SERVICE_NAME}"
echo "Logs:      journalctl -u ${SERVICE_NAME} -f"
echo "Update:    sudo bash $0"
echo "============================================================"
