#!/bin/bash
# Calendar Management System — Deploy Script (with watchdog)
# Usage: bash deploy.sh [start|stop|restart|status|logs|watchdog]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# If run from skill dir, point to actual scripts location
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
RUN_DIR="$HERMES_HOME/scripts"
VENV="$HERMES_HOME/hermes-agent/venv/bin/python3"
LOG_DIR="/tmp"

WATCHDOG_PID_FILE="$SCRIPT_DIR/.watchdog.pid"

# Use running scripts if they exist, otherwise use skill copies
TG_SCRIPT="$RUN_DIR/telegram_calendar_agent.py"
AM_SCRIPT="$RUN_DIR/agentmail_calendar_worker.py"
[ -f "$TG_SCRIPT" ] || TG_SCRIPT="$SCRIPT_DIR/telegram_calendar_agent.py"
[ -f "$AM_SCRIPT" ] || AM_SCRIPT="$SCRIPT_DIR/agentmail_calendar_worker.py"

is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

start_process() {
    local script="$1" name="$2"
    if is_running "$(basename $script)"; then
        echo "[SKIP] $name already running"
    else
        PYTHONUNBUFFERED=1 nohup "$VENV" "$script" \
            >> "$LOG_DIR/$(basename $script .py).log" 2>&1 &
        echo "[OK] $name started (PID $!)"
    fi
}

stop_process() {
    local name="$1"
    pkill -f "$name" 2>/dev/null && echo "[OK] Stopped $name" || echo "[SKIP] $name not running"
}

start() {
    start_process "$TG_SCRIPT" "Telegram Agent"
    start_process "$AM_SCRIPT" "AgentMail Worker"
}

stop() {
    # Stop watchdog first
    if [ -f "$WATCHDOG_PID_FILE" ]; then
        local wp=$(cat "$WATCHDOG_PID_FILE")
        kill "$wp" 2>/dev/null && echo "[OK] Stopped watchdog (PID $wp)" || true
        rm -f "$WATCHDOG_PID_FILE"
    fi
    stop_process "telegram_calendar_agent.py"
    stop_process "agentmail_calendar_worker.py"
}

status() {
    echo "=== Telegram Agent ==="
    if is_running "telegram_calendar_agent.py"; then
        echo "Running (PID $(pgrep -f telegram_calendar_agent.py))"
    else
        echo "Not running"
    fi
    echo ""
    echo "=== AgentMail Worker ==="
    if is_running "agentmail_calendar_worker.py"; then
        echo "Running (PID $(pgrep -f agentmail_calendar_worker.py))"
    else
        echo "Not running"
    fi
    echo ""
    echo "=== Watchdog ==="
    if [ -f "$WATCHDOG_PID_FILE" ] && kill -0 "$(cat $WATCHDOG_PID_FILE)" 2>/dev/null; then
        echo "Running (PID $(cat $WATCHDOG_PID_FILE))"
    else
        echo "Not running"
    fi
}

logs() {
    echo "=== Telegram Agent (last 20 lines) ==="
    tail -20 "$LOG_DIR/telegram_calendar_agent.log" 2>/dev/null || echo "(no log)"
    echo ""
    echo "=== AgentMail Worker (last 20 lines) ==="
    tail -20 "$LOG_DIR/agentmail_calendar_worker.log" 2>/dev/null || echo "(no log)"
}

watchdog() {
    echo "[WATCHDOG] Started (PID $$), checking every 30s"
    echo $$ > "$WATCHDOG_PID_FILE"
    trap "rm -f $WATCHDOG_PID_FILE; exit 0" SIGTERM SIGINT

    while true; do
        if ! is_running "telegram_calendar_agent.py"; then
            echo "[$(date '+%H:%M:%S')] [WATCHDOG] Telegram Agent down, restarting..."
            start_process "$TG_SCRIPT" "Telegram Agent"
        fi
        if ! is_running "agentmail_calendar_worker.py"; then
            echo "[$(date '+%H:%M:%S')] [WATCHDOG] AgentMail Worker down, restarting..."
            start_process "$AM_SCRIPT" "AgentMail Worker"
        fi
        sleep 30
    done
}

case "${1:-start}" in
    start)
        start
        # Start watchdog in background
        if ! is_running "deploy.sh.*watchdog"; then
            nohup bash "$SCRIPT_DIR/deploy.sh" watchdog \
                >> "$LOG_DIR/calendar_watchdog.log" 2>&1 &
            echo "[OK] Watchdog started (PID $!)"
        fi
        ;;
    stop)    stop ;;
    restart) stop; sleep 1; "$0" start ;;
    status)  status ;;
    logs)    logs ;;
    watchdog) watchdog ;;
    *)       echo "Usage: $0 {start|stop|restart|status|logs}"; exit 1 ;;
esac
