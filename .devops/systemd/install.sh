#!/usr/bin/env bash
# Install api-ai maintenance scripts + systemd timer on a host. Run as root.
# Idempotent — safe to re-run.
set -euo pipefail

SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPTS_DIR="$(cd "$SRC_DIR/../scripts" && pwd -P)"

echo "→ installing scripts to /usr/local/bin/"
install -o root -g root -m 0755 "$SCRIPTS_DIR/api-ai-venv-update.sh"        /usr/local/bin/api-ai-venv-update.sh
install -o root -g root -m 0755 "$SCRIPTS_DIR/api-ai-models-discovery.py"  /usr/local/bin/api-ai-models-discovery.py
install -o root -g root -m 0755 "$SCRIPTS_DIR/api-ai-models-diff-alert.py" /usr/local/bin/api-ai-models-diff-alert.py

echo "→ installing systemd units"
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance.service" /etc/systemd/system/api-ai-maintenance.service
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance.timer"   /etc/systemd/system/api-ai-maintenance.timer

echo "→ preparing state dir"
install -o root -g root -m 0755 -d /var/lib/api-ai
touch /var/log/api-ai-maintenance.log
chmod 0640 /var/log/api-ai-maintenance.log
chown root:adm /var/log/api-ai-maintenance.log 2>/dev/null || true

echo "→ enabling timer"
systemctl daemon-reload
systemctl enable --now api-ai-maintenance.timer

echo
echo "✓ install complete"
echo "  next run:    $(systemctl list-timers api-ai-maintenance.timer --no-pager | sed -n '2p')"
echo "  manual run:  systemctl start api-ai-maintenance.service"
echo "  tail log:    tail -f /var/log/api-ai-maintenance.log"
