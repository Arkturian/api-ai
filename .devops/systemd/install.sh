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
# Keep the legacy all-in-one service for manual full-chain runs
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance.service" /etc/systemd/system/api-ai-maintenance.service
# Daily 03:00 discover (silent) + 10:00 alert (Telegram during the day)
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance-discover.service" /etc/systemd/system/api-ai-maintenance-discover.service
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance-discover.timer"   /etc/systemd/system/api-ai-maintenance-discover.timer
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance-alert.service"    /etc/systemd/system/api-ai-maintenance-alert.service
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance-alert.timer"      /etc/systemd/system/api-ai-maintenance-alert.timer
# Remove the obsolete combined timer if present (we now split discover + alert)
if systemctl list-unit-files api-ai-maintenance.timer --no-legend 2>/dev/null | grep -q .; then
  echo "→ disabling legacy combined timer (replaced by discover+alert split)"
  systemctl disable --now api-ai-maintenance.timer 2>/dev/null || true
  rm -f /etc/systemd/system/timers.target.wants/api-ai-maintenance.timer
fi
# The legacy .timer file itself stays uninstalled — only the .service remains
# for manual full-chain triggers via `systemctl start api-ai-maintenance.service`.
rm -f /etc/systemd/system/api-ai-maintenance.timer

echo "→ preparing state dir"
install -o root -g root -m 0755 -d /var/lib/api-ai
touch /var/log/api-ai-maintenance.log
chmod 0640 /var/log/api-ai-maintenance.log
chown root:adm /var/log/api-ai-maintenance.log 2>/dev/null || true

echo "→ enabling timers"
systemctl daemon-reload
systemctl enable --now api-ai-maintenance-discover.timer
systemctl enable --now api-ai-maintenance-alert.timer

echo
echo "✓ install complete"
echo "  discover timer (03:00 silent):"
systemctl list-timers api-ai-maintenance-discover.timer --no-pager | sed -n '2p'
echo "  alert timer    (10:00 Telegram):"
systemctl list-timers api-ai-maintenance-alert.timer --no-pager | sed -n '2p'
echo
echo "  manual full chain:  systemctl start api-ai-maintenance.service"
echo "  manual discover:    systemctl start api-ai-maintenance-discover.service"
echo "  manual alert:       systemctl start api-ai-maintenance-alert.service"
echo "  tail log:           tail -f /var/log/api-ai-maintenance.log"
