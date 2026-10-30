#!/usr/bin/env bash
# Install the api-ai models diff alert on a host. Run as root. Idempotent.
#
# Seit #5740 (30.10.2026) gibt es keine Modell-Discovery in api-ai mehr:
# Die Listen kommen aus der CLI-Erhebung der cloud-api (cli-models.json),
# das alte Discovery-Skript (codex exec, veraltete Kandidaten, verbrauchte
# Kontingent) und seine Units sind entfernt. Uebrig bleibt der 10:00-Alarm,
# der models.json mit models.prev.json vergleicht, solange Automation pusht.
set -euo pipefail

SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPTS_DIR="$(cd "$SRC_DIR/../scripts" && pwd -P)"

echo "→ removing retired discovery units"
for u in api-ai-maintenance.timer api-ai-maintenance-discover.timer \
         api-ai-maintenance.service api-ai-maintenance-discover.service; do
  systemctl disable --now "$u" 2>/dev/null || true
  rm -f "/etc/systemd/system/$u" "/etc/systemd/system/timers.target.wants/$u"
done
rm -f /usr/local/bin/api-ai-models-discovery.py

echo "→ installing alert"
install -o root -g root -m 0755 "$SCRIPTS_DIR/api-ai-models-diff-alert.py" /usr/local/bin/api-ai-models-diff-alert.py
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance-alert.service" /etc/systemd/system/api-ai-maintenance-alert.service
install -o root -g root -m 0644 "$SRC_DIR/api-ai-maintenance-alert.timer"   /etc/systemd/system/api-ai-maintenance-alert.timer
install -o root -g root -m 0755 -d /var/lib/api-ai

systemctl daemon-reload
systemctl enable --now api-ai-maintenance-alert.timer
systemctl list-timers api-ai-maintenance-alert.timer --no-pager | sed -n '2p'
