#!/bin/bash
# #3165 Schritt B: api-ai auf apiai-codex umschalten. Als root auf arkserver,
# nach Schritt A und nach dem Release, das codex_konten enthaelt.
#
#   sudo bash deploy/apiai/3165-umschalten.sh "tsaier"            # nur angemeldete Konten
#   sudo bash deploy/apiai/3165-umschalten.sh "apopovic,tsaier"
#   sudo bash deploy/apiai/3165-umschalten.sh --zurueck            # Rueckweg
#
# Danach kennt sudoers keinen Weg mehr von apiai zu alex, und die Sandbox des
# Dienstes schreibt nicht mehr in /home/alex.
set -euo pipefail
umask 077
REPO=$(cd "$(dirname "$0")/../.." && pwd)
D=/etc/systemd/system/api-ai.service.d
DROPIN=$D/zzzz-codex-konten.conf
SUDOERS=/etc/sudoers.d/api-ai
SICHER=/root/backups/3165-$(date +%Y%m%d-%H%M%S)

if [[ "${1:-}" == "--zurueck" ]]; then
  letzte=$(ls -d /root/backups/3165-* | tail -1)
  install -m 0440 -o root -g root "$letzte/sudoers-api-ai" "$SUDOERS"
  visudo -cf "$SUDOERS"
  rm -f "$DROPIN"
  systemctl daemon-reload && systemctl restart api-ai
  echo "zurueck auf codex-als-alex (Sicherung $letzte)"; exit 0
fi

KONTEN=${1:?Konten angeben, z. B. tsaier oder apopovic,tsaier}
[[ "$KONTEN" =~ ^[a-z][a-z0-9]{1,31}(,[a-z][a-z0-9]{1,31})*$ ]] || { echo "Kontenliste" >&2; exit 64; }
for k in ${KONTEN//,/ }; do
  [[ -s /var/lib/api-ai-codex/konten/$k/auth.json ]] || { echo "$k ist nicht angemeldet" >&2; exit 1; }
done

install -d -m 0700 "$SICHER"
cp -p "$SUDOERS" "$SICHER/sudoers-api-ai"

# sudoers: erst pruefen, dann ersetzen
install -m 0440 -o root -g root "$REPO/deploy/apiai/sudoers-api-ai-konten" "$SUDOERS.neu"
visudo -cf "$SUDOERS.neu"
mv -f "$SUDOERS.neu" "$SUDOERS"

cat > "$DROPIN" <<EOF
# #3165: codex als apiai-codex mit eigenen Anmeldungen je Konto, Kontowahl in api-ai
[Service]
Environment=CODEX_RUN_AS_USER=apiai-codex
Environment=CODEX_HOME=/var/lib/api-ai-codex
Environment=API_AI_CODEX_KONTEN=$KONTEN
# /var/lib/api-ai-codex bleibt beschreibbar: codex laeuft per sudo als apiai-codex,
# aber im Mount-Namensraum dieses Dienstes; ProtectSystem=strict gilt auch fuer
# ihn (Token-Erneuerung in auth.json, generated_images, sessions). apiai selbst
# hat dort nur Durchgang (0710).
ReadWritePaths=
ReadWritePaths=/var/lib/api-ai /var/lib/api-ai-cli /var/lib/apiai-agy /var/lib/api-ai-codex /tmp
EOF
chmod 0644 "$DROPIN"
systemctl daemon-reload
systemctl restart api-ai
sleep 5
systemctl is-active api-ai
systemctl show api-ai -p Environment | tr ' ' '\n' | grep -E '^(CODEX_RUN_AS_USER|CODEX_HOME|API_AI_CODEX_KONTEN)='
echo "Sicherung: $SICHER"
