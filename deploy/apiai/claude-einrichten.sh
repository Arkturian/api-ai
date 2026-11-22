#!/bin/bash
# #3658: Claude-CLI fuer den Dienst apiai, wie codex in 3165-einrichten.sh.
# Einmalig als root auf arkserver, wiederholbar (Aktualisierung = erneut ausfuehren).
# - eigene Programmkopie per npm unter /usr/local/lib/api-ai/claude-npm.XXXXXX,
#   Verweis /usr/local/lib/api-ai/claude, Aufruf /usr/local/lib/api-ai/bin/claude
# - Fassung = die der laufenden Agenten-CLI von alex (Verzeichnisname unter
#   ~alex/.local/share/claude/versions, nur gelesen, nie ausgefuehrt)
# - root-eigen, fuer alle nur lesbar/ausfuehrbar; kein Home von alex im Pfad
set -euo pipefail
umask 077
BASIS=/usr/local/lib/api-ai
PROG=$BASIS/claude
V=${CLAUDE_FASSUNG:-$(ls -1 /home/alex/.local/share/claude/versions 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)}
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Claude-Fassung unklar: $V" >&2; exit 1; }
IST=$(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$PROG/package.json" 2>/dev/null || true)
if [[ "$IST" != "$V" ]]; then
  NEU=$(mktemp -d "$BASIS/claude-npm.XXXXXX")
  ( umask 022; npm install --global --prefix "$NEU" --no-fund --no-audit "@anthropic-ai/claude-code@$V" >/dev/null )
  chmod -R a+rX,go-w "$NEU"; chmod 755 "$NEU"
  ln -sfn "$NEU/lib/node_modules/@anthropic-ai/claude-code" "$PROG.neu" && mv -Tf "$PROG.neu" "$PROG"
  echo "installiert: $V in $NEU (vorher: ${IST:-keine})"
fi
install -d -m 0755 -o root -g root "$BASIS/bin"
B=$(python3 -I -c 'import json,sys;b=json.load(open(sys.argv[1])).get("bin");print(b["claude"] if isinstance(b,dict) else b)' "$PROG/package.json")
ln -sfn "$PROG/$B" "$BASIS/bin/claude.neu" && mv -Tf "$BASIS/bin/claude.neu" "$BASIS/bin/claude"
echo "Aufruf: $BASIS/bin/claude -> $(readlink -f "$BASIS/bin/claude")"
echo "Fassung als apiai: $(sudo -n -u apiai env -i HOME=/var/lib/api-ai PATH=$BASIS/bin:/usr/bin:/bin "$BASIS/bin/claude" --version 2>&1 | tail -1)"
