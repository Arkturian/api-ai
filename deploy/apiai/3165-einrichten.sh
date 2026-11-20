#!/bin/bash
# #3165 Schritt A: apiai-codex einrichten. Einmalig als root auf arkserver,
# wiederholbar. Aendert am laufenden api-ai NICHTS (der Dienst nutzt weiter
# codex-als-alex, bis Schritt B umschaltet).
#
#   sudo bash deploy/apiai/3165-einrichten.sh
#
# - Benutzer apiai-codex (System, ohne Login, Hauptgruppe apiai-codex, dazu
#   apiai-cli zum Lesen von eingang/ und arbeit/)
# - codex als eigene Programmkopie unter /usr/local/lib/api-ai/codex
#   (npm, gleiche Fassung wie bei alex; root-eigen, fuer alle nur lesbar)
# - je Konto ein Home /var/lib/api-ai-codex/konten/<konto>:
#   0710 apiai-codex:apiai-cli (apiai darf hindurch, nicht auflisten),
#   config.toml nur mit model + model_reasoning_effort (keine MCP, keine Plugins),
#   generated_images 2750 (Gruppe apiai-cli: api-ai holt dort das Abo-Bild ab)
# - eine Geraetecode-Anmeldung aus der Ablage /var/lib/api-ai-codex-anmeldung/<konto>
#   wird nach konten/<konto>/auth.json (0600 apiai-codex) verschoben, die Ablage
#   danach geloescht. Nie eine Datei aus /home/alex.
# - Wrapper codex-als-dienst nach /usr/local/libexec/api-ai (root 0755)
set -euo pipefail
umask 077
REPO=$(cd "$(dirname "$0")/../.." && pwd)
KONTEN=(apopovic tsaier)
BASIS=/var/lib/api-ai-codex
ABLAGE=/var/lib/api-ai-codex-anmeldung
PROG=/usr/local/lib/api-ai/codex
# Nur LESEN, nie als root ausfuehren (Cloud 20.11.): die Datei gehoert alex,
# jeder Prozess als alex koennte codex.js vorher aendern.
ALEX_PAKET=/home/alex/.local/lib/node_modules/@openai/codex/package.json

# 1. Benutzer
if ! id apiai-codex >/dev/null 2>&1; then
  useradd --system --user-group --home-dir "$BASIS" --no-create-home --shell /usr/sbin/nologin apiai-codex
fi
usermod -aG apiai-cli apiai-codex
id apiai-codex

# 2. Programmkopie, gleiche Fassung wie alex' codex
V=$(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$ALEX_PAKET")
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "codex-Fassung unklar: $V" >&2; exit 1; }
IST=$(python3 -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$PROG/package.json" 2>/dev/null || true)
if [[ "$IST" != "$V" ]]; then
  NEU=$(mktemp -d /usr/local/lib/api-ai/codex-npm.XXXXXX)
  ( umask 022; npm install --global --prefix "$NEU" --ignore-scripts --no-fund --no-audit "@openai/codex@$V" >/dev/null )
  chmod -R a+rX,go-w "$NEU"
  ln -sfn "$NEU/lib/node_modules/@openai/codex" "$PROG.neu" && mv -Tf "$PROG.neu" "$PROG"
fi
echo "codex-Kopie: $(/usr/bin/node "$PROG/bin/codex.js" --version)"

# 3. Verzeichnisse und config.toml je Konto
install -d -m 0710 -o apiai-codex -g apiai-cli "$BASIS" "$BASIS/konten"
CFG=$(awk '/^[[:space:]]*\[/{exit} /^(model|model_reasoning_effort)[[:space:]]*=/{print}' /home/alex/.codex/config.toml)
for k in "${KONTEN[@]}"; do
  H=$BASIS/konten/$k
  install -d -m 0710 -o apiai-codex -g apiai-cli "$H"
  install -d -m 2750 -o apiai-codex -g apiai-cli "$H/generated_images"
  printf '%s\n' "$CFG" | install -m 0600 -o apiai-codex -g apiai-codex /dev/stdin "$H/config.toml"
  # 4. Anmeldung aus der Ablage uebernehmen (nur, wenn dort eine liegt)
  if [[ -s "$ABLAGE/$k/auth.json" ]]; then
    install -m 0600 -o apiai-codex -g apiai-codex "$ABLAGE/$k/auth.json" "$H/auth.json"
    shred -u "$ABLAGE/$k/auth.json"
    rm -rf -- "${ABLAGE:?}/$k" "$ABLAGE/$k.log"
    echo "$k: Anmeldung uebernommen"
  fi
  if [[ -s "$H/auth.json" ]]; then echo "$k: angemeldet"; else echo "$k: noch ohne Anmeldung"; fi
done

# 5. Wrapper
install -m 0755 -o root -g root "$REPO/deploy/apiai/codex-als-dienst" /usr/local/libexec/api-ai/codex-als-dienst
sha256sum /usr/local/libexec/api-ai/codex-als-dienst | cut -c1-8

# 6. Probe als apiai-codex: Fassung und Kontingent je angemeldetem Konto (keine Token in der Ausgabe)
for k in "${KONTEN[@]}"; do
  [[ -s "$BASIS/konten/$k/auth.json" ]] || continue
  echo "$k: $(sudo -n -u apiai-codex /usr/local/libexec/api-ai/codex-als-dienst "$k" text --version)"
  sudo -n -u apiai-codex /usr/local/libexec/api-ai/codex-als-dienst "$k" limits \
    | python3 -I -c 'import json,sys
for z in sys.stdin:
    o=json.loads(z); r=o.get("result") or {}
    if o.get("id")==2: print("  konto:", ((r.get("account") or {}).get("email")), o.get("error",{}).get("message",""))
    if o.get("id")==3: print("  limits:", [ (f or {}).get("usedPercent") for f in ((r.get("rateLimits") or {}).get("primary"),(r.get("rateLimits") or {}).get("secondary"))])'
done
