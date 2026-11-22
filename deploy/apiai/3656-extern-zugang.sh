#!/usr/bin/env bash
# #3656: Zugang fuer eine andere Instanz auf /ai/extern/hydrogen anlegen.
# Nur auf arkserver, als root. Aufruf: 3656-extern-zugang.sh <kennung> [tag_max]
#
# Legt den Schluessel als Datei /root/api-ai-extern-<kennung>.key (0600) an.
# In /etc/api-ai/extern_zugang.json (root:apiai 0640) steht nur sein SHA-256.
# Der Schluessel erscheint nie auf dem Bildschirm, nur ein Fingerabdruck
# (die ersten 12 Zeichen des Hashes). Widerruf: Eintrag aus der JSON loeschen.
set -euo pipefail

kennung="${1:?Kennung fehlt, z. B. p3}"
tag_max="${2:-100}"
[[ "$kennung" =~ ^[a-z0-9-]{1,32}$ ]] || { echo "Kennung nur a-z, 0-9, -" >&2; exit 2; }
[[ "$tag_max" =~ ^[0-9]+$ ]] && (( tag_max > 0 && tag_max <= 1000 )) || { echo "tag_max 1 bis 1000" >&2; exit 2; }
[[ $EUID -eq 0 ]] || { echo "nur als root" >&2; exit 2; }

datei=/etc/api-ai/extern_zugang.json
schluessel="/root/api-ai-extern-${kennung}.key"
umask 077
if [[ -e "$schluessel" ]]; then
  echo "$schluessel gibt es schon. Fuer einen neuen Schluessel die Datei erst entfernen." >&2
  exit 3
fi

python3 -I - "$datei" "$kennung" "$tag_max" "$schluessel" <<'PY'
import grp, hashlib, json, os, secrets, sys, tempfile
datei, kennung, tag_max, schluessel = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
wert = secrets.token_hex(32)
fd = os.open(schluessel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as f:
    f.write(wert)
try:
    with open(datei, encoding="utf-8") as f:
        eintraege = json.load(f)
except FileNotFoundError:
    eintraege = {}
h = hashlib.sha256(wert.encode("ascii")).hexdigest()
eintraege[kennung] = {"sha256": h, "tag_max": tag_max}
wfd, tmp = tempfile.mkstemp(dir=os.path.dirname(datei), prefix=".extern_zugang.")
with os.fdopen(wfd, "w") as f:
    json.dump(eintraege, f, indent=1, sort_keys=True)
os.chown(tmp, 0, grp.getgrnam("apiai").gr_gid)
os.chmod(tmp, 0o640)
os.replace(tmp, datei)
print(f"Zugang {kennung}: Deckel {tag_max} am Tag, Schluessel in {schluessel}, Fingerabdruck {h[:12]}")
PY
