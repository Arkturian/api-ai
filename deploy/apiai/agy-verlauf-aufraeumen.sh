#!/usr/bin/env bash
# #3656: agy legt jede Unterhaltung im eigenen Home ab (conversations/, brain/,
# log/, Zusammenfassungen). api-ai setzt keine Unterhaltung fort, braucht also
# nichts davon. Auf dem Kern sollen keine Prompt-Inhalte liegen bleiben, auch
# nicht die von P3 (P3-Cloud, 22.11.). Laeuft als apiai-agy alle 5 Minuten und
# loescht, was aelter als 10 Minuten ist. Die Zusammenfassungs-Datenbanken nur,
# solange gerade kein agy laeuft.
set -euo pipefail
H="${AGY_HOME:-/var/lib/apiai-agy}/.gemini/antigravity-cli"
MIN="${AGY_AUFBEWAHRUNG_MIN:-10}"
[[ "$MIN" =~ ^[0-9]+$ ]] && (( MIN >= 5 )) || { echo "AGY_AUFBEWAHRUNG_MIN mindestens 5" >&2; exit 2; }
[[ -d "$H" ]] || exit 0
n=0
for d in conversations brain log; do
  [[ -d "$H/$d" ]] || continue
  k=$(find "$H/$d" -mindepth 1 -type f -mmin +"$MIN" -print -delete | wc -l)
  n=$((n + k))
  find "$H/$d" -mindepth 1 -type d -empty -mmin +"$MIN" -delete
done
if ! pgrep -u "$(id -u)" -x agy >/dev/null; then
  for f in conversation_summaries.db conversation_summaries.db-wal conversation_summaries.db-shm jetbox_summaries_proto.pb; do
    [[ -e "$H/$f" ]] && rm -f -- "$H/$f" && n=$((n + 1))
  done
fi
echo "agy-Verlauf: $n Dateien entfernt (aelter als $MIN min)"
