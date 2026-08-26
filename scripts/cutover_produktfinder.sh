#!/usr/bin/env bash
# Probe nach dem Umstellen von REALTIME_PROFILE_ID auf arkturian.
#
# Beantwortet GENAU eine Frage: Kommt ein product-finder-Grant durch
# meinen Verifier? Alles andere ist schon belegt (AuthApi hat den
# Grant gegen die Produktion ausgestellt und den Widerruf gezeigt).
#
# Ausfall != leeres Regal: Jede Probe unterscheidet hier ausdruecklich
# zwischen "abgelehnt" und "nicht erreichbar". Ein Verbindungsfehler,
# der als "keine Freigabe" gelesen wird, schickt die Suche an die
# falsche Stelle.
set -uo pipefail

HOST="${HOST:-https://api-ai.arkturian.com}"
JWT_DATEI="${JWT_DATEI:-/root/.secrets/productfinder-principal-jwt.txt}"

[ -r "$JWT_DATEI" ] || { echo "FEHLT: $JWT_DATEI nicht lesbar (sudo?)"; exit 2; }
JWT="$(tr -d '\r\n' < "$JWT_DATEI")"

echo "== 1. Profil steht im laufenden Prozess =="
systemctl show -p MainPID --value api-ai \
  | xargs -I{} sh -c 'tr "\0" "\n" < /proc/{}/environ' \
  | grep '^REALTIME_PROFILE_ID=' || echo "  nicht lesbar"

echo
echo "== 2. Grant wird eingetauscht (der eigentliche Test) =="
# `confirm_api_billing: true` ist PFLICHT — das Kosten-Gate sitzt HINTER
# der Grant-Pruefung. Fehlt das Feld, kommt 403 und sieht aus, als waere
# der Grant abgelehnt worden. Er war es nicht.
ANTWORT="$(curl -sS -m 15 -o /tmp/cutover_body.$$ -w '%{http_code}' \
  -X POST "$HOST/ai/realtime/token" \
  -H "Authorization: Bearer $JWT" \
  -H 'Content-Type: application/json' \
  -d '{"companion_mode":"product-finder","language":"de","confirm_api_billing":true}' 2>/tmp/cutover_err.$$)"
RC=$?
if [ $RC -ne 0 ]; then
  echo "  NICHT ERREICHBAR (curl rc=$RC) — das ist KEINE Ablehnung:"
  sed 's/^/    /' /tmp/cutover_err.$$
else
  case "$ANTWORT" in
    200) echo "  OK 200 — Grant akzeptiert, Sitzung gemintet" ;;
    401|403) echo "  ABGELEHNT $ANTWORT:"; head -c 400 /tmp/cutover_body.$$ | sed 's/^/    /' ;;
    *) echo "  UNERWARTET $ANTWORT:"; head -c 400 /tmp/cutover_body.$$ | sed 's/^/    /' ;;
  esac
fi
rm -f /tmp/cutover_body.$$ /tmp/cutover_err.$$

echo
echo "== 3. Altwerkzeug-Befund (braucht X-Dev-Secret) =="
echo "  curl -sS $HOST/ai/realtime/config-health -H 'X-Dev-Secret: <geheim>' | jq .legacy_tool_auth"
