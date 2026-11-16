# api-ai als Dienstbenutzer `apiai` (#2937, Beitrag 6150)

Diese Dateien werden erst beim Cutover installiert, root-eigen:

| Datei | Ziel | Rechte |
|---|---|---|
| `codex-als-alex` | `/usr/local/libexec/api-ai/codex-als-alex` | root:root 0755 |
| `agy-als-dienst` | `/usr/local/libexec/api-ai/agy-als-dienst` | root:root 0755 |
| `sudoers-api-ai` | `/etc/sudoers.d/api-ai` | root:root 0440, `visudo -c` |

Den Code-Pfad schaltet `API_AI_CLI_WRAPPER_DIR=/usr/local/libexec/api-ai` im
Drop-in der Unit ein. Fehlt die Variable, laeuft api-ai wie bisher als root.
Dazu im Cutover-Drop-in: `CLI_HOME=/var/lib/api-ai`, `HOME=/var/lib/api-ai`
(zz-decouple setzt heute /root), `CODEX_TEXT_HOME=/var/lib/api-ai-cli/codex-text`.

Laufzeit-Dateien, die der Prozess selbst liest (nicht systemd): beim Cutover
auf `root:apiai 0640` stellen, sonst sperrt der jeweilige Weg (fail-closed).
- `/etc/api-ai/google-ultra.env` (#2867, Gemini-Schluessel fuer das
  Ultra-Guthaben, bis 13.12.2026; danach loeschen). Ohne Leserecht: 503.

Verzeichnisse, in die der Prozess schreibt: beim Cutover `apiai:apiai 0700`.
- `/var/lib/api-ai/agentos_jwt` (#3155, Widerrufsstand von agentos_jwt).
  Ohne Schreibrecht: JWT-Pruefung 503, Etiketten „unbekannt“.

Die Tests `tests/test_2937_dienstmodus.py` pruefen die Wrapper gegen die echten
und gegen feindliche Argumentlisten. Benutzer, Gruppen, Verzeichnisse und Unit:
siehe Beitrag 6150.
