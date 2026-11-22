# api-ai auf einer Kundeninstanz

Was eine isolierte AgentOS-Instanz braucht, damit api-ai läuft — und
womit man prüft, dass sie wirklich isoliert ist.

## Ausliefern

```bash
git fetch --tags
git checkout tenant-release      # NICHT main, NICHT dev
```

`tenant-release` zeigt auf einen Stand, der auf der Föderation läuft und
dessen Tests grün sind. Der Tag wird bewegt, nicht neu erfunden.

## Pflicht-Umgebung

```bash
# --- Zugang -------------------------------------------------------------
API_ACCESS_KEY=<je Instanz erzeugen>       # openssl rand -hex 24
```

Ohne `API_ACCESS_KEY` antworten die Modellrouten (`/ai/chatgpt`, `/ai/claude`,
`/ai/gemini`, `/ai/grok`) einer Instanz mit eigener auth-api mit 503 (#3607), die
Bezahlpfade sind dann offen; der Start warnt laut ins Journal. **Je Instanz erzeugen, niemals aus einer
Vorlage erben** — sonst tragen alle Kunden denselben Schlüssel.

```bash
# --- Anbieter (nur was die Instanz nutzt) --------------------------------
OPENAI_API_KEY=…
ELEVENLABS_API_KEY=…          # TTS
MINIMAX_MULTIMODAL_API_KEY=…  # MiniMax
KLING_API_KEY=…               # Video
TENCENT_SECRET_ID=… TENCENT_SECRET_KEY=…   # 3D
```

Fehlende Anbieterschlüssel sind unkritisch: Der Dienst startet, und der
betroffene Endpunkt scheitert **laut** beim Aufruf (`openai_api_key_missing`
→ 500). Ein Fehlen ist sichtbar, nicht still.

```bash
# --- Fremde Dienste: MÜSSEN auf die lokale Instanz zeigen ----------------
CLOUD_API_URL=https://<lokal>
STORAGE_API_URL=https://<lokal>
STORAGE_API_KEY=<je Instanz>       # PFLICHT, siehe unten
KNOWLEDGE_API_URL=https://<lokal>   # ohne Wert: knowledge_query aus
ARTRACK_API_URL=https://<lokal>     # ohne Wert: pois_near und osm_nearby aus
```

Eine Instanz mit eigener auth-api faellt bei Knowledge und ArTrack nicht auf uns
zurueck (#3604): Ohne eigenen Wert sind die Werkzeuge aus und melden
`status: unavailable`, die Koordinaten gehen nirgends hin.

**`STORAGE_API_KEY` ist Pflicht, auch wenn die Instanz Storage nicht nutzt.**
Ist er unbesetzt, greift ein fest eingebauter Rückfallwert, und die Instanz
meldet sich mit dem Schlüssel des Betreibers gegen dessen storage-api an —
still, mit einer einzigen Warnzeile.

```bash
# --- Eigene auth-api: PFLICHT auf jeder Instanz (seit fffc7f3, 16.11.) ---
REALTIME_AUTH_ISSUER=<kunde>.example.com
REALTIME_AUTH_JWKS_URL=http://127.0.0.1:8010/api/v1/auth/.well-known/jwks.json
REALTIME_AUTH_GRANT_URL=http://127.0.0.1:8010/api/v1/auth/realtime-grant
AGENTOS_JWT_STATE_DIR=/var/lib/api-ai/agentos_jwt   # beschreibbar fuer den Dienst
```

Trotz des Namens gelten diese drei nicht mehr nur fuer Sprache. Ab
`fffc7f3` prueft api-ai **jedes** mitgeschickte JWT gegen sie (`agentos_jwt`):
/ai/jev, /ai/decisions, /ai/genimage (Zuordnung) und /ai/verbrauch (Admin-
Pruefung ueber `<GRANT_URL ohne letztes Segment>/me`). Die Widerrufsliste kommt
von `<JWKS_URL ohne letztes Segment>/revoked-agent-jtis`
(`AGENTOS_JWT_REVOCATION_URL` ueberschreibt).
**Fehlen alle drei, faellt die Instanz auf auth-api.arkturian.com zurueck:**
Sie nimmt dann JWTs des Betreibers an, holt Schluessel und Sperrliste bei ihm,
und /ai/verbrauch schickt das Bearer-Token des Kunden an die auth-api des
Betreibers. Das sieht im Code aus wie unser Kern, deshalb prueft es der Rollout.

**Fehlt nur ein Teil oder zeigt ein Wert auf den Betreiber (#3604):** Tausch und
Token-Pruefung sind aus, Antwort 503 `realtime_auth_unvollstaendig`, `/health`
meldet `realtime_auth: unvollstaendig`. Eine Adresse unter arkturian.com ist nur
erlaubt, wenn ihr Host der eigene Aussteller ist (z. B. `ele.arkturian.com`).

```bash
# --- Realtime (nur wenn Sprache genutzt wird) ---------------------------
REALTIME_PROFILE_ID=<profil[,profil]>
REALTIME_GRANT_SERVICE_KEY=<identisch mit der lokalen auth-api>
OPENAI_REALTIME_MONTHLY_BUDGET_EUR=<UNTER dem Guthaben des Kunden>
REALTIME_BUDGET_WINDOW=daily|monthly        # Vorgabe: daily
```

**Aussteller und Schlüsselquelle immer gemeinsam setzen.** Ein fremder
Aussteller mit fremder Schlüsselquelle ist eine Prüfung, die nur so
aussieht.

**Das Monatsbudget hat die Vorgabe 100.** Liegt das Guthaben des Kunden
darunter, bremst der Wächter erst, wenn das Anbieterkonto längst leer ist.

```bash
# --- Urteile (nur wenn genutzt, #3580) -----------------------------------
JEV_DEFAULT_ENGINE=jev|openai     # Vorgabe jev; "aus" kennt api-ai NICHT
TYPESAFE_API_KEY=…                # eigener Vertrag, sonst 503 typesafe_key_missing
# engine=openai (/ai/decisions) nutzt OPENAI_API_KEY der Instanz
```

„aus“ ist kein Wert fuer `JEV_DEFAULT_ENGINE` (unbekannte Werte laufen wie
`jev`). Abschalten heisst: den Aufrufer nicht anbieten (cloud-api/MCP) bzw.
ohne Schluessel lassen — dann antwortet api-ai laut mit 503.

```bash
# --- Textweg ueber den OpenAI-Schluessel (nur wenn genannt, #3640) -------
OPENAI_TEXT_DAILY_MAX_USD=0.5     # Pflicht fuer diesen Weg, ohne Wert 503
# OPENAI_TEXT_DAILY_MAX_REQUESTS=2000   # Vorgabe
```

`POST /ai/openai` laeuft ueber den `OPENAI_API_KEY` der Instanz (Responses API,
`gpt-6-luna`, `store: false`). Ohne Schluessel oder ohne gueltige Tagesgrenze
antwortet er 503, ein Rueckfall auf einen anderen Schluessel existiert nicht. Ist
die Grenze erreicht, kommt 429 mit `error: kontingent` und `resets_at`.

`GET /ai/wege` zeigt je Weg (`agy`, `codex`, `openai_schluessel`) den Zustand,
ohne Modellaufruf. Der Pfad verlangt den Betreiber- oder Instanzschluessel und
kennt keine Ausnahme fuer den eigenen Host. `/health` nennt die Wege bewusst nicht.

## Verify-Checkliste

Nach jedem Ausrollen, in dieser Reihenfolge:

```bash
# 1 Dienst lebt
curl -fsS http://127.0.0.1:8000/health

# 2 Sperre greift
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8000/ai/chatgpt \
     -H 'Content-Type: application/json' -d '{"prompt":"x"}'
#   erwartet: 401   (200 = API_ACCESS_KEY fehlt, Instanz ist offen)

# 3 Realtime ist NICHT von der Sperre betroffen
curl -s -X POST http://127.0.0.1:8000/ai/realtime/token \
     -H 'Content-Type: application/json' -d '{"companion_mode":"arcturian"}'
#   erwartet: realtime_user_jwt_required
#   NICHT:    api_key_required   ← dann sperrt der Schlüssel den Browser aus

# 4 Umgebung im LAUFENDEN Prozess, nicht in der Datei
PID=$(systemctl show -p MainPID --value <dienst>)
tr '\0' '\n' < /proc/$PID/environ | grep -E '^(REALTIME_AUTH_|AGENTOS_JWT_|API_ACCESS_KEY|STORAGE_API_KEY)' | sed -E 's/=.*/=…/'
#   erwartet: REALTIME_AUTH_ISSUER, _JWKS_URL und _GRANT_URL alle drei gesetzt
```

Punkt 4 ist kein Formalismus: Eine Variable in der `.env`, die der Dienst
nie geladen hat, sieht bei jeder Dateiprüfung richtig aus.

### Isolationsprüfung

```bash
curl -s -X POST http://127.0.0.1:8000/ai/realtime/token \
     -H "Authorization: Bearer <JWT DES BETREIBERS>" \
     -H 'Content-Type: application/json' -d '{"companion_mode":"arcturian"}'
```

Er **muss** abgewiesen werden — und zwar am **Aussteller**. Kommt
`realtime_profile_misconfigured` oder ein anderer Konfigurationsfehler, ist
die Prüfung **nicht** bestanden: Sie ist an einem früheren Riegel gescheitert,
und über den Aussteller ist damit nichts bewiesen.

## Was auf einer Kundeninstanz NICHT gilt

- **Der GCP-Budget-Webhook** (`/ai/gemini/gcp-budget-webhook`) ist von der
  Zugangssperre ausgenommen, weil Google keinen Schlüssel tragen kann. Wer
  ihn nicht braucht, sollte den Pfad im vorgelagerten Server blockieren.
- **Die Text-CLI-Pfade** (`/ai/claude`, `/ai/chatgpt`, `/ai/gemini`,
  `/ai/grok`) laufen über Anmeldungen im Home des Dienstnutzers, nicht über
  Umgebungsvariablen. Ohne eigene Logins existieren diese Endpunkte auf der
  Instanz faktisch nicht.

## Gemini über den Kern (#3656, nur P3)

Eine Instanz ohne eigenes agy (P3) nutzt Gemini Hydrogen und Carbon über den Kern. Ohne `GEMINI_KERN_URL` ändert sich nichts.

| Variable | Bedeutung |
|---|---|
| `GEMINI_KERN_URL` | z. B. `https://api-ai.arkturian.com`. `/ai/gemini` reicht dann reine Textaufrufe an `<URL>/ai/extern/hydrogen` weiter. |
| `GEMINI_KERN_SCHLUESSEL_DATEI` | Datei mit dem Zugangsschlüssel, Vorgabe `/etc/agentos/kern-hydrogen.key` (0600). |
| `GEMINI_KERN_AUFRUFER` | Kommaliste freigegebener Aufrufer (geprüftes JWT, sonst `X-Agent-Name`). Leer heißt: niemand. P3: zuerst nur `Cloud` (Sprechstile). Weitere erst nach Jaschas Zustimmung. |

Nicht über den Kern: Bilder, `conversation_history`, `persona_variant`, Werkzeuge und andere Modelle als `gemini-4-hydrogen[-low|-medium|-high|-xhigh]` und `gemini-4-carbon[-low|-medium|-high]` (422 `gemini_kern_nicht_moeglich`). Nicht freigegebene Aufrufer bekommen 403 `gemini_kern_aufrufer_gesperrt`, bevor etwas die Instanz verlässt.

Auf dem Kern: Zugang je Instanz in `/etc/api-ai/extern_zugang.json` (nur Hash, `tag_max`, optional `aufrufer`), Protokoll nur mit Metadaten, agy-Verlauf wird alle 5 Minuten gelöscht (`agy-verlauf-aufraeumen.timer`).
