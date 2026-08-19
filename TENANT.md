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

Ohne `API_ACCESS_KEY` ist **jeder** `/ai/*`-Endpunkt offen erreichbar; der
Start warnt dann laut ins Journal. **Je Instanz erzeugen, niemals aus einer
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
KNOWLEDGE_API_URL=https://<lokal>
ARTRACK_API_URL=https://<lokal>
```

**`STORAGE_API_KEY` ist Pflicht, auch wenn die Instanz Storage nicht nutzt.**
Ist er unbesetzt, greift ein fest eingebauter Rückfallwert, und die Instanz
meldet sich mit dem Schlüssel des Betreibers gegen dessen storage-api an —
still, mit einer einzigen Warnzeile.

```bash
# --- Realtime (nur wenn Sprache genutzt wird) ---------------------------
REALTIME_AUTH_ISSUER=<kunde>.example.com
REALTIME_AUTH_JWKS_URL=http://127.0.0.1:8010/api/v1/auth/.well-known/jwks.json
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
tr '\0' '\n' < /proc/$PID/environ | grep -E '^(REALTIME_AUTH_|API_ACCESS_KEY|STORAGE_API_KEY)' 
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
