"""Dienst-JWT-Weg fuer einzelne Routen (#3958, AuthAPI 25.11.2026).

Ein Dienst der Federation (auth-api `type=service`, z. B. `service:youtube-api`)
bekommt fuer genau benannte Routen ein eigenes Recht, statt den geteilten
`API_ACCESS_KEY` zu tragen:

    POST /ai/transcribe            -> `ai:transcribe`
    POST /ai/chatgpt               -> `ai:chatgpt`
    GET  /ai/chatgpt/{request_id}  -> `ai:chatgpt`
    POST /ai/audio/abschrift       -> `ai:transcribe` (Projekt 32, A2)
    GET  /ai/audio/abschrift/...   -> `ai:transcribe`

Regeln (fail-closed):
- Geprueft wird mit agentos_jwt (JWKS der auth-api, RS256, kid, Aussteller,
  Widerruf, Typ). Nur `type=service` ohne `aud` geht diesen Weg. Agenten- und
  Menschen-Tokens bleiben beim bisherigen Verhalten.
- Das Recht steht woertlich in `permissions` (auth-api `create_service_token`).
- Ein Dienst-JWT mit einem dieser Rechte ist auf seine Routen begrenzt: an jeder
  anderen /ai/-Route 403, auch wenn das Tor dort sonst offen waere.
- Ein Dienst-JWT ohne das Recht der Route: 403.
- Der Mandant des Tokens muss in DIENST_JWT_MANDANTEN stehen (Vorgabe arkturian).
- Tagesgrenzen je Dienst (DIENST_TAGESGRENZEN, JSON): Audiominuten fuer
  transcribe, Anfragen fuer chatgpt. Gebucht wird vor dem Anbieteraufruf.
- `confirm_api_billing` bleibt unveraendert in den Routen.

Dienst-JWTs ohne eines dieser Rechte (storage-api, tools-api,
conversation-api an /ai/jev und /ai/decisions) aendern nichts an ihrem Weg.

Protokolliert werden sub, Route und Grund, nie das Token.
"""
from __future__ import annotations

import contextvars
import fcntl
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import jwt
from fastapi import HTTPException

logger = logging.getLogger("api-ai.dienst_jwt")

RECHT_TRANSCRIBE = "ai:transcribe"
RECHT_CHATGPT = "ai:chatgpt"
DIENST_RECHTE = frozenset({RECHT_TRANSCRIBE, RECHT_CHATGPT})

# Statische Geschwister von /ai/chatgpt/{request_id}, die keine Kennung sind.
_CHATGPT_KEINE_KENNUNG = frozenset({"cost-status"})

VORGABE_GRENZEN = {"transcribe_minuten": 120, "chatgpt_anfragen": 200}

#: Geprueftes Ergebnis des laufenden Aufrufs ({"sub", "recht"}), fuer die Routen.
AKTUELL: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("dienst_jwt", default=None)


def recht_fuer(methode: str, pfad: str) -> Optional[str]:
    """Welches Recht braucht diese Route auf dem Dienst-Weg? None: keine Dienst-Route."""
    methode = (methode or "").upper()
    if methode == "POST" and pfad == "/ai/transcribe":
        return RECHT_TRANSCRIBE
    # Projekt 32 (A2): Abschrift langer Sendungen ist dieselbe bezahlte STT.
    if methode == "POST" and pfad == "/ai/audio/abschrift":
        return RECHT_TRANSCRIBE
    if methode == "GET" and pfad.startswith("/ai/audio/abschrift/"):
        return RECHT_TRANSCRIBE
    if methode == "POST" and pfad == "/ai/chatgpt":
        return RECHT_CHATGPT
    if methode == "GET" and pfad.startswith("/ai/chatgpt/"):
        rest = pfad[len("/ai/chatgpt/"):]
        if rest and "/" not in rest and rest not in _CHATGPT_KEINE_KENNUNG:
            return RECHT_CHATGPT
    return None


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if not isinstance(authorization, str) or not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip() or None


def _roh(token: str) -> dict:
    """Ungepruefte Claims, NUR um zu entscheiden, ob gepruft werden muss.
    Nie Grundlage eines Ja."""
    try:
        d = jwt.decode(token, options={"verify_signature": False})
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _rechte(claims: dict) -> frozenset:
    p = claims.get("permissions")
    if not isinstance(p, list):
        return frozenset()
    return frozenset(x for x in p if isinstance(x, str))


def _mandanten() -> frozenset:
    roh = os.getenv("DIENST_JWT_MANDANTEN") or "arkturian"
    return frozenset(t.strip() for t in roh.split(",") if t.strip())


def _abweisen(status: int, fehler: str, sub: str, pfad: str, **extra) -> HTTPException:
    logger.warning("DIENST-JWT %s sub=%s path=%s", fehler, sub[:64], pfad)
    return HTTPException(status_code=status, detail={"error": fehler, **extra})


def entscheiden(methode: str, pfad: str, authorization: Optional[str]) -> Optional[dict]:
    """None: dieser Weg ist nicht zustaendig (bisheriges Verhalten).
    dict {"sub", "recht"}: Dienst-JWT geprueft, Route freigegeben.
    HTTPException: 401/403/503."""
    if (methode or "").upper() == "OPTIONS" or not pfad.startswith("/ai/"):
        return None
    token = _bearer(authorization)
    if token is None:
        return None
    roh = _roh(token)
    if roh.get("type") != "service":
        return None      # Agenten-, Menschen- und andere Tokens: unveraendert
    soll = recht_fuer(methode, pfad)
    roh_rechte = _rechte(roh) & DIENST_RECHTE
    if soll is None and not roh_rechte:
        return None      # anderer Dienst (z. B. storage-api an /ai/jev): unveraendert

    from ai.services import jwt_pruefung
    claims = jwt_pruefung.pruefen(authorization)     # 401 ungueltig, 503 nicht pruefbar
    if not claims or claims.get("type") != "service":
        raise _abweisen(401, "invalid_token", str(roh.get("sub") or "?"), pfad)
    sub = str(claims.get("sub") or "")
    rechte = _rechte(claims)
    if soll is None:
        # Dienst mit Rechten fuer AiApi-Routen an einer anderen Route.
        raise _abweisen(403, "dienst_route_nicht_freigegeben", sub, pfad,
                        hint="Dieses Dienst-JWT gilt nur fuer die Routen seiner ai:-Rechte.")
    if soll not in rechte:
        raise _abweisen(403, "dienst_recht_fehlt", sub, pfad, required=soll)
    if str(claims.get("tenant") or "") not in _mandanten():
        raise _abweisen(403, "dienst_mandant_nicht_freigegeben", sub, pfad)
    if not sub.startswith("service:"):
        raise _abweisen(403, "dienst_sub_ungueltig", sub, pfad)
    return {"sub": sub, "recht": soll}


# ---------------------------------------------------------------- Tagesgrenzen

def grenzen(sub: str) -> dict:
    """Grenzen je Dienst aus DIENST_TAGESGRENZEN (JSON), sonst VORGABE_GRENZEN.
    Kaputtes JSON gilt als nicht gesetzt (Vorgabe greift, nie unbegrenzt)."""
    g = dict(VORGABE_GRENZEN)
    try:
        tabelle = json.loads(os.getenv("DIENST_TAGESGRENZEN") or "{}")
    except ValueError:
        logger.error("DIENST_TAGESGRENZEN ist kein gueltiges JSON, Vorgabe gilt")
        return g
    eigen = tabelle.get(sub) if isinstance(tabelle, dict) else None
    if isinstance(eigen, dict):
        for k in g:
            v = eigen.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                g[k] = v
    return g


def _name(sub: str) -> str:
    n = re.sub(r"[^a-z0-9-]+", "-", sub.lower()).strip("-")[:56]
    return f"dienst_{n or 'unbekannt'}"


def _verzeichnis() -> Path:
    return Path(os.getenv("VERBRAUCH_DIR") or "/var/lib/api-ai/verbrauch")


def _naechster_tag(jetzt: datetime) -> tuple:
    naechster = (jetzt.replace(tzinfo=None) + timedelta(days=1)).replace(hour=0, minute=0, second=0)
    bis = naechster.astimezone()
    return bis.isoformat(), max(1, int((bis - jetzt).total_seconds()))


def buchen(sub: str, feld: str, menge: float, grenze: float) -> dict:
    """Bucht `menge` auf den Tageszaehler des Dienstes, wenn die Grenze reicht.
    Sonst 429. Zaehler nicht nutzbar: 503 (nie ungezaehlt durch)."""
    jetzt = datetime.now().astimezone().replace(microsecond=0)
    name = _name(sub)
    verzeichnis = _verzeichnis()
    tmp = None
    try:
        verzeichnis.mkdir(parents=True, exist_ok=True)
        fd = os.open(verzeichnis / f".{name}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "r+") as sperre:
            fcntl.flock(sperre, fcntl.LOCK_EX)
            pfad = verzeichnis / f"{name}_{jetzt.date().isoformat()}.json"
            try:
                stand = json.loads(pfad.read_text())
                if not isinstance(stand, dict):
                    raise ValueError("kein Objekt")
            except FileNotFoundError:
                stand = {}
            bisher = float(stand.get(feld) or 0)
            if bisher + menge > grenze:
                bis, sek = _naechster_tag(jetzt)
                logger.warning("DIENST-JWT tagesgrenze sub=%s feld=%s stand=%s menge=%s grenze=%s",
                               sub[:64], feld, bisher, menge, grenze)
                raise HTTPException(status_code=429, headers={"Retry-After": str(sek)}, detail={
                    "error": "dienst_tagesgrenze", "caller": sub, "feld": feld,
                    "stand": round(bisher, 2), "angefragt": round(menge, 2), "grenze": grenze,
                    "resets_at": bis})
            stand[feld] = round(bisher + menge, 3)
            wfd, tmp = tempfile.mkstemp(dir=verzeichnis, prefix=f".{name}.")   # 0600
            with os.fdopen(wfd, "w") as f:
                json.dump(stand, f)
            os.replace(tmp, pfad)
            tmp = None
            return stand
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, AttributeError) as e:
        logger.error("DIENST-JWT Zaehler %s nicht nutzbar (%s)", name, type(e).__name__)
        raise HTTPException(status_code=503, detail={"error": "dienst_zaehler_fehlt"})
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def vor_aufruf(erg: dict, methode: str, pfad: str) -> None:
    """Grenze fuer chatgpt je Anfrage (POST), vor dem Rumpf. Statusabfragen zaehlen nicht."""
    if erg.get("recht") == RECHT_CHATGPT and (methode or "").upper() == "POST":
        buchen(erg["sub"], "chatgpt_anfragen", 1, grenzen(erg["sub"])["chatgpt_anfragen"])


def transcribe_buchen(dauer_sekunden: Optional[float]) -> None:
    """Aus der transcribe-Route, nach dem Messen der Dauer und VOR dem Anbieter.
    Kein Dienst-Aufruf: nichts. Dauer unbekannt: 422 (ohne Dauer keine Grenze)."""
    erg = AKTUELL.get()
    if not erg or erg.get("recht") != RECHT_TRANSCRIBE:
        return
    if not isinstance(dauer_sekunden, (int, float)) or dauer_sekunden <= 0:
        raise HTTPException(status_code=422, detail={
            "error": "audio_dauer_unbekannt",
            "hint": "Die Dauer der Aufnahme liess sich nicht messen; ohne sie gilt keine Tagesgrenze."})
    minuten = float(dauer_sekunden) / 60.0
    buchen(erg["sub"], "transcribe_minuten", minuten, grenzen(erg["sub"])["transcribe_minuten"])
