"""Bilder je Aufrufer (Auftrag Alex 13.11.2026 ueber CloudV2).

Das Info-Fenster eines Agenten zeigt Tokens (cloud-api) und Bilder (hier)
je Zeitraum. Bis hierher kannte api-ai nur eine Monatssumme ohne Aufrufer
(`openai_cost_tracker`). Diese Datei fuehrt je Erzeugung eine Zeile:
wer, wann, Modell, Weg (Abo oder bezahlt), Kosten.

Ablage: JSONL je Monat unter VERBRAUCH_DIR (Vorgabe
/var/lib/api-ai/verbrauch), `bilder_<YYYY-MM>.jsonl`. Eine Zeile ist
weit unter PIPE_BUF; mehrere Worker haengen mit O_APPEND an, dazu eine
flock-Sperre.

Wer zaehlt: das GEPRUEFTE JWT (Agent -> Name, Mensch -> Mail). Ohne JWT
der Kopf X-Agent-Name, den das MCP-Gateway aus dem geprueften JWT setzt.
Direkt gesetzt ist er ein freies Textfeld; die Zeile sagt deshalb
`quelle` (jwt | header | intern | keine). Nur Etikett, nie Recht.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

WIEN = ZoneInfo("Europe/Vienna")
UNBEKANNT = "(unbekannt)"
INTERN = "(intern)"


def verzeichnis() -> Path:
    return Path(os.getenv("VERBRAUCH_DIR") or "/var/lib/api-ai/verbrauch")


# --- Wer ruft? ------------------------------------------------------------

def _claims_geprueft(authorization: Optional[str]) -> Optional[dict]:
    """Claims eines JWT NUR nach Signatur- und Ausstellerpruefung gegen die
    gepinnte auth-api-JWKS (wie /ai/jev). Ungeprueft waere es Freitext."""
    if not isinstance(authorization, str) or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        import jwt
        from ai.services import realtime_grant_verifier as v
        key = v._jwks().get_signing_key_from_jwt(token)
        return jwt.decode(token, key.key, algorithms=["RS256"], issuer=v.AUTH_ISSUER,
                          options={"verify_aud": False})
    except Exception:
        return None


def name_aus_claims(claims: dict) -> Optional[str]:
    sub = str(claims.get("sub") or "").strip()
    if sub.startswith("agent:"):
        return sub[len("agent:"):][:64] or None
    mail = str(claims.get("email") or "").strip().lower()
    return mail[:128] or None


def aufrufer(authorization: Optional[str], x_agent_name: Optional[str]) -> Tuple[str, str]:
    """(name, quelle). Das gepruefte JWT gewinnt gegen den Kopf."""
    claims = _claims_geprueft(authorization)
    if claims:
        n = name_aus_claims(claims)
        if n:
            return n, "jwt"
    if isinstance(x_agent_name, str) and x_agent_name.strip():
        n = x_agent_name.strip()
        if n.startswith("agent:"):
            n = n[len("agent:"):]
        return n[:64], "header"
    return UNBEKANNT, "keine"


# --- Schreiben ------------------------------------------------------------

def buchen(*, agent: str, quelle: str, art: str = "bild", anzahl: int = 1,
           model: Optional[str], routed_via: Optional[str], kosten_usd: Optional[float],
           storage_id=None, request_id: Optional[str] = None,
           jetzt: Optional[float] = None) -> None:
    """Eine Erzeugung festhalten. Ein Fehler hier darf nie den Aufruf
    treffen: das Bild ist schon erzeugt und bezahlt."""
    try:
        t = jetzt if jetzt is not None else time.time()
        abo = routed_via == "subscription"
        zeile = {
            "ts": datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds"),
            "agent": agent, "quelle": quelle, "art": art, "anzahl": int(anzahl),
            "model": model, "routed_via": routed_via,
            "kosten_usd": 0.0 if abo else (round(float(kosten_usd), 6) if kosten_usd is not None else None),
            "storage_id": storage_id, "request_id": request_id,
        }
        d = verzeichnis()
        d.mkdir(parents=True, exist_ok=True)
        pfad = d / f"bilder_{datetime.fromtimestamp(t, WIEN):%Y-%m}.jsonl"
        with open(pfad, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(zeile, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.warning("Bildverbrauch nicht gebucht: %s", type(e).__name__)


# --- Lesen ----------------------------------------------------------------

def _zeilen() -> Iterable[dict]:
    d = verzeichnis()
    if not d.is_dir():
        return
    for p in sorted(d.glob("bilder_*.jsonl")):
        try:
            with open(p, encoding="utf-8") as f:
                for z in f:
                    z = z.strip()
                    if not z:
                        continue
                    try:
                        yield json.loads(z)
                    except ValueError:
                        continue
        except OSError:
            continue


def _zeit(ts: str) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def bericht(agent: Optional[str] = None, von: Optional[datetime] = None,
            bis: Optional[datetime] = None) -> dict:
    """Summen je Wiener Tag, Agent und Art im Bereich [von, bis).

    `seit` ist die erste Zeile ueberhaupt (unabhaengig vom Filter): erst ab
    da wird gezaehlt, das Portal soll nicht „seit Beginn“ behaupten."""
    seit = None
    gruppen: dict = {}
    for z in _zeilen():
        t = _zeit(z.get("ts") or "")
        if t is None:
            continue
        if seit is None or t < seit:
            seit = t
        if agent and z.get("agent") != agent:
            continue
        if von and t < von:
            continue
        if bis and t >= bis:
            continue
        schluessel = (t.astimezone(WIEN).date().isoformat(), z.get("agent") or UNBEKANNT, z.get("art") or "bild")
        g = gruppen.setdefault(schluessel, {"anzahl": 0, "abo": 0, "bezahlt": 0, "kosten_usd": 0.0,
                                            "kosten_unbekannt": 0})
        n = int(z.get("anzahl") or 1)
        g["anzahl"] += n
        if z.get("routed_via") == "subscription":
            g["abo"] += n
        else:
            g["bezahlt"] += n
            if z.get("kosten_usd") is None:
                g["kosten_unbekannt"] += n
            else:
                g["kosten_usd"] += float(z["kosten_usd"])
    zeilen = []
    for (tag, a, art), g in sorted(gruppen.items()):
        zeile = {"t": tag, "agent": a, "art": art, "anzahl": g["anzahl"], "abo": g["abo"],
                 "bezahlt": g["bezahlt"], "kosten_usd": round(g["kosten_usd"], 4)}
        if g["kosten_unbekannt"]:
            zeile["kosten_unbekannt"] = g["kosten_unbekannt"]
        zeilen.append(zeile)
    return {"seit": seit.isoformat(timespec="seconds") if seit else None, "zeilen": zeilen}
