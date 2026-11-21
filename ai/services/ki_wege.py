"""Welche Modellwege hat diese Instanz? (#3639, Alex 21.11.)

Die Pipeline misst damit vor und nach jedem Lauf, ob jeder genannte Weg da ist.
Ohne Modellaufruf. Nicht im oeffentlichen /health, denn das wuerde zeigen,
welche Abos ein Kunde hat.

`zustand`: fehlt | nicht_angemeldet | angemeldet (vorhanden, unbewiesen) |
ok (letzter echter Aufruf gelang) | kontingent (zeitweise, `bis` falls bekannt)
| fehler.

agy und codex buchen ihre Ausgaenge hier (#3645). Damit zeigt `/ai/wege` auch
dort `ok`, wo der Dienst die Anmeldung nicht lesen darf (#2937), und die Kette
der cloud-api bekommt feste Fehlercodes statt Freitext:

- `cli_fehlt`: nicht installiert oder nicht angemeldet.
- `kontingent`: zeitweise leer, mit `resets_at`, falls die CLI es nennt.

Die HTTP-Status bleiben, und der alte Wortlaut steht in `hint`, weil die cloud-api
bis zu ihrem Umbau daran erkennt. Gespeichert werden nie Prompt oder Fehlertext.
"""
from __future__ import annotations

import json
import logging
import os
import pwd
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from fastapi import HTTPException

from ai.services import cli_dienst, openai_text

logger = logging.getLogger(__name__)

_AGY_TOKEN = (".gemini", "antigravity-cli", "antigravity-oauth-token")
_CLI_WEGE = ("agy", "codex")
# Ein Fehlschlag gilt so lange. Danach zaehlt wieder die Anmeldung auf der Platte,
# sonst bliebe ein Weg nach neuer Anmeldung rot, bis zufaellig jemand ihn ruft.
_FEHLER_GILT_S = 3600
_KONTINGENT_OHNE_BIS_GILT_S = 24 * 3600

# Wortlaute aus dem Journal von arkserver, 07.-21.11.2026.
_QUOTA = re.compile(r"individual quota|resource_exhausted|usage.?limit|quota (?:reached|exceeded)|"
                    r"out of credits", re.I)
_ANMELDUNG = re.compile(r"not logged in|please (?:run|use) .{0,20}login|could not be refreshed|"
                        r"invalid[_ ]grant|unauthorized|token_expired", re.I)
# agy meldet Anmeldung und Zeitueberschreitung mit demselben Satz.
_UNKLAR = re.compile(r"authentication failed", re.I)
_AGY_RESET = re.compile(r"resets in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", re.I)
_CODEX_RESET = re.compile(r"try again at\s+([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+"
                          r"(\d{4})\s+(\d{1,2}):(\d{2})\s*([AP]M)", re.I)
_MONATE = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                       "sep", "oct", "nov", "dec"), start=1)}


def einordnen(text: str) -> Optional[tuple]:
    """(Fehlercode, Zustand fuer /ai/wege) oder None, wenn es keins von beiden ist."""
    t = text or ""
    if _QUOTA.search(t):
        return "kontingent", "kontingent"
    if _UNKLAR.search(t):
        return "cli_fehlt", "fehler"
    if _ANMELDUNG.search(t):
        return "cli_fehlt", "nicht_angemeldet"
    return None


def resets_aus(text: str, jetzt: Optional[datetime] = None) -> Optional[str]:
    """ISO-8601 aus dem Fehlertext, sonst None. Die Uhrzeit von codex ist Ortszeit des Hosts."""
    t = text or ""
    basis = (jetzt or datetime.now().astimezone()).replace(microsecond=0)
    m = _AGY_RESET.search(t)
    if m and any(m.groups()):
        h, mi, s = (int(x or 0) for x in m.groups())
        try:
            return (basis + timedelta(hours=h, minutes=mi, seconds=s)).isoformat()
        except OverflowError:
            return None
    m = _CODEX_RESET.search(t)
    if not m:
        return None
    monat = _MONATE.get(m.group(1).lower())
    if not monat:
        return None
    stunde = int(m.group(4)) % 12 + (12 if m.group(6).upper() == "PM" else 0)
    try:
        return datetime(int(m.group(3)), monat, int(m.group(2)), stunde, int(m.group(5))) \
            .astimezone().replace(microsecond=0).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _ausgang_pfad(weg: str) -> Path:
    return Path(os.getenv("VERBRAUCH_DIR") or "/var/lib/api-ai/verbrauch") / f"ki_weg_{weg}.json"


def _ausgang_lesen(weg: str) -> dict:
    try:
        d = json.loads(_ausgang_pfad(weg).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def ausgang(weg: str, art: str, bis: Optional[str] = None) -> None:
    """Bucht den letzten Ausgang. Ein Fehler hier trifft nie den Aufruf."""
    if weg not in _CLI_WEGE:
        return
    tmp = None
    try:
        pfad = _ausgang_pfad(weg)
        pfad.parent.mkdir(parents=True, exist_ok=True)
        a = _ausgang_lesen(weg)
        a.update(art=art, bis=bis, zeit=time.time())
        if art == "ok":
            a["letzter_erfolg"] = datetime.now().astimezone().isoformat(timespec="seconds")
        fd, tmp = tempfile.mkstemp(dir=pfad.parent, prefix=f".ki_weg_{weg}.")   # 0600
        with os.fdopen(fd, "w") as f:
            json.dump(a, f)
        os.replace(tmp, pfad)
        tmp = None
    except Exception as e:                                    # noqa: BLE001
        logger.warning("ki_wege: Ausgang nicht gebucht (%s)", type(e).__name__)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def fehler(weg: str, status: int, text: str, felder: Optional[dict] = None) -> Optional[HTTPException]:
    """HTTPException mit festem Code, oder None, wenn der Text keins von beiden ist."""
    e = einordnen(text)
    if e is None:
        return None
    code, zustand = e
    bis = resets_aus(text) if code == "kontingent" else None
    ausgang(weg, zustand, bis)
    return HTTPException(status_code=status, detail={
        "error": code, "weg": weg, "resets_at": bis, "hint": text, **(felder or {})})


def nicht_installiert(weg: str, text: str) -> HTTPException:
    ausgang(weg, "fehlt")
    return HTTPException(status_code=500, detail={"error": "cli_fehlt", "weg": weg, "resets_at": None,
                                                  "hint": text})


def _weg(zustand: str, **extra) -> dict:
    return {"zustand": zustand, "bis": None, "letzter_erfolg": None, **extra}


def _agy_home() -> str:
    user = (os.getenv("AGY_RUN_AS_USER") or "").strip()
    if user:
        return pwd.getpwnam(user).pw_dir
    return os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir


def agy() -> dict:
    if not shutil.which("agy") and not cli_dienst.aktiv():
        return _weg("fehlt")
    try:
        os.stat(os.path.join(_agy_home(), *_AGY_TOKEN))
    except FileNotFoundError:
        return _weg("nicht_angemeldet")
    except KeyError:
        return _weg("fehler", grund="agy_benutzer_fehlt")
    except PermissionError:
        # Fehlendes Leserecht sagt nichts ueber die Anmeldung.
        return _weg("fehler", grund="anmeldung_nicht_lesbar")
    except OSError:
        return _weg("fehler", grund="anmeldung_nicht_pruefbar")
    return _weg("angemeldet")


def codex(cli_status: Callable[[str], dict]) -> dict:
    s = cli_status("codex") or {}
    if not s.get("ok"):
        fehler = str(s.get("error") or "")
        if "not found" in fehler:
            return _weg("fehlt")
        return _weg("fehler", grund="cli_nicht_startbar")
    auth = s.get("auth") or {}
    if auth.get("ok") is True:
        return _weg("angemeldet")
    grund = auth.get("reason")
    if grund == "not_readable_by_service" or auth.get("ok") is None:
        return _weg("fehler", grund="anmeldung_nicht_lesbar")
    if grund == "expired" and auth.get("has_refresh_token"):
        # Der Zugriffstoken laeuft regelmaessig ab, codex erneuert ihn beim naechsten Aufruf.
        return _weg("angemeldet", grund="access_token_abgelaufen")
    if grund in ("absent", "expired", "unreadable"):
        return _weg("nicht_angemeldet", grund=grund)
    return _weg("fehler", grund=str(grund or "unbekannt")[:40])


def _mit_ausgang(weg: str, basis: dict) -> dict:
    """Legt den letzten echten Aufruf ueber das, was auf der Platte sichtbar ist."""
    a = _ausgang_lesen(weg)
    if not a:
        return basis
    aus = {**basis, "letzter_erfolg": a.get("letzter_erfolg")}
    z = basis["zustand"]
    # Darf der Dienst die Anmeldung nicht lesen, sagt die Platte nichts: Dann zaehlt der Aufruf.
    offen = z == "angemeldet" or basis.get("grund") == "anmeldung_nicht_lesbar"
    if z == "fehlt" or not offen:
        return aus
    art = a.get("art")
    try:
        alter = time.time() - float(a.get("zeit") or 0)
    except (TypeError, ValueError):
        alter = float("inf")

    def setzen(zustand: str, **extra) -> dict:
        neu = {k: v for k, v in aus.items() if k != "grund"}
        neu.update(zustand=zustand, **extra)
        return neu

    if art == "ok":
        return setzen("ok")
    if art == "kontingent":
        bis = a.get("bis")
        try:
            laeuft = datetime.fromisoformat(bis) > datetime.now().astimezone() if bis \
                else alter < _KONTINGENT_OHNE_BIS_GILT_S
        except (ValueError, TypeError):
            laeuft = alter < _KONTINGENT_OHNE_BIS_GILT_S
        # Ein Kontingentfehler belegt auch: Die Anmeldung gilt.
        return setzen("kontingent", bis=bis) if laeuft else setzen("angemeldet")
    if art in ("fehlt", "nicht_angemeldet", "fehler") and alter < _FEHLER_GILT_S:
        return setzen(art, grund="letzter_aufruf")
    return aus


def wege(cli_status: Callable[[str], dict]) -> dict:
    return {"wege": {"agy": _mit_ausgang("agy", agy()),
                     "codex": _mit_ausgang("codex", codex(cli_status)),
                     "openai_schluessel": openai_text.weg_zustand()}}
