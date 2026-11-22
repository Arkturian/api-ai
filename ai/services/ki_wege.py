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

import fcntl
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
_CLI_WEGE = ("agy", "codex", "claude")
# Ein Fehlschlag gilt so lange. Danach zaehlt wieder die Anmeldung auf der Platte,
# sonst bliebe ein Weg nach neuer Anmeldung rot, bis zufaellig jemand ihn ruft.
_FEHLER_GILT_S = 3600
_KONTINGENT_OHNE_BIS_GILT_S = 24 * 3600
CLAUDE_HYDROGEN_MAX_ENV = "CLAUDE_ROUTE_HYDROGEN_TAG_MAX"
CLAUDE_HYDROGEN_MAX_VORGABE = 85

# Wortlaute aus dem Journal von arkserver, 07.-21.11.2026.
_QUOTA = re.compile(r"individual quota|resource_exhausted|usage.?limit|rate.?limit|insufficient_quota|"
                    r"quota (?:reached|exceeded)|exhausted your|out of credits", re.I)
_ANMELDUNG = re.compile(r"not logged in|please (?:run|use) .{0,20}login|could not be refreshed|"
                        r"invalid[_ ]grant|unauthorized|token_expired", re.I)
# agy meldet Anmeldung und Zeitueberschreitung mit demselben Satz.
_UNKLAR = re.compile(r"authentication failed", re.I)
# Formen wie in cloud-api `wieder_ab` (#3128): "Resets in 42h15m28s", "reset after 9h15m42s",
# "try again at Nov 21st, 2026 1:52 PM", "... Nov 19th 7:20 PM", "... 10:25 PM", "... in 3 hours".
_RESET_ANKER = re.compile(r"(?:try again|resets?|reset will occur|available again)\s+(?:at|after|in|on)\s+"
                          r"([^.\n\"]{2,60})", re.I)
_KURZDAUER = re.compile(r"^\s*(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+)s)?(?=\W|$)", re.I)
_RELATIV = re.compile(r"(\d+)\s*(days?|hours?|hrs?|minutes?|mins?)\b", re.I)
_UHR = re.compile(r"\b(\d{1,2}):(\d{2})\s*([AP]M)?\b", re.I)
_DATUM = re.compile(r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b,?(?:\s+(\d{4})\b)?", re.I)
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
    """ISO-8601 aus dem Fehlertext, sonst None. Uhrzeiten sind Ortszeit des Hosts,
    denn die CLI laeuft auf demselben Rechner."""
    anker = _RESET_ANKER.search(text or "")
    if not anker:
        return None
    roh = anker.group(1).strip()
    basis = (jetzt or datetime.now().astimezone()).replace(microsecond=0)

    def lokal(naiv: datetime) -> str:
        # Ohne vorgegebene Zeit rechnet astimezone die Sommerzeit des Zieltags richtig.
        ziel = naiv.replace(tzinfo=basis.tzinfo) if jetzt else naiv.astimezone()
        return ziel.replace(microsecond=0).isoformat()

    try:
        m = _KURZDAUER.match(roh)
        if m and any(m.groups()):
            h, mi, se = (int(x or 0) for x in m.groups())
            return (basis + timedelta(hours=h, minutes=mi, seconds=se)).isoformat()
        uhr = _UHR.search(roh)
        std = mnt = None
        if uhr:
            std, mnt = int(uhr.group(1)), int(uhr.group(2))
            ampm = (uhr.group(3) or "").upper()
            if ampm == "PM" and std < 12:
                std += 12
            elif ampm == "AM" and std == 12:
                std = 0
            if not (0 <= std <= 23 and 0 <= mnt <= 59):
                return None
        # Vor dem Datum: In "1 day 2 hours" saehe "day 2" sonst wie ein Monat aus.
        rel = _RELATIV.findall(roh) if not uhr else []
        if rel:
            sek = {"d": 86400, "h": 3600, "m": 60}
            return (basis + timedelta(seconds=sum(int(z) * sek[e[0].lower()] for z, e in rel))).isoformat()
        datum = _DATUM.search(roh)
        if datum:
            monat = _MONATE.get(datum.group(1).lower())
            if not monat:
                return None
            jahr = int(datum.group(3)) if datum.group(3) else basis.year
            ziel = datetime(jahr, monat, int(datum.group(2)), std or 0, mnt or 0)
            if not datum.group(3) and ziel < basis.replace(tzinfo=None) - timedelta(days=1):
                ziel = ziel.replace(year=jahr + 1)
            return lokal(ziel)
        if uhr:
            ziel = basis.replace(tzinfo=None, hour=std, minute=mnt, second=0)
            if ziel <= basis.replace(tzinfo=None):
                ziel += timedelta(days=1)
            return lokal(ziel)
    except (ValueError, OverflowError, OSError):
        return None
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


def hydrogen_tag_max() -> int:
    """Tagesdeckel fuer /ai/claude ueber agy. Vorgabe 85 (Cloud, #3652): zusammen mit
    den 500 der Sprechstile deutlich unter der bekannten Grenze von rund 970."""
    try:
        wert = int((os.getenv(CLAUDE_HYDROGEN_MAX_ENV) or "").strip())
    except ValueError:
        return CLAUDE_HYDROGEN_MAX_VORGABE
    return wert if wert > 0 else CLAUDE_HYDROGEN_MAX_VORGABE


def hydrogen_tag_buchen() -> None:
    """Zaehlt einen Aufruf. Am Deckel 429 `kontingent` bis Mitternacht. Ist der Zaehler
    nicht schreibbar, geht nichts hinaus: Eine Sperre von 130 h traefe alle Wege."""
    tag_buchen("claude_hydrogen", hydrogen_tag_max())


def tag_buchen(name: str, tag_max: int) -> None:
    """Tageszaehler `<name>_<tag>.json` unter VERBRAUCH_DIR, gemeinsam fuer
    /ai/claude (#3652) und /ai/extern/hydrogen (#3656, je Aufrufer)."""
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", name or ""):
        raise HTTPException(status_code=503, detail={"error": "hydrogen_zaehler_fehlt"})
    jetzt = datetime.now().astimezone().replace(microsecond=0)
    tag = jetzt.date().isoformat()
    verzeichnis = Path(os.getenv("VERBRAUCH_DIR") or "/var/lib/api-ai/verbrauch")
    tmp = None
    try:
        verzeichnis.mkdir(parents=True, exist_ok=True)
        fd = os.open(verzeichnis / f".{name}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "r+") as sperre:
            fcntl.flock(sperre, fcntl.LOCK_EX)
            pfad = verzeichnis / f"{name}_{tag}.json"
            try:
                n = int(json.loads(pfad.read_text()).get("anfragen") or 0)
            except FileNotFoundError:
                n = 0
            if n >= tag_max:
                naechster = (jetzt.replace(tzinfo=None) + timedelta(days=1)).replace(hour=0, minute=0, second=0)
                bis = naechster.astimezone().isoformat()
                sek = max(1, int((naechster.astimezone() - jetzt).total_seconds()))
                raise HTTPException(status_code=429, headers={"Retry-After": str(sek)}, detail={
                    "error": "kontingent", "weg": "agy", "grund": "tagesdeckel", "resets_at": bis,
                    "hint": "daily quota reached, try again at 00:00"})
            wfd, tmp = tempfile.mkstemp(dir=verzeichnis, prefix=f".{name}.")   # 0600
            with os.fdopen(wfd, "w") as f:
                json.dump({"anfragen": n + 1}, f)
            os.replace(tmp, pfad)
            tmp = None
    except (OSError, ValueError, TypeError, AttributeError) as e:
        logger.error("ki_wege: Hydrogen-Zaehler nicht nutzbar (%s)", type(e).__name__)
        raise HTTPException(status_code=503, detail={"error": "hydrogen_zaehler_fehlt"})
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def tag_stand(name: str) -> int:
    """Heutiger Zaehlerstand, 0 wenn es keinen gibt."""
    tag = datetime.now().astimezone().date().isoformat()
    pfad = Path(os.getenv("VERBRAUCH_DIR") or "/var/lib/api-ai/verbrauch") / f"{name}_{tag}.json"
    try:
        return int(json.loads(pfad.read_text()).get("anfragen") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def tag_zaehlen(name: str) -> None:
    """Zaehlt mit, sperrt nie (#3658: direkte /ai/gemini-Aufrufe im Google-Zaehler)."""
    try:
        tag_buchen(name, 10 ** 9)
    except HTTPException:
        logger.warning("ki_wege: Zaehler %s nicht gebucht", name)


def gesperrt(weg: str) -> tuple:
    """(True, bis) solange der letzte Aufruf des Wegs ein leeres Kontingent meldete."""
    a = _ausgang_lesen(weg)
    if a.get("art") != "kontingent":
        return False, None
    bis = a.get("bis")
    try:
        alter = time.time() - float(a.get("zeit") or 0)
    except (TypeError, ValueError):
        return False, None
    try:
        if bis:
            return (datetime.fromisoformat(bis) > datetime.now().astimezone()), bis
    except (ValueError, TypeError):
        pass
    return alter < _KONTINGENT_OHNE_BIS_GILT_S, None


def _anmeldung_mtime(weg: str) -> Optional[float]:
    """Wann die Anmeldung zuletzt geschrieben wurde, None wenn nicht lesbar."""
    try:
        if weg == "agy":
            pfad = os.path.join(_agy_home(), *_AGY_TOKEN)
        else:
            home = os.getenv("CODEX_HOME") or os.path.join(
                os.getenv("CLI_HOME") or pwd.getpwuid(os.getuid()).pw_dir, ".codex")
            pfad = os.path.join(home, "auth.json")
        return os.stat(pfad).st_mtime
    except (OSError, KeyError):
        return None


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
        # Die Platte zeigt die CLI samt Anmeldung: Das gebuchte „fehlt“ ist ueberholt.
        if art == "fehlt" and z == "angemeldet":
            return aus
        # Seit dem Fehlschlag neu angemeldet. Fuer `kontingent` gilt das bewusst nicht:
        # Eine Token-Erneuerung schreibt die Datei auch, das Kontingent bleibt leer.
        stand = _anmeldung_mtime(weg)
        if stand is not None and stand > time.time() - alter:
            return aus
        return setzen(art, grund="letzter_aufruf")
    return aus


def wege(cli_status: Callable[[str], dict]) -> dict:
    return {"wege": {"agy": _mit_ausgang("agy", agy()),
                     "codex": _mit_ausgang("codex", codex(cli_status)),
                     "openai_schluessel": openai_text.weg_zustand()}}
