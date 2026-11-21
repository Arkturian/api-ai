"""Was wuerde ein Schluesseltor auf unserem Kern sperren? (#1184, #3607)

Dieses Modul sperrt NICHTS. Es beurteilt jede Anfrage auf den Modell- und
Bezahlpfaden so, wie ein spaeteres Tor sie beurteilen wuerde, und schreibt das
Urteil in eine Tagesdatei. Ob das Tor je scharf wird, entscheidet Alexander
anhand dieser Daten (#1184).

Durch kaeme:

- der gueltige `API_ACCESS_KEY`,
- ein gepruefter JWT vom Typ `agent` oder `service`,
- uebergangsweise ein Dienst auf diesem Host (`von_diesem_host`).

Nicht durch kaemen `user` und `delegated`: Dahinter stehen Menschen, auch ueber
Connectoren (#3004). Ebenso wenig Tokens aus API_KEY_TOR_NUR_REALTIME, und zwar
auch vom eigenen Host: Sie gelten nur fuer den eigenen Grant-Weg der Stimme.

Die Hostausnahme gilt nur fuer dieses Urteil. `_ag.betreiber` und
`require_operator_key` bleiben davon unberuehrt.

Ins Protokoll kommen nie Token oder Schluessel. Bei `user` und `delegated`
auch kein `sub`, das waere eine Person. Ein Fehler hier trifft nie die Anfrage.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from fastapi import HTTPException

from ai import auth_guard as _ag
from ai.services import auth_konfig, jwt_pruefung

logger = logging.getLogger("api-ai.authwatch")

LOG_DIR_ENV = "KERN_TOR_LOG_DIR"
NUR_REALTIME_ENV = "API_KEY_TOR_NUR_REALTIME"
SCHLUESSEL_ENV = "API_ACCESS_KEY"

_LOG_DIR_VORGABE = "/var/lib/api-ai/kern_tor"
_NUR_REALTIME_VORGABE = ("agent:wave-3d", "agent:productfinder-internal-demo")
_ZUGELASSENE_TYPEN = ("agent", "service")

DURCH = "durch"
WUERDE_SPERREN = "WUERDE_SPERREN"
UNKLAR = "unklar"

_MAX_GLEICHZEITIG = 8
_MAX_BYTES_JE_TAG = 20 * 1024 * 1024

_laufend = 0
_uebersprungen = 0
_aufgaben: set = set()
_voll_gemeldet: Optional[str] = None


def _nur_realtime() -> frozenset:
    roh = os.getenv(NUR_REALTIME_ENV)
    if roh is None:
        return frozenset(_NUR_REALTIME_VORGABE)
    return frozenset(t.strip() for t in roh.split(",") if t.strip())


def _trifft(pfad: str, kandidaten: Iterable[str]) -> bool:
    return any(pfad == k or pfad.startswith(k + "/") for k in kandidaten)


def gilt(methode: str, pfad: str, schutz: Sequence[str], offen: Sequence[str]) -> bool:
    """Nur auf unserem Kern. Eine Kundeninstanz hat ihr eigenes Modelltor."""
    if methode == "OPTIONS" or _trifft(pfad, offen) or not _trifft(pfad, schutz):
        return False
    return auth_konfig.zustand() == auth_konfig.VORGABE


def _schluessel_stimmt(soll: str, ist: str) -> bool:
    # Als Bytes: compare_digest wirft bei Text ausserhalb von ASCII.
    return bool(soll and ist) and hmac.compare_digest(ist.encode("utf-8", "replace"),
                                                      soll.encode("utf-8", "replace"))


def urteil(schluessel_soll: str, schluessel_ist: str,
           authorization: Optional[str], lokal: bool) -> dict:
    """Wie entschiede das Tor? Liefert urteil, grund, typ, sub und mandant."""
    erg = {"urteil": WUERDE_SPERREN, "grund": "anonym", "typ": None, "sub": None, "mandant": None}
    if _schluessel_stimmt(schluessel_soll, schluessel_ist):
        erg.update(urteil=DURCH, grund="schluessel")
        return erg

    jwt_grund = None
    try:
        claims = jwt_pruefung.pruefen(authorization)
    except HTTPException as e:
        claims = None
        jwt_grund = "jwt_nicht_pruefbar" if e.status_code == 503 else "jwt_ungueltig"
    if claims:
        typ = str(claims.get("type") or "")[:24]
        erg.update(typ=typ, mandant=str(claims.get("tenant") or claims.get("tenant_id") or "")[:48] or None)
        if typ in _ZUGELASSENE_TYPEN:
            sub = str(claims.get("sub") or "")[:64]
            erg["sub"] = sub or None
            if sub in _nur_realtime():
                # Vor der Hostausnahme: Ein Proxy auf diesem Host darf aus einem
                # Stimm-Token keinen allgemeinen Zugang machen.
                erg["grund"] = "nur_realtime"
                return erg
            erg.update(urteil=DURCH, grund="jwt")
            return erg
        jwt_grund = "typ_nicht_zugelassen"

    if lokal:
        erg.update(urteil=DURCH, grund="lokal")
        return erg
    if jwt_grund == "jwt_nicht_pruefbar":
        erg.update(urteil=UNKLAR, grund=jwt_grund)
        return erg
    if schluessel_ist and not schluessel_soll:
        erg.update(urteil=UNKLAR, grund="schluessel_nicht_pruefbar")
        return erg
    erg["grund"] = jwt_grund or ("schluessel_falsch" if schluessel_ist else "anonym")
    return erg


def _schreiben(zeile: dict) -> None:
    global _voll_gemeldet
    jetzt = datetime.now()
    tag = f"{jetzt:%Y-%m-%d}"
    verzeichnis = Path(os.getenv(LOG_DIR_ENV) or _LOG_DIR_VORGABE)
    verzeichnis.mkdir(mode=0o700, parents=True, exist_ok=True)
    datei = verzeichnis / f"{tag}.jsonl"
    try:
        groesse = datei.stat().st_size
    except FileNotFoundError:
        groesse = 0
    if groesse >= _MAX_BYTES_JE_TAG:
        if _voll_gemeldet != tag:
            _voll_gemeldet = tag
            logger.error("Kern-Tor-Protokoll: Tagesgrenze erreicht, weitere Zeilen entfallen")
        return
    daten = (json.dumps(zeile, ensure_ascii=False) + "\n").encode("utf-8", "replace")
    fd = os.open(datei, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, daten)
    finally:
        os.close(fd)


def _auswerten_sync(anfrage: dict) -> None:
    erg = urteil((os.getenv(SCHLUESSEL_ENV) or "").strip(), anfrage["schluessel"],
                 anfrage["authorization"], anfrage["lokal"])
    zeile = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
             "methode": anfrage["methode"], "pfad": anfrage["pfad"], **erg,
             "lokal": anfrage["lokal"], "quelle": anfrage["quelle"], "ua": anfrage["ua"],
             "etikett": anfrage["etikett"], "hat_schluessel": bool(anfrage["schluessel"]),
             "hat_bearer": bool(anfrage["authorization"]),
             "weitergereicht": anfrage["weitergereicht"]}
    if anfrage.get("uebersprungen"):
        zeile["uebersprungen"] = anfrage["uebersprungen"]
    _schreiben(zeile)


async def _auswerten(anfrage: dict) -> None:
    global _laufend
    try:
        await asyncio.to_thread(_auswerten_sync, anfrage)
    except Exception as e:
        logger.warning("Kern-Tor-Beobachtung: %s", type(e).__name__)
    finally:
        _laufend -= 1


def _starten(anfrage: dict) -> None:
    """Laeuft neben der Anfrage: Eine langsame auth-api haelt niemanden auf."""
    global _laufend, _uebersprungen
    if _laufend >= _MAX_GLEICHZEITIG:
        _uebersprungen += 1
        return
    anfrage["uebersprungen"], _uebersprungen = _uebersprungen, 0
    _laufend += 1
    aufgabe = asyncio.get_running_loop().create_task(_auswerten(anfrage))
    _aufgaben.add(aufgabe)
    aufgabe.add_done_callback(_aufgaben.discard)


def beobachten(request, schutz: Sequence[str], offen: Sequence[str]) -> None:
    """Nimmt die Anfrage auf. Wirft nie und veraendert nichts an ihr."""
    try:
        pfad = request.url.path
        if not gilt(request.method, pfad, schutz, offen):
            return
        k = request.headers
        quelle = (k.get("x-real-ip") or (request.client.host if request.client else "") or "?")[:64]
        _starten({"methode": request.method, "pfad": pfad[:120],
                  "schluessel": (k.get("x-api-key") or "").strip(),
                  "authorization": k.get("authorization"),
                  "lokal": _ag.von_diesem_host(request), "quelle": quelle,
                  "ua": (k.get("user-agent") or "")[:60],
                  "etikett": (k.get("x-agent-name") or "").strip()[:64] or None,
                  "weitergereicht": bool(k.get("x-api-ai-forwarded"))})
    except Exception as e:
        logger.warning("Kern-Tor-Beobachtung: %s", type(e).__name__)
