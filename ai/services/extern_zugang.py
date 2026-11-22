"""#3656: Zugang fuer eine andere Instanz (heute P3) auf Hydrogen ueber unseren Kern.

Alex, 21.11.: Arcturian und P3 sollen sich gleich verhalten. P3 bekommt keine
eigene Google-Anmeldung (Bereich cloud-platform auf einer VM mit fremdem root),
sondern ruft Hydrogen hier auf, mit eigenem Schluessel und eigenem Tagesdeckel.

Die Datei `EXTERN_ZUGANG_DATEI` (Vorgabe /etc/api-ai/extern_zugang.json) haelt
je Aufrufer nur den SHA-256 des Schluessels und den Deckel:

    {"p3": {"sha256": "<64 hex>", "tag_max": 100, "aufrufer": ["Cloud"]}}

`aufrufer` ist optional: Fehlt es, entscheidet allein die Instanz (dort
GEMINI_KERN_AUFRUFER). Steht es da, muss der Kopf X-Extern-Aufrufer einen
dieser Namen tragen. Freigabe weiterer Aufrufer ist ein Eintrag, kein Code.

Ohne Datei ist der Weg aus (503 `extern_aus`), so auf allen Instanzen ausser
dem Kern. Widerruf: Eintrag loeschen, wirkt beim naechsten Aufruf, ohne Neustart.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from typing import Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger("api-ai.extern")

DATEI_ENV = "EXTERN_ZUGANG_DATEI"
DATEI_VORGABE = "/etc/api-ai/extern_zugang.json"
TAG_MAX_HOECHSTENS = 1000
_KENNUNG = re.compile(r"[a-z0-9-]{1,32}")
_HASH = re.compile(r"[0-9a-f]{64}")


def _eintraege() -> dict:
    pfad = os.getenv(DATEI_ENV) or DATEI_VORGABE
    try:
        with open(pfad, encoding="utf-8") as f:
            roh = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.error("extern_zugang: Datei nicht lesbar (%s)", type(e).__name__)
        return {}
    gut = {}
    if isinstance(roh, dict):
        for kennung, e in roh.items():
            if not (isinstance(kennung, str) and _KENNUNG.fullmatch(kennung) and isinstance(e, dict)):
                continue
            h = str(e.get("sha256") or "").lower()
            try:
                tag_max = int(e.get("tag_max"))
            except (TypeError, ValueError):
                continue
            liste = e.get("aufrufer")
            if liste is not None and not (isinstance(liste, list) and all(isinstance(x, str) for x in liste)):
                continue
            if _HASH.fullmatch(h) and 0 < tag_max <= TAG_MAX_HOECHSTENS:
                gut[kennung] = (h, tag_max, None if liste is None else frozenset(liste))
    return gut


def pruefen(schluessel: Optional[str], aufrufer: Optional[str] = None) -> Tuple[str, int, str]:
    """(kennung, tag_max, aufrufer) oder HTTPException. Laeuft vor dem Lesen des Rumpfs."""
    eintraege = _eintraege()
    if not eintraege:
        raise HTTPException(status_code=503, detail={
            "error": "extern_aus", "hint": "Diese Instanz bietet keinen externen Zugang an."})
    roh = (schluessel or "").strip()
    try:
        ist = hashlib.sha256(roh.encode("ascii")).hexdigest() if roh else ""
    except UnicodeEncodeError:
        ist = ""
    treffer = None
    for kennung, (soll, tag_max, liste) in sorted(eintraege.items()):
        # Jeden Eintrag vergleichen, damit die Laufzeit nicht verraet, welcher passte.
        if ist and hmac.compare_digest(ist, soll) and treffer is None:
            treffer = (kennung, tag_max, liste)
    if treffer is None:
        raise HTTPException(status_code=401, detail={
            "error": "extern_schluessel_required",
            "hint": "Dieser Weg verlangt den eigenen Schluessel im Kopf `X-API-KEY`."})
    kennung, tag_max, liste = treffer
    wer = (aufrufer or "").strip()[:64] or "(unbekannt)"
    if liste is not None and wer not in liste:
        logger.warning("extern_zugang: %s Aufrufer %r nicht freigegeben", kennung, wer)
        raise HTTPException(status_code=403, detail={
            "error": "extern_aufrufer_gesperrt", "aufrufer": wer,
            "hint": "Dieser Aufrufer ist fuer den externen Zugang nicht freigeschaltet."})
    return kennung, tag_max, wer
