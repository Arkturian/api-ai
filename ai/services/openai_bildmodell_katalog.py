"""Aktuelles OpenAI-Bildmodell automatisch waehlen (Alex, 11.11.2026, #2699).

Intern soll keine Versionsnummer festgeschrieben sein: Gibt es ein neueres
gpt-image-Modell, nimmt AiApi das. Quelle ist GET /v1/models (kostenlos).
Gewaehlt wird je Stufe:

- ``qualitaet``: hoechste Version ohne Endung bzw. mit einer Endung, die
  nicht als schnell bekannt ist (heute ``-sunburst``)
- ``schnell``: hoechste Version mit ``-flare`` oder ``-mini``

Nicht in Frage kommen datierte Schnappschuesse (``…-2026-09-08``) und
Modelle, deren ``shutdown_date`` in weniger als 30 Tagen liegt.

Ist OpenAI nicht erreichbar oder fehlt der Schluessel, gilt der zuletzt
gelesene Stand, sonst die feste Rueckfallwahl unten. Ein Bild scheitert
also nie an der Katalogabfrage.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import re
import time
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

STUFEN = ("qualitaet", "schnell")
SCHNELL_ENDUNGEN = ("flare", "mini", "fast")

# Stand 11.11.2026, gelesen in GET /v1/models. Gilt nur, solange der
# Katalog noch nie gelesen werden konnte.
RUECKFALL = {
    "qualitaet": os.getenv("OPENAI_IMAGE_DEFAULT_MODEL", "gpt-image-2.6-sunburst"),
    "schnell": os.getenv("OPENAI_IMAGE_FAST_MODEL", "gpt-image-2.6-flare"),
}

_NAME_RE = re.compile(r"^gpt-image-(\d+(?:\.\d+)*)(?:-([a-z]+))?$")
_GUELTIG_S = int(os.getenv("OPENAI_IMAGE_CATALOG_TTL_S", "21600"))  # 6 h
_FEHLER_PAUSE_S = 300
_MIN_RESTLAUFZEIT_TAGE = 30

_stand: Dict[str, str] = {}
_gelesen_um: float = 0.0
_letzter_versuch: float = 0.0
_sperre = asyncio.Lock()


def version_von(model: str) -> Optional[Tuple[int, ...]]:
    m = _NAME_RE.match(model or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def stufe_von(model: str) -> str:
    """Welche Stufe meint ein Aufrufer mit diesem Namen?"""
    m = _NAME_RE.match(model or "")
    endung = m.group(2) if m else None
    if endung in SCHNELL_ENDUNGEN or (model or "").endswith(SCHNELL_ENDUNGEN):
        return "schnell"
    return "qualitaet"


def ist_gpt_image(model: str) -> bool:
    return (model or "").startswith("gpt-image")


def waehlen(modelle: Iterable[dict], heute: Optional[dt.date] = None) -> Dict[str, str]:
    """Aus der Liste von /v1/models je Stufe das aktuelle Modell."""
    heute = heute or dt.date.today()
    grenze = heute + dt.timedelta(days=_MIN_RESTLAUFZEIT_TAGE)
    beste: Dict[str, Tuple[Tuple[int, ...], int, str]] = {}
    for m in modelle:
        mid = m.get("id") or ""
        treffer = _NAME_RE.match(mid)
        if not treffer:
            continue
        ende = m.get("shutdown_date")
        if ende:
            try:
                if dt.date.fromisoformat(str(ende)[:10]) <= grenze:
                    continue
            except ValueError:
                continue
        stufe = "schnell" if treffer.group(2) in SCHNELL_ENDUNGEN else "qualitaet"
        schluessel = (version_von(mid), int(m.get("created") or 0), mid)
        if stufe not in beste or schluessel > beste[stufe]:
            beste[stufe] = schluessel
    wahl = {s: v[2] for s, v in beste.items()}
    # Fehlt eine Stufe ganz, lieber das aktuelle Modell der anderen als ein altes.
    for s in STUFEN:
        if s not in wahl:
            andere = [wahl[x] for x in STUFEN if x in wahl]
            if andere:
                wahl[s] = andere[0]
    return wahl


def aktuell(stufe: str = "qualitaet") -> str:
    return _stand.get(stufe) or RUECKFALL[stufe]


def aufloesen(model: Optional[str]) -> Optional[str]:
    """Jeder gpt-image-Name (auch abgeschaltete und kuenftige) -> aktuelles
    Modell seiner Stufe. Andere Namen bleiben unveraendert."""
    if not model or not ist_gpt_image(model):
        return model
    return aktuell(stufe_von(model))


def stand() -> dict:
    return {
        "qualitaet": aktuell("qualitaet"),
        "schnell": aktuell("schnell"),
        "quelle": "openai_models" if _stand else "rueckfall",
        "gelesen_um": (dt.datetime.fromtimestamp(_gelesen_um, dt.timezone.utc).isoformat()
                       if _gelesen_um else None),
    }


async def aktualisieren(erzwingen: bool = False) -> None:
    """Liest /v1/models, wenn der Stand aelter als 6 h ist. Wirft nie."""
    global _gelesen_um, _letzter_versuch
    jetzt = time.time()
    if not erzwingen and _stand and jetzt - _gelesen_um < _GUELTIG_S:
        return
    if not erzwingen and jetzt - _letzter_versuch < _FEHLER_PAUSE_S:
        return
    async with _sperre:
        if not erzwingen and _stand and time.time() - _gelesen_um < _GUELTIG_S:
            return
        _letzter_versuch = time.time()
        schluessel = os.getenv("OPENAI_API_KEY")
        if not schluessel:
            return
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get("https://api.openai.com/v1/models",
                                headers={"Authorization": f"Bearer {schluessel}"})
            if r.status_code != 200:
                logger.warning("Bildmodell-Katalog: /v1/models HTTP %s, behalte %s",
                               r.status_code, stand())
                return
            wahl = waehlen(r.json().get("data") or [])
        except Exception as e:  # Katalog darf nie ein Bild verhindern
            logger.warning("Bildmodell-Katalog nicht lesbar (%s), behalte %s", e, stand())
            return
        if not wahl:
            logger.warning("Bildmodell-Katalog: kein gpt-image-Modell in /v1/models")
            return
        vorher = dict(_stand) or dict(RUECKFALL)
        _stand.update(wahl)
        _gelesen_um = time.time()
        for s in STUFEN:
            if wahl.get(s) and wahl[s] != vorher.get(s):
                logger.warning("Bildmodell-Katalog: Stufe %s wechselt %s -> %s",
                               s, vorher.get(s), wahl[s])
