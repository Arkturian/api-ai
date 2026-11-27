"""6586/I5: Dienstweg je Aufrufer aus dem Zugangsregister der cloud-api.

Vertrag (cloud-api 09a2f55c, backend/app/zugaenge_routes.py):

    GET /api/dienstwege/aufloesen?mandant=…&dienst=…&mensch=…
    -> {weg, anbieter, modell, zugang, version, tresor_verweis, quelle}

Nur mit dem Dienst-JWT `service:api-ai-arkserver` und dem Recht
`zugaenge:dienstweg:aufloesen` (AuthAPI 27.11.). Erst gilt der Weg des Menschen,
sonst die Vorgabe des Mandanten. Der Tresor-Verweis kommt nur bei einem Zugang
`z_…` mit gesetztem Schluessel, nie ein Wert. Den Verweis leiten wir nie selbst ab.

Fehler fuer den Aufrufer (fail-closed, nie ein anderer Weg):
- 403 `dienstweg_nicht_nutzbar` mit `grund`: `kein_dienstweg` (404 der cloud-api)
  oder der Grund aus 409 (`inaktiv`, `nicht_freigegeben`, `fremder_mandant`, `unbekannt`).
- 503 `dienstweg_nicht_lesbar` mit `grund`: Ausweis fehlt, cloud-api nicht
  erreichbar oder Abruf verweigert.

Die Adresse des Menschen geht nur an die cloud-api, nie ins Journal.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import httpx
from fastapi import HTTPException

logger = logging.getLogger("api-ai.dienstweg")

#: wie cloud-api zugaenge.DIENSTE
DIENSTE = ("conversation", "realtime", "bild", "video", "stt", "tts", "projektleitung", "prompt_agent")
_MANDANT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[a-z]{2,}$")
_GRUENDE_409 = ("inaktiv", "nicht_freigegeben", "fremder_mandant", "unbekannt")

_CACHE: Dict[Tuple[str, str, str], Tuple[float, object]] = {}


@dataclass(frozen=True)
class Weg:
    weg: str
    anbieter: Optional[str]
    modell: Optional[str]
    zugang: Optional[str]
    tresor_verweis: Optional[str]
    quelle: str
    version: Optional[int] = None


def basis() -> str:
    u = (os.getenv("DIENSTWEG_URL") or os.getenv("CLOUD_API_URL")
         or "https://cloud-api.arkserver.arkturian.com").strip()
    return u.rstrip("/")


def _ttl() -> float:
    try:
        return max(0.0, float(os.getenv("DIENSTWEG_TTL_S") or 60))
    except ValueError:
        return 60.0


def mensch_pruefen(mensch: Optional[str]) -> Optional[str]:
    m = (mensch or "").strip().lower() or None
    if m is not None and not _EMAIL.match(m):
        raise HTTPException(status_code=422, detail={"error": "mensch_ungueltig"})
    return m


def nicht_nutzbar(grund: str, mandant: str, dienst: str) -> HTTPException:
    return HTTPException(status_code=403, detail={
        "error": "dienstweg_nicht_nutzbar", "grund": grund, "mandant": mandant, "dienst": dienst,
        "hint": "Fuer diesen Dienst gibt es keinen nutzbaren Weg. Es laeuft kein anderer Anbieter."})


def nicht_lesbar(grund: str, mandant: str, dienst: str) -> HTTPException:
    return HTTPException(status_code=503, detail={
        "error": "dienstweg_nicht_lesbar", "grund": grund, "mandant": mandant, "dienst": dienst,
        "hint": "Der Dienstweg war nicht abrufbar. Es laeuft kein anderer Anbieter."})


def leeren() -> None:
    _CACHE.clear()


async def aufloesen(mandant: str, dienst: str, mensch: Optional[str] = None) -> Weg:
    """Weg des Menschen bzw. des Mandanten fuer einen Dienst. Wirft 403/503 (siehe oben)."""
    if not _MANDANT.fullmatch(mandant or ""):
        raise HTTPException(status_code=422, detail={"error": "mandant_ungueltig"})
    if dienst not in DIENSTE:
        raise HTTPException(status_code=422, detail={"error": "dienst_ungueltig", "erlaubt": list(DIENSTE)})
    mensch = mensch_pruefen(mensch)
    schluessel = (mandant, dienst, mensch or "")
    jetzt = time.monotonic()
    treffer = _CACHE.get(schluessel)
    if treffer and treffer[0] > jetzt:
        if isinstance(treffer[1], Weg):
            return treffer[1]
        raise nicht_nutzbar(str(treffer[1]), mandant, dienst)

    from ai.services import tresor
    try:
        tok = await tresor._dienst_token(mandant)
    except HTTPException as e:
        grund = (e.detail or {}).get("grund") if isinstance(e.detail, dict) else None
        raise nicht_lesbar(grund or "dienstausweis", mandant, dienst) from None
    params = {"mandant": mandant, "dienst": dienst}
    if mensch:
        params["mensch"] = mensch
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("DIENSTWEG_TIMEOUT_S") or 8)) as c:
            r = await c.get(f"{basis()}/api/dienstwege/aufloesen", params=params,
                            headers={"Authorization": f"Bearer {tok}"})
    except httpx.HTTPError as e:
        logger.warning("dienstweg: nicht erreichbar (%s) mandant=%s dienst=%s", type(e).__name__, mandant, dienst)
        raise nicht_lesbar("cloud_api_nicht_erreichbar", mandant, dienst) from None

    if r.status_code == 200:
        try:
            d = r.json() or {}
        except ValueError:
            raise nicht_lesbar("antwort_kein_json", mandant, dienst) from None
        weg = str(d.get("weg") or "").strip().lower()
        if not weg:
            raise nicht_lesbar("antwort_ohne_weg", mandant, dienst)
        w = Weg(weg=weg, anbieter=(str(d.get("anbieter")).lower() if d.get("anbieter") else None),
                modell=d.get("modell") or None, zugang=d.get("zugang") or None,
                tresor_verweis=d.get("tresor_verweis") or None,
                quelle=str(d.get("quelle") or "mandant"), version=d.get("version"))
        _CACHE[schluessel] = (jetzt + _ttl(), w)
        logger.info("dienstweg: mandant=%s dienst=%s mensch=%s weg=%s anbieter=%s quelle=%s", mandant, dienst,
                    "ja" if mensch else "nein", w.weg, w.anbieter, w.quelle)
        return w
    if r.status_code in (404, 409):
        grund = "kein_dienstweg"
        if r.status_code == 409:
            try:
                g = ((r.json() or {}).get("detail") or {}).get("grund")
            except (ValueError, AttributeError):
                g = None
            grund = g if g in _GRUENDE_409 else "nicht_nutzbar"
        _CACHE[schluessel] = (jetzt + _ttl(), grund)
        logger.info("dienstweg: mandant=%s dienst=%s mensch=%s %s", mandant, dienst,
                    "ja" if mensch else "nein", grund)
        raise nicht_nutzbar(grund, mandant, dienst)
    if r.status_code in (401, 403):
        tresor._TOKEN["wert"] = None
        logger.warning("dienstweg: Abruf verweigert (%s) mandant=%s", r.status_code, mandant)
        raise nicht_lesbar("aufloesen_verweigert", mandant, dienst)
    logger.warning("dienstweg: Fehler %s mandant=%s dienst=%s", r.status_code, mandant, dienst)
    raise nicht_lesbar(f"cloud_api_{r.status_code}", mandant, dienst)
