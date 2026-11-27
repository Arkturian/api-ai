"""6586/I5: Kein Kunden-Mandant rechnet auf dem Kern mit Abos oder Schluesseln des Betreibers.

Jeder POST /ai/* mit einem GEPRUEFTEN JWT eines anderen Mandanten als dem Betreiber
geht hier durch, bevor ein Anbieter laeuft. Der Pfad bestimmt den Dienst (bild,
video, stt, tts, realtime, conversation), der Dienstweg kommt aus dem Zugangsregister
der cloud-api (ai/services/dienstweg.py).

Auf dem Kern laeuft fuer Kunden heute nur der Schluesselweg von /ai/verarbeitung
(eigene Pruefung dort, ebenso /ai/jev und /ai/decisions ueber urteil_pruefen) und
der Pool-Weg: `weg: pool` heisst, der Mandant gehoert zum Pool des Kontoregisters
und darf die Wege des Betreibers nutzen (edera, p3; Cloud 27.11.). Dann gilt ein
Tagesdeckel je Mandant und Dienst (MANDANT_POOL_TAG, Vorgabe bild 50, video 10,
sonst 200 Aufrufe). Fuer alle anderen Wege gibt es auf dem Kern noch nichts mit
dem Schluessel des Kunden: Abo- und lokale Wege laufen auf seinen Rechnern
(Knotenweg folgt). Die Antwort nennt den Grund, es laeuft nie ein Weg des Betreibers.

MANDANT_WEGE:
- `beobachten` (Vorgabe): nichts wird abgewiesen, jede Abweisung, die `an` braeuchte,
  steht im Journal und in MANDANT_WEGE_DIR/<datum>.jsonl. Erst messen, dann verlangen.
- `an`: abweisen. Ein Fehler hier weist ebenfalls ab (fail-closed).
- `aus`: keine Pruefung.

Betreiber und Aufrufe ohne geprueftes JWT laufen unveraendert.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("api-ai.mandant_wege")

MODUS_ENV = "MANDANT_WEGE"
MODI = ("aus", "beobachten", "an")

#: pruefen selbst (Schluesselweg bzw. urteil_pruefen) oder sind Verwaltung des Mandanten
EIGENE = ("/ai/verarbeitung", "/ai/jev", "/ai/decisions", "/ai/mandanten/")

#: (Praefix, Dienst) -- das erste passende gilt
DIENST_JE_PFAD = (
    ("/ai/genimage", "bild"), ("/ai/generate-images", "bild"), ("/ai/upscale", "bild"),
    ("/ai/gendepth", "bild"), ("/ai/scene/", "bild"),
    ("/ai/genvideo", "video"),
    ("/ai/transcribe", "stt"),
    ("/ai/tts", "tts"), ("/ai/generate_speech", "tts"),
    ("/ai/realtime", "realtime"),
    ("/ai/claude", "conversation"), ("/ai/chatgpt", "conversation"), ("/ai/gemini", "conversation"),
    ("/ai/kette", "conversation"), ("/ai/deepseek", "conversation"), ("/ai/m3", "conversation"),
    ("/ai/openai", "conversation"), ("/ai/grok", "conversation"), ("/ai/translate", "conversation"),
    ("/ai/dialog/", "conversation"),
)


def modus() -> str:
    m = (os.getenv(MODUS_ENV) or "beobachten").strip().lower()
    return m if m in MODI else "an"           # Tippfehler: lieber streng


def dienst_fuer(pfad: str) -> Optional[str]:
    for praefix, dienst in DIENST_JE_PFAD:
        if _unter(pfad, praefix):
            return dienst
    return None


def _unter(pfad: str, praefix: str) -> bool:
    """/ai/tts deckt /ai/tts und /ai/tts/minimax, aber nicht /ai/ttsx oder /ai/openai-cost-…"""
    p = praefix.rstrip("/")
    return pfad == p or pfad.startswith(p + "/")


def _eigene(pfad: str) -> bool:
    return any(_unter(pfad, e) for e in EIGENE)


def _vermerken(zeile: dict) -> None:
    try:
        d = Path(os.getenv("MANDANT_WEGE_DIR") or "/var/lib/api-ai/mandant_wege")
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{datetime.now():%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(zeile, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("mandant_wege: Vermerk nicht geschrieben (%s)", type(e).__name__)


POOL_DECKEL_VORGABE = {"bild": 50, "video": 10}
POOL_DECKEL_SONST = 200


def pool_deckel(dienst: str) -> int:
    """Tagesdeckel je Mandant und Dienst auf dem Pool-Weg (MANDANT_POOL_TAG als JSON, z. B. {"bild": 50})."""
    werte = dict(POOL_DECKEL_VORGABE)
    roh = (os.getenv("MANDANT_POOL_TAG") or "").strip()
    if roh:
        try:
            d = json.loads(roh)
            if isinstance(d, dict):
                werte.update({str(k): int(v) for k, v in d.items() if int(v) >= 0})
        except (ValueError, TypeError):
            logger.warning("mandant_wege: MANDANT_POOL_TAG ungueltig, Vorgabe gilt")
    return int(werte.get(dienst, werte.get("sonst", POOL_DECKEL_SONST)))


def _zaehlen(mandant: str, dienst: str) -> int:
    """Zaehlt einen Aufruf des Tages und gibt den neuen Stand zurueck (nur Zahlen, kein Inhalt)."""
    import fcntl
    d = Path(os.getenv("MANDANT_WEGE_DIR") or "/var/lib/api-ai/mandant_wege") / "zaehler"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{datetime.now():%Y-%m-%d}_{mandant}_{dienst}.txt"
    with open(p, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            n = int((f.read() or "0").strip() or 0)
        except ValueError:
            n = 0
        n += 1
        f.seek(0)
        f.truncate()
        f.write(str(n))
    return n


async def urteil(pfad: str, mandant: str, mensch: Optional[str]) -> Optional[HTTPException]:
    """None: Der Mandant darf diesen Pfad nutzen (Pool-Weg innerhalb des Deckels).
    Sonst die Abweisung mit Grund."""
    dienst = dienst_fuer(pfad)
    if dienst is None:
        return HTTPException(status_code=403, detail={
            "error": "dienstweg_nicht_nutzbar", "grund": "kein_dienst_fuer_pfad", "mandant": mandant,
            "hint": "Dieser Pfad gehoert zu keinem Dienst des Zugangsregisters und laeuft nur fuer den Betreiber."})
    from ai.services import dienstweg
    try:
        w = await dienstweg.aufloesen(mandant, dienst, mensch)
    except HTTPException as e:
        return e
    if w.weg == "pool":
        grenze = pool_deckel(dienst)
        n = _zaehlen(mandant, dienst)
        if n > grenze:
            return HTTPException(status_code=429, detail={
                "error": "kontingent", "grund": "pool_tagesdeckel", "mandant": mandant, "dienst": dienst,
                "weg": "pool", "pool": w.pool, "grenze": grenze,
                "hint": "Der Tagesdeckel dieses Mandanten auf dem gemeinsamen Pool ist erreicht."})
        return None
    return HTTPException(status_code=503, detail={
        "error": "weg_nicht_auf_dem_kern", "mandant": mandant, "dienst": dienst, "weg": w.weg,
        "anbieter": w.anbieter, "quelle": w.quelle,
        "hint": "Dieser Weg laeuft auf den Rechnern des Mandanten bzw. mit seinem Schluessel; auf dem Kern "
                "gibt es ihn fuer diesen Pfad noch nicht. Es laeuft kein Weg des Betreibers."})


async def pruefen(pfad: str, mandant: Optional[str], jwt_typ: Optional[str],
                  mensch_kopf: Optional[str], aufrufer: str) -> Optional[JSONResponse]:
    """None = weiter. Sonst die Abweisung (nur im Modus `an`)."""
    m = modus()
    if m == "aus" or mandant is None or _eigene(pfad):
        return None
    from ai.services import verarbeitung as v
    if mandant == v.betreiber():
        return None
    try:
        mensch = mensch_kopf if jwt_typ == "service" else None
        e = await urteil(pfad, mandant, mensch)
    except Exception as fehler:                                    # noqa: BLE001
        logger.warning("mandant_wege: Pruefung gescheitert (%s) mandant=%s pfad=%s",
                       type(fehler).__name__, mandant, pfad)
        e = HTTPException(status_code=503, detail={"error": "dienstweg_nicht_lesbar", "grund": "pruefung_fehler",
                                                   "mandant": mandant})
    if e is None:
        _vermerken({"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "modus": m, "mandant": mandant,
                    "pfad": pfad, "aufrufer": (aufrufer or "?")[:64], "status": 200, "error": None,
                    "grund": "pool", "dienst": dienst_fuer(pfad), "weg": "pool", "anbieter": None})
        return None
    d = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
    _vermerken({"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "modus": m, "mandant": mandant,
                "pfad": pfad, "aufrufer": (aufrufer or "?")[:64], "status": e.status_code,
                "error": d.get("error"), "grund": d.get("grund"), "dienst": d.get("dienst"),
                "weg": d.get("weg"), "anbieter": d.get("anbieter")})
    if m == "beobachten":
        logger.warning("mandant_wege beobachten: mandant=%s pfad=%s aufrufer=%s wuerde %s %s/%s", mandant, pfad,
                       (aufrufer or "?")[:64], e.status_code, d.get("error"), d.get("grund"))
        return None
    logger.warning("mandant_wege: abgewiesen mandant=%s pfad=%s aufrufer=%s %s %s/%s", mandant, pfad,
                   (aufrufer or "?")[:64], e.status_code, d.get("error"), d.get("grund"))
    return JSONResponse(status_code=e.status_code, content={"detail": d})
