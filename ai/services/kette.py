"""#3658: Kette fuer Text ueber die Abos, Google -> ChatGPT -> Claude.

Alex, 21.11.: API-Kosten nur fuer TTS, STT und Realtime, aller Text ueber die
Abos, zuerst Google, dann ChatGPT, dann Claude, automatisch und ausfallsicher.
Vertrag mit Cloud-worker: #3658, Kommentare 11327, 11330, 11331.

Diese Datei entscheidet nur: welcher Weg als naechstes, was ein Fehler bedeutet,
wie die Kette endet. Die Wege selbst ruft `kette_routes` auf.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from ai.services import ki_wege

logger = logging.getLogger("api-ai.kette")

KETTE_ENV = "KI_KETTE"
WEGE = ("agy", "codex", "claude")
VORGABE = {"agy": ("gemini-4-hydrogen-low", None), "codex": ("gpt-5.6-luna", "low"), "claude": ("haiku", None)}
# Tagesdeckel je Weg (Zaehlername, Umgebung, Vorgabe; 0 = kein Deckel). Der Google-
# Zaehler ist derselbe, in den direkte /ai/gemini-Aufrufe mitzaehlen.
DECKEL = {
    "agy": ("google_hydrogen", "KETTE_AGY_TAG_MAX", 500),
    "codex": ("kette_codex", "KETTE_CODEX_TAG_MAX", 0),
    "claude": ("kette_claude", "KETTE_CLAUDE_TAG_MAX", 100),
}
BUDGET_VORGABE_S = 30.0
BUDGET_HOECHSTENS_S = 120.0          # Vorgabe der Obergrenze, wenn nichts gesetzt ist
BUDGET_HOECHSTENS_ENV = "KETTE_BUDGET_HOECHSTENS_S"
_BUDGET_GRENZEN_S = (30.0, 600.0)   # CLIs enden nach 300 s, nginx nach 600 s
_MINDESTREST_S = 2.0


def budget_hoechstens() -> float:
    """Obergrenze fuer budget_s je Host (Project Manager 25.11.2026: der Planer der
    project-api braucht fuer grosse Befunde mehr als 120 s). Ungueltig: Vorgabe;
    ausserhalb von 30..600 s: an den Rand gezogen."""
    roh = (os.getenv(BUDGET_HOECHSTENS_ENV) or "").strip()
    try:
        wert = float(roh) if roh else BUDGET_HOECHSTENS_S
    except ValueError:
        logger.error("%s=%r ist keine Zahl, Vorgabe %ss gilt", BUDGET_HOECHSTENS_ENV, roh, BUDGET_HOECHSTENS_S)
        wert = BUDGET_HOECHSTENS_S
    if wert != wert:   # NaN
        wert = BUDGET_HOECHSTENS_S
    return min(max(wert, _BUDGET_GRENZEN_S[0]), _BUDGET_GRENZEN_S[1])


def _uhr() -> float:
    return time.monotonic()


def kette() -> List[str]:
    """Reihenfolge aus KI_KETTE, nur bekannte Wege, ohne Doppelte. Leer = aus."""
    out = []
    for w in (os.getenv(KETTE_ENV) or "").split(","):
        w = w.strip().lower()
        if w in WEGE and w not in out:
            out.append(w)
    return out


def deckel(weg: str) -> Tuple[str, int]:
    name, env, vorgabe = DECKEL[weg]
    try:
        wert = int((os.getenv(env) or "").strip())
    except ValueError:
        wert = vorgabe
    return name, max(0, wert)


def einordnen(e: BaseException) -> Tuple[str, Optional[str]]:
    """(art, resets_at). art: kontingent | fehlt | aussetzer | anfrage."""
    if isinstance(e, HTTPException):
        d = e.detail if isinstance(e.detail, dict) else {}
        fehler = d.get("error")
        if fehler == "kontingent":
            return "kontingent", d.get("resets_at")
        if fehler == "cli_fehlt":
            return "fehlt", None
        if fehler == "claude_account_unavailable":
            if d.get("grund") == "limit":
                return "kontingent", d.get("resets")
            return "fehlt", None
        if isinstance(e.detail, str) and "not installed" in e.detail:
            return "fehlt", None
        if e.status_code >= 500 or e.status_code == 429:
            return "aussetzer", None
        return "anfrage", None
    if isinstance(e, (asyncio.TimeoutError, subprocess.TimeoutExpired)):
        return "aussetzer", None
    return "aussetzer", None


def _frueheste(werte: List[str]) -> Optional[str]:
    beste, beste_zeit = None, None
    for w in werte:
        try:
            t = datetime.fromisoformat(w)
            if t.tzinfo is None:
                t = t.astimezone()
        except (TypeError, ValueError):
            continue
        if beste_zeit is None or t < beste_zeit:
            beste, beste_zeit = w, t
    return beste


Rufer = Callable[[str, str, Optional[str]], Awaitable[object]]


async def laufen(rufen: Rufer, wahl: Dict[str, dict], budget_s: float, nur_erster_weg: bool,
                 frist_setzen: Callable[[Optional[float]], object],
                 frist_zuruecksetzen: Callable[[object], None]) -> dict:
    """Geht die Kette durch. Gibt {"out", "weg", "model", "versucht"} zurueck oder wirft
    die feste Endantwort (429 kontingent, 503 wege_fehler/cli_fehlt, 504 budget) bzw.
    den Fehler der Anfrage (4xx) unveraendert."""
    reihe = kette()
    if nur_erster_weg:
        reihe = reihe[:1]
    ende = _uhr() + budget_s
    versucht: List[dict] = []
    bis_liste: List[str] = []
    budget_aus = False
    for weg in reihe:
        rest = ende - _uhr()
        if rest < _MINDESTREST_S:
            budget_aus = True
            break
        gesperrt, bis = ki_wege.gesperrt(weg)
        if gesperrt:
            versucht.append({"weg": weg, "code": "kontingent", "uebersprungen": True})
            if bis:
                bis_liste.append(bis)
            continue
        modell = (wahl.get(weg) or {}).get("model") or VORGABE[weg][0]
        effort = (wahl.get(weg) or {}).get("effort") or VORGABE[weg][1]
        marke = None
        try:
            name, tag_max = deckel(weg)
            if tag_max:
                ki_wege.tag_buchen(name, tag_max)
            marke = frist_setzen(ende)
            out = await asyncio.wait_for(rufen(weg, modell, effort), timeout=rest + 3)
        except BaseException as e:                            # noqa: BLE001
            if isinstance(e, asyncio.CancelledError):
                raise
            art, bis = einordnen(e)
            if art == "anfrage":
                raise
            versucht.append({"weg": weg, "code": art})
            if bis:
                bis_liste.append(bis)
            if art == "kontingent" and weg == "claude":
                # agy und codex vermerken das selbst (#3645), der Claude-Weg nicht.
                ki_wege.ausgang("claude", "kontingent", bis)
            logger.info("kette: %s -> %s", weg, art)
            if art == "aussetzer" and ende - _uhr() < _MINDESTREST_S:
                budget_aus = True
                break
            continue
        finally:
            if marke is not None:
                frist_zuruecksetzen(marke)
        text = getattr(out, "response", None)
        if not (isinstance(text, str) and text.strip()):
            versucht.append({"weg": weg, "code": "aussetzer", "grund": "leer"})
            continue
        versucht.append({"weg": weg, "code": "ok"})
        return {"out": out, "weg": weg, "model": getattr(out, "model", None) or modell, "versucht": versucht}
    resets_at = _frueheste(bis_liste)
    arten = [v["code"] for v in versucht]
    if budget_aus:
        raise HTTPException(status_code=504, detail={
            "error": "budget", "versucht": versucht, "resets_at": resets_at,
            "hint": f"Budget von {budget_s:g} s erschoepft."})
    if arten and all(a == "kontingent" for a in arten):
        sek = None
        if resets_at:
            try:
                t = datetime.fromisoformat(resets_at)
                t = t if t.tzinfo else t.astimezone()
                sek = max(1, int((t - datetime.now().astimezone()).total_seconds()))
            except (TypeError, ValueError):
                sek = None
        raise HTTPException(status_code=429, headers={"Retry-After": str(sek)} if sek else None, detail={
            "error": "kontingent", "versucht": versucht, "resets_at": resets_at,
            "hint": "Alle Abo-Wege sind leer."})
    code = "wege_fehler" if "aussetzer" in arten else "cli_fehlt"
    raise HTTPException(status_code=503, detail={
        "error": code, "versucht": versucht, "resets_at": resets_at,
        "hint": "Kein Abo-Weg hat geantwortet." if code == "wege_fehler" else "Kein Abo-Weg ist verfuegbar."})
