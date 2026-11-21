"""Textweg ueber den OpenAI-Platform-Schluessel der Instanz (#3640, #3639).

Alex, 21.11.: Hat der Kunde nur OpenAI, ist die Abhaengigkeit klar. Der
vorhandene Kundenschluessel und sein Budget tragen die Sprechstile, wenn agy
und codex fehlen.

Grenzen:

- Ohne `OPENAI_API_KEY` 503 `openai_schluessel_fehlt`. Nie ein anderer
  Schluessel, nie unser api-ai.
- Ohne gueltige Tagesgrenze `OPENAI_TEXT_DAILY_MAX_USD` 503. Unser Kern setzt
  sie nicht, dort ist der Weg also aus.
- Nur die Modelle aus PREISE, mit festem Hoechstwert fuer die Ausgabe.
- Vor dem Aufruf wird der hoechstmoegliche Betrag unter flock reserviert (uvicorn
  hat zwei Prozesse), danach auf den Ist-Wert berichtigt. Bei einer
  Zeitueberschreitung bleibt die Reservierung stehen, weil OpenAI den Aufruf
  trotzdem berechnet haben kann.
- `store: false`: Kundentext bleibt nicht als Antwort bei OpenAI gespeichert.

Der Schluessel steht nie in Log oder Antwort. Fehlertexte von OpenAI werden
nicht durchgereicht, sie koennen Teile des Schluessels enthalten.

Preise je 1 Mio. Tokens, gelesen am 21.11.2026 auf
developers.openai.com/api/docs/pricing (Standard, bis 272K Kontext).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

UPSTREAM_URL = "https://api.openai.com/v1/responses"
WIEN = ZoneInfo("Europe/Vienna")

# modell -> (Eingabe, gecachte Eingabe, Ausgabe) in USD je 1 Mio. Tokens
PREISE = {"gpt-6-luna": (0.10, 0.01, 0.50)}
VORGABE_MODELL = "gpt-6-luna"
EFFORTS = ("none", "low", "medium")
VORGABE_EFFORT = "low"

MAX_AUSGABE_TOKENS = 4000
MIN_AUSGABE_TOKENS = 16
MAX_EINGABE_ZEICHEN = 40000
VORGABE_MAX_ANFRAGEN = 2000
TIMEOUT_S = 45.0

GRENZE_ENV = "OPENAI_TEXT_DAILY_MAX_USD"
ANFRAGEN_ENV = "OPENAI_TEXT_DAILY_MAX_REQUESTS"

_ABRECHNUNGS_CODES = frozenset({
    "insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
})


def schluessel() -> Optional[str]:
    return (os.getenv("OPENAI_API_KEY") or "").strip() or None


def fingerabdruck() -> Optional[str]:
    """Kurzer sha256-Praefix: Die Pipeline prueft damit, dass es nicht unserer ist."""
    k = schluessel()
    return hashlib.sha256(k.encode("utf-8")).hexdigest()[:8] if k else None


def grenze_usd() -> Optional[float]:
    """Gueltige Tagesgrenze oder None. Unlesbar heisst aus, nicht offen."""
    try:
        wert = float((os.getenv(GRENZE_ENV) or "").strip())
    except ValueError:
        return None
    return wert if math.isfinite(wert) and wert > 0 else None


def grenze_anfragen() -> int:
    try:
        wert = int((os.getenv(ANFRAGEN_ENV) or "").strip())
    except ValueError:
        return VORGABE_MAX_ANFRAGEN
    return wert if wert > 0 else VORGABE_MAX_ANFRAGEN


def _jetzt() -> datetime:
    return datetime.now(WIEN)


def _naechster_tag(jetzt: Optional[datetime] = None) -> str:
    j = jetzt or _jetzt()
    return (j + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _dir() -> Path:
    return Path(os.getenv("VERBRAUCH_DIR") or "/var/lib/api-ai/verbrauch")


def _tagesdatei(tag: str) -> Path:
    return _dir() / f"openai_text_{tag}.json"


def _ausgangsdatei() -> Path:
    return _dir() / "openai_text_ausgang.json"


class _Sperre:
    def __enter__(self):
        _dir().mkdir(parents=True, exist_ok=True)
        fd = os.open(_dir() / ".openai_text.lock", os.O_RDWR | os.O_CREAT, 0o600)
        self._f = os.fdopen(fd, "r+")
        fcntl.flock(self._f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self._f, fcntl.LOCK_UN)
        self._f.close()
        return False


def _lesen(pfad: Path) -> dict:
    try:
        d = json.loads(pfad.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _schreiben(pfad: Path, d: dict) -> None:
    tmp = pfad.with_name(f".{pfad.name}.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    os.replace(tmp, pfad)


def _kontingent(grund: str, resets_at: Optional[str] = None, retry_after: Optional[str] = None,
                hinweis: str = "quota reached") -> HTTPException:
    """Fester Code fuer die Kette der cloud-api (#3639), `quota` im Text fuer
    deren Erkennung am Wortlaut."""
    kopf = {"Retry-After": retry_after} if retry_after else None
    return HTTPException(status_code=429, headers=kopf, detail={
        "error": "kontingent", "weg": "openai_schluessel", "grund": grund,
        "resets_at": resets_at, "hint": hinweis})


@dataclass
class Reservierung:
    tag: str
    usd: float


def reservieren(usd: float) -> Reservierung:
    grenze = grenze_usd()
    if grenze is None:
        raise HTTPException(status_code=503, detail={"error": "openai_text_grenze_fehlt",
                                                      "hint": f"{GRENZE_ENV} ist nicht gesetzt."})
    jetzt = _jetzt()
    tag = jetzt.date().isoformat()
    with _Sperre():
        d = _lesen(_tagesdatei(tag))
        heute, anfragen = float(d.get("usd") or 0), int(d.get("anfragen") or 0)
        if heute + usd > grenze or anfragen >= grenze_anfragen():
            bis = _naechster_tag(jetzt)
            sek = max(1, int((datetime.fromisoformat(bis) - jetzt).total_seconds()))
            raise _kontingent("tagesgrenze", bis, str(sek), "daily quota reached, try again at 00:00")
        d.update(usd=round(heute + usd, 8), anfragen=anfragen + 1)
        _schreiben(_tagesdatei(tag), d)
    return Reservierung(tag=tag, usd=usd)


def _ausgang(art: str, bis: Optional[str] = None) -> None:
    a = _lesen(_ausgangsdatei())
    a["letzter_ausgang"] = art
    a["bis"] = bis
    if art == "ok":
        a["letzter_erfolg"] = _jetzt().isoformat(timespec="seconds")
    _schreiben(_ausgangsdatei(), a)


def abrechnen(r: Reservierung, ist: float) -> None:
    with _Sperre():
        d = _lesen(_tagesdatei(r.tag))
        d["usd"] = round(max(float(d.get("usd") or 0) + ist - r.usd, 0.0), 8)
        d["erfolg"] = int(d.get("erfolg") or 0) + 1
        _schreiben(_tagesdatei(r.tag), d)
        _ausgang("ok")


def stornieren(r: Reservierung, ausgang: str = "fehler", bis: Optional[str] = None) -> None:
    """Der Aufruf wurde nachweislich nicht berechnet."""
    with _Sperre():
        d = _lesen(_tagesdatei(r.tag))
        d["usd"] = round(max(float(d.get("usd") or 0) - r.usd, 0.0), 8)
        d["anfragen"] = max(int(d.get("anfragen") or 0) - 1, 0)
        d["fehler"] = int(d.get("fehler") or 0) + 1
        _schreiben(_tagesdatei(r.tag), d)
        _ausgang(ausgang, bis)


def behalten(r: Reservierung) -> None:
    """Ausgang unklar (Zeitueberschreitung): Der Betrag bleibt gebucht."""
    with _Sperre():
        d = _lesen(_tagesdatei(r.tag))
        d["fehler"] = int(d.get("fehler") or 0) + 1
        _schreiben(_tagesdatei(r.tag), d)
        _ausgang("fehler")


def kosten(modell: str, usage: dict) -> float:
    ein, gecacht_preis, aus = PREISE[modell]
    usage = usage or {}
    gesamt = int(usage.get("input_tokens") or 0)
    gecacht = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    ausgabe = int(usage.get("output_tokens") or 0)   # enthaelt die Reasoning-Tokens
    return (max(gesamt - gecacht, 0) * ein + gecacht * gecacht_preis + ausgabe * aus) / 1_000_000.0


def hoechstbetrag(modell: str, zeichen: int, max_ausgabe: int) -> float:
    """Obergrenze vor dem Aufruf: je Zeichen hoechstens ein Token, dazu Rahmen."""
    ein, _g, aus = PREISE[modell]
    return ((zeichen + 200) * ein + max_ausgabe * aus) / 1_000_000.0


def text_aus_antwort(antwort: dict) -> str:
    stuecke = []
    for item in antwort.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for teil in item.get("content") or []:
            if isinstance(teil, dict) and teil.get("type") == "output_text" \
                    and isinstance(teil.get("text"), str):
                stuecke.append(teil["text"])
    return "".join(stuecke).strip()


def eingaben_pruefen(modell: Optional[str], effort: Optional[str], max_tokens: Optional[int],
                     zeichen: int) -> tuple:
    m = (modell or VORGABE_MODELL).strip()
    if m not in PREISE:
        raise HTTPException(status_code=422, detail={"error": "modell_nicht_freigegeben", "model": m[:80],
                                                     "available": sorted(PREISE)})
    e = (effort or VORGABE_EFFORT).strip().lower()
    if e not in EFFORTS:
        raise HTTPException(status_code=422, detail={"error": "invalid_effort", "available": list(EFFORTS)})
    if zeichen > MAX_EINGABE_ZEICHEN:
        raise HTTPException(status_code=422, detail={"error": "prompt_zu_lang", "max_zeichen": MAX_EINGABE_ZEICHEN})
    if zeichen == 0:
        raise HTTPException(status_code=422, detail={"error": "prompt_leer"})
    gewuenscht = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 1000
    return m, e, max(MIN_AUSGABE_TOKENS, min(gewuenscht, MAX_AUSGABE_TOKENS))


def _monatsdeckel_greift() -> bool:
    from ai.services.openai_cost_tracker import openai_cost_tracker
    return bool(openai_cost_tracker.should_block_request())


def freigabe() -> str:
    """Schluessel und Tagesgrenze da? Liefert den Schluessel oder wirft 503."""
    key = schluessel()
    if not key:
        raise HTTPException(status_code=503, detail={
            "error": "openai_schluessel_fehlt",
            "hint": "Diese Instanz hat keinen OPENAI_API_KEY fuer den Textweg."})
    if grenze_usd() is None:
        raise HTTPException(status_code=503, detail={"error": "openai_text_grenze_fehlt",
                                                      "hint": f"{GRENZE_ENV} ist nicht gesetzt."})
    return key


async def aufrufen(text: str, system: Optional[str], modell: Optional[str],
                   effort: Optional[str], max_tokens: Optional[int]) -> dict:
    """Ein Aufruf der Responses API. Liefert text, Tokens und Kosten oder wirft."""
    import asyncio
    zeichen = len(text) + len(system or "")
    m, e, max_ausgabe = eingaben_pruefen(modell, effort, max_tokens, len(text) and zeichen)
    key = freigabe()
    try:
        gesperrt = await asyncio.to_thread(_monatsdeckel_greift)
    except Exception as exc:
        logger.error("openai_text: Monatsdeckel nicht pruefbar (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail={"error": "openai_text_deckel_nicht_pruefbar"})
    if gesperrt:
        raise _kontingent("monatsdeckel", hinweis="monthly quota reached")

    r = await asyncio.to_thread(reservieren, hoechstbetrag(m, zeichen, max_ausgabe))
    rumpf = {"model": m, "input": text, "max_output_tokens": max_ausgabe,
             "reasoning": {"effort": e}, "store": False}
    if system:
        rumpf["instructions"] = system
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
            antwort = await c.post(UPSTREAM_URL, json=rumpf, headers={"Authorization": f"Bearer {key}"})
    except httpx.TimeoutException:
        await asyncio.to_thread(behalten, r)
        logger.warning("openai_text: Zeitueberschreitung")
        raise HTTPException(status_code=504, detail={"error": "openai_text_timeout"})
    except httpx.HTTPError as exc:
        # Die Anfrage kann schon gesendet sein: nur ein Verbindungsfehler ist sicher unberechnet.
        if isinstance(exc, httpx.ConnectError):
            await asyncio.to_thread(stornieren, r)
        else:
            await asyncio.to_thread(behalten, r)
        logger.warning("openai_text: Netzfehler %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail={"error": "openai_text_netzfehler"})

    if antwort.status_code != 200:
        raise await _fehler(antwort, r)
    try:
        daten = antwort.json()
        usage = daten.get("usage") or {}
        ist = kosten(m, usage) if usage else r.usd  # ohne usage bleibt der Hoechstbetrag gebucht
        inhalt = text_aus_antwort(daten)
        grund = (daten.get("incomplete_details") or {}).get("reason")
        ausgabe_details = usage.get("output_tokens_details") or {}
        gesamt_tokens = int(usage.get("total_tokens") or 0)
        denk_tokens = int(ausgabe_details.get("reasoning_tokens") or 0)
    except (ValueError, TypeError, AttributeError):
        # Unlesbar heisst nicht unberechnet: Der Betrag bleibt gebucht.
        await asyncio.to_thread(behalten, r)
        raise HTTPException(status_code=502, detail={"error": "openai_text_antwort_unlesbar"})
    if not inhalt:
        # Berechnet wurde trotzdem.
        await asyncio.to_thread(_leer_buchen, r, ist)
        raise HTTPException(status_code=502, detail={"error": "openai_text_leer", "grund": grund})
    await asyncio.to_thread(abrechnen, r, ist)
    fertig = "stop" if daten.get("status") == "completed" else \
        ("length" if grund == "max_output_tokens" else (grund or "incomplete"))
    return {"text": inhalt, "model": daten.get("model") or m, "effort": e,
            "tokens": gesamt_tokens, "thinking": denk_tokens,
            "finish_reason": fertig, "usd": round(ist, 8)}


def _leer_buchen(r: Reservierung, ist: float) -> None:
    with _Sperre():
        d = _lesen(_tagesdatei(r.tag))
        d["usd"] = round(max(float(d.get("usd") or 0) + ist - r.usd, 0.0), 8)
        d["fehler"] = int(d.get("fehler") or 0) + 1
        _schreiben(_tagesdatei(r.tag), d)
        _ausgang("fehler")


async def _fehler(antwort, r: Reservierung) -> HTTPException:
    """Abgelehnte Anfragen berechnet OpenAI nicht. Nur Status und Code, nie den Text."""
    import asyncio
    try:
        roh = antwort.json()
        f = roh.get("error") if isinstance(roh, dict) else None
    except ValueError:
        f = None
    f = f if isinstance(f, dict) else {}
    code = str(f.get("code") or "")[:60]
    typ = str(f.get("type") or "")[:60]
    status = antwort.status_code
    logger.warning("openai_text: OpenAI %s code=%s type=%s", status, code or "-", typ or "-")
    if status == 429:
        retry = antwort.headers.get("retry-after")
        if code in _ABRECHNUNGS_CODES or typ == "insufficient_quota":
            await asyncio.to_thread(stornieren, r, "kontingent")
            return _kontingent("openai_guthaben", hinweis="OpenAI quota exhausted")
        # Retry-After kann auch ein Datum oder Unsinn sein. Nur eine Sekundenzahl gilt.
        sekunden = int(retry) if retry and retry.isascii() and retry.isdigit() and len(retry) <= 6 else None
        bis = (_jetzt() + timedelta(seconds=sekunden)).isoformat(timespec="seconds") \
            if sekunden is not None else None
        await asyncio.to_thread(stornieren, r, "kontingent", bis)
        return _kontingent("openai_rate_limit", bis, str(sekunden) if sekunden is not None else None,
                           "OpenAI rate limit reached")
    await asyncio.to_thread(stornieren, r)
    if status in (401, 403):
        return HTTPException(status_code=502, detail={"error": "openai_schluessel_abgelehnt",
                                                      "upstream_status": status})
    return HTTPException(status_code=502, detail={"error": "openai_text_fehler", "upstream_status": status,
                                                  "upstream_code": code or None})


def stand() -> dict:
    """Fuer cost-status. Ohne Schluessel, ohne Prompts."""
    jetzt = _jetzt()
    d = _lesen(_tagesdatei(jetzt.date().isoformat()))
    a = _lesen(_ausgangsdatei())
    return {"modell": VORGABE_MODELL, "schluessel": bool(schluessel()), "fingerabdruck": fingerabdruck(),
            "grenze_usd": grenze_usd(), "grenze_anfragen": grenze_anfragen(),
            "heute_usd": round(float(d.get("usd") or 0), 6), "heute_anfragen": int(d.get("anfragen") or 0),
            "heute_erfolg": int(d.get("erfolg") or 0), "heute_fehler": int(d.get("fehler") or 0),
            "letzter_erfolg": a.get("letzter_erfolg"), "resets_at": _naechster_tag(jetzt)}


def weg_zustand() -> dict:
    """Zustand fuer /ai/wege (#3639), ohne Modellaufruf."""
    s = stand()
    aus = {"zustand": "angemeldet", "bis": None, "letzter_erfolg": s["letzter_erfolg"],
           "grenze_usd": s["grenze_usd"], "heute_usd": s["heute_usd"], "fingerabdruck": s["fingerabdruck"]}
    if not s["schluessel"]:
        aus["zustand"] = "fehlt"
        return aus
    if s["grenze_usd"] is None:
        aus.update(zustand="fehler", grund="tagesgrenze_fehlt")
        return aus
    if s["heute_usd"] >= s["grenze_usd"] or s["heute_anfragen"] >= s["grenze_anfragen"]:
        aus.update(zustand="kontingent", bis=s["resets_at"], grund="tagesgrenze")
        return aus
    a = _lesen(_ausgangsdatei())
    art = a.get("letzter_ausgang")
    if art == "ok":
        aus["zustand"] = "ok"
    elif art == "fehler":
        aus["zustand"] = "fehler"
    elif art == "kontingent":
        bis = a.get("bis")
        try:
            vorbei = bool(bis) and datetime.fromisoformat(bis) <= _jetzt()
        except (ValueError, TypeError):
            vorbei = False
        if not vorbei:
            aus.update(zustand="kontingent", bis=bis)
    return aus
