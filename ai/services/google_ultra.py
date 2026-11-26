"""Google-Gemini-API auf dem Ultra-Guthaben (#2867, Auftrag Alex 14.11.2026).

Alex: einen Monat lang Transkription und Text-to-Speech ueber die Gemini-API,
sonst nichts (kein Gemini-Text, kein Imagen, kein Veo). Bezahlt wird aus dem
Cloud-Guthaben des Ultra-Abos (125,77 EUR auf „Mein Maps-Rechnungskonto“,
eingeloest 16.11.). Kosten ueber das Guthaben hinaus duerfen nicht entstehen
(Mai-Vorfall: 209,56 EUR). Ein GCP-Budget warnt nur, darum sitzt die harte
Bremse hier:

- Schluessel nur aus der Datei GEMINI_ULTRA_KEY_FILE (Vorgabe
  /etc/api-ai/google-ultra.env, root 600), NIE in os.environ. So erbt kein
  CLI-Kindprozess (codex, claude, agy) den Schluessel; die bekommen alle
  os.environ.copy().
- Ab GOOGLE_ULTRA_BIS (Vorgabe 2026-12-13, 00:00 Wien) 503, auch wenn
  niemand daran denkt.
- Tagesdeckel GOOGLE_ULTRA_TAG_USD (5 $) und Gesamtdeckel
  GOOGLE_ULTRA_GESAMT_USD (80 $). Das Guthaben gilt auch fuer die Maps-
  Projekte am selben Konto; 80 $ lassen dafuer Luft.
- Nur die Modelle aus MODELLE. Ein anderes Modell ergibt 422 mit der Liste,
  kein stiller Rueckfall.
- Vor dem Aufruf wird der geschaetzte Betrag reserviert (unter flock, also
  ueber beide Worker hinweg), danach mit dem Ist-Wert berichtigt oder
  storniert.

Preise je Minute Audio, gelesen am 14.11. auf der Gemini-Preisseite (gueltig
bis 31.12.). Gemini rechnet nach Tokens ab; die Minutenpreise sind die
Umrechnung dafuer und hier bewusst die Grundlage, weil die Dauer vor dem
Aufruf bekannt ist.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import HTTPException

logger = logging.getLogger(__name__)

WIEN = ZoneInfo("Europe/Vienna")

# art -> modell -> USD je Minute Audio
MODELLE = {
    "transcribe": {
        "gemini-3.5-flash-lite": 0.0009,
        "gemini-3.5-transcribe": 0.0051,
    },
    "tts": {
        "gemini-3.8-flash-tts": 0.0135,
        "gemini-3.8-flash-lite-tts": 0.009,
    },
}
VORGABE = {"transcribe": "gemini-3.5-transcribe", "tts": "gemini-3.8-flash-tts"}


def _float_env(name: str, vorgabe: float) -> float:
    try:
        return float(os.getenv(name) or vorgabe)
    except ValueError:
        return vorgabe


def tag_deckel() -> float:
    return _float_env("GOOGLE_ULTRA_TAG_USD", 5.0)


def gesamt_deckel() -> float:
    return _float_env("GOOGLE_ULTRA_GESAMT_USD", 80.0)


def bis() -> date:
    roh = (os.getenv("GOOGLE_ULTRA_BIS") or "2026-12-13").strip()
    try:
        return date.fromisoformat(roh)
    except ValueError:
        # Unlesbar heisst gesperrt, nicht offen.
        return date.min


def _jetzt() -> datetime:
    return datetime.now(WIEN)


def abgelaufen(jetzt: Optional[datetime] = None) -> bool:
    return (jetzt or _jetzt()).date() >= bis()


def _key_datei() -> Path:
    return Path(os.getenv("GEMINI_ULTRA_KEY_FILE") or "/etc/api-ai/google-ultra.env")


def schluessel() -> Optional[str]:
    """Liest den Schluessel bei jedem Aufruf aus der Datei. Loeschen der Datei
    sperrt also sofort, ohne Neustart."""
    try:
        for zeile in _key_datei().read_text().splitlines():
            zeile = zeile.strip()
            if zeile.startswith("GEMINI_ULTRA_API_KEY="):
                wert = zeile.split("=", 1)[1].strip().strip('"').strip("'")
                return wert or None
    except OSError:
        return None
    return None


def _buch() -> Path:
    return Path(os.getenv("VERBRAUCH_DIR") or "/var/lib/api-ai/verbrauch") / "google_ultra.jsonl"


def freigabe(art: str, modell: Optional[str]) -> str:
    """Prueft Ablauf, Schluessel und Modell. Liefert das Modell oder wirft."""
    if art not in MODELLE:
        raise HTTPException(status_code=503, detail={"error": "google_api_gesperrt", "art": art})
    if abgelaufen():
        raise HTTPException(status_code=503, detail={
            "error": "google_api_gesperrt",
            "grund": f"Die Google-API war nur bis {bis().isoformat()} freigegeben (#2867).",
        })
    if not schluessel():
        raise HTTPException(status_code=503, detail={
            "error": "google_api_gesperrt",
            "grund": "Kein Gemini-Schluessel fuer das Ultra-Guthaben hinterlegt (#2867).",
        })
    modell = modell or VORGABE[art]
    if modell not in MODELLE[art]:
        raise HTTPException(status_code=422, detail={
            "error": "modell_nicht_freigegeben",
            "model": modell,
            "available": sorted(MODELLE[art]),
            "hint": "Ueber die Gemini-API sind nur diese Modelle freigegeben (#2867).",
        })
    return modell


@dataclass
class Reservierung:
    id: str
    art: str
    modell: str
    usd: float


def _zeilen(f) -> list:
    f.seek(0)
    aus = []
    for z in f.read().splitlines():
        try:
            aus.append(json.loads(z))
        except ValueError:
            continue
    return aus


def _summen(zeilen: list, heute: str) -> tuple[float, float]:
    gesamt = sum(float(z.get("usd") or 0) for z in zeilen)
    tag = sum(float(z.get("usd") or 0) for z in zeilen if z.get("tag") == heute)
    return round(tag, 6), round(gesamt, 6)


def _anhaengen(eintrag: dict, pruefen: Optional[float] = None) -> None:
    pfad = _buch()
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if pruefen is not None:
                tag, gesamt = _summen(_zeilen(f), eintrag["tag"])
                if tag + pruefen > tag_deckel() or gesamt + pruefen > gesamt_deckel():
                    raise HTTPException(status_code=429, detail={
                        "error": "google_budget_erschoepft",
                        "tag_usd": tag, "tag_deckel_usd": tag_deckel(),
                        "gesamt_usd": gesamt, "gesamt_deckel_usd": gesamt_deckel(),
                        "angefragt_usd": round(pruefen, 6),
                    })
            f.seek(0, os.SEEK_END)
            f.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def kosten(art: str, modell: str, minuten: float) -> float:
    return round(max(minuten, 0.0) * MODELLE[art][modell], 6)


def reservieren(art: str, modell: str, minuten: float, aufrufer: str = "(unbekannt)") -> Reservierung:
    usd = kosten(art, modell, minuten)
    r = Reservierung(id=uuid.uuid4().hex[:12], art=art, modell=modell, usd=usd)
    _anhaengen({"ts": time.time(), "tag": _jetzt().date().isoformat(), "id": r.id,
                "art": art, "modell": modell, "minuten": round(minuten, 3),
                "usd": usd, "status": "reserviert", "aufrufer": aufrufer[:128]},
               pruefen=usd)
    return r


def abrechnen(r: Reservierung, minuten_ist: float, **extra) -> float:
    """Bucht die Differenz zwischen Ist und Reservierung. Liefert die Ist-Kosten."""
    ist = kosten(r.art, r.modell, minuten_ist)
    _anhaengen({"ts": time.time(), "tag": _jetzt().date().isoformat(), "id": r.id,
                "art": r.art, "modell": r.modell, "minuten": round(minuten_ist, 3),
                "usd": round(ist - r.usd, 6), "status": "berichtigt", **extra})
    from ai.services import kostenbuch as _kb   # #4127: Ultra-Guthaben, kein Geld vom Konto
    _kb.buchen(anbieter="google_ultra", art="guthaben", modell=r.modell, usd=0.0, gegenwert_usd=ist,
               minuten=minuten_ist if r.art == "transcribe" or r.art == "tts" else 0.0,
               zeichen=int(extra.get("zeichen") or 0), tokens_ein=int(extra.get("tokens_in") or 0),
               tokens_aus=int(extra.get("tokens_out") or 0))
    return ist


def stornieren(r: Reservierung, grund: str = "") -> None:
    _anhaengen({"ts": time.time(), "tag": _jetzt().date().isoformat(), "id": r.id,
                "art": r.art, "modell": r.modell, "usd": -r.usd,
                "status": "storniert", "grund": grund[:200]})


# --- Gemini-REST ---------------------------------------------------------------

API_BASIS = "https://generativelanguage.googleapis.com/v1beta"
# Anfrage-Grenze der Gemini-API liegt bei 20 MB; Base64 blaeht um ein Drittel.
INLINE_MAX_BYTES = 14 * 1024 * 1024


async def generate(modell: str, rumpf: dict, timeout: float = 300.0) -> dict:
    """generateContent ueber REST. Der Schluessel geht im Kopf mit, nie in der
    Adresse (sonst stuende er in Fehlermeldungen und Protokollen)."""
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{API_BASIS}/models/{modell}:generateContent",
                         headers={"x-goog-api-key": schluessel() or ""}, json=rumpf)
    if r.status_code != 200:
        try:
            meldung = (r.json().get("error") or {}).get("message", "")
        except ValueError:
            meldung = r.text
        raise HTTPException(status_code=502, detail={
            "error": "gemini_fehler", "upstream_status": r.status_code,
            "upstream": str(meldung)[:300]})
    return r.json()


def text_aus_antwort(antwort: dict) -> str:
    """Text aus generateContent. gemini-3.5-transcribe legt ihn in
    parts[].audioTranscription.text, die anderen Modelle in parts[].text."""
    stuecke = []
    for kand in antwort.get("candidates") or []:
        for teil in ((kand.get("content") or {}).get("parts") or []):
            if teil.get("thought"):
                continue
            if isinstance(teil.get("text"), str):
                stuecke.append(teil["text"])
            elif isinstance((teil.get("audioTranscription") or {}).get("text"), str):
                stuecke.append(teil["audioTranscription"]["text"])
    return "".join(stuecke).strip()


def hochladen(data: bytes, mime: str, endung: str = ".mp3") -> tuple:
    """Grosse Aufnahmen ueber die File-API (SDK). Liefert (name, uri)."""
    import tempfile
    from google import genai
    c = genai.Client(api_key=schluessel())
    with tempfile.NamedTemporaryFile(suffix=endung or ".mp3") as tf:
        tf.write(data)
        tf.flush()
        f = c.files.upload(file=tf.name, config={"mime_type": mime})
    for _ in range(60):
        if str(getattr(f, "state", "")).endswith("ACTIVE"):
            break
        time.sleep(2)
        f = c.files.get(name=f.name)
    return f.name, f.uri


def datei_loeschen(name: str) -> None:
    try:
        from google import genai
        genai.Client(api_key=schluessel()).files.delete(name=name)
    except Exception:
        pass


def stand() -> dict:
    heute = _jetzt().date().isoformat()
    try:
        with open(_buch()) as f:
            zeilen = _zeilen(f)
    except OSError:
        zeilen = []
    tag, gesamt = _summen(zeilen, heute)
    je_art: dict = {}
    for z in zeilen:
        a = je_art.setdefault(z.get("art") or "?", {"usd": 0.0, "aufrufe": 0})
        a["usd"] = round(a["usd"] + float(z.get("usd") or 0), 6)
        if z.get("status") == "berichtigt":
            a["aufrufe"] += 1
    return {
        "freigegeben": bool(schluessel()) and not abgelaufen(),
        "schluessel": bool(schluessel()),
        "bis": bis().isoformat(),
        "tag_usd": tag, "tag_deckel_usd": tag_deckel(),
        "gesamt_usd": gesamt, "gesamt_deckel_usd": gesamt_deckel(),
        "je_art": je_art,
        "modelle": {a: {m: {"usd_je_minute": p} for m, p in ms.items()} for a, ms in MODELLE.items()},
    }
