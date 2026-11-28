"""POST /ai/audio/abschrift — Abschrift langer Sendungen (Projekt 32, Paket A2).

Vertrag `radio-schnitt-v2` (Abschrift unverändert seit v1), Content-Beitrag 6630 (bestätigt von Story-Codex,
Stand v35), Abschnitt 1. Asynchron mit Pflicht-`request_id`:

- gleiche `request_id`, gleicher fachlicher Rumpf → derselbe Job, keine
  zweite Buchung; eine erneuerte signierte `quelle_url` zählt nicht zum Rumpf
- gleiche `request_id`, anderer Rumpf → 409 `request_id_konflikt`
- Abschrift ist bezahlte STT (zwei Durchläufe: whisper-1 für Wörter,
  gpt-4o-transcribe-diarize für Sprecher) → `confirm_api_billing: true`

Gebucht wird erst, wenn der Lauf wirklich einen Anbieter aufruft. Bricht er
vorher ab (Quelle nicht lesbar, zu lang, Tagesgrenze), kann dieselbe
`request_id` neu anlaufen; danach nicht mehr ohne neue Kennung.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ai.services import abschrift_lang as al

logger = logging.getLogger("api-ai.abschrift")
router = APIRouter()

KIND = "abschrift"
MAX_MB = int(os.getenv("ABSCHRIFT_MAX_MB", "1024"))
MAX_MIN = int(os.getenv("ABSCHRIFT_MAX_MIN", "180"))
USD_JE_MIN = 0.006            # whisper-1 und gpt-4o-transcribe-diarize, je Durchlauf
DURCHLAEUFE = 2
DIARIZE_MODELL = "gpt-4o-transcribe-diarize"
WORT_MODELL = "whisper-1"
_REF_MIN_MS, _REF_MAX_MS, _REF_MAX = 2000, 10000, 4

_LAEUFE: set = set()          # starke Referenzen auf laufende Tasks


class SprecherName(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    start_ms: int = Field(..., ge=0)
    ende_ms: int = Field(..., gt=0)


class AbschriftAnfrage(BaseModel):
    request_id: str
    quelle_storage_id: int = Field(..., ge=1)
    quelle_url: Optional[str] = None
    sprache: Optional[str] = Field("de", max_length=8)
    sprecher_namen: Optional[List[SprecherName]] = Field(None, max_length=_REF_MAX)
    fuelllaute_vorgabe: bool = True
    confirm_api_billing: bool = False


# ── Zugang ───────────────────────────────────────────────────────────────────

def _aufrufer(request: Request, authorization: Optional[str], x_api_key: Optional[str]) -> str:
    """Dienst-JWT mit ai:transcribe (Middleware hat geprüft), Federation-JWT
    oder X-API-KEY. Die Route kostet Geld und liest Kundendaten: anonym nie."""
    dienst = getattr(request.state, "dienst", None)
    if dienst:
        return f"service:{dienst.get('sub')}"
    from ai.auth_guard import angemeldet
    return angemeldet(authorization=authorization, x_api_key=x_api_key)


# ── Anbieter (in Tests ersetzt) ──────────────────────────────────────────────

def _openai():
    from openai import AsyncOpenAI
    if not os.getenv("OPENAI_API_KEY"):
        from ai.provider_config import provider_missing
        raise provider_missing("openai")
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _buchen(modell: str, minuten: float) -> None:
    try:
        from ai.services import kostenbuch as kb
        kb.buchen(anbieter="openai_stt", art="geld", modell=modell, minuten=minuten,
                  usd=minuten * USD_JE_MIN, genauigkeit="gezaehlt")
    except Exception:
        logger.exception("kostenbuch abschrift")


async def _whisper_woerter(mp3: bytes, sprache: Optional[str], vorgabe: Optional[str], minuten: float) -> list:
    buf = io.BytesIO(mp3)
    buf.name = "stueck.mp3"
    kw = {"model": WORT_MODELL, "file": buf, "response_format": "verbose_json",
          "timestamp_granularities": ["word"]}
    if sprache:
        kw["language"] = sprache
    if vorgabe:
        kw["prompt"] = vorgabe
    erg = await _openai().audio.transcriptions.create(**kw)
    _buchen(WORT_MODELL, minuten)
    from fastapi.encoders import jsonable_encoder
    return list(jsonable_encoder(erg).get("words") or [])


async def _diarize(mp3: bytes, sprache: Optional[str], namen: list, referenzen: list, minuten: float) -> list:
    buf = io.BytesIO(mp3)
    buf.name = "stueck.mp3"
    kw = {"model": DIARIZE_MODELL, "file": buf, "response_format": "diarized_json",
          "chunking_strategy": "auto"}
    if sprache:
        kw["language"] = sprache
    if namen:
        kw["extra_body"] = {"known_speaker_names": namen, "known_speaker_references": referenzen}
    erg = await _openai().audio.transcriptions.create(**kw)
    _buchen(DIARIZE_MODELL, minuten)
    from fastapi.encoders import jsonable_encoder
    return list(jsonable_encoder(erg).get("segments") or [])


async def _quelle_holen(url: str, ziel: Path) -> tuple[int, str]:
    """Streamt auf Platte, mit Deckel und derselben Zielsperre wie /ai/transcribe."""
    import httpx
    from ai.clients.storage_abruf import (GesperrteDatei, PrivateDatei, ZielGesperrt, gesperrt_detail,
                                          oeffne, ohne_anfrage, private_media_detail, ziel_gesperrt_detail)
    deckel = MAX_MB * 1024 * 1024
    h, n = hashlib.sha256(), 0
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as client:
            async with oeffne(client, url) as r:
                with ziel.open("wb") as f:
                    async for chunk in r.aiter_bytes(1 << 16):
                        n += len(chunk)
                        if n > deckel:
                            raise HTTPException(status_code=413, detail={"error": "quelle_zu_gross", "max_mb": MAX_MB})
                        h.update(chunk)
                        f.write(chunk)
    except HTTPException:
        raise
    except PrivateDatei:
        raise HTTPException(status_code=403, detail=private_media_detail(url))
    except GesperrteDatei:
        raise HTTPException(status_code=451, detail=gesperrt_detail(url))
    except ZielGesperrt as e:
        raise HTTPException(status_code=422, detail=ziel_gesperrt_detail(ZielGesperrt(url, e.grund), "quelle_url_blocked"))
    except Exception as e:
        raise HTTPException(status_code=502, detail={"error": "quelle_nicht_lesbar",
                                                     "grund": str(e).replace(url, ohne_anfrage(url))[:300]})
    if n == 0:
        raise HTTPException(status_code=502, detail={"error": "quelle_leer"})
    return n, h.hexdigest()


# ── ffmpeg ───────────────────────────────────────────────────────────────────

def _probe(pfad: Path) -> tuple[int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=sample_rate,channels", "-of", "json", str(pfad)],
                       capture_output=True, text=True, timeout=60)
    try:
        s = json.loads(r.stdout)["streams"][0]
        return int(s["sample_rate"]), int(s["channels"])
    except Exception:
        raise HTTPException(status_code=422, detail={"error": "quelle_kein_audio"})


def _pcm_mono(pfad: Path, sr: int) -> bytes:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(pfad), "-ac", "1", "-ar", str(sr),
                        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"], capture_output=True, timeout=1800)
    if r.returncode != 0 or not r.stdout:
        raise HTTPException(status_code=422, detail={"error": "quelle_nicht_dekodierbar",
                                                     "upstream": r.stderr[-300:].decode("utf-8", "replace")})
    return r.stdout


def _stueck_mp3(pfad: Path, von_ms: int, bis_ms: int) -> bytes:
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{von_ms / 1000:.3f}", "-t", f"{(bis_ms - von_ms) / 1000:.3f}",
                        "-i", str(pfad), "-ac", "1", "-ar", "16000", "-b:a", "64k", "-f", "mp3", "pipe:1"],
                       capture_output=True, timeout=600)
    if r.returncode != 0 or not r.stdout:
        raise HTTPException(status_code=500, detail={"error": "stueck_schnitt_fehlgeschlagen", "von_ms": von_ms})
    return r.stdout


def _referenzen_aus(mp3: bytes, segmente: list) -> tuple[list, list]:
    """Je Sprecher das längste Segment (2–10 s) als bekannte Referenz."""
    import base64
    from ai.routes.audio_ai_routes import _ausschnitt_wav
    bestes: dict = {}
    for s in segmente:
        d = al._ms(s["end"]) - al._ms(s["start"])
        if d >= _REF_MIN_MS and d > bestes.get(s["speaker"], (0, None))[0]:
            bestes[s["speaker"]] = (d, s)
    namen, refs = [], []
    for etikett, (d, s) in list(bestes.items())[:_REF_MAX]:
        start = float(s["start"])
        ende = start + min(d, _REF_MAX_MS) / 1000
        try:
            wav = _ausschnitt_wav(mp3, start, ende)
        except HTTPException:
            continue
        namen.append(str(etikett))
        refs.append("data:audio/wav;base64," + base64.b64encode(wav).decode("ascii"))
    return namen, refs


def _roh_pfad(rid: str) -> Path:
    from ai.services import narrate_jobs as nj
    return nj.jobs_dir() / f"{rid}.roh.json"


def _peaks_pfad(rid: str) -> Path:
    from ai.services import narrate_jobs as nj
    return nj.jobs_dir() / f"{rid}.peaks.bin"


# ── Lauf ─────────────────────────────────────────────────────────────────────

async def _lauf(rid: str, req: AbschriftAnfrage, url: str) -> None:
    from ai.services import narrate_jobs as nj
    try:
        with tempfile.TemporaryDirectory(prefix="abschrift-") as d:
            quelle = Path(d) / "quelle"
            nj.stufe(rid, "laden")
            groesse, sha = await _quelle_holen(url, quelle)
            sr, kanaele = await asyncio.to_thread(_probe, quelle)
            pcm = await asyncio.to_thread(_pcm_mono, quelle, sr)
            samples = len(pcm) // 2
            dauer_ms = samples * 1000 // sr
            if dauer_ms > MAX_MIN * 60_000:
                raise HTTPException(status_code=422, detail={"error": "quelle_zu_lang", "max_min": MAX_MIN,
                                                             "dauer_ms": dauer_ms})
            peaks, werte = await asyncio.to_thread(al.peaks, pcm, sr)
            del pcm
            _peaks_pfad(rid).write_bytes(peaks)

            # #3958: Tagesgrenze eines Dienst-JWT, vor dem ersten Anbieteraufruf
            from ai.services import dienst_jwt as dj
            dj.transcribe_buchen(dauer_ms / 1000 * DURCHLAEUFE)

            teile_w, teile_d = [], []
            namen, refs = [], []
            if req.sprecher_namen:
                mp3_ganz = await asyncio.to_thread(_stueck_mp3, quelle, 0, dauer_ms)
                segs = [{"speaker": s.name, "start": s.start_ms / 1000, "end": s.ende_ms / 1000}
                        for s in req.sprecher_namen]
                namen, refs = _referenzen_aus(mp3_ganz, segs)
                del mp3_ganz
            nj.stufe(rid, "anbieter")
            for k, (von, bis) in enumerate(al.stuecke(dauer_ms)):
                mp3 = await asyncio.to_thread(_stueck_mp3, quelle, von, bis)
                minuten = (bis - von) / 60_000
                vorgabe = al.FUELL_VORGABE if req.fuelllaute_vorgabe else None
                words = await _whisper_woerter(mp3, req.sprache, vorgabe, minuten)
                segs = await _diarize(mp3, req.sprache, namen, refs, minuten)
                if k == 0 and not namen:
                    namen, refs = _referenzen_aus(mp3, segs)
                teile_w.append({"von_ms": von, "bis_ms": bis, "words": words})
                teile_d.append({"von_ms": von, "bis_ms": bis, "segments": segs})
                nj.stufe(rid, "anbieter")          # hält updated_at frisch
            nj.stufe(rid, "speichern")
            # Rohantworten behalten: ohne sie ließ sich die Sprecherzuordnung
            # des Langtests (28.11.) nicht mehr nachvollziehen.
            try:
                _roh_pfad(rid).write_text(json.dumps({"woerter": teile_w, "sprecher": teile_d},
                                                     ensure_ascii=False), encoding="utf-8")
            except Exception:
                logger.warning("abschrift rohdaten rid=%s nicht geschrieben", rid)

            woerter = al.woerter_zusammenfuehren(teile_w)
            segmente = al.segmente_aus_diarize(teile_d)
            namen_map = {s.name: s.name for s in (req.sprecher_namen or [])}
            woerter, sprecher, unsicher = al.sprecher_zuordnen(woerter, segmente, namen_map)
            ergebnis = al.abschrift_bauen(
                quelle={"storage_id": req.quelle_storage_id, "sha256": sha, "bytes": groesse,
                        "sample_rate": sr, "kanaele": kanaele, "samples": samples,
                        "dauer_s": round(samples / sr, 3)},
                woerter=woerter, sprecher=sprecher, unsicher=unsicher,
                peaks_info={"storage_id": None, "je_sekunde": al.PEAKS_JE_SEKUNDE, "werte": werte,
                            "format": "int8-minmax-mono", "abruf": f"/ai/audio/abschrift/{rid}/peaks"},
                herkunft={"woerter": WORT_MODELL, "sprecher": DIARIZE_MODELL,
                          "fuelllaute_vorgabe": req.fuelllaute_vorgabe,
                          "stuecke": [{"von_ms": a, "bis_ms": b} for a, b in al.stuecke(dauer_ms)]},
            )
            ergebnis["abschrift_storage_id"] = None
            await _speichern(rid, ergebnis, peaks)
        nj.abschliessen(rid, ergebnis)
        logger.info("abschrift fertig rid=%s woerter=%d dauer_ms=%d", rid, len(ergebnis["woerter"]), dauer_ms)
    except HTTPException as e:
        nj.scheitern(rid, e.status_code, e.detail)
        logger.warning("abschrift gescheitert rid=%s status=%s", rid, e.status_code)
    except Exception as e:
        nj.scheitern(rid, 500, {"error": "abschrift_fehlgeschlagen", "grund": type(e).__name__})
        logger.exception("abschrift fehlgeschlagen rid=%s", rid)


async def _speichern(rid: str, ergebnis: dict, peaks: bytes) -> None:
    """Private Ablage im Storage. Scheitert sie, bleibt das Ergebnis über GET
    lesbar; die bezahlte Abschrift ist damit nicht verloren."""
    try:
        from ai.clients.storage_client import save_file_and_record
        p = await save_file_and_record(data=peaks, original_filename=f"abschrift_{rid}.peaks.bin",
                                       context="radio-schnitt", is_public=False, private=True, analyze=False)
        ergebnis["peaks"]["storage_id"] = p.id
        a = await save_file_and_record(data=json.dumps(ergebnis, ensure_ascii=False).encode("utf-8"),
                                       original_filename=f"abschrift_{rid}.json", context="radio-schnitt",
                                       is_public=False, private=True, analyze=False)
        ergebnis["abschrift_storage_id"] = a.id
    except Exception as e:
        ergebnis["storage_fehler"] = type(e).__name__
        logger.warning("abschrift storage rid=%s %s", rid, type(e).__name__)


def _antwort(d: dict) -> dict:
    zustand = {"running": "laeuft", "done": "fertig", "failed": "fehler"}.get(d.get("state"), d.get("state"))
    out = {"vertrag": al.VERTRAG, "job_id": d["request_id"], "request_id": d["request_id"],
           "status": zustand, "stufe": d.get("stage")}
    if d.get("stale") and zustand == "laeuft":
        out["haengt"] = True
    if zustand == "fertig":
        out["ergebnis"] = d.get("result")
    if zustand == "fehler":
        out["fehler"] = (d.get("error") or {}).get("detail")
        out["fehler_stufe"] = d.get("failed_stage")
        out["neuanlauf_moeglich"] = d.get("failed_stage") in ("pre_tts", "laden")
    return out


# ── Routen ───────────────────────────────────────────────────────────────────

@router.post("/audio/abschrift", status_code=202)
async def abschrift_starten(req: AbschriftAnfrage, request: Request,
                            authorization: Optional[str] = Header(None),
                            x_api_key: Optional[str] = Header(None, alias="X-API-KEY")):
    from ai.services import narrate_jobs as nj
    wer = _aufrufer(request, authorization, x_api_key)
    rid = req.request_id
    if not nj.request_id_ok(rid):
        raise HTTPException(status_code=422, detail={"error": "invalid_request_id"})
    if not req.confirm_api_billing:
        raise HTTPException(status_code=403, detail={
            "error": "billing_noetig", "usd_je_minute": USD_JE_MIN * DURCHLAEUFE,
            "hinweis": "Zwei bezahlte STT-Durchläufe (Wörter und Sprecher). confirm_api_billing: true mitschicken."})
    for s in req.sprecher_namen or []:
        if not (_REF_MIN_MS <= s.ende_ms - s.start_ms):
            raise HTTPException(status_code=422, detail={"error": "sprecher_referenz_zu_kurz", "name": s.name,
                                                         "min_ms": _REF_MIN_MS})
    h = al.anfrage_hash(req.model_dump())
    alt = nj.lesen(rid)
    if alt:
        if alt.get("kind") != KIND:
            raise HTTPException(status_code=409, detail={"error": "request_id_gehoert_anderer_route", "request_id": rid})
        if alt.get("payload_hash") != h:
            raise HTTPException(status_code=409, detail={"error": "request_id_konflikt", "job_id": rid})
        if not (alt.get("state") == "failed" and alt.get("failed_stage") in ("pre_tts", "laden")):
            return JSONResponse(status_code=200, content={**_antwort(alt), "wiederholt": True})
    gewonnen = None
    try:
        gewonnen = nj.reservieren(rid, h, kind=KIND, extra={"aufrufer": wer[:64]})
    except nj.ClaimConflict:
        pass
    if not gewonnen:
        try:
            gewonnen = nj.uebernehmen(rid, h, KIND, lambda a: a.get("state") == "failed"
                                      and a.get("failed_stage") in ("pre_tts", "laden")
                                      and a.get("payload_hash") == h)
        except nj.ClaimConflict:
            gewonnen = None
        if not gewonnen:
            d = nj.lesen(rid) or {"request_id": rid, "state": "running"}
            return JSONResponse(status_code=200, content={**_antwort(d), "wiederholt": True})
    url = req.quelle_url
    if not url:
        base = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com").rstrip("/")
        url = f"{base}/storage/media/{int(req.quelle_storage_id)}?variant=full"
    task = asyncio.create_task(_lauf(rid, req, url))
    _LAEUFE.add(task)
    task.add_done_callback(_LAEUFE.discard)
    logger.info("abschrift gestartet rid=%s aufrufer=%s quelle=%s", rid, wer[:64], req.quelle_storage_id)
    return {**_antwort(nj.lesen(rid) or gewonnen), "wiederholt": False}


@router.get("/audio/abschrift/{request_id}")
async def abschrift_status(request_id: str, request: Request,
                           authorization: Optional[str] = Header(None),
                           x_api_key: Optional[str] = Header(None, alias="X-API-KEY")):
    from ai.services import narrate_jobs as nj
    _aufrufer(request, authorization, x_api_key)
    d = nj.lesen(request_id) if nj.request_id_ok(request_id) else None
    if not d or d.get("kind") != KIND:
        raise HTTPException(status_code=404, detail={"error": "unbekannte_request_id", "request_id": request_id})
    return _antwort(d)


@router.get("/audio/abschrift/{request_id}/peaks")
async def abschrift_peaks(request_id: str, request: Request,
                          authorization: Optional[str] = Header(None),
                          x_api_key: Optional[str] = Header(None, alias="X-API-KEY")):
    from ai.services import narrate_jobs as nj
    _aufrufer(request, authorization, x_api_key)
    d = nj.lesen(request_id) if nj.request_id_ok(request_id) else None
    if not d or d.get("kind") != KIND or d.get("state") != "done":
        raise HTTPException(status_code=404, detail={"error": "keine_peaks", "request_id": request_id})
    p = _peaks_pfad(request_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "keine_peaks", "request_id": request_id})
    return Response(content=p.read_bytes(), media_type="application/octet-stream",
                    headers={"X-Peaks-Werte": str(len(p.read_bytes()) // 2)})
