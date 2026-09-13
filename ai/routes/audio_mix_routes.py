"""POST /ai/audio/mix — deterministisches Mischen gespeicherter Audio-IDs.

Story-Codex (13.09., unter Alex' Audio-Auftrag; Vertrag Content #4976,
p-b12b0d4d6062): Sprache, Atmosphaere, Geraeusche und Musik liegen als
einzelne Storage-Objekte vor; der Player braucht eine Datei mit
Startzeiten. Bis heute gab es dafuer nur den Mischer des Hoerspielpfads,
der auf lokalen Pfaden arbeitet und die Zeilen immer neu spricht.

Hier wird NICHTS erzeugt, nur gerechnet: kein Anbieteraufruf, kein
Zaehler, keine Sperre. Dieselben Eingaben ergeben dieselben Bytes
(`-fflags +bitexact`); der Filtergraph steht in der Antwort.

Quellen kommen ueber `audio_id` aus dem Storage. Der Dienstschluessel
geht nur an den eigenen Storage-Ursprung — dieselbe Pruefung wie bei
Referenzbildern (#1794), nicht neu gebaut, sondern importiert.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_TRACKS = 32
MAX_DURATION_S = 1800.0
MAX_SOURCE_BYTES = 100 * 1024 * 1024
_FORMATE = {"mp3": ("libmp3lame", "mp3"), "wav": ("pcm_s16le", "wav")}


class MixTrack(BaseModel):
    audio_id: int
    start_s: float = Field(0.0, ge=0.0, description="Platz auf der Zeitleiste")
    source_offset_s: float = Field(0.0, ge=0.0, description="Einstieg in die Quelle")
    duration_s: Optional[float] = Field(None, gt=0.0, description="Laenge des Ausschnitts; None = bis Quellende")
    gain_db: float = Field(0.0, ge=-60.0, le=20.0)
    fade_in_s: float = Field(0.0, ge=0.0)
    fade_out_s: float = Field(0.0, ge=0.0)


class MixRequest(BaseModel):
    request_id: Optional[str] = None
    tracks: List[MixTrack]
    duration_s: Optional[float] = Field(None, gt=0.0, description="Gesamtlaenge; None = spaetestes Spurende")
    output_format: str = "mp3"
    sample_rate: int = Field(44100, ge=8000, le=48000)
    normalize: bool = False
    save_options: Optional[dict] = None
    collection_id: Optional[str] = "ai-generated-audio"
    link_id: Optional[str] = None


def mix_hash(req: MixRequest) -> str:
    """Reihenfolge der Spuren ist egal: gleiche Spuren, gleiche Bytes."""
    d = req.model_dump()
    d.pop("request_id", None)
    d.pop("save_options", None)
    d["tracks"] = sorted(d["tracks"], key=lambda t: (t["audio_id"], t["start_s"], t["source_offset_s"]))
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pruefe_grenzen(req: MixRequest) -> None:
    if not req.tracks:
        raise HTTPException(status_code=422, detail={"error": "no_tracks"})
    if len(req.tracks) > MAX_TRACKS:
        raise HTTPException(status_code=422, detail={"error": "too_many_tracks", "max": MAX_TRACKS, "given": len(req.tracks)})
    if req.output_format not in _FORMATE:
        raise HTTPException(status_code=422, detail={"error": "unsupported_output_format", "supported": sorted(_FORMATE)})
    if req.duration_s and req.duration_s > MAX_DURATION_S:
        raise HTTPException(status_code=422, detail={"error": "duration_too_long", "max_s": MAX_DURATION_S})
    for t in req.tracks:
        if t.duration_s and t.start_s + t.duration_s > MAX_DURATION_S:
            raise HTTPException(status_code=422, detail={"error": "duration_too_long", "max_s": MAX_DURATION_S, "audio_id": t.audio_id})


def filtergraph(tracks: List[MixTrack], quellen_dauer: List[float], sample_rate: int,
                duration_s: Optional[float], normalize: bool) -> tuple:
    """(filter_complex, gesamtdauer). Deterministisch aus den Eingaben.

    Je Spur: atrim (Ausschnitt) -> asetpts -> volume -> afade -> adelay
    (Platz auf der Zeitleiste) -> apad auf die Gesamtlaenge. Dann amix
    und volume=N.

    Warum apad + volume=N statt `amix=normalize=0`: `normalize` gibt es
    erst ab ffmpeg 4.4. arkturian laeuft aelter (gemessen 13.09., der
    erste Live-Mix scheiterte dort mit "Option 'normalize' not found").
    Das alte amix teilt die Summe durch die Zahl der gerade AKTIVEN
    Eingaenge — endet eine Spur, springt der Pegel. Sind alle Spuren per
    apad gleich lang, ist der Teiler konstant N, und volume=N hebt ihn
    wieder auf: exakte Summe, auf jeder ffmpeg-Version gleich.
    """
    enden = []
    laengen = []
    for t, qd in zip(tracks, quellen_dauer):
        ende_quelle = min(qd, t.source_offset_s + t.duration_s) if t.duration_s else qd
        laenge = max(0.0, ende_quelle - t.source_offset_s)
        laengen.append((ende_quelle, laenge))
        enden.append(t.start_s + laenge)
    gesamt = duration_s if duration_s else (max(enden) if enden else 0.0)
    teile, labels = [], []
    for i, (t, (ende_quelle, laenge)) in enumerate(zip(tracks, laengen)):
        kette = [
            f"[{i}:a]aresample={sample_rate}",
            f"atrim=start={t.source_offset_s:.3f}:end={ende_quelle:.3f}",
            "asetpts=PTS-STARTPTS",
        ]
        if t.gain_db:
            kette.append(f"volume={t.gain_db:.2f}dB")
        if t.fade_in_s:
            kette.append(f"afade=t=in:st=0:d={t.fade_in_s:.3f}")
        if t.fade_out_s and laenge > t.fade_out_s:
            kette.append(f"afade=t=out:st={laenge - t.fade_out_s:.3f}:d={t.fade_out_s:.3f}")
        ms = int(round(t.start_s * 1000))
        kette.append(f"adelay={ms}|{ms}")
        kette.append(f"apad=whole_dur={gesamt:.3f}")
        teile.append(",".join(kette) + f"[t{i}]")
        labels.append(f"[t{i}]")
    n = len(tracks)
    mix = f"{''.join(labels)}amix=inputs={n}:dropout_transition=0,volume={n}"
    if normalize:
        mix += ",loudnorm=I=-16:TP=-1.5:LRA=11"
    mix += f",atrim=end={gesamt:.3f}[out]"
    return ";".join(teile + [mix]), gesamt


def _dauer(pfad: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(pfad)],
                       capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def mischen(pfade: List[Path], tracks: List[MixTrack], sample_rate: int, output_format: str,
            duration_s: Optional[float], normalize: bool, ziel: Path) -> tuple:
    dauern = [_dauer(p) for p in pfade]
    graph, gesamt = filtergraph(tracks, dauern, sample_rate, duration_s, normalize)
    codec, fmt = _FORMATE[output_format]
    cmd = ["ffmpeg", "-y", "-v", "error", "-fflags", "+bitexact", "-flags:a", "+bitexact"]
    for p in pfade:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", graph, "-map", "[out]", "-ar", str(sample_rate), "-ac", "2",
            "-c:a", codec, "-f", fmt, str(ziel)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(status_code=500, detail={"error": "ffmpeg_failed", "stderr": (r.stderr or "")[-600:]})
    return graph, gesamt


async def _hole_quelle(audio_id: int, ziel: Path) -> int:
    """Laedt ein Storage-Objekt; Schluessel nur an den eigenen Ursprung."""
    from ai.clients.storage_client import storage_api_key
    from ai.routes.image_ai_routes import _hole_referenz

    base = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com").rstrip("/")
    url = f"{base}/storage/media/{int(audio_id)}?variant=full"
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as client:
        r = await _hole_referenz(client, url, storage_api_key())
    if r.status_code != 200:
        raise HTTPException(status_code=422, detail={"error": "source_unavailable", "audio_id": audio_id, "status": r.status_code})
    if len(r.content) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=422, detail={"error": "source_too_large", "audio_id": audio_id, "max_bytes": MAX_SOURCE_BYTES})
    ziel.write_bytes(r.content)
    return len(r.content)


@router.post("/audio/mix")
async def audio_mix(req: MixRequest):
    from ai.clients.storage_client import save_file_and_record
    from ai.services import narrate_jobs as nj

    pruefe_grenzen(req)
    rid = req.request_id
    h = mix_hash(req)
    if rid:
        if not nj.request_id_ok(rid):
            raise HTTPException(status_code=422, detail={"error": "invalid_request_id"})
        alt = nj.lesen(rid)
        if alt:
            if alt.get("payload_hash") != h:
                raise HTTPException(status_code=409, detail={"error": "request_id_payload_mismatch", "request_id": rid})
            if alt.get("state") == "done":
                out = dict(alt["result"]); out["replayed"] = True
                return out
            if alt.get("state") == "running" and not alt.get("stale"):
                raise HTTPException(status_code=409, detail={"error": "mix_in_progress", "request_id": rid}, headers={"Retry-After": "5"})
        nj.anlegen(rid, h)

    try:
        with tempfile.TemporaryDirectory(prefix="audio-mix-") as d:
            d = Path(d)
            pfade = []
            for i, t in enumerate(req.tracks):
                p = d / f"src_{i}_{t.audio_id}"
                await _hole_quelle(t.audio_id, p)
                pfade.append(p)
            ziel = d / f"mix.{req.output_format}"
            graph, gesamt = await asyncio.to_thread(
                mischen, pfade, req.tracks, req.sample_rate, req.output_format, req.duration_s, req.normalize, ziel)
            daten = ziel.read_bytes()
            echte_dauer = await asyncio.to_thread(_dauer, ziel)

        saved = None
        if req.save_options is not None:
            saved = await save_file_and_record(
                data=daten, original_filename=f"mix_{uuid.uuid4().hex[:8]}.{req.output_format}",
                context="audio-mix", is_public=bool(req.save_options.get("is_public", True)),
                collection_id=req.collection_id, link_id=req.link_id)
        result = {
            "id": saved.id if saved else None,
            "audio_url": saved.file_url if saved else None,
            "file_url": saved.file_url if saved else None,
            "saved": saved is not None,
            "duration_seconds": round(echte_dauer, 3),
            "planned_duration_seconds": round(gesamt, 3),
            "tracks_used": [{"audio_id": t.audio_id, "start_s": t.start_s} for t in req.tracks],
            "ffmpeg_filter": graph,
            "sample_rate": req.sample_rate,
            "output_format": req.output_format,
            "request_id": rid,
            "replayed": False,
            "provider_calls": 0,
        }
        if rid:
            nj.abschliessen(rid, result)
        return result
    except HTTPException as e:
        if rid:
            nj.scheitern(rid, e.status_code, e.detail)
        raise
    except Exception as e:
        if rid:
            nj.scheitern(rid, None, str(e)[:300])
        logger.exception("audio mix failed")
        raise HTTPException(status_code=500, detail={"error": "mix_failed", "exc": str(e)[:200]})
