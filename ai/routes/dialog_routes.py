"""
Dialog System Routes
====================

Multi-character dialog generation with TTS.

Features:
- Background job processing for long-running dialog generation
- Real-time status updates via polling
- Temporary audio chunk streaming
- Job cancellation support
"""

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
import time
import json as _json
import asyncio
import hmac
import re

logger = logging.getLogger(__name__)
router = APIRouter()

# Import services
from ai.services.tts_models import SpeechRequest
from ai.services.audio_drama_service import AudioDramaGenerator
from fastapi.encoders import jsonable_encoder


# In-memory storage for dialog jobs
TEMP_DIALOG_REGISTRY: Dict[str, List[Dict[str, Any]]] = {}
TEMP_DIALOG_STATUS: Dict[str, Dict[str, Any]] = {}
TEMP_DIALOG_RESULTS: Dict[str, Dict[str, Any]] = {}
TEMP_DIALOG_HISTORY: Dict[str, List[Dict[str, Any]]] = {}
TASKS: Dict[str, Any] = {}
STATUS_DIR = "/tmp/dialog_status"
EVENTS_MAX = 300

# Ensure status directory exists
os.makedirs(STATUS_DIR, exist_ok=True)


def get_api_key():
    return "placeholder"  # not verified anywhere; see #514 / auth decision


# ── Absicherung (#2935, Befund 56) ───────────────────────────────────────
# Die Kennung wurde ungeprueft an STATUS_DIR gehaengt; der Dienst laeuft als
# root. `../../root/.codex/auth` las so beliebige .json-Dateien, clear_temp
# loeschte sie. Jetzt: strenge Kennung, Pfad bleibt in STATUS_DIR, Zugriff nur
# angemeldet und nur auf eigene Auftraege.
_DIALOG_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _pruefe_id(req_id: Any) -> str:
    if not isinstance(req_id, str) or not _DIALOG_ID.match(req_id):
        raise HTTPException(status_code=422, detail={
            "error": "invalid_dialog_id", "hint": "Erlaubt: A-Z a-z 0-9 _ -, 1 bis 64 Zeichen."})
    return req_id


def _status_pfad(req_id: str, endung: str) -> str:
    """Datei in STATUS_DIR fuer diese Kennung; wirft bei jeder anderen Form."""
    _pruefe_id(req_id)
    pfad = os.path.join(STATUS_DIR, req_id + endung)
    basis = os.path.realpath(STATUS_DIR) + os.sep
    if not os.path.realpath(pfad).startswith(basis):
        raise HTTPException(status_code=422, detail={"error": "invalid_dialog_id"})
    return pfad


def _dialog_aufrufer(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-KEY"),
) -> str:
    """Wer ruft? Betreiber-Schluessel oder ein gepruefter JWT. Sonst 401."""
    erwartet = os.environ.get("REALTIME_GRANT_SERVICE_KEY", "")
    if erwartet and x_api_key and hmac.compare_digest(x_api_key, erwartet):
        return "betreiber"
    from ai.services import bild_verbrauch as bv
    claims = bv._claims_geprueft(authorization)
    name = bv.name_aus_claims(claims) if claims else None
    if name:
        return name
    raise HTTPException(status_code=401, detail={
        "error": "dialog_auth_required",
        "hint": "Bearer-JWT der Foederation oder Betreiber-Schluessel (X-API-KEY) mitschicken."})


def _besitzer_setzen(req_id: str, wer: str) -> None:
    pfad = _status_pfad(req_id, ".owner")
    try:
        with open(pfad, "x") as f:
            f.write(wer)
        return
    except FileExistsError:
        pass
    with open(pfad) as f:
        alt = f.read().strip()
    if alt != wer and wer != "betreiber":
        raise HTTPException(status_code=409, detail={"error": "dialog_id_taken"})


def _besitzer_pruefen(req_id: str, wer: str) -> None:
    """Nur der Starter (oder der Betreiber) sieht oder steuert einen Auftrag.
    Unbekannt und fremd sehen gleich aus (404), damit nichts verraten wird."""
    pfad = _status_pfad(req_id, ".owner")
    if wer == "betreiber":
        return
    try:
        with open(pfad) as f:
            if f.read().strip() == wer:
                return
    except OSError:
        pass
    raise HTTPException(status_code=404, detail="Dialog not found")


def register_temp_dialog_chunks(req_id: str, items: List[Dict[str, Any]]):
    """Register temp dialog chunks in memory AND persist to disk for multi-worker access."""
    TEMP_DIALOG_REGISTRY[req_id] = items
    # Persist to disk for cross-worker visibility (uvicorn --workers 2)
    try:
        with open(_status_pfad(req_id, ".chunks.json"), "w") as f:
            _json.dump(items, f)
    except Exception:
        pass


def _load_chunks_from_disk(req_id: str) -> List[Dict[str, Any]]:
    """Load chunks from disk if not in memory (for multi-worker scenarios)."""
    chunks_path = _status_pfad(req_id, ".chunks.json")
    if not os.path.exists(chunks_path):
        return []
    try:
        with open(chunks_path, "r") as f:
            return _json.load(f)
    except Exception:
        return []


def _append_status_event(req_id: str, event: Dict[str, Any]):
    ev = dict(event)
    ev["ts"] = time.time()
    hist = TEMP_DIALOG_HISTORY.get(req_id) or []
    hist.append(ev)
    if len(hist) > EVENTS_MAX:
        hist = hist[-EVENTS_MAX:]
    TEMP_DIALOG_HISTORY[req_id] = hist
    # Append to file as JSONL for robustness
    try:
        with open(_status_pfad(req_id, ".events.jsonl"), "a") as f:
            f.write(_json.dumps(ev) + "\n")
    except Exception:
        pass


def _load_events_from_disk(req_id: str) -> List[Dict[str, Any]]:
    # Load all events for this req_id from file if present
    events_path = _status_pfad(req_id, ".events.jsonl")
    if not os.path.exists(events_path):
        return []
    try:
        events: List[Dict[str, Any]] = []
        with open(events_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(_json.loads(line))
                except Exception:
                    pass
        if len(events) > EVENTS_MAX:
            events = events[-EVENTS_MAX:]
        return events
    except Exception:
        return []


def set_dialog_status(req_id: str, **fields):
    st = TEMP_DIALOG_STATUS.get(req_id) or {"updated_at": time.time()}
    st.update(fields)
    st["updated_at"] = time.time()
    TEMP_DIALOG_STATUS[req_id] = st
    _append_status_event(req_id, fields)
    # Persist to disk for multi-worker visibility
    try:
        with open(_status_pfad(req_id, ".json"), "w") as f:
            _json.dump(st, f)
    except Exception:
        pass


def set_dialog_result(req_id: str, result: Dict[str, Any]):
    TEMP_DIALOG_RESULTS[req_id] = result
    try:
        with open(_status_pfad(req_id, ".result.json"), "w") as f:
            _json.dump(result, f)
    except Exception:
        pass


@router.get("/tempchunk")
def get_temp_dialog_chunk(id: str, index: int, wer: str = Depends(_dialog_aufrufer)):
    """
    Get a temporary audio chunk from dialog generation.

    Used for streaming dialog as it's being generated.
    Returns the audio file for a specific chunk index.

    Note: Uses disk-based fallback to support multi-worker uvicorn deployments.
    """
    _besitzer_pruefen(id, wer)
    try:
        # Try in-memory first, then disk fallback for multi-worker scenarios
        items = TEMP_DIALOG_REGISTRY.get(id)
        if not items:
            # Fallback to disk for cross-worker access
            items = _load_chunks_from_disk(id)
            if items:
                # Cache in memory for subsequent requests to this worker
                TEMP_DIALOG_REGISTRY[id] = items

        if not items:
            raise HTTPException(status_code=404, detail="Chunk not found")

        # Find by original sequence index (robust to filtered registry order)
        match = None
        for it in items:
            if int(it.get("index", -1)) == int(index):
                match = it
                break
        if not match:
            raise HTTPException(status_code=404, detail="Chunk not found")
        path = match.get("path")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File missing")
        # Only allow files from /tmp for safety
        if not os.path.realpath(path).startswith("/tmp/"):
            raise HTTPException(status_code=403, detail="Forbidden path")
        # Guess media type by extension (mp3 default)
        ext = os.path.splitext(path)[1].lower()
        media = "audio/mpeg" if ext == ".mp3" else "application/octet-stream"
        return FileResponse(path, media_type=media)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear_temp")
def clear_temp_dialog_chunks(id: str, wer: str = Depends(_dialog_aufrufer)):
    """
    Clear temporary dialog chunks after job completion.

    Cleanup temp files to save disk space.
    """
    _besitzer_pruefen(id, wer)
    TEMP_DIALOG_REGISTRY.pop(id, None)
    TEMP_DIALOG_STATUS.pop(id, None)
    TEMP_DIALOG_HISTORY.pop(id, None)
    try:
        os.remove(_status_pfad(id, ".json"))
    except Exception:
        pass
    try:
        os.remove(_status_pfad(id, ".events.jsonl"))
    except Exception:
        pass
    try:
        os.remove(_status_pfad(id, ".chunks.json"))
    except Exception:
        pass
    return {"status": "cleared"}


@router.get("/status")
def get_dialog_status(id: str, wer: str = Depends(_dialog_aufrufer)):
    """
    Check status of dialog generation job.

    Returns status, progress, and event history.
    """
    _besitzer_pruefen(id, wer)
    st = TEMP_DIALOG_STATUS.get(id)
    if not st:
        # Try file-backed status for other workers
        try:
            with open(_status_pfad(id, ".json"), "r") as f:
                st = _json.load(f)
        except Exception:
            st = {"status": "unknown"}
    # Attach result if present (memory or file)
    res = TEMP_DIALOG_RESULTS.get(id)
    if not res:
        try:
            with open(_status_pfad(id, ".result.json"), "r") as f:
                res = _json.load(f)
        except Exception:
            res = None
    if res:
        st = {**st, "result": res}
    # Attach history (memory or disk)
    hist = TEMP_DIALOG_HISTORY.get(id)
    if not hist:
        hist = _load_events_from_disk(id)
        if hist:
            TEMP_DIALOG_HISTORY[id] = hist
    if hist:
        st = {**st, "history": hist[-100:]}
    return st


@router.post("/cancel")
def cancel_dialog_job(id: str, wer: str = Depends(_dialog_aufrufer)):
    """
    Cancel a running dialog generation job.

    Sets status to cancelled and attempts to cancel the background task.
    """
    _besitzer_pruefen(id, wer)
    task = TASKS.get(id)
    if task and not task.done():
        try:
            task.cancel()
        except Exception:
            pass
        TEMP_DIALOG_STATUS[id] = {"phase": "cancelled", "updated_at": time.time()}
        return {"status": "cancelled"}
    return {"status": "not_running"}


@router.post("/start")
async def start_dialog_job(
    req: SpeechRequest,
    api_key: str = Depends(get_api_key),
    wer: str = Depends(_dialog_aufrufer),
):
    """
    Start a dialog generation job.

    Process:
    1. Validate dialog_mode is enabled
    2. Create background task for audio drama generation
    3. Return job_id for status polling
    """
    _besitzer_setzen(_pruefe_id(req.id), wer)
    if not getattr(req.config, 'dialog_mode', False):
        raise HTTPException(status_code=400, detail="dialog_mode must be true")

    try:
        # Initialize status
        try:
            set_dialog_status(req.id, phase="queued")
        except Exception:
            # Fallback file write in case helper fails
            try:
                with open(_status_pfad(req.id, ".json"), "w") as f:
                    _json.dump({"phase": "queued", "updated_at": time.time()}, f)
            except Exception:
                pass

        async def run_job():
            try:
                # Force analyze_only False for the generation run
                req.config.analyze_only = False
                set_dialog_status(req.id, phase="analyze", subphase="start")

                generator = AudioDramaGenerator(req, api_key, image_gen_func=None)

                # Harte Laufzeitgrenze, damit kein Arbeiter haengenbleibt.
                # 180 s reichten fuer die Analyse per Claude Opus; seit der
                # Umstellung auf gpt-6-astra mit Effort "high" (Alexanders
                # Vorgabe, 12.09.) denkt das Modell laenger, und danach
                # kommt erst die eigentliche Produktion. Ueber die Umgebung
                # setzbar, damit eine Grenze kein Deploy braucht.
                _ttl = float(os.getenv("DIALOG_JOB_TTL_S", "900"))
                saved_audio_obj, production_plan, generated_image_obj = await asyncio.wait_for(
                    generator.generate(),
                    timeout=_ttl
                )

                # Build response compatible with /ai/generate_speech
                if saved_audio_obj is not None:
                    response_data = {
                        "id": saved_audio_obj.id,
                        "file_url": saved_audio_obj.file_url,
                        "object_key": saved_audio_obj.object_key,
                        "original_filename": saved_audio_obj.original_filename,
                        "mime_type": saved_audio_obj.mime_type,
                        "file_size": saved_audio_obj.file_size,
                        "context": saved_audio_obj.context,
                        "is_public": saved_audio_obj.is_public,
                        "collection_id": saved_audio_obj.collection_id
                    }
                else:
                    response_data = {}

                response_data['analysis_result'] = production_plan

                if generated_image_obj:
                    response_data['generated_image'] = {
                        "id": generated_image_obj.id,
                        "file_url": generated_image_obj.file_url
                    }

                set_dialog_result(req.id, response_data)
                set_dialog_status(req.id, phase="done", result_ready=True)

            except asyncio.CancelledError:
                set_dialog_status(req.id, phase="cancelled")
            except asyncio.TimeoutError:
                set_dialog_status(req.id, phase="error", error="deadline_exceeded")
            except HTTPException as e:
                # Fehlercode maschinenlesbar weitergeben statt ihn in einen
                # String zu giessen: der Poller auf /ai/dialog/status soll
                # "no_dialog_cues" auswerten koennen, ohne Text zu parsen.
                _detail = e.detail
                _code = _detail.get("error") if isinstance(_detail, dict) else None
                set_dialog_status(
                    req.id,
                    phase="error",
                    error=_code or str(_detail),
                    error_status=e.status_code,
                    error_detail=_detail if isinstance(_detail, dict) else {"message": str(_detail)},
                )
            except Exception as e:
                set_dialog_status(req.id, phase="error", error=str(e))

        task = asyncio.create_task(run_job())
        TASKS[req.id] = task
        return {"job_id": req.id, "phase": "queued"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
