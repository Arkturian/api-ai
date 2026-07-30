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

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
import time
import json as _json
import asyncio

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


def register_temp_dialog_chunks(req_id: str, items: List[Dict[str, Any]]):
    """Register temp dialog chunks in memory AND persist to disk for multi-worker access."""
    TEMP_DIALOG_REGISTRY[req_id] = items
    # Persist to disk for cross-worker visibility (uvicorn --workers 2)
    try:
        with open(os.path.join(STATUS_DIR, f"{req_id}.chunks.json"), "w") as f:
            _json.dump(items, f)
    except Exception:
        pass


def _load_chunks_from_disk(req_id: str) -> List[Dict[str, Any]]:
    """Load chunks from disk if not in memory (for multi-worker scenarios)."""
    chunks_path = os.path.join(STATUS_DIR, f"{req_id}.chunks.json")
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
        with open(os.path.join(STATUS_DIR, f"{req_id}.events.jsonl"), "a") as f:
            f.write(_json.dumps(ev) + "\n")
    except Exception:
        pass


def _load_events_from_disk(req_id: str) -> List[Dict[str, Any]]:
    # Load all events for this req_id from file if present
    events_path = os.path.join(STATUS_DIR, f"{req_id}.events.jsonl")
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
        with open(os.path.join(STATUS_DIR, f"{req_id}.json"), "w") as f:
            _json.dump(st, f)
    except Exception:
        pass


def set_dialog_result(req_id: str, result: Dict[str, Any]):
    TEMP_DIALOG_RESULTS[req_id] = result
    try:
        with open(os.path.join(STATUS_DIR, f"{req_id}.result.json"), "w") as f:
            _json.dump(result, f)
    except Exception:
        pass


@router.get("/tempchunk")
def get_temp_dialog_chunk(id: str, index: int):
    """
    Get a temporary audio chunk from dialog generation.

    Used for streaming dialog as it's being generated.
    Returns the audio file for a specific chunk index.

    Note: Uses disk-based fallback to support multi-worker uvicorn deployments.
    """
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
        if not os.path.abspath(path).startswith("/tmp/"):
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
def clear_temp_dialog_chunks(id: str):
    """
    Clear temporary dialog chunks after job completion.

    Cleanup temp files to save disk space.
    """
    TEMP_DIALOG_REGISTRY.pop(id, None)
    TEMP_DIALOG_STATUS.pop(id, None)
    TEMP_DIALOG_HISTORY.pop(id, None)
    try:
        os.remove(os.path.join(STATUS_DIR, f"{id}.json"))
    except Exception:
        pass
    try:
        os.remove(os.path.join(STATUS_DIR, f"{id}.events.jsonl"))
    except Exception:
        pass
    try:
        os.remove(os.path.join(STATUS_DIR, f"{id}.chunks.json"))
    except Exception:
        pass
    return {"status": "cleared"}


@router.get("/status")
def get_dialog_status(id: str):
    """
    Check status of dialog generation job.

    Returns status, progress, and event history.
    """
    st = TEMP_DIALOG_STATUS.get(id)
    if not st:
        # Try file-backed status for other workers
        try:
            with open(os.path.join(STATUS_DIR, f"{id}.json"), "r") as f:
                st = _json.load(f)
        except Exception:
            st = {"status": "unknown"}
    # Attach result if present (memory or file)
    res = TEMP_DIALOG_RESULTS.get(id)
    if not res:
        try:
            with open(os.path.join(STATUS_DIR, f"{id}.result.json"), "r") as f:
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
def cancel_dialog_job(id: str):
    """
    Cancel a running dialog generation job.

    Sets status to cancelled and attempts to cancel the background task.
    """
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
    api_key: str = Depends(get_api_key)
):
    """
    Start a dialog generation job.

    Process:
    1. Validate dialog_mode is enabled
    2. Create background task for audio drama generation
    3. Return job_id for status polling
    """
    if not getattr(req.config, 'dialog_mode', False):
        raise HTTPException(status_code=400, detail="dialog_mode must be true")

    try:
        # Initialize status
        try:
            set_dialog_status(req.id, phase="queued")
        except Exception:
            # Fallback file write in case helper fails
            try:
                with open(os.path.join(STATUS_DIR, f"{req.id}.json"), "w") as f:
                    _json.dump({"phase": "queued", "updated_at": time.time()}, f)
            except Exception:
                pass

        async def run_job():
            try:
                # Force analyze_only False for the generation run
                req.config.analyze_only = False
                set_dialog_status(req.id, phase="analyze", subphase="start")

                generator = AudioDramaGenerator(req, api_key, image_gen_func=None)

                # Hard job TTL to avoid wedging a worker (e.g., 180s)
                saved_audio_obj, production_plan, generated_image_obj = await asyncio.wait_for(
                    generator.generate(),
                    timeout=180
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
            except Exception as e:
                set_dialog_status(req.id, phase="error", error=str(e))

        task = asyncio.create_task(run_job())
        TASKS[req.id] = task
        return {"job_id": req.id, "phase": "queued"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
