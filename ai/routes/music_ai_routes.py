"""
Music AI Routes
===============

Endpoints for AI music generation. Currently MiniMax Music 2.6 is the
only provider (cover-gen + full track synthesis). Endpoint family
mirrors the video pattern: synchronous polling by default, optional
``wait_for_result=false`` for caller-driven polling.

POST /ai/music                        — generate full track / cover
POST /ai/music/status/{request_id}    — poll-and-fetch for async submissions
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


def get_api_key():
    return "placeholder"


class MusicGenRequest(BaseModel):
    """Body for ``POST /ai/music``.

    MiniMax Music 2.6 supports two generation modes:
      * **full**:  generate an original track from a prompt
                   (genre, mood, tempo description)
      * **cover**: cover an uploaded reference song with the
                   prompt-described style (requires reference_audio_url)
    """

    prompt: str = Field(..., description="Style / genre / mood description")
    lyrics: str = Field(
        ...,
        description="Sung lyrics (MiniMax Music 2.6 pflicht-feld). Für reine instrumental tracks: '[Instrumental]' oder einen kurzen Marker übergeben.",
    )
    mode: Literal["full", "cover"] = Field(
        default="full",
        description="full = original generation, cover = re-stylise reference_audio_url",
    )
    reference_audio_url: Optional[str] = Field(
        default=None,
        description="Required when mode='cover' — reference song to cover",
    )
    duration: int = Field(
        default=30, ge=10, le=240,
        description="Track length in seconds (10..240). MiniMax may clamp.",
    )
    model: str = Field(
        default="minimax-music-2.6",
        description="Currently only minimax-music-2.6 is supported",
    )
    wait_for_result: bool = Field(
        default=True,
        description="If true poll inline up to ~10 min; else return task_id for caller-driven poll",
    )
    collection_id: Optional[str] = Field(
        default="ai-generated-music", description="Storage collection for the MP3"
    )
    link_id: Optional[str] = Field(default=None)
    confirm_api_billing: bool = Field(
        default=False,
        description="Required true — MiniMax Music is pay-as-you-go billed per track",
    )


class MusicGenResponse(BaseModel):
    request_id: str
    status: str
    audio_url: Optional[str] = None
    storage_object_id: Optional[int] = None
    model: str
    duration: int
    mode: str
    message: Optional[str] = None


@router.post("/music", response_model=MusicGenResponse)
async def generate_music_endpoint(
    request: MusicGenRequest, api_key: str = Depends(get_api_key)
):
    """Generate AI music via MiniMax Music 2.6.

    Sync mode polls for up to ~10 min. Caller can opt out via
    ``wait_for_result=false`` and poll the MiniMax task themselves.
    """
    import asyncio
    import httpx
    from ai.clients.minimax_client import post_json, get_json, base_resp_failed
    from ai.clients.storage_client import save_file_and_record
    from ai.services.minimax_cost_tracker import minimax_cost_tracker
    from ai.routes.text_ai_routes import _check_minimax_billing_gate

    _check_minimax_billing_gate(
        request.confirm_api_billing, endpoint="minimax-music"
    )

    if request.mode == "cover" and not request.reference_audio_url:
        raise HTTPException(
            status_code=400,
            detail="mode='cover' requires reference_audio_url",
        )

    submit_payload: dict = {
        "model": "music-2.6",
        "prompt": request.prompt,
        "lyrics": request.lyrics,
        "duration": request.duration,
        "mode": request.mode,
    }
    if request.reference_audio_url:
        submit_payload["reference_audio_url"] = request.reference_audio_url

    logger.info(
        f"MiniMax music gen: model={request.model} mode={request.mode} "
        f"duration={request.duration}s wait={request.wait_for_result}"
    )

    submit = await post_json("music_generation", submit_payload, timeout=60.0)
    err = base_resp_failed(submit)
    if err:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_music_submit_failed", "upstream_msg": err},
        )

    task_id = submit.get("task_id") or (submit.get("data") or {}).get("task_id")
    if not task_id:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_task_id", "upstream_body": submit},
        )

    if not request.wait_for_result:
        return MusicGenResponse(
            request_id=task_id,
            status="queued",
            audio_url=None,
            storage_object_id=None,
            model=request.model,
            duration=request.duration,
            mode=request.mode,
            message=(
                f"MiniMax queued music task {task_id}. Poll "
                f"/ai/music/status/{task_id} or query MiniMax directly."
            ),
        )

    # Sync poll — up to ~10 min (60 * 10s)
    file_id: Optional[str] = None
    for attempt in range(60):
        await asyncio.sleep(10.0)
        status_body = await get_json(
            "query/music_generation",
            params={"task_id": task_id},
            timeout=30.0,
        )
        status = status_body.get("status") or (status_body.get("data") or {}).get("status")
        if status == "Success":
            file_id = status_body.get("file_id") or (
                status_body.get("data") or {}
            ).get("file_id")
            break
        if status == "Fail":
            raise HTTPException(
                status_code=502,
                detail={"error": "minimax_music_failed", "upstream_body": status_body},
            )
        logger.info(f"MiniMax music {task_id} status={status} (attempt {attempt+1}/60)")

    if not file_id:
        raise HTTPException(
            status_code=504,
            detail={"error": "minimax_music_timeout", "task_id": task_id, "after_s": 600},
        )

    file_body = await get_json("files/retrieve", params={"file_id": file_id}, timeout=30.0)
    download_url = (file_body.get("file") or {}).get("download_url") or file_body.get(
        "download_url"
    )
    if not download_url:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_download_url", "upstream_body": file_body},
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(download_url)
        r.raise_for_status()
        audio_bytes = r.content
        content_type = r.headers.get("content-type", "audio/mpeg")
    ext = "mp3" if "mpeg" in content_type or "mp3" in content_type else "wav"
    filename = f"music_minimax_{task_id[:12]}.{ext}"

    saved_obj = await save_file_and_record(
        data=audio_bytes,
        original_filename=filename,
        context="music-generation",
        is_public=True,
        collection_id=request.collection_id or "ai-generated-music",
        link_id=request.link_id,
    )

    minimax_cost_tracker.track_music(model=request.model, num_tracks=1)
    logger.info(f"Saved MiniMax music to storage: ID={saved_obj.id}")

    return MusicGenResponse(
        request_id=task_id,
        status="completed",
        audio_url=saved_obj.file_url,
        storage_object_id=saved_obj.id,
        model=request.model,
        duration=request.duration,
        mode=request.mode,
        message="Music generation completed",
    )


@router.get("/music/status/{request_id}", response_model=MusicGenResponse)
async def get_music_status(
    request_id: str,
    save_on_complete: bool = True,
    collection_id: str = "ai-generated-music",
    link_id: Optional[str] = None,
    api_key: str = Depends(get_api_key),
):
    """Check status of an async ``wait_for_result=false`` music submission.

    If ``save_on_complete=true`` (default) and the task has completed,
    we download + persist + cost-track in this call so the caller doesn't
    have to handle the download themselves.
    """
    import httpx
    from ai.clients.minimax_client import get_json, base_resp_failed
    from ai.clients.storage_client import save_file_and_record
    from ai.services.minimax_cost_tracker import minimax_cost_tracker

    status_body = await get_json(
        "query/music_generation", params={"task_id": request_id}, timeout=30.0
    )
    err = base_resp_failed(status_body)
    if err:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_status_failed", "upstream_msg": err},
        )

    status = status_body.get("status") or (status_body.get("data") or {}).get("status")
    file_id = status_body.get("file_id") or (status_body.get("data") or {}).get("file_id")

    if status != "Success" or not file_id or not save_on_complete:
        return MusicGenResponse(
            request_id=request_id,
            status=str(status or "unknown"),
            audio_url=None,
            storage_object_id=None,
            model="minimax-music-2.6",
            duration=0,
            mode="unknown",
            message=f"Status: {status}",
        )

    file_body = await get_json("files/retrieve", params={"file_id": file_id}, timeout=30.0)
    download_url = (file_body.get("file") or {}).get("download_url") or file_body.get(
        "download_url"
    )
    if not download_url:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_download_url"},
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(download_url)
        r.raise_for_status()
        audio_bytes = r.content
        content_type = r.headers.get("content-type", "audio/mpeg")
    ext = "mp3" if "mpeg" in content_type or "mp3" in content_type else "wav"
    filename = f"music_minimax_{request_id[:12]}.{ext}"

    saved_obj = await save_file_and_record(
        data=audio_bytes,
        original_filename=filename,
        context="music-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id,
    )

    minimax_cost_tracker.track_music(model="minimax-music-2.6", num_tracks=1)

    return MusicGenResponse(
        request_id=request_id,
        status="completed",
        audio_url=saved_obj.file_url,
        storage_object_id=saved_obj.id,
        model="minimax-music-2.6",
        duration=0,
        mode="unknown",
        message="Music generation completed (retrieved via status endpoint)",
    )
