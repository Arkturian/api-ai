"""
Kling Routes — Kling AI: image → video (and text → video).

POST /ai/genvideo/kling            — submit an image-to-video job;
                                     optional sync wait + save to storage
GET  /ai/genvideo/kling/status/{id} — poll a job, returns the video URL when done

Upstream: `https://api.klingai.com/v1/videos/image2video`.

Auth note (verified empirically 2026-08, NOT what the public docs say):
Kling's documentation describes an AccessKey/SecretKey pair from which the
caller signs a short-lived HS256 JWT per request. The key issued for this
account is instead accepted DIRECTLY as `Authorization: Bearer <key>` —
`GET /account/costs` returns code 0/SUCCEED with it. So no JWT machinery
here. If Kling ever tightens this, the failure will be a 401 from their
side and this module is where the JWT builder would go.

Billing: Kling is prepaid (resource packs / units) → default-deny
`confirm_api_billing` gate, same house rule as the other paid endpoints.
Result URLs from Kling expire, so results are persisted to storage-api.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio
import logging
import os
import time
import uuid

from ai.clients.storage_client import storage_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

KLING_BASE = os.getenv("KLING_API_BASE", "https://api.klingai.com")


def get_api_key():
    return "placeholder"  # not verified anywhere; see #514 / auth decision


def _kling_key() -> str:
    key = os.getenv("KLING_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "kling_not_configured",
                "hint": "KLING_API_KEY is not set in this host's environment.",
            },
        )
    return key


async def _kling_call(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One Kling API call. Unwraps the {code, message, data} envelope."""
    import httpx
    headers = {
        "Authorization": f"Bearer {_kling_key()}",
        "Content-Type": "application/json",
    }
    url = f"{KLING_BASE}{path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        if method == "POST":
            r = await client.post(url, json=payload or {}, headers=headers)
        else:
            r = await client.get(url, headers=headers)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"Kling returned non-JSON (http {r.status_code}): {r.text[:200]}",
        )
    # Kling wraps everything: code 0 = ok, anything else is an error even on HTTP 200.
    if data.get("code") not in (0, None):
        logger.error(f"Kling {path} error: code={data.get('code')} msg={data.get('message')}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": "kling_error",
                "kling_code": data.get("code"),
                "kling_message": data.get("message"),
                "request_id": data.get("request_id"),
            },
        )
    return data.get("data") or {}


class KlingVideoRequest(BaseModel):
    image: Optional[str] = Field(
        default=None,
        description=(
            "Source image: public https URL, a storage-api media URL, or raw "
            "base64 (no data: prefix). Required for image-to-video."
        ),
    )
    prompt: Optional[str] = Field(default=None, description="Motion/scene description (max ~2500 chars)")
    negative_prompt: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Kling model. Omit to let the server pick: kling-v3 when an "
            "image_tail (end frame) is given, otherwise the account default. "
            "Only kling-v3 supports start+end frames — verified against the "
            "live API: kling-v2-master rejects image_tail with 'Image tail is "
            "not supported by the current model'."
        ),
    )
    mode: Optional[str] = Field(default=None, description="std (cheaper) or pro (higher quality)")
    duration: Optional[str] = Field(default=None, description="'5' or '10' seconds")
    cfg_scale: Optional[float] = Field(default=None, description="0.0-1.0, how strictly to follow the prompt")
    image_tail: Optional[str] = Field(default=None, description="Optional end-frame image (same formats as image)")
    wait_for_result: bool = Field(default=True, description="true = poll until done (video gen takes minutes)")
    timeout_s: int = Field(default=600, description="Max seconds to wait when wait_for_result=true")
    save_to_storage: bool = Field(default=True, description="Persist the video to storage-api (Kling URLs expire)")
    collection_id: Optional[str] = Field(default="ai-generated-video")
    link_id: Optional[str] = Field(default=None)
    confirm_api_billing: Optional[bool] = Field(
        default=False,
        description="Required true — Kling is prepaid; each job consumes account units.",
    )


async def _save_video_to_storage(url: str, collection_id: Optional[str],
                                 link_id: Optional[str]) -> Dict[str, Any]:
    """Stream the finished video into storage-api (size-capped)."""
    import httpx
    storage_url = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
    max_bytes = int(os.getenv("KLING_MAX_VIDEO_MB", "500")) * 1024 * 1024
    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                clen = r.headers.get("content-length")
                if clen and int(clen) > max_bytes:
                    return {"storage_object_id": None, "kling_url": url,
                            "save_error": f"video exceeds {max_bytes}-byte cap"}
                buf, total = bytearray(), 0
                async for chunk in r.aiter_bytes(65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"stream exceeded {max_bytes}-byte cap")
                    buf.extend(chunk)
            data = {"is_public": "true", "analyze": "false", "ai_mode": "none",
                    "reuse_existing": "false"}
            if collection_id:
                data["collection_id"] = collection_id
            if link_id:
                data["link_id"] = link_id
            up = await client.post(
                f"{storage_url}/storage/upload",
                files={"file": (f"kling_{uuid.uuid4().hex[:8]}.mp4", bytes(buf), "video/mp4")},
                data=data,
                headers={"X-API-KEY": storage_api_key()},
            )
            up.raise_for_status()
            sid = up.json().get("id")
            logger.info(f"kling: saved video ({total} bytes) -> media/{sid}")
            return {
                "storage_object_id": sid,
                "storage_url": f"{storage_url}/storage/media/{sid}" if sid else None,
                "kling_url": url,
            }
    except Exception as e:
        logger.warning(f"kling: persist failed for {url[:80]}: {e}")
        return {"storage_object_id": None, "kling_url": url, "save_error": str(e)}


def _extract_video(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    videos = ((task.get("task_result") or {}).get("videos")) or []
    return videos[0] if videos else None


@router.post("/genvideo/kling")
async def kling_image_to_video(request: KlingVideoRequest, api_key: str = Depends(get_api_key)):
    """
    Generate a video from an image (or a prompt) via Kling AI.

    Prepaid — requires `confirm_api_billing: true`. Video generation takes
    minutes; sync by default, set wait_for_result=false for a task_id to poll
    via GET /ai/genvideo/kling/status/{task_id}. The finished video is saved
    to storage by default because Kling's own URLs expire.
    """
    from .text_ai_routes import _check_api_billing_gate
    _check_api_billing_gate(request.confirm_api_billing, endpoint="kling-video")

    if not request.image and not request.prompt:
        raise HTTPException(
            status_code=422,
            detail="Provide at least `image` (image-to-video) or `prompt`.",
        )

    # Start+end-frame loops (the Realtime-avatar use case: same image in both
    # roles so the clip starts and ends in an identical pose and can be hard-cut
    # without a black flash) only work on kling-v3. Other models reject
    # image_tail outright. Pick it automatically instead of letting the caller
    # hit "Image tail is not supported by the current model".
    if request.image_tail and not request.model_name:
        request.model_name = os.getenv("KLING_TAIL_MODEL", "kling-v3")
        logger.info("kling: image_tail given -> model_name=%s", request.model_name)

    payload: Dict[str, Any] = {}
    for field, key in (
        ("image", "image"), ("image_tail", "image_tail"), ("prompt", "prompt"),
        ("negative_prompt", "negative_prompt"), ("model_name", "model_name"),
        ("mode", "mode"), ("duration", "duration"), ("cfg_scale", "cfg_scale"),
    ):
        val = getattr(request, field)
        if val is not None:
            payload[key] = val

    data = await _kling_call("POST", "/v1/videos/image2video", payload)
    task_id = data.get("task_id")
    if not task_id:
        raise HTTPException(status_code=502, detail=f"Kling returned no task_id: {data}")
    logger.info(f"kling: submitted task {task_id} (model={request.model_name or 'default'})")

    if not request.wait_for_result:
        return {"task_id": task_id, "status": data.get("task_status") or "submitted"}

    deadline = time.monotonic() + max(60, request.timeout_s)
    task: Dict[str, Any] = {}
    status = "submitted"
    while time.monotonic() < deadline:
        await asyncio.sleep(10)
        task = await _kling_call("GET", f"/v1/videos/image2video/{task_id}")
        status = task.get("task_status") or ""
        if status in ("succeed", "failed"):
            break
    if status == "failed":
        raise HTTPException(
            status_code=502,
            detail={"error": "kling_task_failed", "task_id": task_id,
                    "message": task.get("task_status_msg")},
        )
    if status != "succeed":
        return {"task_id": task_id, "status": status or "processing",
                "hint": f"still running after {request.timeout_s}s — poll GET /ai/genvideo/kling/status/{task_id}"}

    video = _extract_video(task)
    result: Dict[str, Any] = {"task_id": task_id, "status": "succeed", "video": video}
    if request.save_to_storage and video and video.get("url"):
        result["saved"] = await _save_video_to_storage(
            video["url"], request.collection_id, request.link_id
        )
    return result


@router.get("/genvideo/kling/status/{task_id}")
async def kling_status(task_id: str, save_to_storage: bool = False,
                       collection_id: Optional[str] = "ai-generated-video",
                       link_id: Optional[str] = None,
                       api_key: str = Depends(get_api_key)):
    """Poll a Kling job. Optionally persist the video on success."""
    task = await _kling_call("GET", f"/v1/videos/image2video/{task_id}")
    status = task.get("task_status") or "unknown"
    video = _extract_video(task)
    out: Dict[str, Any] = {"task_id": task_id, "status": status, "video": video}
    if status == "failed":
        out["message"] = task.get("task_status_msg")
    if status == "succeed" and save_to_storage and video and video.get("url"):
        out["saved"] = await _save_video_to_storage(video["url"], collection_id, link_id)
    return out


@router.get("/genvideo/kling/account")
async def kling_account(api_key: str = Depends(get_api_key)):
    """Remaining Kling units per resource pack — check before burning budget."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 30 * 24 * 3600 * 1000
    data = await _kling_call("GET", f"/account/costs?start_time={start_ms}&end_time={now_ms}")
    packs = [
        {
            "name": p.get("resource_pack_name"),
            "total": p.get("total_quantity"),
            "remaining": p.get("remaining_quantity"),
            "status": p.get("status"),
            "expires_at": p.get("invalid_time"),
        }
        for p in (data.get("resource_pack_subscribe_infos") or [])
    ]
    return {"packs": packs, "remaining_total": sum((p["remaining"] or 0) for p in packs)}
