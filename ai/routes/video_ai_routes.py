"""
Video AI Routes
===============

Endpoints for video generation using Higgsfield API:
- Image-to-Video transformation
- Multiple model support (DoP, Kling, Seedance)
- Async job management with polling
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging
import asyncio

from ai.clients.higgsfield_client import (
    HiggsFieldClient,
    HiggsFieldResponse,
    HiggsFieldStatus,
    HiggsFieldVideoModel,
    get_client
)
from ai.clients.storage_client import save_file_and_record

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class VideoGenRequest(BaseModel):
    """Request model for video generation"""
    image_url: str = Field(..., description="URL of source image to animate")
    prompt: str = Field(..., description="Motion/animation description")
    duration: int = Field(default=5, ge=1, le=10, description="Video duration in seconds (1-10)")
    model: str = Field(
        default="higgsfield-ai/dop/standard",
        description="Model: higgsfield-ai/dop/standard, kling-video/v2.1/pro/image-to-video, bytedance/seedance/v1/pro/image-to-video, minimax-hailuo-pro, minimax-hailuo-fast"
    )
    wait_for_result: bool = Field(
        default=True,
        description="If true, wait for result (sync). If false, return request_id immediately (async)."
    )
    collection_id: Optional[str] = Field(default="ai-generated-videos", description="Storage collection")
    link_id: Optional[str] = Field(default=None, description="Link ID for related objects")
    # MiniMax pay-as-you-go gate (Hailuo). Subscription / free-tier
    # providers (Higgsfield, Kling, Seedance) ignore this flag.
    confirm_api_billing: Optional[bool] = Field(default=False, description="Required true for MiniMax Hailuo models (pay-as-you-go)")


class VideoGenResponse(BaseModel):
    """Response model for video generation"""
    request_id: str
    status: str
    video_url: Optional[str] = None
    storage_object_id: Optional[int] = None
    model: str
    duration: int
    message: Optional[str] = None


class VideoStatusResponse(BaseModel):
    """Response for status check endpoint"""
    request_id: str
    status: str
    video_url: Optional[str] = None
    storage_object_id: Optional[int] = None
    error: Optional[str] = None
    progress: Optional[float] = None


def get_api_key():
    """Placeholder for API key validation"""
    return "placeholder"


# ── MiniMax Hailuo video dispatch ─────────────────────────────────────

MINIMAX_VIDEO_MODEL_MAPPING = {
    "minimax-hailuo-pro":  "MiniMax-Hailuo-02-Pro",
    "minimax-hailuo-fast": "MiniMax-Hailuo-02-Fast",
    "minimax-hailuo":      "MiniMax-Hailuo-02-Pro",  # alias to pro
    "hailuo-pro":          "MiniMax-Hailuo-02-Pro",
    "hailuo-fast":         "MiniMax-Hailuo-02-Fast",
}


def is_minimax_video_model(model: str) -> bool:
    return model.startswith("minimax-hailuo") or model.startswith("hailuo")


async def generate_with_minimax_video(
    image_url: str,
    prompt: str,
    duration: int,
    model: str,
    collection_id: str,
    link_id: Optional[str],
    wait_for_result: bool,
) -> dict:
    """MiniMax Hailuo image-to-video (sync polling).

    Async-mode for MiniMax is not yet exposed via ``/genvideo/status/``
    (would need provider-aware routing in get_video_status). For now,
    ``wait_for_result=false`` is honoured by returning the queued task_id
    and letting the caller poll MiniMax directly — most callers should
    just use sync with the 8-minute polling window.
    """
    import asyncio
    import httpx
    from ai.clients.minimax_client import post_json, get_json, base_resp_failed
    from ai.services.minimax_cost_tracker import minimax_cost_tracker
    from ai.routes.text_ai_routes import _check_minimax_billing_gate

    actual_model = MINIMAX_VIDEO_MODEL_MAPPING.get(model, model)
    logger.info(
        f"MiniMax video gen: model={model} -> {actual_model}, "
        f"duration={duration}s, wait={wait_for_result}"
    )

    submit = await post_json(
        "video_generation",
        {
            "model": actual_model,
            "prompt": prompt,
            "first_frame_image": image_url,
            "duration": duration,
        },
        timeout=60.0,
    )
    err = base_resp_failed(submit)
    if err:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_video_submit_failed", "upstream_msg": err},
        )

    task_id = submit.get("task_id") or (submit.get("data") or {}).get("task_id")
    if not task_id:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_task_id", "upstream_body": submit},
        )

    if not wait_for_result:
        return {
            "request_id": task_id,
            "status": "queued",
            "video_url": None,
            "storage_object_id": None,
            "message": (
                f"MiniMax queued task {task_id}. Poll status at "
                f"https://api.minimax.io/v1/query/video_generation?task_id={task_id} "
                f"directly (provider-aware /ai/genvideo/status routing is a follow-up)."
            ),
        }

    # Sync polling — up to 8 minutes
    file_id = None
    for attempt in range(48):  # 48 * 10s = 480s = 8min
        await asyncio.sleep(10.0)
        status_body = await get_json(
            "query/video_generation", params={"task_id": task_id}, timeout=30.0
        )
        # ``status`` and ``file_id`` live at the top level of the response
        status = status_body.get("status") or (status_body.get("data") or {}).get("status")
        if status == "Success":
            file_id = status_body.get("file_id") or (status_body.get("data") or {}).get("file_id")
            break
        if status == "Fail":
            raise HTTPException(
                status_code=502,
                detail={"error": "minimax_video_failed", "upstream_body": status_body},
            )
        # else: Queueing / Processing — keep polling
        logger.info(f"MiniMax video {task_id} status={status} (attempt {attempt+1}/48)")

    if not file_id:
        raise HTTPException(
            status_code=504,
            detail={"error": "minimax_video_timeout", "task_id": task_id, "after_s": 480},
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
        video_bytes = r.content

    filename = f"video_minimax_{task_id[:12]}.mp4"
    saved_obj = await save_file_and_record(
        data=video_bytes,
        original_filename=filename,
        context="video-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id,
    )

    minimax_cost_tracker.track_video(model=f"minimax-hailuo-{'fast' if 'Fast' in actual_model else 'pro'}", seconds=duration)
    logger.info(f"Saved MiniMax video to storage: ID={saved_obj.id}")

    return {
        "request_id": task_id,
        "status": "completed",
        "video_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "message": "Video generation completed",
    }


async def download_and_save_video(
    video_url: str,
    request_id: str,
    collection_id: str = "ai-generated-videos",
    link_id: Optional[str] = None
) -> int:
    """
    Download video from Higgsfield and save to Storage API.

    Args:
        video_url: URL of generated video
        request_id: Higgsfield request ID (used in filename)
        collection_id: Storage collection
        link_id: Optional link ID

    Returns:
        Storage object ID
    """
    import httpx

    logger.info(f"Downloading video from Higgsfield: {video_url}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(video_url)
        response.raise_for_status()
        video_bytes = response.content

    # Generate filename
    filename = f"video_{request_id[:8]}.mp4"

    # Save to storage
    saved_obj = await save_file_and_record(
        data=video_bytes,
        original_filename=filename,
        context="video-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id
    )

    logger.info(f"Saved video to storage: ID={saved_obj.id}, URL={saved_obj.file_url}")
    return saved_obj.id, saved_obj.file_url


@router.post("/genvideo", response_model=VideoGenResponse)
async def generate_video_endpoint(
    request: VideoGenRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate video from image using Higgsfield AI.

    Image-to-Video animation with multiple model options:
    - **higgsfield-ai/dop/standard** (default): High-quality animation
    - **higgsfield-ai/dop/preview**: Fast preview quality
    - **kling-video/v2.1/pro/image-to-video**: Cinematic animations
    - **bytedance/seedance/v1/pro/image-to-video**: Professional-grade

    Set `wait_for_result=false` for async mode (returns immediately with request_id).
    Use `/ai/genvideo/status/{request_id}` to check progress.

    Example:
    ```json
    {
        "image_url": "https://example.com/photo.jpg",
        "prompt": "camera slowly zooms in, gentle wind moves the hair",
        "duration": 5,
        "model": "higgsfield-ai/dop/standard"
    }
    ```
    """
    try:
        # MiniMax dispatch — Hailuo I2V, pay-as-you-go gated.
        if is_minimax_video_model(request.model):
            from ai.routes.text_ai_routes import _check_minimax_billing_gate
            _check_minimax_billing_gate(
                request.confirm_api_billing, endpoint="minimax-video"
            )
            result = await generate_with_minimax_video(
                image_url=request.image_url,
                prompt=request.prompt,
                duration=request.duration,
                model=request.model,
                collection_id=request.collection_id or "ai-generated-videos",
                link_id=request.link_id,
                wait_for_result=request.wait_for_result,
            )
            return VideoGenResponse(
                request_id=result["request_id"],
                status=result["status"],
                video_url=result.get("video_url"),
                storage_object_id=result.get("storage_object_id"),
                model=request.model,
                duration=request.duration,
                message=result.get("message"),
            )

        client = get_client()

        logger.info(
            f"Video generation request: model={request.model}, "
            f"duration={request.duration}s, wait={request.wait_for_result}"
        )

        if request.wait_for_result:
            # Synchronous: wait for result
            result = await client.generate_video(
                image_url=request.image_url,
                prompt=request.prompt,
                duration=request.duration,
                model=request.model,
                poll_interval=5.0,
                max_wait=300.0  # 5 minutes max
            )

            if result.status != HiggsFieldStatus.COMPLETED:
                raise HTTPException(
                    status_code=500,
                    detail=f"Video generation failed: {result.error or result.status}"
                )

            # Download and save to storage
            storage_id, storage_url = await download_and_save_video(
                video_url=result.result_url,
                request_id=result.request_id,
                collection_id=request.collection_id,
                link_id=request.link_id
            )

            return VideoGenResponse(
                request_id=result.request_id,
                status="completed",
                video_url=storage_url,
                storage_object_id=storage_id,
                model=request.model,
                duration=request.duration,
                message="Video generation completed"
            )

        else:
            # Asynchronous: return request_id immediately
            submit_result = await client.submit_video_generation(
                image_url=request.image_url,
                prompt=request.prompt,
                duration=request.duration,
                model=request.model
            )

            return VideoGenResponse(
                request_id=submit_result.request_id,
                status="queued",
                video_url=None,
                storage_object_id=None,
                model=request.model,
                duration=request.duration,
                message="Video generation queued. Use /ai/genvideo/status/{request_id} to check progress."
            )

    except TimeoutError as e:
        logger.error(f"Video generation timeout: {e}")
        raise HTTPException(status_code=504, detail="Video generation timed out")
    except RuntimeError as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video generation error: {str(e)}")


@router.get("/genvideo/status/{request_id}", response_model=VideoStatusResponse)
async def get_video_status(
    request_id: str,
    save_on_complete: bool = True,
    collection_id: str = "ai-generated-videos",
    link_id: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    Check status of video generation request.

    Args:
        request_id: Higgsfield request ID
        save_on_complete: Auto-save to storage when completed (default: true)
        collection_id: Storage collection for auto-save
        link_id: Link ID for auto-save

    Returns:
        Current status with video_url if completed
    """
    try:
        client = get_client()
        result = await client.get_status(request_id)

        response = VideoStatusResponse(
            request_id=request_id,
            status=result.status.value,
            error=result.error
        )

        # If completed and save requested, download and save
        if result.status == HiggsFieldStatus.COMPLETED and result.result_url:
            if save_on_complete:
                storage_id, storage_url = await download_and_save_video(
                    video_url=result.result_url,
                    request_id=request_id,
                    collection_id=collection_id,
                    link_id=link_id
                )
                response.video_url = storage_url
                response.storage_object_id = storage_id
            else:
                response.video_url = result.result_url

        return response

    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genvideo/cancel/{request_id}")
async def cancel_video_generation(
    request_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Cancel a pending video generation request.

    Only works if the request is still queued (not yet processing).

    Returns:
        Success status
    """
    try:
        client = get_client()
        cancelled = await client.cancel_request(request_id)

        if cancelled:
            return {"status": "cancelled", "request_id": request_id}
        else:
            return {
                "status": "cannot_cancel",
                "request_id": request_id,
                "message": "Request is already processing and cannot be cancelled"
            }

    except Exception as e:
        logger.error(f"Cancel error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/genvideo/models")
async def list_video_models():
    """List available video generation models"""
    return {
        "models": [
            {
                "id": "higgsfield-ai/dop/standard",
                "name": "DoP Standard",
                "provider": "Higgsfield",
                "description": "High-quality image animation",
                "max_duration": 10
            },
            {
                "id": "higgsfield-ai/dop/preview",
                "name": "DoP Preview",
                "provider": "Higgsfield",
                "description": "Fast preview quality",
                "max_duration": 5
            },
            {
                "id": "kling-video/v2.1/pro/image-to-video",
                "name": "Kling 2.1 Pro",
                "provider": "Kling Video",
                "description": "Advanced cinematic animations",
                "max_duration": 10
            },
            {
                "id": "bytedance/seedance/v1/pro/image-to-video",
                "name": "Seedance Pro",
                "provider": "ByteDance",
                "description": "Professional-grade video generation",
                "max_duration": 10
            }
        ]
    }
