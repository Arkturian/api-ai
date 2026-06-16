"""
Image AI Routes
===============

Endpoints for image generation and processing:
- Image generation (Higgsfield, Gemini, DALL-E)
- Image upscaling
- Depth map generation
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    aspect_ratio: Optional[str] = Field(default="1:1", description="Aspect ratio (1:1, 16:9, 9:16, 1:4, 4:1, 1:8, 8:1)")
    image_size: Optional[str] = Field(default=None, description="Resolution: 1K, 2K, 4K (Nano Banana 2 only)")
    model: Optional[str] = Field(
        default="nano-banana-2",
        description="Model: nano-banana-2 (default), nano-banana-pro, imagen-4, higgsfield, higgsfield-reve, minimax-image-01"
    )
    collection_id: Optional[str] = Field(default="ai-generated-images", description="Storage collection")
    link_id: Optional[str] = Field(default=None, description="Link ID for related objects")
    # MiniMax API-billed (pay-as-you-go). The MiniMax dispatch branch
    # enforces this via `_check_minimax_billing_gate` — default deny.
    # Subscription / free-tier providers (Higgsfield, Gemini-CLI) ignore
    # this flag.
    confirm_api_billing: Optional[bool] = Field(default=False, description="Required true for MiniMax models (pay-as-you-go)")
    # OpenAI gpt-image-* quality knob. Default null → server picks "auto"
    # (which itself auto-scales by resolution). Set to "low" / "medium" /
    # "high" if you explicitly want OpenAI's quality tier (high consistently
    # >180s for 2048², 4K may exceed the 300s upstream timeout).
    quality: Optional[str] = Field(default=None, description="OpenAI gpt-image-* quality: low/medium/high/auto. Null = server picks 'auto'.")
    # Reference images for style-transfer / image-to-image. Currently only
    # supported by the OpenAI gpt-image-* family (routed via /v1/images/edits
    # multipart). HTTP(S) URLs allowed; storage-api URLs are fetched with
    # X-API-KEY so private/quarantined assets work too. Max 16 references
    # (OpenAI hard limit). dall-e-3 does NOT support edits — rejected at
    # dispatch. Non-OpenAI models reject too with a 400 pointing at gpt-image-2.
    reference_image_urls: Optional[List[str]] = Field(
        default=None,
        description=(
            "Reference images as visual style/composition anchors. Only "
            "supported with gpt-image-1 / gpt-image-1.5 / gpt-image-2 — "
            "switches dispatch to OpenAI /v1/images/edits. Max 16 URLs."
        ),
    )


class ImageResponse(BaseModel):
    image_url: str
    storage_object_id: Optional[int] = None
    model: str
    width: int
    height: int


class UpscaleRequest(BaseModel):
    image_url: str
    scale_factor: Optional[int] = 4
    model: Optional[str] = "real-esrgan"


class DepthRequest(BaseModel):
    image_url: str
    model: Optional[str] = "depth-anything"


def get_api_key():
    return "placeholder"


# Model mapping: user-friendly names to provider-specific IDs
MODEL_MAPPING = {
    # Higgsfield Models
    "higgsfield": "higgsfield-ai/soul/standard",
    "higgsfield-soul": "higgsfield-ai/soul/standard",
    "higgsfield-reve": "reve/text-to-image",
    "reve": "reve/text-to-image",

    # Gemini Image Models
    "nano-banana-2": "gemini-3.1-flash-image-preview",
    "nano-banana": "gemini-3.1-flash-image-preview",  # alias -> Nano Banana 2
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "gemini": "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",

    # Imagen 4
    "imagen-4": "imagen-4.0-generate-001",
    "imagen-4.0-generate-001": "imagen-4.0-generate-001",

    # MiniMax Image-01 (pay-as-you-go, gated via confirm_api_billing)
    "minimax-image-01": "image-01",
    "minimax-image": "image-01",
    "image-01": "image-01",

    # OpenAI GPT Image family (pay-as-you-go, gated via confirm_api_billing)
    "gpt-image-2": "gpt-image-2",
    "gpt-image-1": "gpt-image-1",
    "gpt-image-1.5": "gpt-image-1.5",
    "gpt-image-1-mini": "gpt-image-1-mini",

    # Others
    "dall-e-3": "dall-e-3",
}


def is_higgsfield_model(model: str) -> bool:
    """Check if model should use Higgsfield provider"""
    higgsfield_prefixes = ["higgsfield", "reve"]
    return any(model.startswith(p) for p in higgsfield_prefixes)


def is_gemini_model(model: str) -> bool:
    """Check if model should use Gemini/Google provider"""
    gemini_prefixes = ["gemini", "imagen", "nano-banana"]
    return any(model.startswith(p) for p in gemini_prefixes)


def is_minimax_model(model: str) -> bool:
    """Check if model should use MiniMax provider."""
    minimax_prefixes = ["minimax-image", "image-01"]
    return any(model.startswith(p) for p in minimax_prefixes)


def is_openai_image_model(model: str) -> bool:
    """Check if model should use OpenAI Images API provider."""
    return model.startswith("gpt-image-") or model == "dall-e-3"


async def _download_reference_images(urls: List[str]) -> List[tuple]:
    """Fetch reference image bytes for OpenAI /v1/images/edits multipart upload.

    Returns [(filename, bytes, mime_type), ...] suitable for httpx files-param.
    Storage-api URLs are fetched with X-API-KEY so quarantined/private assets
    work; other URLs are fetched anonymously. Fail fast — if any URL 404s we
    raise 422 with the bad URL, so the caller sees which reference was wrong
    instead of getting a confidently wrong generated image.
    """
    import httpx
    import os
    import uuid

    out = []
    storage_key = os.getenv("STORAGE_API_KEY", "Inetpass1")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for i, url in enumerate(urls):
            headers = {}
            if "/storage/media/" in url:
                headers["X-API-KEY"] = storage_key
            try:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "reference_image_fetch_failed",
                        "url": url,
                        "exc": str(e)[:200],
                    },
                )
            ct = r.headers.get("content-type", "image/png")
            # OpenAI Images API accepts png / jpg / webp on edits.
            mime = "image/png" if "png" in ct else (
                "image/webp" if "webp" in ct else "image/jpeg"
            )
            ext = "png" if mime == "image/png" else (
                "webp" if mime == "image/webp" else "jpg"
            )
            filename = f"ref_{i}_{uuid.uuid4().hex[:6]}.{ext}"
            out.append((filename, r.content, mime))
            logger.info(
                f"OpenAI edits ref[{i}]: {url} -> {len(r.content)} bytes ({mime})"
            )
    return out


async def generate_with_openai_image(
    prompt: str,
    model: str,
    collection_id: str,
    link_id: Optional[str],
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    quality: Optional[str] = None,
    reference_image_urls: Optional[List[str]] = None,
) -> dict:
    """Generate image via OpenAI Images API (gpt-image-2 / gpt-image-1 / dall-e-3).

    Pay-as-you-go: the route handler enforces ``_check_openai_billing_gate``
    before this is reached. Sizes accepted by the OpenAI Images API:
    ``1024x1024``, ``1536x1024``, ``1024x1536``, ``2048x2048``,
    ``2048x1152``, ``3840x2160``, ``2160x3840``, plus ``auto``.

    We map our (width, height) form to the closest supported size — falling
    back to ``2048x2048`` for the default ``1024×1024``-default-but-Story-wants-2k
    use case, or to ``auto`` when the caller passes nothing.

    When ``reference_image_urls`` is set the call switches from
    ``/v1/images/generations`` (JSON, text-only) to ``/v1/images/edits``
    (multipart, with up to 16 image[] file uploads as visual anchors).
    dall-e-3 is rejected on the edits path — only gpt-image-* supports it.
    """
    import base64
    import httpx
    import os
    from ai.clients.storage_client import save_file_and_record

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "openai_api_key_missing",
                "hint": "OPENAI_API_KEY not configured in service env.",
            },
        )

    use_edits = bool(reference_image_urls)
    if use_edits and model == "dall-e-3":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "reference_image_not_supported",
                "model": model,
                "hint": (
                    "dall-e-3 does not support /v1/images/edits. Switch to "
                    "gpt-image-2 (or gpt-image-1 / gpt-image-1.5) to use "
                    "reference_image_urls as visual style anchors."
                ),
            },
        )
    if use_edits and len(reference_image_urls) > 16:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "too_many_reference_images",
                "got": len(reference_image_urls),
                "max": 16,
                "hint": "OpenAI /v1/images/edits accepts at most 16 image[] uploads.",
            },
        )

    # Map (width, height) → an OpenAI-supported size. We accept the
    # caller's exact dims if they already match, otherwise round to the
    # nearest supported preset.
    supported_sizes = {
        (1024, 1024), (1536, 1024), (1024, 1536),
        (2048, 2048), (2048, 1152), (3840, 2160), (2160, 3840),
    }
    if width and height and (width, height) in supported_sizes:
        size = f"{width}x{height}"
    elif aspect_ratio:
        ar_map = {
            "1:1": "2048x2048",
            "16:9": "2048x1152",
            "9:16": "1024x1536",
            "4:3": "1536x1024",
            "3:4": "1024x1536",
        }
        size = ar_map.get(aspect_ratio, "auto")
    else:
        size = "auto"

    # Quality knob: ``auto`` is the safe default (server picks band by
    # resolution — 1024² stays ~30s, 2048² runs ~200s). Callers can now
    # opt-in to ``low``/``medium``/``high`` explicitly; ``high`` at 2048²
    # consistently >180s and may approach the 300s upstream timeout for
    # 4K. Story IACP b3244014 + Knowledge IACP 20ec50db requested the
    # exposed knob so Cover-Plates / Botanik-Plates can pick the
    # quality/latency tradeoff per call.
    requested_quality = (quality or "").strip().lower() or "auto"
    if requested_quality not in {"auto", "low", "medium", "high"}:
        logger.warning(
            f"OpenAI image gen: unknown quality={requested_quality!r}, "
            f"falling back to 'auto'"
        )
        requested_quality = "auto"

    # 300s upstream timeout — gpt-image-2 at 2048² with quality=auto
    # measured 30-90s, but 4K + complex prompts push past 120s. The
    # storage-api caller already has its own retry-on-timeout layer.
    if use_edits:
        # /v1/images/edits — multipart upload, model+prompt+size+quality
        # as form fields, image[] for each reference. OpenAI accepts the
        # `image` field repeated (sent as `image` + `image` + ...) or as
        # `image[]`; httpx encodes a list of ("image", (...)) tuples as
        # the repeated form which OpenAI parses correctly.
        ref_files = await _download_reference_images(reference_image_urls)
        form: dict = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": requested_quality,
            "n": "1",
        }
        if model.startswith("gpt-image-"):
            form["output_format"] = "png"
            form["background"] = "auto"
        # OpenAI /v1/images/edits accepts the `image` field as either a
        # SINGLE upload (field name `image`) or an ARRAY (field name
        # `image[]` repeated). Sending `image` more than once gets you
        # a 400 invalid_request_error with code `duplicate_parameter`.
        # So: pick the field name based on count. Verified live by CHAP2
        # IACP 6aee6916 — single ref worked with `image`, 3 refs failed
        # until switched to `image[]`.
        field_name = "image" if len(ref_files) == 1 else "image[]"
        files = [(field_name, (fn, content, mime)) for fn, content, mime in ref_files]
        logger.info(
            f"OpenAI image edit: model={model}, size={size}, "
            f"quality={requested_quality}, refs={len(ref_files)}"
        )
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                r = await client.post(
                    "https://api.openai.com/v1/images/edits",
                    data=form,
                    files=files,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "openai_upstream_unreachable",
                        "exc": str(e)[:200],
                    },
                )
    else:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "output_format": "png",
            "quality": requested_quality,
        }
        # gpt-image-1.5 and gpt-image-2 accept "background", dall-e-3 does not.
        if model.startswith("gpt-image-"):
            payload["background"] = "auto"

        logger.info(
            f"OpenAI image gen: model={model}, size={size}, quality={payload['quality']}"
        )
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                r = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "openai_upstream_unreachable",
                        "exc": str(e)[:200],
                    },
                )

    if r.status_code >= 400:
        try:
            upstream = r.json()
        except Exception:
            upstream = {"raw": r.text[:500]}
        logger.error(f"OpenAI image gen {model} → {r.status_code}: {upstream}")
        raise HTTPException(
            status_code=502 if r.status_code >= 500 else r.status_code,
            detail={
                "error": "openai_upstream_error",
                "upstream_status": r.status_code,
                "upstream_body": upstream,
            },
        )

    body = r.json()
    data_arr = body.get("data") or []
    if not data_arr:
        raise HTTPException(
            status_code=502,
            detail={"error": "openai_no_image_returned", "upstream_body": body},
        )

    entry = data_arr[0]

    # OpenAI returns either ``url`` (older) or ``b64_json`` (default for
    # gpt-image-*). Prefer b64 to skip a separate download round-trip.
    if entry.get("b64_json"):
        image_bytes = base64.b64decode(entry["b64_json"])
        content_type = "image/png"
    elif entry.get("url"):
        async with httpx.AsyncClient(timeout=60.0) as client:
            dl = await client.get(entry["url"])
            dl.raise_for_status()
            image_bytes = dl.content
            content_type = dl.headers.get("content-type", "image/png")
    else:
        raise HTTPException(
            status_code=502,
            detail={"error": "openai_no_image_data", "upstream_body": body},
        )

    ext = "png" if "png" in content_type else "jpg"
    # ``created`` is a unix timestamp from the OpenAI response — use it as
    # a stable suffix so re-runs don't clash but we don't fabricate dates
    # ourselves.
    request_id = str(body.get("created") or "openai")
    filename = f"img_openai_{request_id}.{ext}"

    saved_obj = await save_file_and_record(
        data=image_bytes,
        original_filename=filename,
        context="image-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id,
    )

    # Track AFTER successful save so a half-failed call doesn't bill
    # against the cap. Federation-shared via openai_cost_tracker.
    from ai.services.openai_cost_tracker import openai_cost_tracker
    openai_cost_tracker.track_image(model=model, num_images=1)

    logger.info(f"Saved OpenAI image to storage: ID={saved_obj.id}")

    return {
        "id": saved_obj.id,
        "image_url": saved_obj.file_url,
        "file_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "request_id": request_id,
    }


def _check_openai_billing_gate(confirmed: Optional[bool], endpoint: str) -> None:
    """Default-deny gate for OpenAI Images API.

    Two independent block conditions, either of which trips the gate:
      1. ``confirm_api_billing`` body-flag missing → 403
      2. Federation-shared 50 EUR/month cap reached (master arkserver,
         client arkturian via ``/internal/openai-cost-shared-state``) →
         429, even for confirmed callers

    Story IACP b3244014 (2026-06-13) explicitly asked for the 50 EUR/month
    pattern analog to Gemini (15 EUR) and MiniMax (25 EUR). The three caps
    are independent so OpenAI burn doesn't accidentally block a MiniMax
    Image-01 generation and vice versa.
    """
    from ..services.openai_cost_tracker import openai_cost_tracker

    if not confirmed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "api_billing_confirmation_required",
                "endpoint": endpoint,
                "provider": "openai",
                "hint": (
                    "OpenAI Images API is pay-as-you-go billed against "
                    "OPENAI_API_KEY (separate from MiniMax + GCP caps). "
                    "Approx pricing: gpt-image-2 high-q 2048² ≈ $0.15-0.20, "
                    "gpt-image-1 medium ≈ $0.05. Send "
                    "`confirm_api_billing: true` to acknowledge."
                ),
            },
        )

    if openai_cost_tracker.should_block_request():
        status = openai_cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "monthly_api_cap_reached",
                "endpoint": endpoint,
                "provider": "openai",
                "spent_eur": round(status.get("total_cost_eur", 0.0), 2),
                "budget_eur": status.get("monthly_budget_eur"),
                "hint": ("OpenAI monthly cap reached. Cap resets at the "
                         "start of the next calendar month. Subscription "
                         "endpoints (/ai/claude, /ai/chatgpt) are unaffected, "
                         "MiniMax + Gemini have their own separate caps."),
            },
        )


async def generate_with_higgsfield(
    prompt: str,
    model: str,
    aspect_ratio: str,
    collection_id: str,
    link_id: Optional[str]
) -> dict:
    """Generate image using Higgsfield API"""
    import httpx
    import uuid
    from ai.clients.higgsfield_client import get_client, HiggsFieldStatus
    from ai.clients.storage_client import save_file_and_record

    client = get_client()

    # Higgsfield only accepts "720p" or "1080p"
    resolution = "1080p"

    logger.info(f"Higgsfield image gen: model={model}, aspect={aspect_ratio}")

    # Generate image (with polling)
    result = await client.generate_image(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        poll_interval=2.0,
        max_wait=120.0
    )

    if result.status != HiggsFieldStatus.COMPLETED:
        raise HTTPException(
            status_code=500,
            detail=f"Higgsfield image generation failed: {result.error or result.status}"
        )

    # Download image from Higgsfield
    logger.info(f"Downloading image from Higgsfield: {result.result_url}")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.get(result.result_url)
        response.raise_for_status()
        image_bytes = response.content

    # Determine file extension from content-type
    content_type = response.headers.get("content-type", "image/png")
    ext = "png" if "png" in content_type else "jpg"
    filename = f"img_{result.request_id[:8]}.{ext}"

    # Save to storage
    saved_obj = await save_file_and_record(
        data=image_bytes,
        original_filename=filename,
        context="image-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id
    )

    logger.info(f"Saved Higgsfield image to storage: ID={saved_obj.id}")

    return {
        "id": saved_obj.id,
        "image_url": saved_obj.file_url,
        "file_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "request_id": result.request_id
    }


async def generate_with_minimax(
    prompt: str,
    model: str,
    aspect_ratio: str,
    collection_id: str,
    link_id: Optional[str],
) -> dict:
    """Generate image using MiniMax Image-01 (pay-as-you-go REST API).

    Caller-side billing gate is enforced by the route handler before this
    function is reached. The MiniMax response is the canonical URL list
    on a successful ``base_resp.status_code == 0`` — we download the
    first image and save it via storage-api.

    Cost-tracking: one ``track_image`` call per request. Federation-
    shared via ``minimax_cost_tracker`` (master arkserver).
    """
    import httpx
    from ai.clients.minimax_client import post_json, base_resp_failed
    from ai.clients.storage_client import save_file_and_record
    from ai.services.minimax_cost_tracker import minimax_cost_tracker

    logger.info(f"MiniMax image gen: model={model}, aspect={aspect_ratio}")

    payload = {
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio or "1:1",
        "response_format": "url",
        "n": 1,
        "prompt_optimizer": True,
    }
    body = await post_json("image_generation", payload, timeout=120.0)
    err = base_resp_failed(body)
    if err:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_image_generation_failed", "upstream_msg": err},
        )

    # Response shape (MiniMax Image-01): ``data.image_urls: [<url>, ...]``
    data = body.get("data") or {}
    urls = data.get("image_urls") or []
    if not urls:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_image_returned", "upstream_body": body},
        )

    image_url = urls[0]
    logger.info(f"MiniMax returned image URL: {image_url}")

    # Download + save to storage. MiniMax-hosted CDN URLs are short-lived,
    # so we always persist locally before responding to the caller.
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        r = await http_client.get(image_url)
        r.raise_for_status()
        image_bytes = r.content
        content_type = r.headers.get("content-type", "image/png")

    ext = "png" if "png" in content_type else "jpg"
    request_id = body.get("id") or body.get("request_id") or "unknown"
    filename = f"img_minimax_{request_id[:12]}.{ext}"

    saved_obj = await save_file_and_record(
        data=image_bytes,
        original_filename=filename,
        context="image-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id,
    )

    # Track usage AFTER successful save so a half-failed call doesn't
    # bill against the cap.
    minimax_cost_tracker.track_image(model=f"minimax-{model}" if not model.startswith("minimax-") else model)

    logger.info(f"Saved MiniMax image to storage: ID={saved_obj.id}")

    return {
        "id": saved_obj.id,
        "image_url": saved_obj.file_url,
        "file_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "request_id": request_id,
    }


async def generate_with_gemini(
    prompt: str,
    model: str,
    collection_id: str,
    link_id: Optional[str],
    aspect_ratio: Optional[str] = None,
    image_size: Optional[str] = None
) -> dict:
    """Generate image using Google Gemini/Imagen API"""
    from google import genai
    from google.genai import types
    import os
    import uuid
    from ai.clients.storage_client import save_file_and_record
    from ai.services.cost_tracker import cost_tracker

    # Check budget before processing
    if cost_tracker.should_block_request():
        status = cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail=f"Monthly Gemini API budget exceeded: {status['total_cost_eur']:.2f}/{status['monthly_budget_eur']:.2f} EUR."
        )

    logger.info(f"Gemini image gen: model={model}, aspect_ratio={aspect_ratio}, image_size={image_size}")

    # Use Google GenAI SDK
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Build image config for Nano Banana 2 features
    image_config = None
    if aspect_ratio or image_size:
        config_kwargs = {}
        if aspect_ratio:
            config_kwargs["aspect_ratio"] = aspect_ratio
        if image_size:
            config_kwargs["image_size"] = image_size
        image_config = types.ImageConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=image_config
        )
    )

    # Extract image from response
    image_bytes = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            import base64
            raw_data = part.inline_data.data

            # Handle various encodings from Gemini API
            if isinstance(raw_data, str):
                # String: decode from base64
                image_bytes = base64.b64decode(raw_data)
                logger.info("Decoded base64 string from Gemini")
            elif isinstance(raw_data, bytes):
                # Check if bytes are base64-encoded (starts with common image headers in base64)
                # JPEG base64 starts with /9j/, PNG with iVBO
                if raw_data[:4] in (b'/9j/', b'iVBO', b'/9j/'):
                    image_bytes = base64.b64decode(raw_data)
                    logger.info("Decoded base64 bytes from Gemini")
                else:
                    image_bytes = raw_data
            else:
                image_bytes = raw_data
            break

    if not image_bytes:
        raise HTTPException(status_code=500, detail="No image found in Gemini response.")

    # Generate filename
    filename = f"img_{str(uuid.uuid4())[:8]}.png"

    # Save to storage
    saved_obj = await save_file_and_record(
        data=image_bytes,
        original_filename=filename,
        context="image-generation",
        is_public=True,
        collection_id=collection_id,
        link_id=link_id
    )

    logger.info(f"Saved Gemini image to storage: ID={saved_obj.id}")

    # The storage-api's upload response does not populate file_url with the
    # checksum-versioned URL; only a follow-up GET on the asset returns it.
    # Refetch so the response carries a usable URL clients can hit directly
    # instead of always having to round-trip via assets_get.
    from ai.clients.storage_client import get_storage_object
    file_url = saved_obj.file_url
    width = None
    height = None
    try:
        full_obj = await get_storage_object(saved_obj.id)
        if full_obj.file_url:
            file_url = full_obj.file_url
        # Real generated dimensions live on the storage record; the request
        # body's width/height are user hints and may not match (Gemini picks
        # output size from aspect_ratio + image_size).
        width = getattr(full_obj, "width", None)
        height = getattr(full_obj, "height", None)
    except Exception as e:
        logger.warning(f"Could not refetch storage object {saved_obj.id} for full URL/dims: {e}")

    # Fallback for width/height if storage didn't return them — read from
    # the in-memory bytes via PIL (zero extra IO).
    if not width or not height:
        try:
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(image_bytes)) as im:
                width, height = im.size
        except Exception as e:
            logger.warning(f"PIL fallback for dims failed: {e}")

    # Track cost
    cost_tracker.track_image_generation(model, num_images=1)

    return {
        "id": saved_obj.id,
        "image_url": file_url,
        "file_url": file_url,
        "storage_object_id": saved_obj.id,
        "width": width,
        "height": height,
    }


@router.post("/genimage")
async def generate_image_endpoint(
    request: ImageGenRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate an image from text prompt using various AI models.

    **Default: nano-banana-2 (Gemini 3.1 Flash Image)** - fast, free, up to 4K

    Supported models:
    - **nano-banana-2** (default): Gemini 3.1 Flash Image - fast, free, up to 4K
    - **nano-banana-pro**: Gemini 3 Pro Image - best Gemini quality
    - **imagen-4**: Google Imagen 4.0 - photorealistic
    - **higgsfield**: Higgsfield Soul - high quality (requires credits)
    - **higgsfield-reve**: Reve model via Higgsfield (requires credits)

    Example:
    ```json
    {
        "prompt": "A serene mountain landscape at sunset",
        "model": "nano-banana-2",
        "aspect_ratio": "16:9",
        "image_size": "2K"
    }
    ```
    """
    try:
        # Determine model
        model_name = request.model or "nano-banana"
        actual_model = MODEL_MAPPING.get(model_name, model_name)

        logger.info(f"Image gen request: model={model_name} -> {actual_model}")

        # Reject reference_image_urls early if the chosen model can't use
        # them. Only OpenAI gpt-image-* takes refs (via /v1/images/edits);
        # everything else is text-to-image only on this endpoint.
        if request.reference_image_urls:
            if not (
                is_openai_image_model(model_name)
                or is_openai_image_model(actual_model)
            ) or actual_model == "dall-e-3":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "reference_image_not_supported_for_model",
                        "model": model_name,
                        "hint": (
                            "reference_image_urls is only wired for "
                            "gpt-image-2 / gpt-image-1 / gpt-image-1.5 "
                            "(OpenAI /v1/images/edits). Pick one of those, "
                            "or drop reference_image_urls and use the model "
                            "with text-only generation."
                        ),
                    },
                )

        # Route to appropriate provider
        if is_higgsfield_model(actual_model):
            result = await generate_with_higgsfield(
                prompt=request.prompt,
                model=actual_model,
                aspect_ratio=request.aspect_ratio or "1:1",
                collection_id=request.collection_id or "ai-generated-images",
                link_id=request.link_id
            )

        elif is_gemini_model(model_name) or actual_model.startswith("gemini") or actual_model.startswith("imagen"):
            result = await generate_with_gemini(
                prompt=request.prompt,
                model=actual_model,
                collection_id=request.collection_id or "ai-generated-images",
                link_id=request.link_id,
                aspect_ratio=request.aspect_ratio,
                image_size=request.image_size
            )

        elif is_minimax_model(model_name) or is_minimax_model(actual_model):
            # MiniMax Image-01 is pay-as-you-go API — require explicit
            # billing opt-in, then run the federation-shared cap check.
            from ai.routes.text_ai_routes import _check_minimax_billing_gate
            _check_minimax_billing_gate(
                request.confirm_api_billing, endpoint="minimax-image"
            )
            result = await generate_with_minimax(
                prompt=request.prompt,
                model=actual_model,
                aspect_ratio=request.aspect_ratio or "1:1",
                collection_id=request.collection_id or "ai-generated-images",
                link_id=request.link_id,
            )

        elif is_openai_image_model(model_name) or is_openai_image_model(actual_model):
            # OpenAI Images API (gpt-image-2, gpt-image-1, gpt-image-1.5,
            # dall-e-3) — pay-as-you-go, billed against OPENAI_API_KEY.
            _check_openai_billing_gate(
                request.confirm_api_billing, endpoint=f"openai-{actual_model}"
            )
            result = await generate_with_openai_image(
                prompt=request.prompt,
                model=actual_model,
                collection_id=request.collection_id or "ai-generated-images",
                link_id=request.link_id,
                width=request.width,
                height=request.height,
                aspect_ratio=request.aspect_ratio,
                quality=request.quality,
                reference_image_urls=request.reference_image_urls,
            )

        elif model_name == "dall-e-3":
            raise HTTPException(status_code=501, detail="DALL-E 3 support coming soon")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

        # Don't echo request.width/height — the actual model picks output
        # dimensions from aspect_ratio + image_size, so request.width is just
        # a default placeholder (1024) that would lie about reality. The
        # provider handler puts real dims into `result`; preserve those.
        return {
            **result,
            "model": model_name,
            "actual_model": actual_model,
        }

    except HTTPException:
        raise
    except TimeoutError as e:
        logger.error(f"Image generation timeout: {e}")
        raise HTTPException(status_code=504, detail="Image generation timed out")
    except RuntimeError as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")


@router.get("/genimage/models")
async def list_image_models():
    """List available image generation models"""
    return {
        "models": [
            {
                "id": "nano-banana-2",
                "name": "Nano Banana 2",
                "provider": "Google Gemini",
                "description": "Gemini 3.1 Flash Image - fast, free, up to 4K, character consistency (default)",
                "default": True
            },
            {
                "id": "nano-banana-pro",
                "name": "Nano Banana Pro",
                "provider": "Google Gemini",
                "description": "Best Gemini image quality"
            },
            {
                "id": "imagen-4",
                "name": "Imagen 4",
                "provider": "Google",
                "description": "Photorealistic image generation"
            },
            {
                "id": "higgsfield",
                "name": "Higgsfield Soul",
                "provider": "Higgsfield",
                "description": "High quality text-to-image (requires credits)"
            },
            {
                "id": "higgsfield-reve",
                "name": "Reve",
                "provider": "Higgsfield",
                "description": "Versatile text-to-image (requires credits)"
            },
            {
                "id": "minimax-image-01",
                "name": "MiniMax Image-01",
                "provider": "MiniMax",
                "description": (
                    "Pay-as-you-go ($0.003/image, ~1/10 of typical price). "
                    "Requires confirm_api_billing=true. Counts against the "
                    "shared 25 EUR/month MiniMax cap."
                ),
                "billing": "payg"
            },
            {
                "id": "gpt-image-2",
                "name": "GPT Image 2 (OpenAI)",
                "provider": "OpenAI",
                "description": (
                    "OpenAI gpt-image-2 — best for diagrams, photoreal, up to 4K. "
                    "Pay-as-you-go (~$0.15-0.20/image high-quality 2048²). "
                    "Requires confirm_api_billing=true."
                ),
                "billing": "payg"
            },
            {
                "id": "gpt-image-1",
                "name": "GPT Image 1 (OpenAI, legacy)",
                "provider": "OpenAI",
                "description": (
                    "Older gpt-image-1 — cheaper (~$0.05/image medium). "
                    "Supports native transparent alpha (gpt-image-2 does not). "
                    "Requires confirm_api_billing=true."
                ),
                "billing": "payg"
            }
        ]
    }


@router.post("/upscale", response_model=ImageResponse)
async def upscale_image_endpoint(
    request: UpscaleRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Upscale an image using AI models

    Supports:
    - Real-ESRGAN
    - GFPGAN (face enhancement)
    - Other upscaling models
    """
    try:
        # TODO: Implement actual image upscaling
        # 1. Download input image
        # 2. Call upscaling model (Replicate)
        # 3. Upload result to Storage API
        # 4. Return storage object info

        return ImageResponse(
            image_url=request.image_url,
            storage_object_id=None,
            model=request.model,
            width=1024 * request.scale_factor,
            height=1024 * request.scale_factor
        )
    except Exception as e:
        logger.error(f"Image upscale error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gendepth", response_model=dict)
async def generate_depth_endpoint(
    request: DepthRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate depth map from image

    Useful for:
    - 3D reconstruction
    - AR/VR applications
    - Image analysis
    """
    try:
        # TODO: Implement depth generation
        # Could be async job-based like in old API

        return {
            "job_id": "placeholder",
            "status": "pending",
            "message": "Depth generation job started"
        }
    except Exception as e:
        logger.error(f"Depth generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gendepth/result/{job_id}")
async def get_depth_result(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """Get depth generation job result"""
    # TODO: Implement job result retrieval
    return {
        "job_id": job_id,
        "status": "completed",
        "depth_map_url": "https://placeholder.com/depth.png",
        "storage_object_id": None
    }
