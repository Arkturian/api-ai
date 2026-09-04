"""
Narration / TTS Routes

POST /ai/tts/narrate                 — Full pipeline: text → AI dramatic preprocessing → ElevenLabs TTS → audio
POST /ai/tts/narrate/preview         — Preview only: returns the dramatic script without generating audio
POST /ai/tts/minimax                 — Direct MiniMax Speech-02 TTS (no dramatic preprocessing) — pay-as-you-go
POST /ai/tts/clone                   — MiniMax voice-cloning: ref-audio + name → voice_id — pay-as-you-go
POST /ai/tts/design                  — MiniMax voice design: text description → voice_id — pay-as-you-go
GET  /ai/tts/elevenlabs/subscription — quota/tier proxy so batch scripts can pre-flight without the key
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai.services.narration_service import (
    NarrationService,
    NarrationRequest,
    NarrationResponse,
)
from ai.provider_config import provider_missing

logger = logging.getLogger(__name__)
router = APIRouter()


def get_api_key():
    """Dependency placeholder — matches existing auth pattern."""
    import os
    return os.getenv("AI_API_KEY", "")


@router.get("/tts/elevenlabs/subscription")
async def elevenlabs_subscription(api_key: str = Depends(get_api_key)):
    """Proxy for ElevenLabs ``GET /v1/user/subscription``.

    Lets batch scripts (ArTrack's pre-flight quota check, IACP
    2026-07-03) compare remaining character credits against the
    estimated batch size WITHOUT holding the ELEVENLABS_API_KEY
    themselves — the key stays server-side.

    Returns a trimmed view: tier, character_count, character_limit,
    remaining, next_reset_unix. Full upstream body under ``raw``.
    """
    import os
    import httpx

    key = os.getenv("ELEVENLABS_API_KEY", "").strip('"').strip("'")
    if not key:
        raise provider_missing("elevenlabs")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": key},
            )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "elevenlabs_unreachable", "exc": str(e)[:200]},
        )
    if r.status_code != 200:
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:300]}
        raise HTTPException(
            status_code=502,
            detail={
                "error": "elevenlabs_subscription_error",
                "upstream_status": r.status_code,
                "upstream_body": body,
            },
        )
    data = r.json()
    count = int(data.get("character_count") or 0)
    limit = int(data.get("character_limit") or 0)
    return {
        "tier": data.get("tier"),
        "character_count": count,
        "character_limit": limit,
        "remaining": max(0, limit - count),
        "next_reset_unix": data.get("next_character_count_reset_unix"),
        "raw": data,
    }


@router.post("/tts/narrate", response_model=NarrationResponse)
async def narrate(req: NarrationRequest, api_key: str = Depends(get_api_key)):
    """
    Dramaturgical TTS — AI-enriched narration in one call.

    Takes plain text + character profile + context, uses an AI agent to build
    a dramaturgically enriched script, then generates audio via ElevenLabs.

    **Character**: Who speaks (name, voice_id, personality, speaking_style)
    **Context**: What kind of content (story_scene, annotation, audioguide), mood, audience
    **Config**: TTS settings (stability, clarity, speed, preprocessing on/off)

    Returns audio (optionally saved to Storage API) + the dramatic script used.
    """
    service = NarrationService()
    try:
        result = await service.generate(req)
        return result
    except HTTPException:
        raise
    except Exception as e:
        # Classify ElevenLabs failures on the exception's ATTRIBUTES
        # (.status_code / .body), not on its class.
        #
        # Why: the previous version imported APIError / AuthorizationError /
        # RateLimitError from `elevenlabs` and matched with isinstance().
        # None of those names exist in the installed SDK — it raises
        # `elevenlabs.core.api_error.ApiError`. The import therefore threw,
        # the bare `except` fell back to empty tuples, every isinstance()
        # was False, and every ElevenLabs error fell through to the generic
        # 500. The mapping was dead from the day it was written (#527), and
        # the silent import fallback is exactly what hid it. Duck-typing on
        # the attributes survives SDK renames; no import, nothing to break.
        status = getattr(e, "status_code", None)
        body = getattr(e, "body", None)
        code = ""
        if isinstance(body, dict):
            _d = body.get("detail")
            if isinstance(_d, dict):
                code = _d.get("code") or _d.get("status") or ""
            elif isinstance(_d, str):
                code = _d

        # Quota first: ElevenLabs reports an exhausted character quota as
        # HTTP 401 (not 402/429) with body.detail.code=quota_exceeded, so a
        # plain status check would misfile it as an auth failure.
        if code == "quota_exceeded" or code == "quota_exceeded_free_tier":
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "elevenlabs_quota_exceeded",
                    "elevenlabs_status": status,
                    "elevenlabs_code": code,
                    "elevenlabs_body": body,
                    "hint": (
                        "ElevenLabs character quota is exhausted. It reports "
                        "this as HTTP 401 with body.detail.code="
                        "quota_exceeded; we re-map it to 429 so callers can "
                        "distinguish 'out of credits' (retry after top-up or "
                        "next billing cycle) from a real auth failure. "
                        "body.detail.message carries the exact credit counts."
                    ),
                },
            )
        if status == 429:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "elevenlabs_rate_limited",
                    "elevenlabs_status": status,
                    "elevenlabs_code": code,
                    "elevenlabs_body": body,
                    "hint": "ElevenLabs per-minute rate limit hit. Back off and retry.",
                },
            )
        if status in (401, 403):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "elevenlabs_auth_failed",
                    "elevenlabs_status": status,
                    "elevenlabs_code": code,
                    "elevenlabs_body": body,
                    "hint": (
                        "ElevenLabs returned 401/403. A 401 does NOT always "
                        "mean auth — inspect elevenlabs_code: "
                        "'needs_authorization' = missing/bad key (check "
                        "`systemctl show api-ai -p EnvironmentFiles`); "
                        "'missing_permissions' = scoped key lacks a scope; "
                        "'quota_exceeded' is handled as 429 above."
                    ),
                },
            )
        if status is not None:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "elevenlabs_api_error",
                    "elevenlabs_status": status,
                    "elevenlabs_code": code,
                    "elevenlabs_body": body,
                    "hint": "ElevenLabs returned a non-2xx that is not auth, rate-limit or quota.",
                },
            )
        # No status_code attribute -> not an ElevenLabs API error at all.
        logger.exception(f"Narration failed with unclassified exception: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "narration_failed",
                "exception_type": type(e).__name__,
                "message": str(e)[:500],
            },
        )


@router.post("/tts/narrate/preview")
async def narrate_preview(req: NarrationRequest, api_key: str = Depends(get_api_key)):
    """
    Preview the dramatic script without generating audio.

    Useful for reviewing/adjusting the AI's dramaturgical choices
    before spending TTS credits.
    """
    service = NarrationService()
    try:
        script = await service.preprocess_only(req)
        return JSONResponse(content={
            "dramatic_script": script,
            "original_text": req.text,
            "character": req.character.name,
            "mood": req.context.mood,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


# ── MiniMax Speech-02 TTS (PR-3) ──────────────────────────────────────
#
# Separate endpoint family from /tts/narrate because MiniMax-Speech-02
# is a direct text→audio call without the ElevenLabs dramatic-preprocessing
# pipeline. Keeping them split avoids leaking provider-specific knobs into
# the narration_service abstraction.

class MinimaxTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(
        default="male-qn-qingse",
        description="Voice ID — built-in preset or a custom cloned voice from /tts/clone",
    )
    model: str = Field(default="speech-02-hd", description="MiniMax TTS model id")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Playback speed")
    emotion: Optional[str] = Field(
        default=None,
        description="Optional: happy / sad / angry / neutral / surprised / fearful / disgusted",
    )
    language: str = Field(
        default="auto",
        description="ISO-like code or 'auto' — MiniMax detects from text",
    )
    collection_id: str = Field(
        default="tts-audio", description="Storage collection for the generated MP3"
    )
    link_id: Optional[str] = Field(default=None)
    confirm_api_billing: bool = Field(
        default=False,
        description="Required true — MiniMax TTS is pay-as-you-go billed per 1k chars",
    )


@router.post("/tts/minimax")
async def tts_minimax(req: MinimaxTTSRequest, api_key: str = Depends(get_api_key)):
    """Direct MiniMax Speech-02 TTS — no dramatic preprocessing.

    Pay-as-you-go: per-1k-chars pricing tracked via `minimax_cost_tracker`.
    Caller must opt in via `confirm_api_billing=true`; the federation-shared
    25 EUR/month cap applies regardless of the flag.
    """
    import base64
    from ai.clients.minimax_client import post_json, base_resp_failed
    from ai.clients.storage_client import save_file_and_record
    from ai.services.minimax_cost_tracker import minimax_cost_tracker
    from ai.routes.text_ai_routes import _check_minimax_billing_gate

    _check_minimax_billing_gate(req.confirm_api_billing, endpoint="minimax-tts")

    logger.info(
        f"MiniMax TTS: model={req.model}, voice={req.voice_id}, "
        f"chars={len(req.text)}, speed={req.speed}"
    )

    voice_setting = {"voice_id": req.voice_id, "speed": req.speed}
    if req.emotion:
        voice_setting["emotion"] = req.emotion

    payload = {
        "model": req.model,
        "text": req.text,
        "voice_setting": voice_setting,
        "audio_setting": {"format": "mp3", "sample_rate": 32000, "bitrate": 128000},
        "stream": False,
        "language_boost": req.language if req.language and req.language != "auto" else None,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    body = await post_json("t2a_v2", payload, timeout=120.0)
    err = base_resp_failed(body)
    if err:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_tts_failed", "upstream_msg": err},
        )

    # MiniMax returns audio bytes hex-encoded in data.audio
    audio_data = (body.get("data") or {}).get("audio")
    if not audio_data:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_audio_returned", "upstream_body": body},
        )

    try:
        # MiniMax docs: ``audio`` is a hex string of the raw bytes
        audio_bytes = bytes.fromhex(audio_data)
    except ValueError:
        # Some accounts return base64 instead — fall back gracefully
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail={"error": "minimax_audio_decode_failed", "exc": str(e)[:120]},
            )

    request_id = body.get("trace_id") or body.get("request_id") or "unknown"
    filename = f"tts_minimax_{request_id[:12]}.mp3"

    saved_obj = await save_file_and_record(
        data=audio_bytes,
        original_filename=filename,
        context="tts-generation",
        is_public=True,
        collection_id=req.collection_id,
        link_id=req.link_id,
    )

    minimax_cost_tracker.track_tts(model="minimax-speech-02", chars=len(req.text))

    extra_info = body.get("extra_info") or {}
    return {
        "id": saved_obj.id,
        "audio_url": saved_obj.file_url,
        "file_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "model": req.model,
        "voice_id": req.voice_id,
        "char_count": len(req.text),
        "audio_length_ms": extra_info.get("audio_length"),
        "request_id": request_id,
    }


@router.post("/tts/clone")
async def tts_voice_clone(
    name: str = Form(..., description="Display name for the voice profile"),
    file: UploadFile = File(..., description="Reference audio ~10 seconds (mp3/wav/m4a)"),
    confirm_api_billing: str = Form(
        "false", description="Must be 'true'/'1'/'yes' — pay-as-you-go billed per clone job"
    ),
    api_key: str = Depends(get_api_key),
):
    """MiniMax voice-cloning: 10s reference audio + name → voice_id.

    The returned ``voice_id`` is reusable indefinitely in subsequent
    ``/ai/tts/minimax`` calls. The clone job itself is billed once
    (~$0.10); subsequent TTS at standard per-1k-chars pricing.
    """
    from ai.clients.minimax_client import post_multipart, base_resp_failed
    from ai.services.minimax_cost_tracker import minimax_cost_tracker
    from ai.routes.text_ai_routes import _check_minimax_billing_gate

    confirmed_bool = str(confirm_api_billing).lower() in ("true", "1", "yes", "y")
    _check_minimax_billing_gate(confirmed_bool, endpoint="minimax-voice-clone")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Reference audio file is empty.")

    logger.info(
        f"MiniMax voice clone: name='{name}', ref_audio={file.filename}, bytes={len(data)}"
    )

    body = await post_multipart(
        path="voice_clone",
        fields={"voice_name": name, "purpose": "voice_clone"},
        files={
            "file": (
                file.filename or "reference.mp3",
                data,
                file.content_type or "audio/mpeg",
            )
        },
        timeout=180.0,
    )
    err = base_resp_failed(body)
    if err:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_voice_clone_failed", "upstream_msg": err},
        )

    voice_id = body.get("voice_id") or (body.get("data") or {}).get("voice_id")
    if not voice_id:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_voice_id_returned", "upstream_body": body},
        )

    minimax_cost_tracker.track_voice_clone()

    return {
        "voice_id": voice_id,
        "name": name,
        "ref_audio_filename": file.filename,
        "ref_audio_bytes": len(data),
        "hint": f"Use voice_id='{voice_id}' in subsequent /ai/tts/minimax calls",
    }


# ── MiniMax Voice Design (PR #99) ─────────────────────────────────────
#
# Design a NEW voice from a free-text description — no reference audio
# needed (that's /tts/clone's job). MiniMax's /v1/voice_design takes
# {prompt, preview_text} and returns a persistent voice_id plus a
# hex-encoded trial audio speaking the preview_text with the freshly
# designed voice. The voice_id is immediately usable in /tts/minimax.
#
# Use case (Alex, 2026-07-03): character voices for the Tscheppaschlucht
# audio guides — "Eule Tschauko" + "Tschauko" run on MiniMax-designed
# voices; only Dr. Peter Tschauko stays on ElevenLabs.
#
# Verified live against api.minimax.io before writing this endpoint:
#   POST /v1/voice_design {"prompt": "...", "preview_text": "..."}
#   → {"voice_id": "ttv-voice-...", "trial_audio": "<hex>",
#      "base_resp": {"status_code": 0}}

class MinimaxVoiceDesignRequest(BaseModel):
    prompt: str = Field(
        ...,
        description=(
            "Free-text voice description — character, age, tone, accent, "
            "speaking style. English prompts work best, the designed "
            "voice speaks whatever language preview_text/TTS text is in."
        ),
    )
    preview_text: str = Field(
        ...,
        description="Text the trial audio will speak (in the target language).",
    )
    voice_name: Optional[str] = Field(
        default=None,
        description="Display name stored alongside the trial audio in Storage.",
    )
    confirm_api_billing: bool = Field(
        default=False,
        description="Must be true — pay-as-you-go billed per design call.",
    )


@router.post("/tts/design")
async def tts_voice_design(
    req: MinimaxVoiceDesignRequest, api_key: str = Depends(get_api_key)
):
    """MiniMax voice design: free-text description → persistent voice_id.

    Returns the new ``voice_id`` (reusable in ``/ai/tts/minimax``) plus
    the trial audio saved to Storage so the caller can listen before
    committing to the voice.
    """
    from ai.clients.minimax_client import post_json, base_resp_failed
    from ai.clients.storage_client import save_file_and_record
    from ai.services.minimax_cost_tracker import minimax_cost_tracker
    from ai.routes.text_ai_routes import _check_minimax_billing_gate

    _check_minimax_billing_gate(req.confirm_api_billing, endpoint="minimax-voice-design")

    logger.info(
        f"MiniMax voice design: name='{req.voice_name or 'unnamed'}', "
        f"prompt={len(req.prompt)} chars, preview={len(req.preview_text)} chars"
    )

    body = await post_json(
        "voice_design",
        {"prompt": req.prompt, "preview_text": req.preview_text},
        timeout=120.0,
    )
    err = base_resp_failed(body)
    if err:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_voice_design_failed", "upstream_msg": err},
        )

    voice_id = body.get("voice_id") or ""
    trial_hex = body.get("trial_audio") or ""
    if not voice_id:
        raise HTTPException(
            status_code=502,
            detail={"error": "minimax_no_voice_id_returned", "upstream_keys": list(body.keys())},
        )

    # Persist the trial audio so the operator can listen via a plain URL.
    audio_url = None
    storage_id = None
    if trial_hex:
        try:
            audio_bytes = bytes.fromhex(trial_hex)
            safe_name = (req.voice_name or "voice_design").replace(" ", "_")[:40]
            obj = await save_file_and_record(
                data=audio_bytes,
                original_filename=f"voice_design_{safe_name}_{voice_id[-8:]}.mp3",
                context=f"MiniMax voice design trial: {req.voice_name or voice_id}",
                is_public=True,
            )
            storage_id = obj.id
            audio_url = obj.file_url
        except Exception as e:
            # Trial-audio persistence is a nicety — the voice_id is the
            # actual product. Don't fail the request over a storage hiccup.
            logger.warning(f"voice design trial-audio save failed: {e}")

    # Cost posture: MiniMax bills voice design per call in the same
    # PAYG lane as cloning; reuse the clone counter until the tracker
    # grows a dedicated bucket.
    minimax_cost_tracker.track_voice_clone(model="minimax-voice-design")

    return {
        "voice_id": voice_id,
        "voice_name": req.voice_name,
        "trial_audio_storage_id": storage_id,
        "trial_audio_url": audio_url,
        "hint": (
            f"Use voice_id='{voice_id}' in subsequent /ai/tts/minimax calls. "
            "Listen to trial_audio_url first — if the voice isn't right, "
            "call /tts/design again with a refined prompt."
        ),
    }
