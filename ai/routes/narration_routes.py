"""
Narration / TTS Routes

POST /ai/tts/narrate           — Full pipeline: text → AI dramatic preprocessing → ElevenLabs TTS → audio
POST /ai/tts/narrate/preview   — Preview only: returns the dramatic script without generating audio
POST /ai/tts/minimax           — Direct MiniMax Speech-02 TTS (no dramatic preprocessing) — pay-as-you-go
POST /ai/tts/clone             — MiniMax voice-cloning: ref-audio + name → voice_id — pay-as-you-go
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

logger = logging.getLogger(__name__)
router = APIRouter()


def get_api_key():
    """Dependency placeholder — matches existing auth pattern."""
    import os
    return os.getenv("AI_API_KEY", "")


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
        # Surface ElevenLabs' own error taxonomy so upstream sees the real
        # cause instead of a generic 500 (ArTrack IACP 2026-07-01: the
        # opaque "Narration failed: ..." string cost hours of wrong-lead
        # debugging on what was really a plain 401 from ElevenLabs).
        try:
            from elevenlabs import (
                APIError as _ELApiError,
                AuthorizationError as _ELAuthError,
                RateLimitError as _ELRateError,
            )
        except Exception:
            _ELApiError = _ELAuthError = _ELRateError = ()
        # The ElevenLabs SDK's ApiError exposes .status_code + .body + .headers
        # as attributes; str(e) is basically 'headers: {...}' — useless. Pull
        # the real JSON body out. (ArTrack IACP 2026-07-01: PR #96 v1 gave
        # ArTrack 'headers: {...}' with no body context, hiding the actual
        # 'needs_authorization' / 'missing permission models_read' payload
        # that would have pointed straight at the env-loading bug.)
        def _elevenlabs_body(err):
            return getattr(err, "body", None) or None

        if isinstance(e, _ELAuthError):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "elevenlabs_auth_failed",
                    "elevenlabs_message": str(e)[:200],
                    "elevenlabs_body": _elevenlabs_body(e),
                    "hint": (
                        "ElevenLabs returned 401. Common causes: "
                        "ELEVENLABS_API_KEY missing/expired/revoked, "
                        "the key doesn't have permission for this voice_id, "
                        "OR the api-ai service isn't loading .env (check "
                        "'systemctl show api-ai -p EnvironmentFiles' on the "
                        "target host). Look at elevenlabs_body.detail.message "
                        "for the exact ElevenLabs classification."
                    ),
                },
            )
        if isinstance(e, _ELRateError):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "elevenlabs_rate_or_quota_exceeded",
                    "elevenlabs_message": str(e)[:200],
                    "elevenlabs_body": _elevenlabs_body(e),
                    "hint": (
                        "ElevenLabs returned 429. Either the per-minute "
                        "rate-limit or the monthly character quota is "
                        "exceeded. Back off + retry, or top up the plan."
                    ),
                },
            )
        if isinstance(e, _ELApiError):
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "elevenlabs_api_error",
                    "elevenlabs_status": getattr(e, "status_code", None),
                    "elevenlabs_message": str(e)[:200],
                    "elevenlabs_body": _elevenlabs_body(e),
                    "hint": "ElevenLabs returned a non-2xx that isn't auth or rate-limit.",
                },
            )
        # Unknown exception — keep the generic 500 path but log richly.
        logger.exception(f"Narration failed with unclassified exception: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "narration_failed",
                "exception_type": type(e).__name__,
                "message": str(e),
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
