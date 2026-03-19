"""
Narration Routes — Dramaturgical TTS Endpoint

POST /ai/tts/narrate — Full pipeline: text → AI dramatic preprocessing → ElevenLabs TTS → audio
POST /ai/tts/narrate/preview — Preview only: returns the dramatic script without generating audio
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ai.services.narration_service import (
    NarrationService,
    NarrationRequest,
    NarrationResponse,
)

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Narration failed: {str(e)}")


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
