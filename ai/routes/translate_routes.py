"""
Translation Routes
==================

Endpoint for text translation via Google Translate.
Uses deep-translator for reliable, free translations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class TranslateRequest(BaseModel):
    text: str
    source: Optional[str] = "auto"
    target: str = "en"


class TranslateBatchRequest(BaseModel):
    texts: List[str]
    source: Optional[str] = "auto"
    target: str = "en"


class TranslateResponse(BaseModel):
    translated: str
    source: str
    target: str
    original: str


class TranslateBatchResponse(BaseModel):
    translations: List[TranslateResponse]
    source: str
    target: str


@router.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    """
    Translate text via Google Translate.

    - source: Language code (e.g. "de", "en", "sl") or "auto" for auto-detect
    - target: Target language code (e.g. "en", "de", "sl")
    - text: Text to translate

    Language codes: https://cloud.google.com/translate/docs/languages
    """
    from deep_translator import GoogleTranslator

    try:
        translator = GoogleTranslator(source=req.source, target=req.target)
        result = translator.translate(req.text)

        logger.info(f"Translated {len(req.text)} chars {req.source}->{req.target}")

        return TranslateResponse(
            translated=result,
            source=req.source,
            target=req.target,
            original=req.text
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@router.post("/translate/batch", response_model=TranslateBatchResponse)
async def translate_batch(req: TranslateBatchRequest):
    """
    Translate multiple texts in one request.

    - texts: List of strings to translate
    - source/target: Same as /translate
    """
    from deep_translator import GoogleTranslator

    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per batch")

    try:
        translator = GoogleTranslator(source=req.source, target=req.target)
        translations = []

        for text in req.texts:
            result = translator.translate(text)
            translations.append(TranslateResponse(
                translated=result,
                source=req.source,
                target=req.target,
                original=text
            ))

        logger.info(f"Batch translated {len(req.texts)} texts {req.source}->{req.target}")

        return TranslateBatchResponse(
            translations=translations,
            source=req.source,
            target=req.target
        )
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@router.get("/translate/languages")
async def list_languages():
    """List all supported language codes."""
    from deep_translator import GoogleTranslator

    try:
        langs = GoogleTranslator().get_supported_languages(as_dict=True)
        return {"languages": langs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch languages: {str(e)}")
