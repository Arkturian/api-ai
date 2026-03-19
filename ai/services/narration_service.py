"""
Narration Service — Dramaturgical TTS Primitive

Takes plain text + character profile + context, uses an AI agent to build
a dramaturgically enriched script, then sends it to ElevenLabs TTS in one call.

This is the base primitive that:
- content-api uses directly for story scene narration
- AudioDramaService can use per chunk for better dialog quality

Architecture:
  Text + Character + Context
    → AI Agent (Gemini) builds dramatic script with ElevenLabs-compatible markup
    → ElevenLabs TTS (single call, one voice, one audio)
    → Audio bytes returned (+ optional Storage upload)
"""

import json
import os
import logging
import time
import asyncio
from typing import Optional
from pydantic import BaseModel, Field

import google.generativeai as genai

logger = logging.getLogger(__name__)


# ── Request / Response Models ────────────────────────────────────

class NarrationCharacter(BaseModel):
    """Who is speaking."""
    name: str = Field(description="Character name, e.g. 'tschauko'")
    voice_id: str = Field(description="ElevenLabs voice ID")
    personality: Optional[str] = Field(default=None, description="Character personality traits")
    speaking_style: Optional[str] = Field(default=None, description="How this character narrates, e.g. 'warm storyteller for children'")


class NarrationContext(BaseModel):
    """Context for the narration."""
    type: str = Field(default="story_scene", description="story_scene | annotation | audioguide | freeform")
    title: Optional[str] = Field(default=None, description="Scene/chapter title")
    mood: Optional[str] = Field(default=None, description="dramatisch | mystisch | friedlich | dunkel | fröhlich")
    position: Optional[str] = Field(default=None, description="opening | middle | climax | closing")
    audience: str = Field(default="families", description="kids | families | adults | expert")
    language: str = Field(default="de", description="Language code")
    additional_instructions: Optional[str] = Field(default=None, description="Extra instructions for the AI agent")


class NarrationConfig(BaseModel):
    """TTS generation config."""
    stability: float = Field(default=0.3, description="ElevenLabs stability (lower = more expressive)")
    clarity: float = Field(default=0.75, description="ElevenLabs similarity boost")
    model_id: str = Field(default="eleven_multilingual_v2", description="ElevenLabs model")
    speed: float = Field(default=1.0, description="Speaking speed")
    preprocessing: bool = Field(default=True, description="Enable AI dramatic preprocessing")
    output_format: str = Field(default="mp3", description="Audio format")


class NarrationRequest(BaseModel):
    """Full narration request."""
    text: str = Field(description="Plain text to narrate")
    character: NarrationCharacter
    context: NarrationContext = NarrationContext()
    config: NarrationConfig = NarrationConfig()
    save_options: Optional[dict] = Field(default=None, description="Storage API save options")
    collection_id: Optional[str] = Field(default=None, description="Storage collection ID")


class NarrationResponse(BaseModel):
    """Narration result."""
    audio_id: Optional[int] = None
    audio_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    dramatic_script: str = Field(description="The enriched script that was spoken")
    original_text: str = Field(description="Original input text")
    preprocessing_model: Optional[str] = None


# ── Dramatic Script Agent ────────────────────────────────────────

DRAMATIC_AGENT_PROMPT = """Du bist ein Dramaturg und Audio-Regisseur. Deine Aufgabe ist es, einen Text für einen Sprecher aufzubereiten,
so dass er beim Vorlesen dramaturgisch wirksam ist.

DER SPRECHER:
- Name: {character_name}
- Persönlichkeit: {personality}
- Erzählstil: {speaking_style}

KONTEXT:
- Typ: {context_type}
- Titel: {title}
- Stimmung: {mood}
- Position in der Geschichte: {position}
- Publikum: {audience}
- Sprache: {language}

{additional_instructions}

DEINE AUFGABE:
Bereite den folgenden Text so auf, dass ElevenLabs ihn optimal vorlesen kann.

REGELN für die Textaufbereitung:
1. **Pausen**: Verwende "..." für kurze dramatische Pausen (0.5s) und "—" für längere Denkpausen
2. **Tempo**: Kurze Sätze = schnelleres Tempo. Lange Sätze mit Kommas = langsamer, nachdenklicher
3. **Betonung**: Einzelne wichtige Wörter können durch Isolation hervorgehoben werden (eigener kurzer Satz)
4. **Spannung**: Bei dramatischen Stellen die Sätze kürzer machen, mehr Pausen
5. **Ruhe**: Bei friedlichen/mystischen Stellen längere, fließende Sätze verwenden
6. **Emotionalität**: Der Text soll die Stimmung ({mood}) transportieren
7. **Natürlichkeit**: Es muss sich wie natürliches Erzählen anhören, nicht wie Vorlesen
8. **Sprache beibehalten**: Der Text bleibt in {language}, nichts übersetzen!

WICHTIG:
- Ändere NICHT den Inhalt oder die Fakten
- Füge KEINE neuen Informationen hinzu
- Der Kern-Text bleibt gleich, nur die Aufbereitung ändert sich
- Gib NUR den aufbereiteten Text zurück, keine Erklärungen
- Kein JSON, kein Markdown, nur den reinen aufbereiteten Text

ORIGINALTEXT:
{text}

AUFBEREITETER TEXT:"""


class NarrationService:
    """Dramaturgical TTS — AI-enriched text → ElevenLabs → Audio."""

    async def generate(self, request: NarrationRequest) -> NarrationResponse:
        """Full pipeline: preprocess → TTS → optional save."""
        t_start = time.time()

        # Step 1: Dramatic preprocessing (optional)
        if request.config.preprocessing:
            dramatic_script = await self._preprocess_text(request)
        else:
            dramatic_script = request.text

        logger.info(f"[Narration] Script ready ({len(dramatic_script)} chars, {int((time.time()-t_start)*1000)}ms)")

        # Step 2: Generate TTS via ElevenLabs
        audio_bytes = await self._generate_tts(dramatic_script, request)
        logger.info(f"[Narration] TTS done ({len(audio_bytes)} bytes, {int((time.time()-t_start)*1000)}ms)")

        # Step 3: Optional save to Storage API
        audio_id = None
        audio_url = None
        if request.save_options:
            audio_id, audio_url = await self._save_audio(audio_bytes, request)
            logger.info(f"[Narration] Saved to storage: id={audio_id}")

        return NarrationResponse(
            audio_id=audio_id,
            audio_url=audio_url,
            dramatic_script=dramatic_script,
            original_text=request.text,
            preprocessing_model="gemini" if request.config.preprocessing else None,
        )

    async def preprocess_only(self, request: NarrationRequest) -> str:
        """Only run the dramatic preprocessing, return enriched text."""
        return await self._preprocess_text(request)

    async def _preprocess_text(self, request: NarrationRequest) -> str:
        """AI agent enriches text with dramatic markup for TTS."""
        prompt = DRAMATIC_AGENT_PROMPT.format(
            character_name=request.character.name,
            personality=request.character.personality or "natürlich, freundlich",
            speaking_style=request.character.speaking_style or "Geschichtenerzähler",
            context_type=request.context.type,
            title=request.context.title or "(kein Titel)",
            mood=request.context.mood or "neutral",
            position=request.context.position or "middle",
            audience=request.context.audience,
            language=request.context.language,
            additional_instructions=request.context.additional_instructions or "",
            text=request.text,
        )

        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = await asyncio.to_thread(model.generate_content, prompt)
            result = response.text.strip()
            # Clean any markdown wrapping
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result[3:]
            if result.endswith("```"):
                result = result[:-3].strip()
            return result
        except Exception as e:
            logger.warning(f"[Narration] Preprocessing failed, using original text: {e}")
            return request.text  # Fallback: use original text

    async def _generate_tts(self, text: str, request: NarrationRequest) -> bytes:
        """Generate audio via ElevenLabs."""
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ModuleNotFoundError:
            raise RuntimeError("ElevenLabs package required. Install with 'pip install elevenlabs'.")

        client = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=request.character.voice_id,
            model_id=request.config.model_id,
            voice_settings={
                "stability": request.config.stability,
                "similarity_boost": request.config.clarity,
            }
        )

        audio_bytes = b""
        async for chunk in audio_stream:
            audio_bytes += chunk

        return audio_bytes

    async def _save_audio(self, audio_bytes: bytes, request: NarrationRequest) -> tuple:
        """Save audio to Storage API. Returns (storage_id, url)."""
        import httpx

        storage_url = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
        storage_key = os.getenv("STORAGE_API_KEY", "Inetpass1")

        boundary = "----NarrationUpload"
        filename = f"narration_{request.character.name}_{int(time.time())}.mp3"

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: audio/mpeg\r\n\r\n"
        ).encode() + audio_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="is_public"\r\n\r\n'
            f'{str(request.save_options.get("is_public", True)).lower()}\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="ai_mode"\r\n\r\nnone\r\n'
        ).encode()

        if request.save_options.get("link_id"):
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="link_id"\r\n\r\n'
                f'{request.save_options["link_id"]}\r\n'
            ).encode()

        if request.collection_id:
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="collection_id"\r\n\r\n'
                f'{request.collection_id}\r\n'
            ).encode()

        body += f"--{boundary}--\r\n".encode()

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{storage_url}/storage/upload",
                content=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "X-API-KEY": storage_key,
                }
            )
            resp.raise_for_status()
            result = resp.json()
            storage_id = result.get("id")
            url = f"{storage_url}/storage/media/{storage_id}"
            return storage_id, url
