"""
Audio AI Routes
===============

Endpoints for audio generation:
- Text-to-Speech (TTS) & Dialog Generation
- Sound Effects (SFX)
- Music Generation
"""

from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
import logging
import re
import io
import os

logger = logging.getLogger(__name__)
router = APIRouter()


# Import models and services
import sys
sys.path.insert(0, '/Volumes/DatenAP/Code/api-ai/ai/services')

from ai.services.tts_models import SpeechRequest
from ai.services.speech_service import SpeechGenerator
from ai.services.audio_drama_service import AudioDramaGenerator
from ai.clients.storage_client import StorageObject
from ai.routes.music_generation import generate_music_stable_audio, generate_music_elevenlabs
from openai import AsyncOpenAI


# Response Models
class AudioResponse(BaseModel):
    audio_url: str
    file_url: Optional[str] = None  # Alias for audio_url for frontend compatibility
    storage_object_id: Optional[int] = None
    duration_seconds: Optional[float] = None
    format: str = "mp3"


class SFXRequest(BaseModel):
    prompt: str
    duration: Optional[float] = 5.0
    model: Optional[str] = "audio-ldm"


class MusicRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 30
    model: Optional[str] = "suno"  # suno or eleven


class AudioGenRequest(BaseModel):
    """Request model for audio generation (SFX/Music) - compatible with legacy API"""
    prompt: str
    duration_ms: Optional[int] = None
    link_id: Optional[str] = None
    owner_user_id: Optional[int] = None


def get_api_key():
    return "Inetpass1"


def is_likely_json(text: str) -> bool:
    """Check if text looks like JSON to prevent expensive TTS mistakes."""
    trimmed = text.strip()

    # Check for obvious JSON structures
    if (trimmed.startswith('{') and trimmed.endswith('}')) or \
       (trimmed.startswith('[') and trimmed.endswith(']')):
        return True

    # Check for JSON patterns
    json_patterns = [
        r'"production_cues"\s*:',
        r'"cues"\s*:',
        r'"background_music"\s*:',
        r'"music"\s*:',
        r'"type"\s*:\s*"dialog"',
        r'"audio_url"\s*:',
        r'"pause_after_ms"\s*:'
    ]

    for pattern in json_patterns:
        if re.search(pattern, trimmed):
            return True

    return False


@router.post("/generate_speech")
async def generate_speech_endpoint(
    req: SpeechRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Handles both single voice TTS and complex audio drama generation.

    Accepts the full SpeechRequest model from dialog.php with:
    - id, timestamp, content (text, language, voice, speed)
    - config (dialog_mode, voice_mapping, add_sfx, add_music, etc.)
    - collection_id, save_options
    """
    try:
        # CRITICAL VALIDATION: Prevent JSON from being synthesized (costs money!)
        if req.content and req.content.text and is_likely_json(req.content.text):
            print(f"🚫 TTS BLOCKED: Text looks like JSON structure!")
            print(f"🚫 First 200 chars: {req.content.text[:200]}")
            raise HTTPException(
                status_code=400,
                detail="TTS generation blocked: Text appears to be JSON. JSON should be parsed, not spoken."
            )

        if req.config.dialog_mode:
            # Dialog/Audio Drama Generation
            generator = AudioDramaGenerator(req, api_key, image_gen_func=None)
            saved_audio_obj, production_plan, generated_image_obj = await generator.generate()

            # Analyze-only mode: return just the analysis
            if saved_audio_obj is None:
                return JSONResponse(content={
                    "analysis_result": production_plan,
                    "message": "analysis_only"
                })

            # Convert StorageObject to dict for JSON response
            response_data = {
                "id": saved_audio_obj.id,
                "file_url": saved_audio_obj.file_url,
                "audio_url": saved_audio_obj.file_url,  # Legacy compatibility
                "object_key": saved_audio_obj.object_key,
                "original_filename": saved_audio_obj.original_filename,
                "mime_type": saved_audio_obj.mime_type,
                "file_size": saved_audio_obj.file_size,
                "thumbnail_url": saved_audio_obj.thumbnail_url,
                "context": saved_audio_obj.context,
                "is_public": saved_audio_obj.is_public,
                "collection_id": saved_audio_obj.collection_id,
                "analysis_result": production_plan
            }

            if generated_image_obj:
                response_data['generated_image'] = {
                    "id": generated_image_obj.id,
                    "file_url": generated_image_obj.file_url
                }

            return JSONResponse(content=response_data)

        else:
            # Simple TTS Generation
            generator = SpeechGenerator(req, api_key, image_gen_func=None)
            saved_audio_obj, _, _ = await generator.generate()

            return {
                "id": saved_audio_obj.id,
                "file_url": saved_audio_obj.file_url,
                "audio_url": saved_audio_obj.file_url,
                "object_key": saved_audio_obj.object_key,
                "original_filename": saved_audio_obj.original_filename,
                "mime_type": saved_audio_obj.mime_type,
                "file_size": saved_audio_obj.file_size,
                "context": saved_audio_obj.context,
                "is_public": saved_audio_obj.is_public,
                "collection_id": saved_audio_obj.collection_id
            }

    except Exception as e:
        print(f"--- ERROR [TTS Endpoint]: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during TTS generation: {str(e)}")


@router.post("/gensfx")
async def generate_sfx_endpoint(
    request: SFXRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generates a sound effect using ElevenLabs, analyzes its volume, and saves it to storage if it's not silent.

    Examples:
    - "dog barking"
    - "thunder and rain"
    - "car engine starting"
    """
    import os
    import tempfile
    from pathlib import Path
    from ai.services import tts_service
    from ai.clients.storage_client import save_file_and_record

    try:
        from elevenlabs.client import AsyncElevenLabs

        print(f"--- SFX Gen: Generating SFX for prompt: '{request.prompt[:80]}...'")
        client = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        audio_stream = client.text_to_sound_effects.convert(text=request.prompt)

        audio_bytes = b""
        async for chunk in audio_stream:
            audio_bytes += chunk

        if not audio_bytes:
            raise HTTPException(status_code=500, detail="ElevenLabs SFX generation returned no data.")

        # --- Audio Analysis ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        mean_volume = tts_service.analyze_audio_level(temp_path)
        print(f"--- SFX Gen: Analyzed audio level: {mean_volume} dB")

        # Discard silent or near-silent SFX
        SILENCE_THRESHOLD = -60.0
        if mean_volume < SILENCE_THRESHOLD:
            print(f"--- SFX Gen: SFX is too quiet ({mean_volume} dB), discarding.")
            temp_path.unlink()
            raise HTTPException(status_code=422, detail=f"Generated SFX was silent and has been discarded.")

        # --- Save to Storage via HTTP ---
        filename = f"sfx_{request.prompt.replace(' ', '_')[:20]}.mp3"

        saved_obj = await save_file_and_record(
            data=audio_bytes,
            original_filename=filename,
            context="sfx-generation",
            is_public=True,
            collection_id="ai-generated-sfx"
        )

        temp_path.unlink()
        print(f"--- SFX Gen: Saved SFX to storage object ID {saved_obj.id}")

        return {
            "id": saved_obj.id,
            "file_url": saved_obj.file_url,
            "audio_url": saved_obj.file_url,
            "storage_object_id": saved_obj.id,
            "format": "mp3"
        }

    except Exception as e:
        logger.error(f"SFX generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    # GCP-cost-incident hardening (2026-05-30): when `model` starts with
    # `gemini`, the request hits the Gemini API (GCP-billed) and must
    # carry this opt-in form-field set to "true" / "1" / "yes". Whisper
    # and gpt-4o models route to OpenAI and are unaffected by this flag.
    confirm_api_billing: Optional[str] = Form(None),
    api_key: str = Depends(get_api_key)
):
    """
    Transcribe an uploaded audio file using OpenAI Whisper / GPT-4o / Gemini.

    Models (OpenAI):
    - whisper-1: OpenAI Whisper (default, classic)
    - gpt-4o-transcribe / gpt-4o-mini-transcribe: newer gpt-4o transcription
    - gpt-4o-transcribe-diarize: transcription WITH speaker separation —
      use with response_format="diarized_json" to get speaker-labeled segments

    Models (Google):
    - gemini-1.5-flash / gemini-1.5-pro / gemini-2.0-flash-exp

    Parameters:
    - language (optional): ISO code like "de", "en" — improves accuracy
    - response_format (optional): text | json | diarized_json.
      For speaker separation set model=gpt-4o-transcribe-diarize +
      response_format=diarized_json.

    Supports typical audio MIME types (mp3, m4a, wav, webm, etc.).
    """
    try:
        if model.startswith("gemini"):
            # GCP-cost-incident hardening: require explicit opt-in + cap check.
            from .text_ai_routes import _check_api_billing_gate
            confirmed_bool = str(confirm_api_billing).lower() in ("true", "1", "yes", "y")
            _check_api_billing_gate(confirmed_bool, endpoint="transcribe-gemini")
            return await _transcribe_with_gemini(file, model, prompt)
        else:
            return await _transcribe_with_whisper(
                file, model, prompt,
                language=language,
                response_format=response_format,
            )

    except HTTPException:
        raise
    except Exception as e:
        # Map OpenAI/upstream errors to actionable HTTP status codes so the
        # frontend can branch on the real cause instead of opaque 500.
        try:
            from openai import RateLimitError, BadRequestError, APIStatusError
        except Exception:
            RateLimitError = BadRequestError = APIStatusError = ()

        msg = str(e)
        if isinstance(e, RateLimitError) or "insufficient_quota" in msg or "exceeded your current quota" in msg:
            logger.error(f"Transcription quota/rate-limit: {e}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "OpenAI quota exhausted or rate-limited. "
                             "Top up the OPENAI billing balance or switch model to "
                             "'gemini-2.0-flash-exp' (no API cost on free tier).",
                    "code": "openai_quota_exhausted",
                    "upstream": msg[:500],
                },
            )
        # More specific BadRequest classification: differentiate between
        # duration overflow, unsupported format, and other "audio not OK".
        if "Invalid file format" in msg or "Supported formats" in msg:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        "OpenAI rejected the audio format. Allowed: flac, m4a, "
                        "mp3, mp4, mpeg, mpga, oga, ogg, wav, webm. Rename the "
                        "file with a proper extension or transcode it first."
                    ),
                    "code": "audio_unsupported_format",
                    "upstream": msg[:500],
                },
            )
        if "audio duration" in msg or "is longer than 1400 seconds" in msg:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        "Audio too long for the chosen model. Diarization "
                        "models have a hard cap of 23 min (1400 s) per request. "
                        "Split the audio into shorter chunks or use a non-"
                        "diarize model (e.g. whisper-1 handles longer files)."
                    ),
                    "code": "audio_duration_too_long",
                    "upstream": msg[:500],
                },
            )
        if isinstance(e, BadRequestError):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "OpenAI rejected the request — see upstream for details.",
                    "code": "audio_invalid_for_model",
                    "upstream": msg[:500],
                },
            )
        logger.error(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Codec → safe extension mapping for OpenAI's whitelist
# (flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm).
#
# IMPORTANT: only list codecs that ride in containers OpenAI parses
# cleanly *without* re-muxing. Anything not listed here (or mapped to
# ``None``) gets force-transcoded to MP3. AAC for example *can* live in
# an M4A container but a lot of in-the-wild AAC files are raw ADTS or
# have broken MOOV atoms — renaming the extension is not enough and
# OpenAI returns "Invalid file format" + zero-duration. Same story for
# Opus/Vorbis/ALAC: better to spend the 1-2s on ffmpeg than gamble.
_CODEC_TO_EXT = {
    "mp3": ".mp3",
    "flac": ".flac",
    "pcm_s16le": ".wav", "pcm_s24le": ".wav", "pcm_s32le": ".wav",
    "pcm_f32le": ".wav", "pcm_u8":    ".wav",
    # Force-transcode list (kept here for documentation/intent clarity)
    "aac":    None, "alac":   None,
    "opus":   None, "vorbis": None,
    "wmav2":  None, "amr_nb": None, "amr_wb": None,
}


def _normalize_audio_for_openai(data: bytes, basename: str = "audio") -> tuple[bytes, str]:
    """Ensure ``data`` is in a format OpenAI's audio.transcriptions accepts.

    Returns ``(bytes, ext)`` where ``ext`` is a leading-dot extension. Strategy:

    1. ffprobe the buffer to read the real codec (ignores filename lies).
    2. Codec in :data:`_CODEC_TO_EXT` and not ``None`` → keep bytes as-is,
       attach the matching extension.
    3. Otherwise → ffmpeg-transcode to MP3 (libmp3lame, VBR q4 → ~128 kbps).

    Falls back to returning ``(data, ".mp3")`` (last-resort guess) if both
    ffprobe and ffmpeg fail. The transcribe endpoint then surfaces the real
    OpenAI error to the caller.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as src:
        src.write(data)
        src_path = src.name

    try:
        # 1) Probe codec
        codec = None
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=nokey=1:noprint_wrappers=1",
                    src_path,
                ],
                capture_output=True, text=True, timeout=15,
            )
            if probe.returncode == 0:
                codec = (probe.stdout or "").strip().splitlines()[0:1]
                codec = codec[0] if codec else None
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}")

        logger.info(f"Audio codec detected: {codec!r}")

        # 2) Mapped codec we can use as-is
        if codec and _CODEC_TO_EXT.get(codec):
            return data, _CODEC_TO_EXT[codec]

        # 3) Unknown / unmapped / needs-transcode → MP3
        dst_path = src_path + ".mp3"
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", src_path,
                 "-vn",                  # drop any video stream
                 "-ac", "1",             # mono is plenty for speech
                 "-ar", "16000",         # 16 kHz, standard ASR rate
                 "-c:a", "libmp3lame",
                 "-q:a", "4",
                 dst_path],
                capture_output=True, timeout=180,
            )
            if r.returncode == 0 and os.path.exists(dst_path):
                with open(dst_path, "rb") as f:
                    new_data = f.read()
                logger.info(
                    f"Transcoded {len(data)} bytes ({codec or 'unknown'}) "
                    f"→ {len(new_data)} bytes mp3"
                )
                return new_data, ".mp3"
            logger.warning(
                f"ffmpeg transcode failed rc={r.returncode}: {r.stderr[:300]!r}"
            )
        except Exception as e:
            logger.warning(f"ffmpeg transcode raised: {e}")
        finally:
            if os.path.exists(dst_path):
                try: os.unlink(dst_path)
                except Exception: pass

        # Last-resort: pretend it's mp3 and let OpenAI's error be authoritative
        return data, ".mp3"
    finally:
        try: os.unlink(src_path)
        except Exception: pass


async def _transcribe_with_whisper(
    file: UploadFile,
    model: str,
    prompt: Optional[str] = None,
    language: Optional[str] = None,
    response_format: Optional[str] = None,
):
    """Transcribe audio using OpenAI Whisper / GPT-4o transcribe family."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # Normalize the audio for OpenAI:
    #   1) Probe the actual codec via ffprobe (ignores filename + MIME lies).
    #   2) If codec is in OpenAI's whitelist → keep bytes, just rename the
    #      buffer to the matching extension.
    #   3) If codec is anything else → transcode to MP3 via ffmpeg.
    # This robustly handles: misnamed extensions, raw AAC, weird containers,
    # Voice-Memos .m4a-with-quirks, browser-uploaded blobs etc.
    base, _ = os.path.splitext(file.filename or "audio")
    data, ext = _normalize_audio_for_openai(data, base)
    audio_buffer = io.BytesIO(data)
    audio_buffer.name = (base or "audio") + ext

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build kwargs conditionally — the OpenAI SDK is strict about unexpected params
    kwargs = {"model": model, "file": audio_buffer}
    if prompt:
        kwargs["prompt"] = prompt
    if language:
        kwargs["language"] = language
    if response_format:
        kwargs["response_format"] = response_format

    # Diarization models REQUIRE chunking_strategy. Without it OpenAI returns
    # 400: 'chunking_strategy is required for diarization models'. Default to
    # "auto" (server-side chunking) which is correct for arbitrary-length
    # meeting recordings; callers can override via the Form field if they
    # need a different strategy.
    if "diarize" in model:
        kwargs.setdefault("chunking_strategy", "auto")

    result = await client.audio.transcriptions.create(**kwargs)

    # diarized_json → return raw payload (segments[] with speaker labels)
    if response_format == "diarized_json":
        raw = jsonable_encoder(result)
        return {
            "model": model,
            "prompt": prompt,
            "language": language,
            "filename": file.filename,
            "response_format": "diarized_json",
            "diarized": raw,
        }

    text = getattr(result, "text", None)
    if text is None:
        return {
            "model": model,
            "filename": file.filename,
            "raw": jsonable_encoder(result),
        }

    return {
        "text": text,
        "model": model,
        "prompt": prompt,
        "language": language,
        "filename": file.filename,
    }


async def _transcribe_with_gemini(
    file: UploadFile,
    model: str,
    prompt: Optional[str] = None
):
    """Transcribe audio using Google Gemini"""
    import tempfile
    from pathlib import Path
    from ai.services.cost_tracker import cost_tracker

    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Google API key not configured.")

    # Check budget before processing
    if cost_tracker.should_block_request():
        status = cost_tracker.get_status()
        raise HTTPException(
            status_code=429,
            detail=f"Monthly Gemini API budget exceeded: {status['total_cost_eur']:.2f}/{status['monthly_budget_eur']:.2f} EUR."
        )

    try:
        # NOTE: switched from the deprecated `google-generativeai` SDK to the
        # new `google-genai` SDK. The old package is EOL'd by Google and its
        # `upload_file` attribute has been stripped in recent installs (we hit
        # AttributeError at runtime even though dir() listed it). The new SDK
        # uses Client().files.upload + client.models.generate_content.
        from google import genai as genai_new

        client = genai_new.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # Read audio data
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        # Save to temporary file (Gemini's file-upload API expects a path)
        suffix = Path(file.filename).suffix if file.filename else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)

        try:
            # Upload to Gemini File API
            logger.info(f"Uploading audio file to Gemini: {temp_path}")
            uploaded = client.files.upload(file=str(temp_path))

            # Build the prompt
            transcription_prompt = (
                "Transcribe this audio file accurately. "
                "Provide only the transcription text."
            )
            if prompt:
                transcription_prompt = f"{prompt}\n\nTranscribe this audio file accurately."

            logger.info(f"Starting Gemini transcription with model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=[transcription_prompt, uploaded],
            )

            # Extract text. The new SDK exposes `.text` the same way; if not
            # present we fall back to the first candidate part.
            transcription_text = getattr(response, "text", None) or ""
            if not transcription_text and getattr(response, "candidates", None):
                try:
                    parts = response.candidates[0].content.parts
                    transcription_text = "".join(
                        getattr(p, "text", "") or "" for p in parts
                    )
                except Exception:
                    transcription_text = ""
            transcription_text = (transcription_text or "").strip()

            if not transcription_text:
                raise HTTPException(status_code=500, detail="Gemini returned empty response.")

            # Track usage - extract token counts if available
            input_tokens = 0
            output_tokens = 0
            usage = getattr(response, "usage_metadata", None)
            if usage:
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                logger.info(f"Gemini STT tokens: {input_tokens}in/{output_tokens}out")

            model_key = f"{model}-audio" if not model.endswith("-audio") else model
            if input_tokens > 0 or output_tokens > 0:
                cost_tracker.track_usage(model_key, input_tokens, output_tokens)
            else:
                audio_size_mb = len(data) / (1024 * 1024)
                estimated_input = int(audio_size_mb * 5000)  # ~5000 tokens per MB
                estimated_output = len(transcription_text) // 4  # ~4 chars per token
                cost_tracker.track_usage(model_key, estimated_input, estimated_output)
                logger.info(
                    f"Gemini STT estimated tokens: {estimated_input}in/{estimated_output}out"
                )

            logger.info(f"Gemini transcription completed: {len(transcription_text)} characters")

            return {
                "text": transcription_text,
                "model": model,
                "prompt": prompt,
                "filename": file.filename,
                "provider": "gemini",
            }

        finally:
            # Best-effort cleanup of the uploaded file on Gemini-side + local
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
            temp_path.unlink(missing_ok=True)

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="google-genai package not installed. Run: pip install google-genai",
        )
    except Exception as e:
        logger.error(f"Gemini transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gemini transcription failed: {str(e)}")


@router.post("/genmusic", response_model=AudioResponse)
async def generate_music_endpoint(
    request: MusicRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate music from text prompt using Stable Audio (AIMLAPI)

    Examples:
    - "upbeat electronic dance music"
    - "calm piano melody"
    - "epic orchestral soundtrack"
    """
    try:
        duration_ms = request.duration * 1000  # Convert seconds to milliseconds
        result = await generate_music_stable_audio(request.prompt, duration_ms)

        return AudioResponse(
            audio_url=result["audio_url"],
            file_url=result["audio_url"],  # For frontend compatibility
            storage_object_id=result["storage_object_id"],
            duration_seconds=float(request.duration),
            format=result["format"]
        )
    except Exception as e:
        logger.error(f"Music generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genmusic_eleven", response_model=AudioResponse)
async def generate_music_eleven_endpoint(
    request: MusicRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate music using ElevenLabs

    Features:
    - High quality music generation
    - Style control
    - Instrumental and vocal options
    - Auto-retry with suggested prompt if flagged
    """
    try:
        duration_ms = request.duration * 1000  # Convert seconds to milliseconds
        result = await generate_music_elevenlabs(request.prompt, duration_ms)

        return AudioResponse(
            audio_url=result["audio_url"],
            file_url=result["audio_url"],  # For frontend compatibility
            storage_object_id=result["storage_object_id"],
            duration_seconds=float(request.duration),
            format=result["format"]
        )
    except Exception as e:
        logger.error(f"ElevenLabs music error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
