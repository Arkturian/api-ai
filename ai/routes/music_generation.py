"""
Music Generation Service - Extracted from Legacy API
Implements Stable Audio (AIMLAPI) and ElevenLabs music generation
"""
import os
import asyncio
import httpx
from pathlib import Path
from urllib.parse import urlparse
from fastapi import HTTPException
from ai.clients.storage_client import save_file_and_record
from typing import Optional
import base64 as _b64
from ai.provider_config import provider_missing


async def generate_music_stable_audio(prompt: str, duration_ms: Optional[int] = 30000) -> dict:
    """Generate music using Stable Audio via AIMLAPI"""
    print(f"--- Music Gen: Generating music with Stable Audio for prompt: '{prompt[:80]}...'")

    aimlapi_key = (os.getenv("AIMLAPI_KEY") or "").strip()
    if not aimlapi_key:
        raise provider_missing("aimlapi")

    payload = {
        "model": "stable-audio",
        "prompt": prompt,
        "seconds_total": max(5, int(duration_ms / 1000)),
        "steps": 100,
    }

    header_variants = [
        {"Authorization": f"Bearer {aimlapi_key}", "Content-Type": "application/json"},
        {"access-token": aimlapi_key, "Content-Type": "application/json"},
        {"X-API-Key": aimlapi_key, "Content-Type": "application/json"},
    ]

    base_url = os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com")
    submit_url = f"{base_url}/v2/generate/audio"
    poll_url = f"{base_url}/v2/generate/audio"

    timeout = httpx.Timeout(connect=10.0, read=25.0, write=25.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        last_error = None
        used_headers = None
        submit_resp = None

        for headers in header_variants:
            try:
                submit_resp = await client.post(submit_url, headers=headers, json=payload)
                if submit_resp.status_code in (401, 403):
                    last_error = f"{submit_resp.status_code} {submit_resp.text}"
                    continue
                submit_resp.raise_for_status()
                used_headers = headers
                break
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if e.response is not None and e.response.status_code in (401, 403):
                    continue
                raise

        if used_headers is None:
            raise HTTPException(status_code=403, detail=f"Stable Audio auth failed: {last_error}")

        job = submit_resp.json()
        generation_id = job.get("id") or job.get("generation_id") or job.get("generationId")
        if not generation_id:
            raise HTTPException(status_code=500, detail="Stable Audio API did not return a generation ID.")

        audio_url = None
        for _ in range(90):
            poll_resp = await client.get(poll_url, headers=used_headers, params={"generation_id": generation_id})
            poll_resp.raise_for_status()
            result = poll_resp.json()
            status = result.get("status")

            if status in ("completed", "success", "succeeded"):
                file_obj = result.get("audio_file") or {}
                audio_url = file_obj.get("url")
                if audio_url:
                    break
            elif status in ("failed", "error"):
                raise HTTPException(status_code=500, detail=f"Stable Audio job failed: {result}")
            await asyncio.sleep(2)

        if not audio_url:
            raise HTTPException(status_code=504, detail="Timed out waiting for Stable Audio result.")

        audio_resp = await client.get(audio_url, follow_redirects=True)
        audio_resp.raise_for_status()
        music_bytes = audio_resp.content

    path = urlparse(audio_url).path
    ext = os.path.splitext(path)[1].lstrip(".") or "mp3"
    filename = f"music_{prompt.replace(' ', '_')[:20]}.{ext}"

    saved_obj = await save_file_and_record(
        data=music_bytes,
        original_filename=filename,
        context="music-generation",
        is_public=True,
        collection_id="ai-generated-music"
    )

    print(f"--- Music Gen: Saved music to storage object ID {saved_obj.id}")
    return {
        "id": saved_obj.id,
        "file_url": saved_obj.file_url,
        "audio_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "format": ext
    }


async def generate_music_elevenlabs(prompt: str, duration_ms: Optional[int] = 30000) -> dict:
    """Generate music using ElevenLabs Music API"""
    print(f"--- Music Gen (ElevenLabs): Generating music for prompt: '{prompt[:80]}...'")

    eleven_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not eleven_key:
        raise provider_missing("elevenlabs")

    payload = {
        "prompt": prompt,
        "musicLengthMs": duration_ms or 30000
    }
    headers = {
        "xi-api-key": eleven_key,
        "Content-Type": "application/json"
    }

    music_bytes: Optional[bytes] = None
    timeout = httpx.Timeout(connect=10.0, read=25.0, write=25.0, pool=None)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def _attempt_request(p: dict) -> Optional[bytes]:
            r = await client.post("https://api.elevenlabs.io/v1/music/compose", json=p, headers=headers)
            ctype_ = r.headers.get("content-type", "")

            if r.status_code == 200 and ctype_.startswith("audio/"):
                return r.content

            # Parse JSON body for non-audio responses
            try:
                body = r.json()
            except Exception:
                raise HTTPException(status_code=r.status_code, detail=f"ElevenLabs music API unexpected response: {r.text}")

            # Extract audio from JSON if present
            b64_audio = body.get("audio") or body.get("audioBase64")
            if b64_audio:
                try:
                    return _b64.b64decode(b64_audio)
                except Exception:
                    pass

            audio_url_ = body.get("url") or body.get("audioUrl")
            if audio_url_:
                dl_ = await client.get(audio_url_, follow_redirects=True)
                dl_.raise_for_status()
                return dl_.content

            # If bad_prompt, return marker to allow retry with suggestion
            detail = body.get("detail") or {}
            if isinstance(detail, dict) and detail.get("status") == "bad_prompt":
                suggestion = ((detail.get("data") or {}).get("prompt_suggestion"))
                if suggestion:
                    print("--- Music Gen (ElevenLabs): Bad prompt flagged. Retrying with suggested prompt.")
                    p_retry = dict(p)
                    p_retry["prompt"] = suggestion
                    return await _attempt_request(p_retry)

            # No usable audio
            raise HTTPException(status_code=r.status_code, detail=f"ElevenLabs music API returned no audio data: {body}")

        music_bytes = await _attempt_request(payload)

    if not music_bytes:
        raise HTTPException(status_code=500, detail="ElevenLabs returned no music data")

    filename = f"music_{prompt.replace(' ', '_')[:20]}.mp3"

    saved_obj = await save_file_and_record(
        data=music_bytes,
        original_filename=filename,
        context="music-generation",
        is_public=True,
        collection_id="ai-generated-music"
    )

    print(f"--- Music Gen (ElevenLabs): Saved music to storage object ID {saved_obj.id}")
    return {
        "id": saved_obj.id,
        "file_url": saved_obj.file_url,
        "audio_url": saved_obj.file_url,
        "storage_object_id": saved_obj.id,
        "format": "mp3"
    }
