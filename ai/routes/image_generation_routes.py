"""
Scene Image Generation Routes
==============================

Generates images for audiobook scenes based on word-level timestamps.

Flow:
1. Takes word_timestamps + text from a produced scene
2. Segments into time windows (configurable interval, e.g. every 5s)
3. Claude Opus generates a visual description per window
4. ComfyUI/Seedream generates the image
5. Returns list of {timestamp, prompt, image_url, storage_id}
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import httpx
import asyncio
import math

logger = logging.getLogger(__name__)
router = APIRouter()


class ImageGenerationRequest(BaseModel):
    """Request to generate images for a scene."""
    # Scene text and timestamps
    narrative: str = Field(description="The spoken text of the scene")
    word_timestamps: List[dict] = Field(description="Word-level timestamps [{word, start, end}]")

    # Style
    visual_style: str = Field(
        default="watercolor painting, soft colors, atmospheric, impressionistic",
        description="Global visual style prompt (project-level)"
    )
    color_palette: Optional[str] = Field(
        default=None,
        description="Optional color palette override (e.g. 'warm earth tones' or 'dark, cold blues')"
    )

    # Interval
    interval_s: float = Field(
        default=5.0, ge=2.0, le=30.0,
        description="Generate one image every N seconds. Scene < interval = 1 image."
    )

    # Image settings
    model: str = Field(default="seedream", description="ComfyUI model: seedream, flux-2-pro, gemini-pro")
    width: int = Field(default=1024)
    height: int = Field(default=768)

    # Scene context for better prompts
    scene_title: Optional[str] = None
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    characters: Optional[str] = None


class GeneratedImage(BaseModel):
    start_s: float
    end_s: float
    spoken_text: str
    image_prompt: str
    image_filename: Optional[str] = None
    image_url: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    total_images: int
    interval_s: float
    scene_duration_s: float
    images: List[GeneratedImage]


def _segment_timestamps(word_timestamps: list, interval_s: float) -> list:
    """Split word timestamps into time windows of interval_s seconds."""
    if not word_timestamps:
        return []

    total_duration = max(w.get("end", 0) for w in word_timestamps) if word_timestamps else 0
    num_images = max(1, math.floor(total_duration / interval_s))

    # If scene is shorter than interval, just one image with all text
    if total_duration <= interval_s:
        all_text = " ".join(w["word"] for w in word_timestamps if w.get("word", "").strip() and w["word"] != "...")
        return [{"start_s": 0, "end_s": total_duration, "text": all_text}]

    segments = []
    for i in range(num_images):
        start = i * interval_s
        end = min((i + 1) * interval_s, total_duration)
        # Collect words in this window
        words = [
            w["word"] for w in word_timestamps
            if w.get("start", 0) >= start and w.get("start", 0) < end
            and w.get("word", "").strip() and w["word"] != "..."
        ]
        if words:
            segments.append({"start_s": round(start, 2), "end_s": round(end, 2), "text": " ".join(words)})

    return segments


@router.post("/generate-images", response_model=ImageGenerationResponse)
async def generate_scene_images(req: ImageGenerationRequest):
    """
    Generate images for an audiobook scene based on word timestamps and interval.

    1. Segments text into time windows
    2. Claude Opus generates image descriptions per window
    3. ComfyUI/Seedream generates the images
    """
    # Step 1: Segment timestamps into windows
    segments = _segment_timestamps(req.word_timestamps, req.interval_s)
    if not segments:
        raise HTTPException(status_code=400, detail="No segments could be derived from timestamps")

    total_duration = max(w.get("end", 0) for w in req.word_timestamps) if req.word_timestamps else 0

    # Step 2: Claude Opus generates visual descriptions for all segments at once
    context_parts = []
    if req.scene_title:
        context_parts.append(f"Scene: {req.scene_title}")
    if req.location:
        context_parts.append(f"Location: {req.location}")
    if req.time_of_day:
        context_parts.append(f"Time: {req.time_of_day}")
    if req.characters:
        context_parts.append(f"Characters: {req.characters}")
    context_str = ". ".join(context_parts) + "." if context_parts else ""

    segments_text = "\n".join(
        f"Window {i+1} ({s['start_s']:.1f}s - {s['end_s']:.1f}s): \"{s['text']}\""
        for i, s in enumerate(segments)
    )

    prompt = (
        f"Generate a short visual image description (1-2 sentences, English) for each time window of this audiobook scene. "
        f"Describe WHAT IS VISIBLE in the scene at that moment — setting, characters, actions, lighting, mood. "
        f"Do NOT describe sounds or dialogue. Focus on visual composition.\n\n"
        f"Context: {context_str}\n"
        f"Full scene text: \"{req.narrative[:500]}\"\n\n"
        f"Time windows:\n{segments_text}\n\n"
        f"Return a JSON array of strings, one description per window. Example: [\"A man stands at a kitchen counter in dim morning light\", \"Close-up of a child's drawing on the table\"]"
    )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "http://localhost:8000/ai/claude",
                json={"prompt": prompt, "system": "Return ONLY a JSON array of strings. No commentary.", "max_tokens": 2000, "model": "opus"},
                params={"api_key": "Inetpass1"}
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = (data.get("response") or data.get("message") or "").strip()
    except Exception as e:
        logger.error(f"Claude Opus image prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI prompt generation failed: {e}")

    # Parse descriptions
    import json
    try:
        cleaned = response_text.replace("```json", "").replace("```", "").strip()
        descriptions = json.loads(cleaned)
        if not isinstance(descriptions, list):
            descriptions = [descriptions]
    except Exception:
        # Fallback: use the raw text split by lines
        descriptions = [line.strip().strip('"').strip("'") for line in response_text.split("\n") if line.strip()]

    # Pad if needed
    while len(descriptions) < len(segments):
        descriptions.append(f"Scene continues: {segments[len(descriptions)]['text'][:80]}")

    # Step 3: Generate images via ComfyUI
    images = []
    for i, seg in enumerate(segments):
        desc = descriptions[i] if i < len(descriptions) else seg["text"]

        # Combine style + content
        style_parts = [req.visual_style]
        if req.color_palette:
            style_parts.append(req.color_palette)
        full_prompt = f"{', '.join(style_parts)}. {desc}"

        logger.info(f"Generating image {i+1}/{len(segments)}: {full_prompt[:80]}...")

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://localhost:8190/api/generate",
                    json={"prompt": full_prompt, "model": req.model, "width": req.width, "height": req.height}
                )
                resp.raise_for_status()
                img_data = resp.json()

            img_list = img_data.get("images", [])
            filename = img_list[0]["filename"] if img_list else None
            backend = img_list[0].get("backend", "arkserver") if img_list else "arkserver"
            image_url = f"https://comfy.arkserver.arkturian.com/api/images/{filename}?backend={backend}" if filename else None

        except Exception as e:
            logger.error(f"ComfyUI generation failed for image {i+1}: {e}")
            filename = None
            image_url = None

        images.append(GeneratedImage(
            start_s=seg["start_s"],
            end_s=seg["end_s"],
            spoken_text=seg["text"],
            image_prompt=full_prompt,
            image_filename=filename,
            image_url=image_url,
        ))

        # Small delay between generations to be nice to APIs
        if i < len(segments) - 1:
            await asyncio.sleep(1)

    return ImageGenerationResponse(
        total_images=len(images),
        interval_s=req.interval_s,
        scene_duration_s=round(total_duration, 2),
        images=images,
    )
