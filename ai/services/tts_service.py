import os
import traceback
import uuid
import shutil
import subprocess
from pathlib import Path
from openai import AsyncOpenAI
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
# ElevenLabs is imported lazily inside functions to avoid hard dependency at import time

# --- Text Chunking Helper ---

def chunk_text(text: str, limit: int) -> List[str]:
    """Splits text into chunks under a certain limit, trying to respect sentence boundaries."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while len(text) > 0:
        if len(text) <= limit:
            chunks.append(text)
            break

        # Find the best place to split within the limit
        split_pos = -1
        # Prefer splitting at newlines, then sentence endings
        for delimiter in ['\n', '.', '?', '!']:
            pos = text.rfind(delimiter, 0, limit)
            if pos != -1:
                split_pos = pos
                break
        
        # If no sentence end, try to split at a space
        if split_pos == -1:
            split_pos = text.rfind(' ', 0, limit)
        
        # If no space, force split at the limit
        if split_pos == -1:
            split_pos = limit - 1
            
        # Ensure the split position is valid
        if split_pos == -1: # Should not happen with the force split logic
             split_pos = min(len(text) -1, limit -1)

        chunks.append(text[:split_pos + 1])
        text = text[split_pos + 1:]

    return [c.strip() for c in chunks if c.strip()]

# --- FFMPEG Audio Concatenation Helper ---

def concatenate_audio_chunks(chunk_paths: List[Path], output_format: str, speed: float = 1.0) -> bytes:
    """Concatenates multiple audio files into one using ffmpeg, applying a speed filter."""
    temp_dir = chunk_paths[0].parent
    output_path = temp_dir / f"output.{output_format}"
    file_list_path = temp_dir / "files.txt"

    # Create a file list for ffmpeg
    with open(file_list_path, "w") as f:
        for path in chunk_paths:
            f.write(f"file '{path.resolve()}'\n")

    # Base command
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(file_list_path.resolve()),
    ]

    # Add audio filter for speed if it's not the default
    if speed != 1.0:
        cmd.extend(["-filter:a", f"atempo={speed}"])
    
    # Add codec and output path
    cmd.extend(["-c:a", "libmp3lame", "-q:a", "2", str(output_path.resolve())])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        final_audio_bytes = output_path.read_bytes()
        return final_audio_bytes
    except subprocess.CalledProcessError as e:
        print(f"--- FFMPEG ERROR: Failed to concatenate audio.")
        print(f"--- FFMPEG STDOUT: {e.stdout}")
        print(f"--- FFMPEG STDERR: {e.stderr}")
        raise e
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


def analyze_audio_level(file_path: Path) -> float:
    """Analyzes an audio file to get its mean volume in dB."""
    try:
        import re
        cmd = [
            "ffmpeg", "-i", str(file_path),
            "-af", "volumedetect", "-f", "null", "-"
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Find the mean_volume line in the stderr output
        for line in result.stderr.splitlines():
            if "mean_volume" in line:
                match = re.search(r"mean_volume:\s*(-?\d+\.\d+)", line)
                if match:
                    return float(match.group(1))
        return 0.0 # Should not be reached if volumedetect works
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError) as e:
        print(f"--- FFMPEG ERROR: Failed to analyze audio level for {file_path}. Error: {e}")
        return 0.0 # Return a default value indicating success/loud enough


def create_silence_chunk(duration_ms: int, output_format: str, temp_dir: Path) -> Path:
    """Creates a short audio file of pure silence using ffmpeg."""
    silence_path = temp_dir / f"silence_{duration_ms}ms.{output_format}"
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", f"{duration_ms / 1000.0}", "-c:a", "libmp3lame", "-q:a", "5",
        str(silence_path.resolve())
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return silence_path
    except subprocess.CalledProcessError as e:
        print(f"--- FFMPEG ERROR: Failed to create silence chunk.")
        print(f"--- FFMPEG STDERR: {e.stderr}")
        raise e

def mix_final_audio(
    dialog_chunks: List[Path],
    music_path: Optional[Path],
    sfx_tracks: List[dict],
    output_format: str,
    chunk_kinds: Optional[List[str]] = None,
    music_delay_ms: Optional[int] = None,
    enable_ducking: bool = True,
    request_id: Optional[str] = None,
    total_dialog_duration_s: Optional[float] = None,
    music_duration_s: Optional[float] = None,
    music_loop: bool = True,
    music_stop_at_end: bool = False
) -> bytes:
    """
    Mixes dialog, music, and sound effects into a final audio file using ffmpeg.
    This version normalizes all inputs before concatenation to prevent errors.
    """
    if not dialog_chunks:
        raise ValueError("Cannot mix audio without dialog chunks.")
        
    temp_dir = dialog_chunks[0].parent
    output_path = temp_dir / f"final_mix.{output_format}"
    
    cmd = ["ffmpeg", "-y"]
    inputs = []
    filter_complex_parts = []
    
    # Add all audio chunks as inputs
    # Add all streams as inputs
    for i, chunk_path in enumerate(dialog_chunks):
        cmd.extend(["-i", str(chunk_path.resolve())])
        inputs.append(f"[{i}:a]")
    
    # Prepare durations for optional fades
    def probe_duration_seconds(path: Path) -> float:
        try:
            import json as _json
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                str(path.resolve())
            ], capture_output=True, text=True, check=True)
            data = _json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0.0))
        except Exception:
            return 0.0

    durations = [probe_duration_seconds(p) for p in dialog_chunks]
    kinds = chunk_kinds or ["dialog"] * len(dialog_chunks)
    has_silence = any((k or "").lower() == "silence" for k in kinds)

    # --- Fast path: dialog + silence only (no music, no sfx) → use concat demuxer ---
    no_music = music_path is None
    no_sfx = not any((k or "").lower() == "sfx" for k in kinds)
    if no_music and no_sfx:
        try:
            # Emit status for observability
            if request_id:
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(request_id, phase="mix", subphase="concat_fast")
                except Exception:
                    pass
            file_list_path = temp_dir / "concat_files.txt"
            with open(file_list_path, "w") as f:
                for p in dialog_chunks:
                    f.write(f"file '{p.resolve()}'\n")
            demux_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(file_list_path.resolve()),
                "-c:a", "libmp3lame", "-q:a", "2", str(output_path.resolve())
            ]
            subprocess.run(demux_cmd, check=True, capture_output=True, text=True)
            if request_id:
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(request_id, phase="mix", subphase="concat_fast_done")
                except Exception:
                    pass
            data = output_path.read_bytes()
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
            return data
        except subprocess.CalledProcessError as e:
            print(f"--- FFMPEG ERROR: concat demuxer path failed. STDERR:\n{e.stderr}")
            if request_id:
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(request_id, phase="mix", subphase="concat_fast_error", error=e.stderr[-400:])
                except Exception:
                    pass
            # Fall through to filter graph fallback below

    # If we have explicit silence chunks in the sequence, preserve them by disabling crossfades and using concat
    if has_silence:
        # Normalize each stream, align timestamps, add tiny SFX tail fade to avoid clicks, then concat
        norm_labels = []
        for i, stream in enumerate(inputs):
            norm = f"[n{i}]"
            chain = (
                f"{stream}aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,"
                f"aresample=async=1:min_hard_comp=0.100000:first_pts=0,asetpts=N/SR/TB"
            )
            # Apply a small fade-out on SFX tails (0.2s), computed from probed duration
            if i < len(kinds) and kinds[i] == "sfx":
                fade_d = 0.20
                st = max((durations[i] if i < len(durations) and durations[i] > 0 else 0.25) - fade_d, 0.0)
                chain += f",afade=t=out:st={st:.3f}:d={fade_d:.2f}"
            chain += norm
            filter_complex_parts.append(chain + ";")
            norm_labels.append(norm)
        filter_complex_parts.append("".join(norm_labels) + f"concat=n={len(norm_labels)}:v=0:a=1[cat];")
        final_dialog_label = "[cat]"
    else:
        # Normalize and optionally fade SFX chunks
        stage_labels = []
        for i, stream in enumerate(inputs):
            base = f"[s{i}]"
            chain = f"{stream}aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo"
            if i < len(kinds) and kinds[i] == "sfx":
                # Apply light fades to SFX: very short in, guaranteed short out (0.2s)
                chain += f",afade=t=in:st=0:d=0.10"
                fade_d = 0.20
                if durations[i] and durations[i] > fade_d:
                    st = max(durations[i] - fade_d, 0.0)
                else:
                    st = 0.0
                chain += f",afade=t=out:st={st:.3f}:d={fade_d:.2f}"
            chain += base
            filter_complex_parts.append(chain + ";")
            stage_labels.append(base)

        # Build crossfade chain
        if len(stage_labels) == 1:
            final_dialog_label = stage_labels[0]
        else:
            prev_label = stage_labels[0]
            for i in range(1, len(stage_labels)):
                # pick crossfade duration that both clips can support, capped conservatively
                d_prev = durations[i-1] if i-1 < len(durations) and durations[i-1] > 0 else 0.5
                d_curr = durations[i] if i < len(durations) and durations[i] > 0 else 0.5
                d_val = min(0.30, d_prev - 0.01, d_curr - 0.01)  # cap at 0.30s
                if d_val < 0.05:
                    d_val = 0.05  # minimal safe crossfade
                out_label = f"[xf{i}]"
                filter_complex_parts.append(f"{prev_label}{stage_labels[i]}acrossfade=d={d_val:.2f}:c1=tri:c2=tri{out_label};")
                prev_label = out_label
            final_dialog_label = prev_label
    
    # --- Music mixing (ensure no hang) ---
    music_input_index = len(inputs)
    if music_path:
        cmd.extend(["-i", str(music_path.resolve())])
        # Normalize music and optionally loop to cover dialog; delay the music to start later; then duck against dialog.
        md = int(music_delay_ms or 0)
        # music_norm -> optional loop/trim -> optional adelay -> normalize -> dynamic volume -> label 'musicw'
        chain_music = f"[{music_input_index}:a]aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo"
        # Decide if we need to loop to cover dialog
        need_loop = False
        try:
            if music_loop and total_dialog_duration_s is not None and music_duration_s is not None and total_dialog_duration_s > music_duration_s + 0.05:
                need_loop = True
        except Exception:
            need_loop = False
        if need_loop:
            # Loop and trim so that after dialog ends the current music loop finishes to its natural end
            # Compute total loops as ceil(dialog_duration / music_duration)
            try:
                import math
                loops = max(1, int(math.ceil(float(total_dialog_duration_s or 0.0) / float(music_duration_s or 0.001))))
            except Exception:
                loops = 1
            if music_stop_at_end:
                # User wants music to stop at dialog end ⇒ do not extend beyond dialog
                loop_dur = float(total_dialog_duration_s or 0.0)
                chain_music += ",aloop=loop=-1:size=2e+09,atrim=0:%0.3f" % max(loop_dur, 0.1)
            else:
                total_music_len = float(max(0.1, loops * float(music_duration_s or 0.0)))
                chain_music += ",aloop=loop=-1:size=2e+09,atrim=0:%0.3f" % total_music_len
        if md > 0:
            chain_music += f",adelay={md}|{md}"
        # Keep PTS from adelay; do not reset after delay
        chain_music += ",aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo"
        # Smooth ramp: keep 0.1 during dialog, ramp to 1.0 over 1.5s after dialog end (in music time)
        ramp = 1.5
        try:
            bump_in_music = float(max(0.0, (total_dialog_duration_s or 0.0) - (md/1000.0)))
        except Exception:
            bump_in_music = 0.0
        chain_music += f",volume='0.1+0.9*if(gte(t,{bump_in_music:.3f}), min(1,(t-{bump_in_music:.3f})/{ramp:.3f}), 0)'[musicw];"
        filter_complex_parts.append(chain_music)
        if enable_ducking:
            # Ensure dialog format matches expected before split
            filter_complex_parts.append(f"{final_dialog_label}aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,aresample=async=1,asetpts=N/SR/TB[dlg_norm];")
            # Split dialog into two paths: one for main mix, one for sidechain key
            filter_complex_parts.append(f"[dlg_norm]asplit=2[dlg_main][dlg_side];")
            # Normalize sidechain branch as well to be safe
            filter_complex_parts.append(f"[dlg_side]aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,aresample=async=1,asetpts=N/SR/TB,apad[dlg_key];")
            # Sidechain compress the music using dialog sidechain
            filter_complex_parts.append(f"[musicw][dlg_key]sidechaincompress=threshold=0.03:ratio=10:attack=5:release=200[ducked];")
            # Ensure compressed music stream is in a stable format for amix
            filter_complex_parts.append(f"[ducked]aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,aresample=async=1,asetpts=N/SR/TB[duckedn];")
            # Mix main dialog with ducked music; choose duration per stop/continue flag
            mix_duration = "shortest" if music_stop_at_end else "longest"
            filter_complex_parts.append(f"[dlg_main][duckedn]amix=inputs=2:duration={mix_duration}[out]")
        else:
            mix_duration = "shortest" if music_stop_at_end else "longest"
            filter_complex_parts.append(f"{final_dialog_label}[musicw]amix=inputs=2:duration={mix_duration}[out]")
        final_output = "[out]"
    else:
        final_output = final_dialog_label

    cmd.extend([
        "-filter_complex", "".join(filter_complex_parts),
        "-map", final_output,
        "-c:a", "libmp3lame", "-q:a", "2", str(output_path.resolve())
    ])

    try:
        # Emit debug status for mix graph
        if request_id:
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(request_id, phase="mix", subphase="graph", inputs=len(dialog_chunks), music=bool(music_path), delay_ms=int(music_delay_ms or 0), ducking=bool(enable_ducking))
            except Exception:
                pass
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if request_id:
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(request_id, phase="mix", subphase="ffmpeg_done")
            except Exception:
                pass
        return output_path.read_bytes()
    except subprocess.CalledProcessError as e:
        print(f"--- FFMPEG ERROR: Crossfade mix failed, falling back to simple concat. STDERR:\n{e.stderr}")
        if request_id:
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(request_id, phase="mix", subphase="ffmpeg_error", error=e.stderr[-400:])
            except Exception:
                pass
        # Fallback: simple concat without crossfades
        try:
            # Build a basic filter to normalize and concat all streams
            fallback_cmd = ["ffmpeg", "-y"]
            for p in dialog_chunks:
                fallback_cmd += ["-i", str(p.resolve())]
            if music_path:
                fallback_cmd += ["-i", str(music_path.resolve())]
            fb_parts = []
            fb_labels = []
            for i in range(len(dialog_chunks)):
                # Add tiny SFX tail fade (0.2s) in fallback too
                chain = (
                    f"[{i}:a]aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,"
                    f"aresample=async=1:min_hard_comp=0.100000:first_pts=0,asetpts=N/SR/TB"
                )
                if i < len(kinds) and kinds[i] == "sfx":
                    fade_d = 0.20
                    dur_i = durations[i] if i < len(durations) else 0.0
                    st = max((dur_i if dur_i > 0 else 0.25) - fade_d, 0.0)
                    chain += f",afade=t=out:st={st:.3f}:d={fade_d:.2f}"
                chain += f"[n{i}];"
                fb_parts.append(chain)
                fb_labels.append(f"[n{i}]")
            fb_parts.append("".join(fb_labels) + f"concat=n={len(fb_labels)}:v=0:a=1[cat];")
            map_label = "[cat]"
            if music_path:
                music_in_idx = len(dialog_chunks)
                # Optional loop in fallback as well
                fb_music = f"[{music_in_idx}:a]aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo"
                if music_loop and total_dialog_duration_s and music_duration_s and total_dialog_duration_s > music_duration_s + 0.05:
                    if music_stop_at_end:
                        fb_music += ",aloop=loop=-1:size=2e+09,atrim=0:%0.3f" % max(float(total_dialog_duration_s), 0.1)
                    else:
                        import math
                        loops = max(1, int(math.ceil(float(total_dialog_duration_s) / float(music_duration_s))))
                        total_len = max(0.1, loops * float(music_duration_s))
                        fb_music += ",aloop=loop=-1:size=2e+09,atrim=0:%0.3f" % total_len
                fb_music += ",aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,aresample=async=1,asetpts=N/SR/TB,volume=0.1[music];"
                fb_parts.append(fb_music)
                mix_duration_fb = "shortest" if music_stop_at_end else "longest"
                fb_parts.append(f"[cat][music]amix=inputs=2:duration={mix_duration_fb}[out]")
                map_label = "[out]"
            fallback_cmd += [
                "-filter_complex", "".join(fb_parts),
                "-map", map_label,
                "-c:a", "libmp3lame", "-q:a", "2", str(output_path.resolve())
            ]
            if request_id:
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(request_id, phase="mix", subphase="fallback_start")
                except Exception:
                    pass
            subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
            if request_id:
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(request_id, phase="mix", subphase="fallback_done")
                except Exception:
                    pass
            return output_path.read_bytes()
        except Exception as e2:
            print(f"--- FFMPEG ERROR: Fallback concat also failed: {e2}")
            raise e
    finally:
        shutil.rmtree(temp_dir)

# --- Pydantic Models for Configuration ---
# These models validate the incoming JSON structure for each provider

class OpenAITTSConfig(BaseModel):
    model: str = Field("tts-1", description="The model to use, e.g., 'tts-1' or 'tts-1-hd'")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = Field("nova", description="The voice to use for the audio")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="The speaking rate, from 0.25 to 4.0")
    output_format: Literal["mp3", "opus", "aac", "flac"] = Field("mp3", description="The output format for the audio")

class GeminiTTSConfig(BaseModel):
    # Note: Gemini API does not have as many options as OpenAI's TTS currently
    # This model is here for future expansion and consistency.
    # The 'voice' and 'language' are part of the main content block in the request.
    model: str = Field("text-embedding-004", description="The model to use for TTS") # Placeholder, actual model is in main.py call
    speed: float = Field(1.0, description="Speaking rate. Not directly supported in Gemini API v1, but kept for compatibility.")

class ElevenLabsTTSConfig(BaseModel):
    model_id: str = Field("eleven_multilingual_v2", description="The model to use, e.g., 'eleven_multilingual_v2'")
    voice_id: str = Field("JBFqnCBsd6RMkjVDRZzb", description="The ID of the voice to use")
    stability: float = Field(0.5, ge=0.0, le=1.0, description="Voice stability, from 0.0 to 1.0")
    clarity: float = Field(0.75, ge=0.0, le=1.0, description="Voice clarity/similarity, from 0.0 to 1.0")


# --- Service Functions ---

async def generate_openai_tts(text: str, config: OpenAITTSConfig) -> bytes:
    """
    Generates audio from text using OpenAI's Text-to-Speech API.
    Handles long texts by chunking and concatenating the audio.
    """
    try:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # OpenAI's limit is 4096 characters
        text_chunks = chunk_text(text, 4000)
        
        if len(text_chunks) == 1:
            # Process as a single chunk if text is short
            response = await client.audio.speech.create(
                model=config.model,
                voice=config.voice,
                input=text_chunks[0],
                speed=config.speed,
                response_format=config.output_format
            )
            return await response.aread()

        # --- Handle multiple chunks ---
        print(f"--- OpenAI TTS: Text too long, processing in {len(text_chunks)} chunks.")
        temp_dir = Path(f"/tmp/tts_chunks_{uuid.uuid4()}")
        temp_dir.mkdir()
        audio_chunk_paths = []

        for i, chunk in enumerate(text_chunks):
            response = await client.audio.speech.create(
                model=config.model,
                voice=config.voice,
                input=chunk,
                speed=config.speed,
                response_format=config.output_format
            )
            chunk_path = temp_dir / f"chunk_{i}.{config.output_format}"
            response.stream_to_file(chunk_path)
            audio_chunk_paths.append(chunk_path)
            print(f"--- OpenAI TTS: Generated chunk {i+1}/{len(text_chunks)}")

        return concatenate_audio_chunks(audio_chunk_paths, config.output_format, config.speed)

    except Exception as e:
        # IMPROVED DEBUGGING: Print the full traceback for detailed error analysis
        print(f"--- ERROR [OpenAI TTS]: Failed to generate audio. Full traceback:")
        traceback.print_exc()
        # Re-raise the exception to be caught by the endpoint handler
        raise e


async def generate_elevenlabs_tts(text: str, config: ElevenLabsTTSConfig, with_timestamps: bool = False):
    """
    Generates audio from text using ElevenLabs Text-to-Speech API.
    Handles long texts by chunking and concatenating the audio.

    If with_timestamps=True, returns (audio_bytes, word_timestamps) where
    word_timestamps is a list of {"word": str, "start": float, "end": float}.
    Otherwise returns just audio_bytes (backward compatible).
    """
    try:
        import httpx
        import base64

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured")

        text_chunks = chunk_text(text, 4500)
        all_audio = b""
        all_words = []
        time_offset = 0.0

        for i, chunk_text_str in enumerate(text_chunks):
            if with_timestamps:
                # Use REST API with-timestamps endpoint directly
                async with httpx.AsyncClient(timeout=60) as http:
                    resp = await http.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{config.voice_id}/with-timestamps",
                        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                        json={
                            "text": chunk_text_str,
                            "model_id": config.model_id,
                            "voice_settings": {"stability": config.stability, "similarity_boost": config.clarity}
                        }
                    )
                    resp.raise_for_status()
                    data = resp.json()

                audio_b64 = data.get("audio_base64", "")
                chunk_audio = base64.b64decode(audio_b64)
                all_audio += chunk_audio

                # Parse character-level alignment into word-level timestamps
                alignment = data.get("alignment") or {}
                chars = alignment.get("characters", [])
                starts = alignment.get("character_start_times_seconds", [])
                ends = alignment.get("character_end_times_seconds", [])

                word = ""
                word_start = 0.0
                for ci, c in enumerate(chars):
                    if c == " " or ci == len(chars) - 1:
                        if ci == len(chars) - 1 and c != " ":
                            word += c
                        if word:
                            word_end = ends[ci - 1] if c == " " else ends[ci]
                            all_words.append({
                                "word": word,
                                "start": round(time_offset + word_start, 3),
                                "end": round(time_offset + word_end, 3)
                            })
                        word = ""
                        if ci + 1 < len(starts):
                            word_start = starts[ci + 1]
                    else:
                        if not word:
                            word_start = starts[ci]
                        word += c

                # Calculate chunk duration for offset of next chunk
                if chunk_audio:
                    chunk_dur = probe_audio_duration(chunk_audio)
                    time_offset += chunk_dur

                print(f"--- ElevenLabs TTS: Chunk {i+1}/{len(text_chunks)} with timestamps ({len(all_words)} words)")
            else:
                # Legacy path: streaming without timestamps
                try:
                    from elevenlabs.client import AsyncElevenLabs
                except ModuleNotFoundError as e:
                    raise RuntimeError("ElevenLabs support requires the 'elevenlabs' package.") from e
                client = AsyncElevenLabs(api_key=api_key)
                audio_stream = client.text_to_speech.convert(
                    text=chunk_text_str,
                    voice_id=config.voice_id,
                    model_id=config.model_id,
                    voice_settings={"stability": config.stability, "similarity_boost": config.clarity}
                )
                chunk_audio = b""
                async for audio_chunk in audio_stream:
                    chunk_audio += audio_chunk
                all_audio += chunk_audio
                print(f"--- ElevenLabs TTS: Chunk {i+1}/{len(text_chunks)}")

        if with_timestamps:
            return all_audio, all_words
        return all_audio

    except Exception as e:
        print(f"--- ERROR [ElevenLabs TTS]: Failed to generate audio. Full traceback:")
        traceback.print_exc()
        raise e


def probe_audio_duration(audio_bytes: bytes) -> float:
    """Get duration of audio bytes using ffprobe."""
    try:
        import subprocess, tempfile, json as _json
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", tmp],
            capture_output=True, text=True
        )
        os.unlink(tmp)
        return float(_json.loads(result.stdout).get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0


async def generate_gemini_tts(text: str, language: str, voice: str, speed: float) -> bytes:
    """
    Generates audio from text using Google Gemini's TTS capabilities.
    
    Note: The Gemini API for TTS is simpler. Voice and language are top-level
    parameters in the content generation request, not separate config objects.
    """
    try:
        # Configure the client with the API key
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        # The Gemini API uses a specific model for TTS generation.
        # The actual generation happens via the standard `generate_content` method
        # with a special configuration. This function just prepares the call.
        # The actual API call is better placed in main.py to follow the existing pattern.
        # This function will therefore be a placeholder for the logic that will be
        # integrated into the main endpoint.
        
        # This is a simplified representation. The actual implementation will be
        # in the main.py endpoint to match the project's style.
        
        # For now, we will simulate the API call structure.
        model = genai.GenerativeModel(model_name="gemini-2.5-flash") # Or another suitable model
        
        # This part of the code is complex and better suited for the main endpoint logic.
        # We will return a placeholder and implement the full logic in the next step in main.py
        
        # This is a conceptual placeholder.
        print(f"--- INFO [Gemini TTS]: Simulating generation for text: '{text[:50]}...'")
        # In the real implementation in main.py, we will make the actual API call.
        # For now, returning empty bytes to signify success in this stage.
        
        # The actual API call will look something like this (and will be in main.py):
        # response = await model.generate_content_async(
        #     contents=[{"parts": [{"text": text}]}],
        #     generation_config={"response_mime_type": "audio/mpeg"} # Simplified
        # )
        # return response.parts[0].blob.data
        
        # This service function's role is to abstract the logic.
        # The final implementation will be slightly different and live in main.py
        # to match the project's architecture.
        
        # Let's put the actual logic here for now.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
            
        # The Gemini TTS API is not directly exposed via the Python SDK in the same way as OpenAI.
        # It's part of the multi-modal generation. The logic is better suited inside the endpoint.
        # This function will therefore be a placeholder.
        
        # Let's raise a NotImplementedError to make it clear this needs to be done in the endpoint.
        raise NotImplementedError("Gemini TTS logic should be implemented directly in the main.py endpoint to match existing patterns.")

    except Exception as e:
        print(f"--- ERROR [Gemini TTS]: Failed to generate audio. Error: {e}")
        raise e

