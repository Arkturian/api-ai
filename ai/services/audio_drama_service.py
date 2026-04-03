import json
import subprocess
from pathlib import Path
import uuid
import httpx
from fastapi import HTTPException
import google.generativeai as genai

from ai.services.speech_service import SpeechGenerator
import time
from ai.services import tts_service
from ai.services import audio_sourcing_service

from ai.services.tts_models import SpeechRequest

from fastapi.encoders import jsonable_encoder
import asyncio

class AudioDramaGenerator(SpeechGenerator):
    def __init__(self, request: SpeechRequest, api_key: str, image_gen_func, db_session=None):
        super().__init__(request, api_key, image_gen_func, db_session)
        # Cache to ensure different speakers use distinct voices by default
        self._speaker_assigned_voice: dict[str, str] = {}
        self._voices_used: set[str] = set()
        # OpenAI available voices (as used elsewhere in the app)
        self._all_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def _normalize_gender_label(self, raw: str) -> str:
        if not raw:
            return 'ai'
        g = str(raw).strip().lower()
        male_syn = {"m", "male", "mann", "männlich", "masculine", "masc", "herr"}
        female_syn = {"f", "female", "frau", "weiblich", "fem", "feminine", "dame"}
        narrator_syn = {"narrator", "erzähler", "erzählerin", "speaker", "voiceover", "voice-over"}
        ai_syn = {"ai", "robot", "maschine", "künstliche intelligenz"}
        if g in male_syn:
            return 'male'
        if g in female_syn:
            return 'female'
        if g in narrator_syn:
            return 'narrator'
        if g in ai_syn:
            return 'ai'
        if any(k in g for k in ["mann", "männ", "male", "masc"]):
            return 'male'
        if any(k in g for k in ["frau", "weib", "female", "fem"]):
            return 'female'
        if "narr" in g:
            return 'narrator'
        return 'ai'

    def _pick_distinct_voice(self, gender: str) -> str:
        # Prefer gender-appropriate ordering but allow fallbacks to keep distinctness
        male_pref = ["onyx", "alloy", "echo", "fable", "shimmer", "nova"]
        female_pref = ["nova", "shimmer", "fable", "echo", "alloy", "onyx"]
        ai_gender = self.request.config.ai_gender if hasattr(self.request.config, 'ai_gender') else 'female'
        ai_pref = male_pref if ai_gender == 'male' else female_pref

        order = female_pref if gender == 'female' else male_pref if gender == 'male' else ai_pref

        # First try a not-yet-used voice
        for v in order:
            if v in self._all_voices and v not in self._voices_used:
                self._voices_used.add(v)
                return v
        # If all preferred have been used, return the first available (will cause reuse)
        fallback = order[0] if order else self._all_voices[0]
        self._voices_used.add(fallback)
        return fallback

    async def generate(self):
        start_overall = time.time()
        print(f"DIALOG[{self.request.id}]: generate() start analyze_only={getattr(self.request.config, 'analyze_only', False)}")
        production_plan = await self._analyze_script()
        
        # New: allow analyze-only mode for UI confirmation in ai.php
        if getattr(self.request.config, 'analyze_only', False):
            return None, production_plan, None

        t0 = time.time()
        audio_bytes = await self._produce_audio_drama(production_plan)
        print(f"DIALOG[{self.request.id}]: _produce_audio_drama done in {int((time.time()-t0)*1000)}ms")
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Audio drama production failed.")

        saved_obj = await self._save_audio(audio_bytes)
        generated_image_obj = None
        if self.request.config.generate_title_image:
            generated_image_obj = await self.generate_title_image()
        print(f"DIALOG[{self.request.id}]: generate() finished in {int((time.time()-start_overall)*1000)}ms")
        return saved_obj, production_plan, generated_image_obj

    async def _analyze_script(self):
        # Allow UI to provide an override analysis result to skip Gemini
        try:
            override = getattr(self.request.config, 'analysis_override', None)
        except Exception:
            override = None
        if isinstance(override, dict) and override.get('production_cues') is not None:
            print(f"DIALOG[{self.request.id}]: Using analysis_override from request (skipping Gemini)")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="analyze", subphase="override_used")
            except Exception:
                pass
            return override
        print(f"DIALOG[{self.request.id}]: Analyzing script with Gemini…")
        try:
            from ai.routes.dialog_routes import set_dialog_status
            set_dialog_status(self.request.id, phase="analyze", subphase="start")
        except Exception:
            pass
        t_start = time.time()
        
        # Load the example structure from the new file
        try:
            with open("prompt_example.json", "r") as f:
                example_json = json.load(f)
            example_str = json.dumps(example_json, indent=2)
        except (FileNotFoundError, json.JSONDecodeError):
            example_str = "{...}" # Fallback

        # Define the base prompt directly in the code
        lang = getattr(self.request.content, 'language', None) or 'original language of the input'
        # Only request a generated music prompt if add_music is true AND no manual storage id is provided
        music_requested = bool(getattr(self.request.config, 'add_music', False)) and not bool(getattr(self.request.config, 'manual_music_storage_id', None))

        sfx_requested = bool(getattr(self.request.config, 'add_sfx', False))
        user_hint = getattr(self.request.config, 'analysis_user_hint', None)
        user_hint_text = f"\nUSER_HINT (optional, do not change schema; consider only as guidance):\n{user_hint}\n" if user_hint else "\n"
        base_prompt = (
            "You are a creative audio drama director. Your task is to analyze the following script and prepare it for production. "
            "1. Correct the text for spelling, grammar, and natural expression, keeping it in the original language. "
            f"Ensure the output remains strictly in the original language of the input (language: {lang}). Do NOT translate any content. Return dialog text exactly in that language. "
            "2. Identify all speakers and background elements described in square brackets (e.g., [Sound of wind]). "
            "3. Determine the likely gender or type for each speaker (male, female, ai, narrator). "
            "4. For each line of dialog, determine if a specific `voice_style` (e.g., 'whispering', 'shouting') is implied by the context. "
            f"5. Background music requested: {music_requested}. If and only if true, generate a detailed English music prompt suitable for a ~30s background track (genre, mood, tempo/BPM, instrumentation, energy, vocals=none). "
            "6. Add timing guidance for production: include optional 'pause_before_ms' and 'pause_after_ms' for dialog cues (integers, milliseconds). Keep pauses short and natural: typical pause_before_ms 0-300ms, typical pause_after_ms 100-400ms. Only use longer pauses (up to 800ms) for dramatic scene breaks or significant mood shifts. Default to 0 if no pause is needed. For any music cue provide 'length_ms' and 'start_offset_ms' (milliseconds from start before music begins), and optionally 'intro_pause_ms' (milliseconds pause before first dialog). "
            "7. Structure the output as a single JSON object with two keys: 'production_cues' and 'music'. "
            "- 'production_cues': A single list containing all dialog and sfx cues in the correct order they appear in the script. "
            "  - Dialog cues should be objects with 'type': 'dialog', 'speaker', 'gender', 'voice_style' (optional), 'text', and optional 'pause_before_ms'/'pause_after_ms'. The 'text' MUST be in the same language as the input. "
            f"  - SFX cues should be objects with 'type': 'sfx' and a short 'description' in English. Only include SFX cues if requested: {sfx_requested}. If false, include none. If true, infer 1-6 SFX cues even if not explicitly bracketed whenever they enhance immersion (e.g., door knock, phone ring, footsteps, ambient wind). Do NOT return zero SFX when requested; include at least one relevant SFX. Keep descriptions concise and production-ready. "
            "- 'music': A list that may be empty or contain one object with keys: 'description' (music prompt in English), 'length_ms' (e.g., 30000), 'start_offset_ms' (e.g., 0), and 'intro_pause_ms' (e.g., 1000)."
            + user_hint_text +
            "IMPORTANT: The JSON schema is FIXED. Only return the specified keys. Do not add extraneous commentary."
        )

        analysis_prompt = f"{base_prompt}\n\n{example_str}\n\nTEXT TO ANALYZE:\n{self.request.content.text}"
        try:
            prompt_size = len(analysis_prompt.encode('utf-8'))
        except Exception:
            prompt_size = len(analysis_prompt)
        print(f"DIALOG[{self.request.id}]: Gemini prompt bytes={prompt_size}")
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        print(f"DIALOG[{self.request.id}]: Calling gemini.generate_content (thread)…")
        try:
            # Use blocking generate_content off the event loop to avoid SDK async incompatibilities
            response = await asyncio.to_thread(model.generate_content, analysis_prompt)
            print(f"DIALOG[{self.request.id}]: Gemini responded in {int((time.time()-t_start)*1000)}ms")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="analyze", subphase="gemini_done", duration_ms=int((time.time()-t_start)*1000))
            except Exception:
                pass
        except Exception as e:
            print(f"DIALOG[{self.request.id}][ERROR]: Gemini call failed after {int((time.time()-t_start)*1000)}ms -> {e}")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="analyze", subphase="gemini_error", error=str(e))
            except Exception:
                pass
            raise
        
        try:
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_text)
            # Enforce SFX off if not requested
            if not sfx_requested:
                try:
                    cues = result.get('production_cues', [])
                    result['production_cues'] = [c for c in cues if (c.get('type') or '').lower() != 'sfx']
                except Exception:
                    pass
            else:
                # If SFX requested but model returned none, inject a safe default ambient suggestion at the start
                try:
                    cues = result.get('production_cues', []) or []
                    sfx_count_now = sum(1 for c in cues if (c.get('type') or '').lower() == 'sfx')
                    if sfx_count_now == 0:
                        default_sfx = {"type": "sfx", "description": "Subtle ambient room tone, low noise floor"}
                        # Insert after any initial silence or at position 0
                        insert_idx = 0
                        result['production_cues'] = cues[:insert_idx] + [default_sfx] + cues[insert_idx:]
                        try:
                            from ai.routes.dialog_routes import set_dialog_status
                            set_dialog_status(self.request.id, phase="analyze", subphase="sfx_injected", reason="none_from_ai")
                        except Exception:
                            pass
                except Exception:
                    pass
            # Emit counts for observability
            try:
                from ai.routes.dialog_routes import set_dialog_status
                cues = result.get('production_cues', [])
                sfx_count = sum(1 for c in cues if (c.get('type') or '').lower() == 'sfx')
                set_dialog_status(self.request.id, phase="analyze", subphase="parsed", cues=len(cues), sfx_count=int(sfx_count))
            except Exception:
                pass
            print(f"DIALOG[{self.request.id}]: Parsed analysis ok in {int((time.time()-t_start)*1000)}ms; cues={len(result.get('production_cues', []))}")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="analyze", subphase="parsed", cues=len(result.get('production_cues', [])))
            except Exception:
                pass

            # If manual music is selected, add it to the analysis result for UI display
            manual_music_id = getattr(self.request.config, 'manual_music_storage_id', None)
            if manual_music_id:
                # Add manual music info to the music array
                if 'music' not in result or not result['music']:
                    result['music'] = []
                result['music'].append({
                    "type": "manual",
                    "storage_id": manual_music_id,
                    "description": f"Manual music selection (Storage ID: {manual_music_id})",
                    "length_ms": 30000,  # Default, will be adjusted based on actual file
                    "start_offset_ms": 0,
                    "intro_pause_ms": 1000
                })
                print(f"DIALOG[{self.request.id}]: Added manual music ID {manual_music_id} to analysis result")

            return result
        except (json.JSONDecodeError, ValueError) as e:
            print(f"DIALOG[{self.request.id}][ERROR]: Failed to parse Gemini response: {e}")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="analyze", subphase="parse_error", error=str(e))
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to parse production plan from Gemini. Raw response: {response.text}. Error: {e}")


    async def _produce_audio_drama(self, production_plan):
        production_cues = production_plan.get("production_cues", [])
        music_cues = production_plan.get("music", [])

        final_sequence_paths = []
        final_sequence_kinds = []  # 'dialog' or 'sfx' or 'silence'
        sequence_meta = []
        sourced_sfx = []
        
        # Generate/Source all audio chunks based on the production cues
        print(f"DIALOG[{self.request.id}]: Producing audio for {len(production_cues)} cues; music_requested={self.request.config.add_music}")
        try:
            from ai.routes.dialog_routes import set_dialog_status
            set_dialog_status(self.request.id, phase="generate", subphase="start", total_cues=len(production_cues), music=self.request.config.add_music)
        except Exception:
            pass
        # Initialize timing controls and adopt any AI suggestions from analysis
        music_delay_ms = 0
        intro_pause_ms = 0
        if self.request.config.add_music and music_cues and isinstance(music_cues, list) and len(music_cues) > 0:
            try:
                music_delay_ms = int(music_cues[0].get('start_offset_ms') or 0)
            except Exception:
                music_delay_ms = 0
            try:
                intro_pause_ms = int(music_cues[0].get('intro_pause_ms') or 0)
            except Exception:
                intro_pause_ms = 0
        # Default intro pause if music enabled but AI did not propose one
        if self.request.config.add_music and intro_pause_ms <= 0:
            intro_pause_ms = 1000
        first_dialog_emitted = False
        for cue in production_cues:
            if cue.get('type') == 'dialog':
                t_seg = time.time()
                # Optional pauses
                before_ms = int(cue.get('pause_before_ms') or 0)
                after_ms = int(cue.get('pause_after_ms') or 0)
                # Inject intro pause only once, at very first dialog line
                if not first_dialog_emitted and intro_pause_ms > 0:
                    before_ms = before_ms + intro_pause_ms
                    first_dialog_emitted = True
                # Insert leading pause if requested
                if before_ms > 0:
                    silence = await asyncio.to_thread(tts_service.create_silence_chunk, before_ms, self.request.config.output_format, self.temp_dir)
                    final_sequence_paths.append(silence)
                    final_sequence_kinds.append('silence')
                    sequence_meta.append({"type":"silence","duration_ms":before_ms,"reason":"pause_before"})

                chunk_path = await self._generate_single_dialog_chunk(cue)
                final_sequence_paths.append(chunk_path)
                final_sequence_kinds.append('dialog')
                sequence_meta.append({
                    "type":"dialog",
                    "speaker": cue.get("speaker"),
                    "voice_style": cue.get("voice_style"),
                    "text": cue.get("text"),
                    "chosen_voice": cue.get("chosen_voice")
                })
                # Insert trailing pause if requested
                if after_ms > 0:
                    silence = await asyncio.to_thread(tts_service.create_silence_chunk, after_ms, self.request.config.output_format, self.temp_dir)
                    final_sequence_paths.append(silence)
                    final_sequence_kinds.append('silence')
                    sequence_meta.append({"type":"silence","duration_ms":after_ms,"reason":"pause_after"})
                print(f"DIALOG[{self.request.id}]: Dialog chunk ready in {int((time.time()-t_seg)*1000)}ms")
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(
                        self.request.id,
                        phase="generate",
                        subphase="dialog_done",
                        last_duration_ms=int((time.time()-t_seg)*1000),
                        chosen_voice=cue.get('chosen_voice'),
                        gender=cue.get('gender'),
                        speaker=cue.get('speaker')
                    )
                except Exception:
                    pass
            elif cue.get('type') == 'sfx':
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(self.request.id, phase="generate", subphase="sfx_start", description=cue.get('description'))
                except Exception:
                    pass
                sfx_storage_obj = await self._source_single_sfx(cue)
                if sfx_storage_obj:
                    # Download the saved SFX to a temporary path for mixing
                    async with httpx.AsyncClient() as client:
                        r = await client.get(sfx_storage_obj.file_url)
                    temp_sfx_path = self.temp_dir / sfx_storage_obj.original_filename
                    temp_sfx_path.write_bytes(r.content)
                    
                    final_sequence_paths.append(temp_sfx_path)
                    final_sequence_kinds.append('sfx')
                    sourced_sfx.append(jsonable_encoder(sfx_storage_obj))
                    sequence_meta.append({
                        "type":"sfx",
                        "description": cue.get("description",""),
                        "storage_object_id": getattr(sfx_storage_obj, 'id', None)
                    })
                    try:
                        from ai.routes.dialog_routes import set_dialog_status
                        set_dialog_status(self.request.id, phase="generate", subphase="sfx_done", sfx_id=getattr(sfx_storage_obj,'id',None))
                    except Exception:
                        pass

        # Add the sourced SFX to the production plan for debugging
        production_plan['sourced_sfx'] = sourced_sfx

        # Honor manual music selection by forcing add_music if an ID is provided
        music_path = None
        manual_id = getattr(self.request.config, 'manual_music_storage_id', None)
        if manual_id and not getattr(self.request.config, 'add_music', False):
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="music_manual_flag_fix")
            except Exception:
                pass
            self.request.config.add_music = True
        if self.request.config.add_music:
            # Manual override: use existing storage object(s) for music if provided
            if manual_id:
                try:
                    print(f"--- Audio Drama: Using manual music storage id {manual_id}...")
                    try:
                        from ai.routes.dialog_routes import set_dialog_status
                        set_dialog_status(self.request.id, phase="generate", subphase="music_manual", storage_id=str(manual_id))
                    except Exception:
                        pass
                    import os as _os
                    # Support multiple IDs separated by semicolons
                    id_list = [s.strip() for s in str(manual_id).split(';') if s.strip()]
                    downloaded_paths: list[Path] = []

                    # Fetch storage objects via HTTP Storage API
                    STORAGE_API_URL = _os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
                    STORAGE_API_KEY = _os.getenv("STORAGE_API_KEY", "Inetpass1")

                    async with httpx.AsyncClient() as client:
                        for idx, mid in enumerate(id_list):
                            # Get storage object info from Storage API
                            try:
                                storage_resp = await client.get(
                                    f"{STORAGE_API_URL}/storage/objects/{mid}",
                                    headers={"X-API-KEY": STORAGE_API_KEY}
                                )
                                storage_resp.raise_for_status()
                                obj_data = storage_resp.json()
                                file_url = obj_data.get('file_url')

                                if not file_url:
                                    raise HTTPException(status_code=404, detail=f"Storage object {mid} has no file_url")
                            except Exception as e:
                                raise HTTPException(status_code=404, detail=f"Storage object {mid} not found: {str(e)}")
                            # HEAD debug
                            try:
                                hr = await client.head(file_url, follow_redirects=True)
                                try:
                                    from ai.routes.dialog_routes import set_dialog_status
                                    set_dialog_status(self.request.id, phase="generate", subphase="music_manual_head", status_code=hr.status_code, content_type=hr.headers.get('content-type'), content_length=int(hr.headers.get('content-length') or -1))
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            # GET download
                            r = await client.get(file_url, follow_redirects=True)
                            r.raise_for_status()
                            p = self.temp_dir / f"manual_music_{idx}.mp3"
                            p.write_bytes(r.content)
                            downloaded_paths.append(p)

                    # If multiple, concatenate into a single track so the mixer can loop the combined program
                    if len(downloaded_paths) > 1:
                        files_txt = self.temp_dir / "music_files.txt"
                        with open(files_txt, 'w') as f:
                            for p in downloaded_paths:
                                f.write(f"file '{str(p.resolve())}'\n")
                        combined = self.temp_dir / "manual_music_combined.mp3"
                        cmd = [
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(files_txt.resolve()),
                            "-c:a", "libmp3lame", "-q:a", "2", str(combined.resolve())
                        ]
                        try:
                            subprocess.run(cmd, check=True, capture_output=True, text=True)
                            music_path = combined
                            try:
                                from ai.routes.dialog_routes import set_dialog_status
                                set_dialog_status(self.request.id, phase="generate", subphase="music_manual_multi_done", count=len(downloaded_paths))
                            except Exception:
                                pass
                        except subprocess.CalledProcessError as e:
                            print(f"--- Audio Drama: Failed to concatenate manual music tracks: {e.stderr}")
                            # Fallback to first track
                            music_path = downloaded_paths[0]
                    else:
                        music_path = downloaded_paths[0]

                    # Probe quick duration for debug
                    def _probe_dur(p: Path) -> float:
                        try:
                            import json as __json
                            res = subprocess.run([
                                "ffprobe","-v","quiet","-print_format","json","-show_format", str(p.resolve())
                            ], capture_output=True, text=True, check=True)
                            data = __json.loads(res.stdout)
                            return float(data.get('format',{}).get('duration',0.0))
                        except Exception:
                            return 0.0
                    try:
                        dur = _probe_dur(music_path)
                        from ai.routes.dialog_routes import set_dialog_status
                        set_dialog_status(self.request.id, phase="generate", subphase="music_manual_downloaded", bytes=int(music_path.stat().st_size), duration_s=round(dur,3))
                    except Exception:
                        pass
                except Exception as e:
                    print(f"--- Audio Drama: Failed to use manual music id {manual_id}: {e}. Falling back to generated music if available.")
                    try:
                        from ai.routes.dialog_routes import set_dialog_status
                        set_dialog_status(self.request.id, phase="generate", subphase="music_manual_error", error=str(e))
                    except Exception:
                        pass

            # If analyzer provided a cue, use it; otherwise fallback to a gentle ambient prompt
            if music_path is None:
                if music_cues and isinstance(music_cues, list) and len(music_cues) > 0 and music_cues[0].get('description'):
                    try:
                        from ai.routes.dialog_routes import set_dialog_status
                        set_dialog_status(self.request.id, phase="generate", subphase="music_start", description=music_cues[0].get('description'))
                    except Exception:
                        pass
                    music_path = await self._source_music(music_cues)
                else:
                    print("--- Audio Drama: No music cue provided by analysis; generating fallback ambient music...")
                    fallback_cue = [{
                        'description': 'Gentle ambient instrumental background, soft piano and warm pads, calm, 70 BPM, no vocals',
                        'length_ms': 30000,
                        'time': 'full_duration'
                    }]
                    try:
                        from ai.routes.dialog_routes import set_dialog_status
                        set_dialog_status(self.request.id, phase="generate", subphase="music_start", description=fallback_cue[0]['description'])
                    except Exception:
                        pass
                    music_path = await self._source_music(fallback_cue)
        # If we have music: smart intro alignment based on music's own leading silence
        def _probe_music_leading_silence_seconds(path: Path) -> float:
            try:
                # Use silencedetect at -40dB; detect leading silence end
                cmd = [
                    "ffmpeg","-hide_banner","-nostats","-i", str(path.resolve()),
                    "-af","silencedetect=n=-40dB:d=0.2","-f","null","-"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                # Parse first silence_end that occurs from 0
                import re
                silence_end = None
                for line in result.stderr.splitlines():
                    if "silence_start: 0" in line:
                        # subsequent lines may contain silence_end
                        continue
                    m = re.search(r"silence_end:\s*([0-9.]+)", line)
                    if m:
                        try:
                            val = float(m.group(1))
                            silence_end = val
                            break
                        except Exception:
                            pass
                return float(silence_end or 0.0)
            except Exception:
                return 0.0

        if music_path:
            try:
                leading_silence_s = _probe_music_leading_silence_seconds(music_path)
                total_intro_ms_needed = int((music_delay_ms or 0)) + int(leading_silence_s * 1000) + int(intro_pause_ms or 0)
                # Compute current intro silence before the first dialog in our sequence
                current_intro_ms = 0
                for i, kind in enumerate(final_sequence_kinds):
                    if kind == 'silence':
                        try:
                            # sequence_meta mirrors paths one-to-one
                            meta = sequence_meta[i]
                            current_intro_ms += int(meta.get('duration_ms') or 0)
                            continue
                        except Exception:
                            break
                    break # first non-silence encountered
                add_ms = max(0, total_intro_ms_needed - current_intro_ms)
                if add_ms > 0:
                    # Prepend an extra silence chunk so audible music -> dialog gap equals intro_pause_ms
                    extra = await asyncio.to_thread(tts_service.create_silence_chunk, add_ms, self.request.config.output_format, self.temp_dir)
                    final_sequence_paths.insert(0, extra)
                    final_sequence_kinds.insert(0, 'silence')
                    sequence_meta.insert(0, {"type":"silence","duration_ms":add_ms, "reason":"intro_align"})
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(self.request.id, phase="generate", subphase="intro_align", leading_silence_ms=int(leading_silence_s*1000), add_ms=int(add_ms))
                except Exception:
                    pass
            except Exception:
                pass

        print(f"DIALOG[{self.request.id}]: Timeline build & mix starting…")
        try:
            from ai.routes.dialog_routes import set_dialog_status
            set_dialog_status(self.request.id, phase="mix", subphase="start")
        except Exception:
            pass

        # --- Build timeline summary before mixing ---
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

        durations = [probe_duration_seconds(p) for p in final_sequence_paths]
        start_times = []
        current_start = 0.0
        for i in range(len(final_sequence_paths)):
            if i == 0:
                start_times.append(0.0)
                continue
            # compute crossfade similar to tts_service
            d_prev = durations[i-1] if durations[i-1] > 0 else 0.5
            d_curr = durations[i] if durations[i] > 0 else 0.5
            d_val = min(0.30, d_prev - 0.01, d_curr - 0.01)
            if d_val < 0.05:
                d_val = 0.05
            # Only apply crossfade between non-silence; for silence, no overlap
            if final_sequence_kinds[i-1] == 'silence' or final_sequence_kinds[i] == 'silence':
                d_val = 0.0
            current_start = start_times[-1] + max(durations[i-1], 0.0) - d_val
            if current_start < 0:
                current_start = 0.0
            start_times.append(current_start)

        mix_timeline = []
        for i, meta in enumerate(sequence_meta):
            item = {
                "index": i,
                "kind": final_sequence_kinds[i],
                "start_s": round(start_times[i], 3),
                "duration_s": round(durations[i], 3)
            }
            item.update(meta)
            mix_timeline.append(item)

        # Compute precise total dialog stream duration (including silences)
        total_dialog_duration = 0.0
        if len(durations) == len(final_sequence_kinds):
            # Sum all clip durations regardless of kind because silences are included as audio
            total_dialog_duration = float(sum(durations))
        # Include music track as separate timeline layer if present
        if music_path:
            music_duration = probe_duration_seconds(music_path)
            mix_timeline.append({
                "index": len(mix_timeline),
                "kind": "music",
                "label": "music",
                "start_s": round((music_delay_ms or 0)/1000.0, 3),
                "duration_s": round(music_duration if music_duration > 0 else (music_cues[0].get('length_ms',30000)/1000.0 if music_cues else 30.0), 3)
            })

        # attach timeline into production plan for frontend visualization
        production_plan['mix_timeline'] = mix_timeline
        # register temp dialog/sfx/music chunks for UI preview
        try:
            from ai.routes.dialog_routes import register_temp_dialog_chunks
            registry = []
            for i in range(len(final_sequence_paths)):
                kind = final_sequence_kinds[i]
                p = final_sequence_paths[i]
                registry.append({
                    'index': i,
                    'kind': kind,
                    'path': str(p.resolve()),
                    'start_s': start_times[i] if i < len(start_times) else 0.0,
                    'duration_s': durations[i] if i < len(durations) else 0.0,
                    'meta': sequence_meta[i] if i < len(sequence_meta) else {}
                })
            if music_path:
                registry.append({
                    'index': len(registry),
                    'kind': 'music',
                    'path': str(music_path.resolve()),
                    'start_s': round((music_delay_ms or 0)/1000.0, 3),
                    'duration_s': probe_duration_seconds(music_path),
                    'meta': {'label':'music'}
                })
            register_temp_dialog_chunks(self.request.id, registry)
            print(f"--- Audio Drama: Registered {len(registry)} chunks for {self.request.id}")
        except Exception as e:
            print(f"--- Audio Drama: Failed to register chunks: {e}")

        try:
            mixed_bytes = await asyncio.wait_for(asyncio.to_thread(tts_service.mix_final_audio,
            dialog_chunks=final_sequence_paths,
            music_path=music_path,
            sfx_tracks=[],
                output_format=self.request.config.output_format,
                chunk_kinds=final_sequence_kinds,
                music_delay_ms=music_delay_ms,
                enable_ducking=False,
                request_id=self.request.id,
                total_dialog_duration_s=total_dialog_duration,
                music_duration_s=music_duration if music_path else None,
                music_loop=bool(getattr(self.request.config, 'music_loop', True)),
                music_stop_at_end=bool(getattr(self.request.config, 'music_stop_at_end', False))
            ), timeout=45)
        except asyncio.TimeoutError:
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="mix", subphase="error", error="mix_timeout")
            except Exception:
                pass
            raise HTTPException(status_code=504, detail="Mixing timed out")
        except Exception as e:
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="mix", subphase="error", error=str(e))
            except Exception:
                pass
            raise
        try:
            from ai.routes.dialog_routes import set_dialog_status
            set_dialog_status(self.request.id, phase="mix", subphase="done")
        except Exception:
            pass
        return mixed_bytes

    def _resolve_voice_profile(self, speaker: str, gender: str):
        """Resolve voice configuration for a speaker from voice_mapping.

        Returns (provider, config_dict) where config_dict has all TTS parameters.
        Supports three voice_mapping formats:
          - str: OpenAI voice name (backward compat)
          - dict with voice_id: ElevenLabs voice profile
          - dict with default/secondary: legacy OpenAI mapping
        """
        from ai.services.tts_models import VoiceProfile
        voice_mapping = self.request.config.voice_mapping or {}
        entry = voice_mapping.get(speaker)

        # Case 1: VoiceProfile-style dict with voice_id (new multi-voice format)
        if isinstance(entry, dict) and 'voice_id' in entry:
            return entry.get('provider', 'elevenlabs'), entry

        # Case 2: Legacy dict with default/secondary OpenAI voices
        if isinstance(entry, dict) and 'default' in entry:
            voice_name = entry.get('default', 'nova')
            return 'openai', {'voice': voice_name}

        # Case 3: Simple string = OpenAI voice name
        if isinstance(entry, str):
            return 'openai', {'voice': entry}

        # Case 4: No mapping — auto-assign
        default_voices = {
            "male": "onyx", "female": "nova",
            "ai": "onyx" if self.request.config.ai_gender == 'male' else "nova",
            "narrator": "shimmer"
        }
        if speaker in self._speaker_assigned_voice:
            voice = self._speaker_assigned_voice[speaker]
        elif speaker.lower() == 'narrator':
            desired = getattr(self.request.config, 'narrator_gender', None)
            if desired in ('male', 'female'):
                voice = default_voices.get(desired, 'shimmer')
            else:
                normalized = self._normalize_gender_label(gender)
                voice = default_voices.get(normalized, 'shimmer')
            self._speaker_assigned_voice[speaker] = voice
        else:
            voice = self._pick_distinct_voice(gender)
            self._speaker_assigned_voice[speaker] = voice

        # If global provider is elevenlabs, use the global voice_id as fallback
        if self.request.config.provider == "elevenlabs":
            return 'elevenlabs', {'voice_id': self.request.content.voice}

        return 'openai', {'voice': voice}

    async def _generate_single_dialog_chunk(self, segment):
        from ai.services.tts_models import OpenAITTSConfig, ElevenLabsTTSConfig

        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "")
        gender = self._normalize_gender_label(segment.get("gender", ""))
        voice_style = segment.get("voice_style")

        # Resolve per-speaker voice profile
        provider, voice_config = self._resolve_voice_profile(speaker, gender)

        # Apply per-line overrides from Gemini analysis (voice_style hints)
        overrides = segment.get("overrides") or {}

        if provider == "elevenlabs":
            voice_id = voice_config.get('voice_id', self.request.content.voice)
            stability = overrides.get('stability', voice_config.get('stability', 0.5))
            clarity = overrides.get('clarity', voice_config.get('clarity', 0.75))
            model_id = voice_config.get('model_id', 'eleven_multilingual_v2')

            # Use NarrationService for dramatic preprocessing if available
            try:
                from ai.services.narration_service import NarrationService, NarrationRequest, NarrationCharacter, NarrationContext, NarrationConfig
                narration = NarrationService()
                narr_req = NarrationRequest(
                    text=text,
                    character=NarrationCharacter(
                        name=speaker,
                        voice_id=voice_id,
                        personality=voice_style or None,
                        speaking_style=None,
                    ),
                    context=NarrationContext(
                        type="dialog_chunk",
                        mood=segment.get("voice_style") or "neutral",
                        language=getattr(self.request.content, 'language', 'de'),
                    ),
                    config=NarrationConfig(
                        stability=stability,
                        clarity=clarity,
                        model_id=model_id,
                        preprocessing=True,
                    ),
                )
                dramatic_text = await narration._preprocess_text(narr_req)
                segment['dramatic_script'] = dramatic_text
                audio_bytes = await narration._generate_tts(dramatic_text, narr_req)
            except ImportError:
                # Fallback: direct ElevenLabs TTS without dramatic preprocessing
                el_config = ElevenLabsTTSConfig(
                    voice_id=voice_id, model_id=model_id,
                    stability=stability, clarity=clarity,
                )
                audio_bytes = await tts_service.generate_elevenlabs_tts(text, el_config)
            chosen_voice = voice_id
        else:
            voice_name = voice_config.get('voice', 'nova')
            speed = overrides.get('speed', voice_config.get('speed', self.request.content.speed))
            config = OpenAITTSConfig(voice=voice_name, speed=speed, output_format=self.request.config.output_format)
            audio_bytes = await tts_service.generate_openai_tts(text, config)
            chosen_voice = voice_name

        chunk_path = self.temp_dir / f"dialog_{uuid.uuid4()}.{self.request.config.output_format}"
        chunk_path.write_bytes(audio_bytes)
        segment['chosen_voice'] = chosen_voice
        return chunk_path

    async def _source_single_sfx(self, cue):
        """Generate SFX using ElevenLabs"""
        description = cue.get('description', '')
        if not description:
            return None

        print(f"--- Audio Drama: Generating SFX for '{description}'...")

        try:
            from elevenlabs.client import AsyncElevenLabs
            from pathlib import Path
            import tempfile
            import os
            from ai.services import tts_service
            from ai.clients.storage_client import save_file_and_record, StorageObject

            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="sfx_start", description=description)
            except Exception:
                pass

            client = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

            audio_stream = client.text_to_sound_effects.convert(text=description)

            audio_bytes = b""
            async for chunk in audio_stream:
                audio_bytes += chunk

            if not audio_bytes:
                raise HTTPException(status_code=500, detail="ElevenLabs SFX generation returned no data.")

            # Audio Analysis
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_bytes)
                temp_path = Path(temp_file.name)

            mean_volume = tts_service.analyze_audio_level(temp_path)
            print(f"--- Audio Drama: SFX audio level: {mean_volume} dB")

            # Discard silent or near-silent SFX
            SILENCE_THRESHOLD = -60.0
            if mean_volume < SILENCE_THRESHOLD:
                print(f"--- Audio Drama: SFX too quiet ({mean_volume} dB), discarding.")
                temp_path.unlink()
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(self.request.id, phase="generate", subphase="sfx_silent", description=description)
                except Exception:
                    pass
                return None

            # Save to Storage
            filename = f"sfx_{description.replace(' ', '_')[:20]}.mp3"

            saved_obj = await save_file_and_record(
                data=audio_bytes,
                original_filename=filename,
                context="dialog-sfx",
                is_public=True,
                collection_id="dialog-generated-sfx",
                link_id=self.request.id
            )

            temp_path.unlink()
            print(f"--- Audio Drama: Saved SFX to storage object ID {saved_obj.id}")

            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="sfx_done", description=description)
            except Exception:
                pass

            return saved_obj

        except HTTPException as he:
            # Gracefully skip silent or invalid SFX prompts
            print(f"--- Audio Drama: Skipping SFX due to HTTPException {he.status_code}: {he.detail}")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="sfx_error", error=str(he.detail))
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"--- Audio Drama: Skipping SFX due to unexpected error: {e}")
            import traceback
            traceback.print_exc()
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="sfx_error", error=str(e))
            except Exception:
                pass
            return None

    async def _source_music(self, music_cues):
        """Generate music using ElevenLabs, Stable Audio, or Freesound fallback"""
        if not music_cues:
            return None
        cue = music_cues[0] if isinstance(music_cues, list) else {}
        description = cue.get('description', '')
        length_ms = cue.get('length_ms') or 30000
        print(f"--- Audio Drama: Sourcing music for '{description}'...")

        # Helper to download music from storage URL to temp file
        async def _download_and_return(file_url: str):
            async with httpx.AsyncClient() as client:
                r = await client.get(file_url)
                r.raise_for_status()
            path = self.temp_dir / "music.mp3"
            path.write_bytes(r.content)
            print(f"--- Audio Drama: Downloaded music to {path}")
            return path

        # 1) Try ElevenLabs Music Generation
        try:
            from ai.routes.music_generation import generate_music_elevenlabs

            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="music_provider_eleven_start")
            except Exception:
                pass

            result = await generate_music_elevenlabs(description, length_ms)

            if result and result.get('audio_url'):
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(self.request.id, phase="generate", subphase="music_provider_eleven_done")
                except Exception:
                    pass
                return await _download_and_return(result['audio_url'])
        except Exception as e:
            print(f"--- Audio Drama: ElevenLabs music failed: {e}")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="music_provider_eleven_error", error=str(e))
            except Exception:
                pass

        # 2) Try Stable Audio via AIMLAPI
        try:
            from ai.routes.music_generation import generate_music_stable_audio

            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="music_provider_aiml_start")
            except Exception:
                pass

            result = await generate_music_stable_audio(description, length_ms)

            if result and result.get('audio_url'):
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(self.request.id, phase="generate", subphase="music_provider_aiml_done")
                except Exception:
                    pass
                return await _download_and_return(result['audio_url'])
        except Exception as e:
            print(f"--- Audio Drama: Stable Audio (AIMLAPI) failed: {e}")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="music_provider_aiml_error", error=str(e))
            except Exception:
                pass

        # 3) Free-sourced fallback (Pixabay/Freesound)
        try:
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="music_provider_fallback_start")
            except Exception:
                pass
            music_url = await audio_sourcing_service.find_music_on_freesound(description)
            if music_url:
                async with httpx.AsyncClient() as client:
                    r = await client.get(music_url, follow_redirects=True)
                    r.raise_for_status()
                music_path = self.temp_dir / "music.mp3"
                music_path.write_bytes(r.content)
                print(f"--- Audio Drama: Downloaded fallback music to {music_path}")
                try:
                    from ai.routes.dialog_routes import set_dialog_status
                    set_dialog_status(self.request.id, phase="generate", subphase="music_provider_fallback_done")
                except Exception:
                    pass
                return music_path
        except Exception as e:
            print(f"--- Audio Drama: Fallback free-sourced music failed: {e}")
            try:
                from ai.routes.dialog_routes import set_dialog_status
                set_dialog_status(self.request.id, phase="generate", subphase="music_provider_fallback_error", error=str(e))
            except Exception:
                pass

        print("--- Audio Drama: No suitable music found or generated. Proceeding without music.")
        return None
