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
from io import BytesIO
from typing import List, Optional

from fastapi import HTTPException

from pydub import AudioSegment
from pydantic import BaseModel, Field

import google.generativeai as genai
from ai.clients.storage_client import storage_api_key

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
    language_code: Optional[str] = Field(
        default=None,
        description=(
            "ISO-Sprachcode, der die Sprache bei ElevenLabs ERZWINGT "
            "(z.B. 'sl'). Standard None = wie bisher, das Modell "
            "detektiert selbst."
        ),
    )
    speed: float = Field(default=1.0, description="Speaking speed")
    preprocessing: bool = Field(default=True, description="Enable AI dramatic preprocessing")
    output_format: str = Field(default="mp3", description="Audio format")
    # Wort-Zeitstempel aus ElevenLabs' Zeichen-Alignment — dieselbe
    # Quelle wie im Dialogpfad. Story (13.09.) braucht sie fuer
    # "synchron zur Bildfolge"; eine nachtraegliche Transkription
    # lieferte Zeiten fuer das, was ein STT-Modell GEHOERT hat, nicht fuer
    # das, was gesprochen wurde. Standard aus: bestehende Aufrufer
    # bekommen weiter den Streaming-Pfad.
    with_timestamps: bool = Field(default=False, description="Return word-level timestamps from ElevenLabs alignment (with-timestamps endpoint)")


class NarrationRequest(BaseModel):
    """Full narration request."""
    text: str = Field(description="Plain text to narrate")
    character: NarrationCharacter
    context: NarrationContext = NarrationContext()
    config: NarrationConfig = NarrationConfig()
    save_options: Optional[dict] = Field(default=None, description="Storage API save options")
    collection_id: Optional[str] = Field(default=None, description="Storage collection ID")
    # Kennung des Aufrufers (8-128 Zeichen, [A-Za-z0-9_.:-]). Mit ihr gibt
    # es dauerhaften Status (GET /ai/tts/narrate/{request_id}) und
    # payloadgebundene Idempotenz: dieselbe Kennung mit demselben Inhalt
    # spricht nie zweimal (Content #4976, p-b12b0d4d6062).
    request_id: Optional[str] = Field(default=None, description="Caller-chosen id for durable status + idempotent replay")
    # Neusprechen nach verwaistem `running` ist eine AUSDRUECKLICHE
    # Handlung, nie ein nackter POST: 10 Minuten beweisen weder Abschluss
    # noch Nichtabrechnung (Review q-d98ba93140f0, Punkt 1).
    respeak_stale: bool = Field(default=False, description="Explicit consent to speak again when the request_id is stuck in stale running")


class NarrationResponse(BaseModel):
    """Narration result."""
    audio_id: Optional[int] = None
    audio_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    dramatic_script: str = Field(description="The enriched script that was spoken")
    original_text: str = Field(description="Original input text")
    preprocessing_model: Optional[str] = None
    # Nur bei config.with_timestamps: [{"word", "start", "end"}], Sekunden
    # ab Audiobeginn, ueber Textstuecke hinweg fortlaufend.
    word_timestamps: Optional[List[dict]] = None
    timestamps_source: Optional[str] = None
    # "word": ElevenLabs liefert Zeichen-Alignment; die Uebersetzung in
    # Wortintervalle passiert bei uns (Wortgrenze am Leerzeichen, erstes
    # Zeichen = start, letztes Zeichen = end). Nullpunkt ist der Anfang
    # der unter audio_id gespeicherten Datei — gespeichert wird der
    # unveraenderte Strom, nichts wird geschnitten.
    timestamp_granularity: Optional[str] = None
    # Ob die Datei im Storage liegt. Ohne `save_options` wird gesprochen,
    # aber nicht gespeichert — ElevenLabs-Zeichen verbraucht, nichts
    # Bindbares hinterlassen (Story/Story-Codex, 13.09.). Sichtbar statt
    # aus `audio_id: null` zu erraten.
    saved: bool = False
    request_id: Optional[str] = None
    # true: Ergebnis aus dem Auftragsbuch, kein Anbieteraufruf, keine Zeichen.
    replayed: bool = False
    # Eintraege, die den Konsumentenvertrag (story-api, 13.09.) verletzen
    # wuerden — 0 <= start < end, endliche Zahlen, nichtleeres Wort —
    # werden verworfen und hier gezaehlt statt still durchgereicht.
    word_timestamps_dropped: Optional[int] = None


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


def bereinige_wortzeiten(woerter) -> tuple:
    """Haelt den Vertrag des Konsumenten (story-api production_windows,
    von Story am 13.09. woertlich uebermittelt): `word` str und nicht
    leer, `start`/`end` int|float, kein bool, endlich, `0 <= start < end`,
    Sekunden. Was das verletzt, fliegt raus und wird gezaehlt — ein
    einziger kaputter Eintrag wuerde dort den ganzen Lauf mit 422 kippen.

    Ein Wort gehoert im Konsumenten zum Fenster seines `start`; `start`
    muss also verlaesslicher sein als `end`. Deshalb wird ein `end`, das
    nicht groesser als `start` ist, NICHT repariert, sondern der Eintrag
    verworfen: eine erfundene Dauer waere schlimmer als ein Loch.
    """
    import math

    sauber = []
    verworfen = 0
    for w in woerter or []:
        if not isinstance(w, dict):
            verworfen += 1
            continue
        wort = w.get("word")
        start, end = w.get("start"), w.get("end")
        ok = (
            isinstance(wort, str) and wort.strip() != ""
            and isinstance(start, (int, float)) and not isinstance(start, bool)
            and isinstance(end, (int, float)) and not isinstance(end, bool)
            and math.isfinite(start) and math.isfinite(end)
            and 0 <= start < end
        )
        if not ok:
            verworfen += 1
            continue
        sauber.append({"word": wort, "start": float(start), "end": float(end)})
    return sauber, verworfen


class NarrationService:
    """Dramaturgical TTS — AI-enriched text → ElevenLabs → Audio."""

    async def generate(self, request: NarrationRequest) -> NarrationResponse:
        """Full pipeline: preprocess → TTS → optional save."""
        t_start = time.time()

        # `{}` ist eine ausdrueckliche Entscheidung zu speichern (mit
        # Vorgaben), `None` ist keine. Bis 13.09. galt `if save_options:`,
        # und ein leeres Objekt hiess: nicht speichern.
        speichern = request.save_options is not None

        # Wort-Zeitstempel beziehen sich per Vertrag auf die gespeicherte
        # Datei unter audio_id. Ohne Datei bedeuten sie nichts — und die
        # Zeichen waeren trotzdem verbraucht. Deshalb 422 VOR dem Sprechen.
        if request.config.with_timestamps and not speichern:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "timestamps_require_save",
                    "hint": ("with_timestamps=true verlangt save_options (z.B. {}), "
                             "weil die Zeiten auf die gespeicherte Datei unter "
                             "audio_id bezogen sind."),
                },
            )

        # Step 1: Dramatic preprocessing (optional). Eigene Stufe: die
        # Aufbereitung ist ein Modellaufruf — `pre_tts` gilt nur DAVOR.
        if request.request_id:
            from ai.services import narrate_jobs
            narrate_jobs.stufe(request.request_id, "prepare")
        if request.config.preprocessing:
            dramatic_script = await self._preprocess_text(request)
        else:
            dramatic_script = request.text

        logger.info(f"[Narration] Script ready ({len(dramatic_script)} chars, {int((time.time()-t_start)*1000)}ms)")

        # Step 2: Generate TTS via ElevenLabs
        if request.request_id:
            from ai.services import narrate_jobs
            narrate_jobs.stufe(request.request_id, "tts")
        audio_bytes, word_timestamps = await self._generate_tts(dramatic_script, request)
        if request.request_id:
            narrate_jobs.stufe(request.request_id, "save")
        verworfen = 0
        if word_timestamps is not None:
            word_timestamps, verworfen = bereinige_wortzeiten(word_timestamps)
        duration_seconds = self._measure_audio_duration(
            audio_bytes,
            request.config.output_format,
        )
        logger.info(f"[Narration] TTS done ({len(audio_bytes)} bytes, {int((time.time()-t_start)*1000)}ms)")

        # Step 3: Optional save to Storage API
        audio_id = None
        audio_url = None
        if speichern:
            audio_id, audio_url = await self._save_audio(audio_bytes, request)
            logger.info(f"[Narration] Saved to storage: id={audio_id}")

        return NarrationResponse(
            audio_id=audio_id,
            audio_url=audio_url,
            duration_seconds=duration_seconds,
            dramatic_script=dramatic_script,
            original_text=request.text,
            preprocessing_model="gemini" if request.config.preprocessing else None,
            word_timestamps=word_timestamps,
            timestamps_source="elevenlabs_alignment" if word_timestamps is not None else None,
            timestamp_granularity="word" if word_timestamps is not None else None,
            saved=audio_id is not None,
            word_timestamps_dropped=verworfen if word_timestamps is not None else None,
        )

    @staticmethod
    def _measure_audio_duration(audio_bytes: bytes, output_format: str) -> Optional[float]:
        """Measure generated audio so callers can persist a complete cue contract."""
        try:
            audio = AudioSegment.from_file(
                BytesIO(audio_bytes),
                format=output_format,
            )
            return len(audio) / 1000.0
        except Exception:
            logger.exception("[Narration] Failed to measure generated audio duration")
            return None

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

    async def _generate_tts(self, text: str, request: NarrationRequest) -> tuple:
        """Generate audio via ElevenLabs. Returns (audio_bytes, word_timestamps|None)."""
        if request.config.with_timestamps:
            # Der Alignment-Pfad lebt in tts_service (with-timestamps-REST,
            # Stueckelung mit fortlaufendem Zeitversatz) und wird vom
            # Dialogpfad genutzt. Nicht nachbauen, wiederverwenden.
            from ai.services import tts_service
            cfg = tts_service.ElevenLabsTTSConfig(
                model_id=request.config.model_id,
                language_code=request.config.language_code,
                voice_id=request.character.voice_id,
                stability=request.config.stability,
                clarity=request.config.clarity,
            )
            audio_bytes, words = await tts_service.generate_elevenlabs_tts(
                text, cfg, with_timestamps=True
            )
            return audio_bytes, list(words or [])
        try:
            from elevenlabs.client import AsyncElevenLabs
        except ModuleNotFoundError:
            raise RuntimeError("ElevenLabs package required. Install with 'pip install elevenlabs'.")

        from ai.services.elevenlabs_cost_tracker import elevenlabs_cost_tracker
        elevenlabs_cost_tracker.pre_check(len(text), endpoint="narrate")

        client = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        # `language_code` NUR, wenn der Aufrufer ihn ausdruecklich setzt.
        #
        # Nicht automatisch aus `context.language` ableiten, obwohl es
        # naheliegt: Der Zwang wirkt nur mit den neueren Modellen, und
        # ein stiller Modellwechsel wuerde den KLANG des bestehenden
        # Bestands aendern — mitten in einer laufenden Vertonung. Alex'
        # ~70 slowenische Toene sind mit dem heutigen Klang aufgenommen;
        # wer sie neu erzeugt, soll das entscheiden, nicht erleiden.
        #
        # Hintergrund (Knowledge, 2026-08-16): Alexanders slowenische
        # Aufnahmen klingen kroatisch. `context.language='sl'` kommt bis
        # zu dieser Funktion — und wurde hier bisher fallen gelassen.
        # `eleven_multilingual_v2` kennt ohnehin keinen `language_code`
        # und raet die Sprache aus dem Text; bei SL-Text liegt HR nahe.
        _tts_args = {
            "text": text,
            "voice_id": request.character.voice_id,
            "model_id": request.config.model_id,
            "voice_settings": {
                "stability": request.config.stability,
                "similarity_boost": request.config.clarity,
            },
        }
        if request.config.language_code:
            _tts_args["language_code"] = request.config.language_code
        audio_stream = client.text_to_speech.convert(**_tts_args)

        audio_bytes = b""
        async for chunk in audio_stream:
            audio_bytes += chunk

        elevenlabs_cost_tracker.track_tts(len(text), caller="narrate")
        return audio_bytes, None

    async def _save_audio(self, audio_bytes: bytes, request: NarrationRequest) -> tuple:
        """Save audio to Storage API. Returns (storage_id, url)."""
        import httpx

        storage_url = os.getenv("STORAGE_API_URL", "https://api-storage.arkturian.com")
        storage_key = storage_api_key()

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
