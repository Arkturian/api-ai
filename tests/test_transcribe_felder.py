"""/ai/transcribe fuer comm-api (#4931): Antwort traegt erkannte Sprache,
Dauer und Latenz. Gegen den echten Fehler gehalten: gpt-4o-transcribe kennt
`verbose_json` nicht — dort darf es NICHT gesetzt werden, whisper-1 ohne
Vorgabe bekommt es (nur so gibt es Sprache + Dauer aus der API)."""

import asyncio
import io
import types

import pytest
from fastapi import UploadFile

from ai.routes import audio_ai_routes as a

_ORIG_DAUER = a._audio_duration_seconds   # vor dem autouse-Patch gesichert


class _FakeTranscriptions:
    def __init__(self, store, result):
        self.store, self.result = store, result

    async def create(self, **kw):
        self.store["kwargs"] = kw
        return self.result


def _client(store, result):
    class _C:
        def __init__(self, *a, **k):
            self.audio = types.SimpleNamespace(transcriptions=_FakeTranscriptions(store, result))
    return _C


def _upload(name="voice.oga"):
    return UploadFile(file=io.BytesIO(b"OggS-fake"), filename=name)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(a, "_normalize_audio_for_openai", lambda d, b: (d, ".ogg"))
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: 12.34)


def test_whisper_ohne_vorgabe_erkennt_sprache_und_liefert_dauer_latenz(monkeypatch):
    store = {}
    res = types.SimpleNamespace(text="Hallo, das ist ein Test.", language="german", duration=12.1)
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, res))
    out = asyncio.run(a._transcribe_with_whisper(_upload(), "whisper-1"))
    assert store["kwargs"]["response_format"] == "verbose_json"
    assert out["text"].startswith("Hallo") and out["model"] == "whisper-1"
    assert out["language"] == "german" and out["language_source"] == "detected"
    assert out["duration_seconds"] == 12.34            # ffprobe hat Vorrang
    assert isinstance(out["latency_ms"], int) and out["latency_ms"] >= 0


def test_gpt4o_bekommt_kein_verbose_json_und_meldet_vorgabe(monkeypatch):
    store = {}
    res = types.SimpleNamespace(text="Hello")
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, res))
    out = asyncio.run(a._transcribe_with_whisper(_upload(), "gpt-4o-transcribe", language="de"))
    assert "response_format" not in store["kwargs"]
    assert store["kwargs"]["language"] == "de"
    assert out["language"] == "de" and out["language_source"] == "requested"
    assert out["duration_seconds"] == 12.34 and "latency_ms" in out


def test_dauer_faellt_auf_api_wert_zurueck_wenn_ffprobe_schweigt(monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: None)
    store = {}
    res = types.SimpleNamespace(text="x", language="english", duration=3.456)
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, res))
    out = asyncio.run(a._transcribe_with_whisper(_upload(), "whisper-1"))
    assert out["duration_seconds"] == 3.46


def test_explizites_format_wird_nicht_ueberschrieben(monkeypatch):
    store = {}
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, types.SimpleNamespace(text="x")))
    asyncio.run(a._transcribe_with_whisper(_upload(), "whisper-1", response_format="text"))
    assert store["kwargs"]["response_format"] == "text"


def test_ffprobe_dauer_echt():
    """Echte Dauer aus ffprobe: 1,5 s Stille als WAV."""
    import subprocess
    # 120 s statt 30 s: der Aufruf erzeugt 1,5 s Stille und braucht im
    # Leerlauf Millisekunden. Am 12.09. lief er waehrend eines parallelen
    # Deploys in den Zeitablauf — der Test misst dann die Auslastung der
    # Maschine, nicht die Dauererkennung. Ein Test, der unter Last kippt,
    # entwertet die ganze Suite als Signal.
    wav = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                          "-t", "1.5", "-f", "wav", "-"], capture_output=True, timeout=120).stdout
    assert wav and abs(_ORIG_DAUER(wav) - 1.5) < 0.05


# --- Segmente + Sprecher-Referenzen (Cloud-Marketing, 06.10.) -------------
# Gegen den echten Fehler gehalten: verbose_json lieferte nur `text`, die
# Segmente gingen verloren; Sprecher-Referenzen gab es nicht, deshalb trugen
# getrennt transkribierte Teile unabhaengige Sprechernamen.

import base64 as _b64
import json as _json
import subprocess as _sp

from fastapi import HTTPException


def _ton(sekunden=12.0):
    """Echte Audiodatei (Sinus, 16 kHz WAV) — der Schnitt laeuft durch ffmpeg."""
    r = _sp.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                 f"sine=frequency=440:duration={sekunden}", "-ac", "1", "-ar", "16000",
                 "-f", "wav", "pipe:1"], capture_output=True, check=True)
    return r.stdout


def test_explizites_verbose_json_liefert_segmente_und_woerter(monkeypatch):
    store = {}
    res = types.SimpleNamespace(text="Hallo Welt", language="german", duration=2.0,
                                segments=[{"id": 0, "start": 0.0, "end": 1.2, "text": "Hallo Welt"}],
                                words=[{"word": "Hallo", "start": 0.0, "end": 0.5}])
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, res))
    out = asyncio.run(a._transcribe_with_whisper(
        _upload(), "whisper-1", response_format="verbose_json",
        timestamp_granularities="segment,word"))
    assert store["kwargs"]["timestamp_granularities"] == ["segment", "word"]
    assert out["segments"][0]["end"] == 1.2 and out["words"][0]["word"] == "Hallo"
    assert out["language"] == "german"


def test_automatisches_verbose_json_bleibt_schlank(monkeypatch):
    res = types.SimpleNamespace(text="x", language="german", duration=1.0,
                                segments=[{"id": 0, "start": 0.0, "end": 1.0, "text": "x"}])
    monkeypatch.setattr(a, "AsyncOpenAI", _client({}, res))
    out = asyncio.run(a._transcribe_with_whisper(_upload(), "whisper-1"))
    assert "segments" not in out


def test_granularitaeten_ohne_verbose_json_422_ohne_anbieteraufruf(monkeypatch):
    store = {}
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, types.SimpleNamespace(text="x")))
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_whisper(_upload(), "whisper-1", timestamp_granularities="word"))
    assert e.value.status_code == 422 and "kwargs" not in store


def test_sprecher_referenzen_nur_mit_diarize(monkeypatch):
    store = {}
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, types.SimpleNamespace(text="x")))
    spec = _json.dumps([{"name": "Alex", "start": 1, "end": 4}])
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_whisper(_upload(), "gpt-4o-transcribe", speaker_references=spec))
    assert e.value.status_code == 422 and "kwargs" not in store


@pytest.mark.parametrize("spec", [
    [{"name": "A", "start": 0, "end": 1.5}],            # zu kurz
    [{"name": "A", "start": 0, "end": 11}],             # zu lang
    [{"name": "", "start": 0, "end": 3}],               # ohne Namen
    [{"name": "A", "start": 0, "end": 3}, {"name": "A", "start": 4, "end": 7}],  # doppelt
    [{"name": n, "start": 0, "end": 3} for n in "ABCDE"],  # mehr als 4
])
def test_ungueltige_sprecher_referenzen_422(monkeypatch, spec):
    store = {}
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, types.SimpleNamespace(text="x")))
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_whisper(
            _upload(), "gpt-4o-transcribe-diarize", response_format="diarized_json",
            speaker_references=_json.dumps(spec)))
    assert e.value.status_code == 422 and "kwargs" not in store


def test_sprecher_referenzen_werden_echt_geschnitten_und_weitergereicht(monkeypatch):
    store = {}
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, {"segments": []}))
    datei = UploadFile(file=io.BytesIO(_ton(12.0)), filename="gespraech.wav")
    spec = _json.dumps([{"name": "Alex", "start": 1.0, "end": 4.0},
                        {"name": "Gast", "start": 6.0, "end": 11.5}])
    out = asyncio.run(a._transcribe_with_whisper(
        datei, "gpt-4o-transcribe-diarize", response_format="diarized_json",
        speaker_references=spec))
    body = store["kwargs"]["extra_body"]
    assert body["known_speaker_names"] == ["Alex", "Gast"]
    laengen = []
    for ref in body["known_speaker_references"]:
        assert ref.startswith("data:audio/wav;base64,")
        wav = _b64.b64decode(ref.split(",", 1)[1])
        laengen.append(round((len(wav) - 44) / 32000, 1))
    assert laengen == [3.0, 5.5]
    assert out["speaker_names"] == ["Alex", "Gast"]


def test_referenz_hinter_dem_dateiende_422(monkeypatch):
    store = {}
    monkeypatch.setattr(a, "AsyncOpenAI", _client(store, {"segments": []}))
    datei = UploadFile(file=io.BytesIO(_ton(5.0)), filename="kurz.wav")
    spec = _json.dumps([{"name": "Alex", "start": 4.0, "end": 9.0}])
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_whisper(
            datei, "gpt-4o-transcribe-diarize", response_format="diarized_json",
            speaker_references=spec))
    assert e.value.detail["error"] == "speaker_reference_beyond_audio" and "kwargs" not in store
