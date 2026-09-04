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
    wav = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                          "-t", "1.5", "-f", "wav", "-"], capture_output=True, timeout=30).stdout
    assert wav and abs(_ORIG_DAUER(wav) - 1.5) < 0.05
