"""`/ai/tts/narrate` liefert Wort-Zeitstempel, wenn der Aufrufer sie will.

Story (13.09.): "synchron zur Bildfolge" hatte bisher keinen Mechanismus —
narrate gab nur Audio, Dauer und Skript zurueck. Die Faehigkeit lag im
Haus: tts_service.generate_elevenlabs_tts(with_timestamps=True), vom
Dialogpfad genutzt. Hier wird sie wiederverwendet, nicht nachgebaut.

Ziel: mit Schalter kommen die Woerter der Alignment-Quelle zurueck.
Gegenfall: ohne Schalter wird der Alignment-Pfad NICHT betreten und die
Antwort traegt keine Zeitstempel — bestehende Aufrufer bleiben auf dem
Streaming-Pfad (anderer Endpunkt bei ElevenLabs, anderer Klang moeglich).
"""

import types

import pytest

from ai.services import narration_service as n
from ai.services import tts_service


def _anfrage(with_timestamps, save=None):
    """Mit Zeitstempeln ist Speichern Pflicht (seit 13.09.): Vorgabe {}."""
    if save is None and with_timestamps:
        save = {}
    return n.NarrationRequest(
        text="Mira trat durch das alte Steintor.",
        character=n.NarrationCharacter(name="Erzaehler", voice_id="voice-test"),
        config=n.NarrationConfig(preprocessing=False, with_timestamps=with_timestamps,
                                 language_code="de"),
        save_options=save,
    )


@pytest.fixture(autouse=True)
def _keine_dauer(monkeypatch):
    monkeypatch.setattr(n.NarrationService, "_measure_audio_duration",
                        staticmethod(lambda b, f: 1.25))

    async def fake_save(self, audio_bytes, request):
        return 4711, "https://api-storage.arkturian.com/storage/media/4711"

    monkeypatch.setattr(n.NarrationService, "_save_audio", fake_save)


@pytest.mark.asyncio
async def test_mit_schalter_kommen_woerter_aus_der_alignment_quelle(monkeypatch):
    gesehen = {}

    async def fake(text, cfg, with_timestamps=False):
        gesehen["text"] = text
        gesehen["cfg"] = cfg
        gesehen["with_timestamps"] = with_timestamps
        return b"MP3", [{"word": "Mira", "start": 0.0, "end": 0.31},
                        {"word": "trat", "start": 0.35, "end": 0.6}]

    monkeypatch.setattr(tts_service, "generate_elevenlabs_tts", fake)
    antwort = await n.NarrationService().generate(_anfrage(True))

    assert gesehen["with_timestamps"] is True
    assert gesehen["cfg"].voice_id == "voice-test"
    assert gesehen["cfg"].language_code == "de"     # der Zwang geht mit, nicht verloren
    assert antwort.word_timestamps == [
        {"word": "Mira", "start": 0.0, "end": 0.31},
        {"word": "trat", "start": 0.35, "end": 0.6},
    ]
    assert antwort.timestamps_source == "elevenlabs_alignment"
    assert antwort.timestamp_granularity == "word"
    assert antwort.word_timestamps_dropped == 0
    assert antwort.duration_seconds == 1.25


@pytest.mark.asyncio
async def test_ohne_schalter_bleibt_der_streaming_pfad(monkeypatch):
    async def darf_nicht(*a, **k):
        raise AssertionError("Alignment-Pfad ohne Schalter betreten")

    monkeypatch.setattr(tts_service, "generate_elevenlabs_tts", darf_nicht)

    class _Stream:
        def __init__(self, *a, **k):
            pass

        def __aiter__(self):
            async def gen():
                yield b"MP"
                yield b"3"
            return gen()

    class _Client:
        def __init__(self, *a, **k):
            self.text_to_speech = types.SimpleNamespace(convert=lambda **kw: _Stream())

    # Das Paket `elevenlabs` ist im Testbaum nicht installiert; der
    # Streaming-Pfad importiert es erst beim Aufruf. Ein Stub-Modul reicht,
    # um zu belegen, dass DIESER Pfad und nicht der Alignment-Pfad laeuft.
    import sys
    stub_paket = types.ModuleType("elevenlabs")
    stub_client = types.ModuleType("elevenlabs.client")
    stub_client.AsyncElevenLabs = _Client
    stub_paket.client = stub_client
    monkeypatch.setitem(sys.modules, "elevenlabs", stub_paket)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", stub_client)

    antwort = await n.NarrationService().generate(_anfrage(False))
    assert antwort.word_timestamps is None
    assert antwort.timestamps_source is None


def test_schalter_ist_standardmaessig_aus():
    assert n.NarrationConfig().with_timestamps is False


# ---------------------------------------------------- Konsumentenvertrag
# story-api production_windows() (Story, 13.09., woertlich): word str,
# start/end int|float ohne bool, endlich, 0 <= start < end. Alles andere
# kippt dort den ganzen Lauf mit 422.


def test_bereinigung_haelt_den_vertrag():
    roh = [
        {"word": "Mira", "start": 0.0, "end": 0.3},        # gut
        {"word": "", "start": 0.3, "end": 0.4},            # leeres Wort
        {"word": "x", "start": 0.5, "end": 0.5},           # start == end
        {"word": "y", "start": 0.7, "end": 0.6},           # rueckwaerts
        {"word": "z", "start": -0.1, "end": 0.2},          # negativ
        {"word": "n", "start": float("nan"), "end": 1.0},  # nicht endlich
        {"word": "b", "start": True, "end": 2.0},          # bool
        {"word": "trat", "start": 1, "end": 1.5},          # int ist erlaubt
        "kaputt",
    ]
    sauber, verworfen = n.bereinige_wortzeiten(roh)
    assert sauber == [{"word": "Mira", "start": 0.0, "end": 0.3},
                      {"word": "trat", "start": 1.0, "end": 1.5}]
    assert verworfen == 7


@pytest.mark.asyncio
async def test_kaputte_eintraege_werden_gezaehlt_nicht_durchgereicht(monkeypatch):
    async def fake(text, cfg, with_timestamps=False):
        return b"MP3", [{"word": "Mira", "start": 0.0, "end": 0.3},
                        {"word": "x", "start": 0.4, "end": 0.4}]

    monkeypatch.setattr(tts_service, "generate_elevenlabs_tts", fake)
    antwort = await n.NarrationService().generate(_anfrage(True))
    assert antwort.word_timestamps == [{"word": "Mira", "start": 0.0, "end": 0.3}]
    assert antwort.word_timestamps_dropped == 1


# ---------------------------------------------------- Speichern sichtbar
# Story/Story-Codex (13.09.): ohne save_options wird gesprochen, aber nicht
# gespeichert — Zeichen verbraucht, audio_id null, nichts Bindbares. Und
# Zeitstempel ohne Datei bedeuten nichts.


@pytest.mark.asyncio
async def test_zeitstempel_ohne_speichern_werden_vor_dem_sprechen_abgewiesen(monkeypatch):
    from fastapi import HTTPException

    async def darf_nicht(*a, **k):
        raise AssertionError("gesprochen, obwohl 422 vorher faellig war")

    monkeypatch.setattr(tts_service, "generate_elevenlabs_tts", darf_nicht)
    with pytest.raises(HTTPException) as exc:
        req = _anfrage(True); req.save_options = None
        await n.NarrationService().generate(req)
    assert exc.value.status_code == 422
    assert exc.value.detail["error"] == "timestamps_require_save"


@pytest.mark.asyncio
async def test_leeres_save_options_heisst_speichern(monkeypatch):
    async def fake(text, cfg, with_timestamps=False):
        return b"MP3", [{"word": "Mira", "start": 0.0, "end": 0.3}]

    gespeichert = {}

    async def fake_save(self, audio_bytes, request):
        gespeichert["bytes"] = audio_bytes
        return 4711, "https://api-storage.arkturian.com/storage/media/4711"

    monkeypatch.setattr(tts_service, "generate_elevenlabs_tts", fake)
    monkeypatch.setattr(n.NarrationService, "_save_audio", fake_save)
    req = _anfrage(True)
    req.save_options = {}
    antwort = await n.NarrationService().generate(req)
    assert gespeichert["bytes"] == b"MP3"
    assert antwort.saved is True and antwort.audio_id == 4711


@pytest.mark.asyncio
async def test_ohne_speichern_sagt_die_antwort_es(monkeypatch):
    """Streaming-Pfad ohne save_options bleibt erlaubt — aber sichtbar."""
    import sys

    class _Stream:
        def __aiter__(self):
            async def gen():
                yield b"MP3"
            return gen()

    class _Client:
        def __init__(self, *a, **k):
            self.text_to_speech = types.SimpleNamespace(convert=lambda **kw: _Stream())

    stub_paket = types.ModuleType("elevenlabs"); stub_client = types.ModuleType("elevenlabs.client")
    stub_client.AsyncElevenLabs = _Client; stub_paket.client = stub_client
    monkeypatch.setitem(sys.modules, "elevenlabs", stub_paket)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", stub_client)

    antwort = await n.NarrationService().generate(_anfrage(False))
    assert antwort.saved is False and antwort.audio_id is None
