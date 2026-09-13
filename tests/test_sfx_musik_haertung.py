"""Haertung von /ai/gensfx und /ai/genmusic_eleven (Review q-d98ba93140f0).

Zwei Befunde von Story-Codex/Story am laufenden Code:
- gensfx reichte `duration` nicht an ElevenLabs durch und mass keine
  Dauer — ein Geraeusch unbekannter Laenge fuer eine bestellte Dauer.
- genmusic_eleven ersetzte bei "bad prompt" den Prompt durch den
  Vorschlag des Anbieters und rief AUTOMATISCH neu: ein zweiter
  bezahlter Aufruf ohne Entscheidung, und der Inhalt war nicht mehr der
  bestellte.

Ziel: Dauer geht an den Anbieter, die Datei wird gemessen; bad_prompt
wird 422 mit Vorschlag, kein zweiter Aufruf. Gegenfall: ohne request_id
wie bisher (ausser den Feldern). Nachbarfall: Idempotenz — dieselbe
Kennung erzeugt nie zweimal; Sperren (429) kommen unveraendert durch.
"""

import json
import sys
import types

import httpx
import pytest
from fastapi import HTTPException

from ai.routes import audio_ai_routes as a
from ai.routes import music_generation as mg
from ai.services import narrate_jobs as nj


@pytest.fixture(autouse=True)
def _ablage(monkeypatch, tmp_path):
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")


def _stub_elevenlabs(monkeypatch, gesehen):
    class _Stream:
        def __aiter__(self):
            async def gen():
                yield b"ID3" + b"\x00" * 64
            return gen()

    class _Client:
        def __init__(self, *a, **k):
            self.text_to_sound_effects = types.SimpleNamespace(convert=lambda **kw: (gesehen.update(kw), _Stream())[1])

    paket = types.ModuleType("elevenlabs"); client = types.ModuleType("elevenlabs.client")
    client.AsyncElevenLabs = _Client; paket.client = client
    monkeypatch.setitem(sys.modules, "elevenlabs", paket)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", client)


def _stub_rest(monkeypatch):
    from ai.services import tts_service
    monkeypatch.setattr(tts_service, "analyze_audio_level", lambda p: -20.0)
    monkeypatch.setattr(tts_service, "probe_audio_duration", lambda b: 4.75)

    async def fake_save(**kw):
        return types.SimpleNamespace(id=9001, file_url="https://s/9001")
    monkeypatch.setattr(a, "save_file_and_record", fake_save, raising=False)
    import ai.clients.storage_client as sc
    monkeypatch.setattr(sc, "save_file_and_record", fake_save)
    from ai.services.elevenlabs_cost_tracker import elevenlabs_cost_tracker as z
    monkeypatch.setattr(z, "pre_check", lambda *a, **k: None)
    monkeypatch.setattr(z, "track_sfx", lambda *a, **k: None)
    monkeypatch.setattr(z, "track_music", lambda *a, **k: None)


# ---------------------------------------------------- gensfx

@pytest.mark.asyncio
async def test_gensfx_reicht_dauer_durch_und_misst(monkeypatch):
    gesehen = {}
    _stub_elevenlabs(monkeypatch, gesehen); _stub_rest(monkeypatch)
    out = await a.generate_sfx_endpoint(a.SFXRequest(prompt="Tuer schlaegt zu", duration=3.5, request_id="sfx-szene-1-tuer"), api_key="x")
    assert gesehen["duration_seconds"] == 3.5              # bestellt geht an den Anbieter
    assert out["duration_seconds"] == 4.75                 # geliefert wird gemessen
    assert out["duration_requested_s"] == 3.5
    assert out["saved"] is True and out["id"] == 9001 and out["replayed"] is False


@pytest.mark.asyncio
async def test_gensfx_idempotent(monkeypatch):
    gesehen = {}; aufrufe = {"n": 0}
    _stub_elevenlabs(monkeypatch, gesehen); _stub_rest(monkeypatch)
    orig = gesehen.update
    req = a.SFXRequest(prompt="Regen", duration=5.0, request_id="sfx-szene-1-regen")
    erste = await a.generate_sfx_endpoint(req, api_key="x")
    gesehen.clear()
    zweite = await a.generate_sfx_endpoint(req, api_key="x")
    assert zweite["replayed"] is True and zweite["id"] == erste["id"]
    assert gesehen == {}                                    # kein zweiter Anbieteraufruf
    st = await a.gensfx_status("sfx-szene-1-regen")
    assert st["kind"] == "sfx" and st["state"] == "done"
    with pytest.raises(HTTPException) as e:
        await a.genmusic_eleven_status("sfx-szene-1-regen")
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_gensfx_laesst_sperre_durch(monkeypatch):
    gesehen = {}
    _stub_elevenlabs(monkeypatch, gesehen); _stub_rest(monkeypatch)
    from ai.services.elevenlabs_cost_tracker import elevenlabs_cost_tracker as z
    def sperre(*a, **k):
        raise HTTPException(status_code=429, detail={"error": "monthly_api_cap_reached"})
    monkeypatch.setattr(z, "pre_check", sperre)
    with pytest.raises(HTTPException) as e:
        await a.generate_sfx_endpoint(a.SFXRequest(prompt="x", request_id="sfx-szene-1-x000"), api_key="x")
    assert e.value.status_code == 429                       # bis 13.09.: 500 mit Text
    assert gesehen == {}
    st = await a.gensfx_status("sfx-szene-1-x000")
    assert st["state"] == "failed" and st["failed_stage"] == "pre_tts"


# ---------------------------------------------------- genmusic_eleven

def _music_transport(antworten):
    """antworten: Liste von (status, json|bytes, content-type); jeder Aufruf nimmt das naechste."""
    calls = {"n": 0}
    def handler(request):
        i = min(calls["n"], len(antworten) - 1); calls["n"] += 1
        status, body, ct = antworten[i]
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, headers={"content-type": ct})
        return httpx.Response(status, json=body, headers={"content-type": ct})
    return handler, calls


@pytest.mark.asyncio
async def test_bad_prompt_ist_422_und_kein_zweiter_aufruf(monkeypatch):
    handler, calls = _music_transport([
        (400, {"detail": {"status": "bad_prompt", "data": {"prompt_suggestion": "calm piano, no vocals"}}}, "application/json"),
        (200, b"ID3musik", "audio/mpeg"),
    ])
    orig = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: orig(transport=httpx.MockTransport(handler)))
    with pytest.raises(HTTPException) as e:
        await mg.generate_music_elevenlabs("weird prompt", 30000)
    assert e.value.status_code == 422
    assert e.value.detail["error"] == "music_prompt_rejected"
    assert e.value.detail["prompt_suggestion"] == "calm piano, no vocals"
    assert calls["n"] == 1                                  # bis 13.09.: 2


@pytest.mark.asyncio
async def test_musik_dauer_wird_gemessen(monkeypatch):
    handler, calls = _music_transport([(200, b"ID3musik", "audio/mpeg")])
    orig = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: orig(transport=httpx.MockTransport(handler)))
    from ai.services import tts_service
    monkeypatch.setattr(tts_service, "probe_audio_duration", lambda b: 29.9)
    async def fake_save(**kw):
        return types.SimpleNamespace(id=9002, file_url="https://s/9002")
    monkeypatch.setattr(mg, "save_file_and_record", fake_save)
    from ai.services.elevenlabs_cost_tracker import elevenlabs_cost_tracker as z
    monkeypatch.setattr(z, "pre_check", lambda *a, **k: None); monkeypatch.setattr(z, "track_music", lambda *a, **k: None)
    out = await mg.generate_music_elevenlabs("calm piano", 30000)
    assert out["duration_seconds"] == 29.9 and out["id"] == 9002


@pytest.mark.asyncio
async def test_genmusic_route_idempotent_und_gemessen(monkeypatch):
    aufrufe = {"n": 0}
    async def fake_gen(prompt, duration_ms):
        aufrufe["n"] += 1
        return {"id": 9003, "audio_url": "https://s/9003", "storage_object_id": 9003, "format": "mp3", "duration_seconds": 31.2}
    monkeypatch.setattr(a, "generate_music_elevenlabs", fake_gen)
    req = a.MusicRequest(prompt="calm piano", duration=30, request_id="music-szene-1-intro")
    erste = await a.generate_music_eleven_endpoint(req, api_key="x")
    zweite = await a.generate_music_eleven_endpoint(req, api_key="x")
    assert erste["duration_seconds"] == 31.2 and erste["duration_requested_s"] == 30.0
    assert zweite["replayed"] is True and aufrufe["n"] == 1


def test_keine_wiederholung_im_quelltext():
    import inspect
    q = inspect.getsource(mg.generate_music_elevenlabs)
    assert "return await _attempt_request(p_retry)" not in q
    assert "music_prompt_rejected" in q
