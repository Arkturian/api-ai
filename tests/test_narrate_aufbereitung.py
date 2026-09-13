"""Dramaturgische Aufbereitung: Abo-Textpfad, Herkunft wahr, strikt vor TTS.

Story-Codex fand am Ziel: `_preprocess_text` rief das Gemini-SDK direkt
(toter Google-Schluessel), fing jeden Fehler, gab den Rohtext zurueck —
und `preprocessing_model` behauptete "gemini". Story (PM) setzte die
Vorgabe: Abo-Textpfad, striktes Verhalten als Option, Herkunft im
Ergebnis wahr.

Ziel: Erfolg -> prepared, Quelle "chatgpt:<modell>". Gegenfall: strikt +
Fehler -> 502 VOR dem Sprechen, TTS nie erreicht. Nachbarfall: nicht
strikt + Fehler -> Rohtext, aber markiert und ohne Modellbehauptung; der
Hoerspielpfad bekommt weiter nur den Text.
"""

import types

import httpx
import pytest
from fastapi import HTTPException

from ai.services import narration_service as n
from ai.services import tts_service


def _req(strikt=False):
    return n.NarrationRequest(
        text="Mira trat durch das Tor.",
        character=n.NarrationCharacter(name="Erz", voice_id="v", personality="warm"),
        config=n.NarrationConfig(preprocessing=True, preparation_strict=strikt),
        save_options={},
    )


@pytest.fixture(autouse=True)
def _rest(monkeypatch):
    monkeypatch.setattr(n.NarrationService, "_measure_audio_duration", staticmethod(lambda b, f: 1.0))

    async def fake_save(self, audio_bytes, request):
        return 4711, "https://s/4711"
    monkeypatch.setattr(n.NarrationService, "_save_audio", fake_save)
    monkeypatch.delenv("NARRATE_PREP_MODEL", raising=False)


_ECHTER_CLIENT = httpx.AsyncClient   # einmal gefangen: ein zweiter Stub darf nicht den ersten kapseln


def _chatgpt(monkeypatch, status=200, body=None):
    def handler(request):
        assert request.url.path == "/ai/chatgpt"
        return httpx.Response(status, json=body if body is not None else {"response": "Mira ... trat durch das Tor.", "model": "gpt-5.6-sol"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _ECHTER_CLIENT(transport=httpx.MockTransport(handler), **{x: y for x, y in k.items() if x != "transport"}))


def _tts(monkeypatch, gezaehlt):
    async def fake_tts(self, text, request, *a, **kw):
        gezaehlt["tts"] = gezaehlt.get("tts", 0) + 1
        gezaehlt["text"] = text
        return b"MP3", None
    monkeypatch.setattr(n.NarrationService, "_generate_tts", fake_tts)


@pytest.mark.asyncio
async def test_erfolg_ueber_abo_pfad_mit_herkunft(monkeypatch):
    _chatgpt(monkeypatch); z = {}; _tts(monkeypatch, z)
    out = await n.NarrationService().generate(_req())
    assert out.prepared is True
    assert out.preparation_source == "chatgpt:gpt-5.6-sol" and out.preprocessing_model == "gpt-5.6-sol"
    assert out.dramatic_script == "Mira ... trat durch das Tor." and z["text"] == out.dramatic_script


@pytest.mark.asyncio
async def test_strikt_scheitert_vor_dem_sprechen(monkeypatch):
    _chatgpt(monkeypatch, status=502, body={"detail": "codex down"}); z = {}; _tts(monkeypatch, z)
    with pytest.raises(HTTPException) as exc:
        await n.NarrationService().generate(_req(strikt=True))
    assert exc.value.status_code == 502 and exc.value.detail["error"] == "preparation_failed"
    assert z.get("tts", 0) == 0                                 # nie gesprochen


@pytest.mark.asyncio
async def test_nicht_strikt_rueckfall_ist_markiert(monkeypatch):
    _chatgpt(monkeypatch, status=502, body={"detail": "codex down"}); z = {}; _tts(monkeypatch, z)
    out = await n.NarrationService().generate(_req(strikt=False))
    assert out.prepared is False
    assert out.preparation_source == "legacy_fallback_original_text"
    assert out.preprocessing_model is None                      # keine Behauptung
    assert out.dramatic_script == "Mira trat durch das Tor." and z["tts"] == 1


@pytest.mark.asyncio
async def test_hoerspielpfad_bekommt_weiter_nur_text(monkeypatch):
    _chatgpt(monkeypatch)
    text = await n.NarrationService()._preprocess_text(_req(strikt=True))   # strikt wird dort NICHT angewandt
    assert isinstance(text, str) and text


def test_kein_gemini_sdk_mehr_im_pfad():
    import inspect
    q = inspect.getsource(n.NarrationService._preprocess_text_mit_herkunft)
    assert "generate_content" not in q and "/ai/chatgpt" in q   # der Docstring nennt das alte SDK als Geschichte


# ---------------------------------------------------- Vorschau (Story-Codex, 13.09.)

@pytest.mark.asyncio
async def test_vorschau_zeigt_herkunft_und_achtet_strikt(monkeypatch):
    from ai.routes import narration_routes as r
    _chatgpt(monkeypatch)
    out = await r.narrate_preview(_req(), api_key="x")
    import json
    d = json.loads(out.body)
    assert d["prepared"] is True and d["preparation_source"] == "chatgpt:gpt-5.6-sol"
    _chatgpt(monkeypatch, status=502, body={"detail": "down"})
    d2 = json.loads((await r.narrate_preview(_req(strikt=False), api_key="x")).body)
    assert d2["prepared"] is False and d2["preparation_source"] == "legacy_fallback_original_text"
    with pytest.raises(HTTPException) as exc:
        await r.narrate_preview(_req(strikt=True), api_key="x")
    assert exc.value.status_code == 502 and exc.value.detail["error"] == "preparation_failed"
