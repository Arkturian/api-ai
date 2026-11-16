"""#2867: Gemini-API auf dem Ultra-Guthaben, nur Transkription und TTS,
nur bis 13.12.2026, mit Tages- und Gesamtdeckel (Mai-Vorfall 209,56 EUR)."""
import asyncio
import io
import json
import os
import struct
import types
from datetime import datetime

import pytest
from fastapi import HTTPException, UploadFile

from ai.routes import audio_ai_routes as a
from ai.routes import narration_routes as n
from ai.services import google_ultra as gu


@pytest.fixture
def schluessel(monkeypatch, tmp_path):
    datei = tmp_path / "google-ultra.env"
    datei.write_text("GEMINI_ULTRA_API_KEY=testschluessel\n")
    monkeypatch.setenv("GEMINI_ULTRA_KEY_FILE", str(datei))
    monkeypatch.setattr(gu, "_jetzt", lambda: datetime(2026, 11, 16, 10, 0, tzinfo=gu.WIEN))
    return datei


def _buchungen():
    with open(gu._buch()) as f:
        return [json.loads(z) for z in f]


# --- Freigabe ---------------------------------------------------------------

def test_ohne_schluesseldatei_gesperrt(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_ULTRA_KEY_FILE", str(tmp_path / "fehlt.env"))
    with pytest.raises(HTTPException) as e:
        gu.freigabe("tts", None)
    assert e.value.status_code == 503


def test_ab_13_12_gesperrt_auch_mit_schluessel(schluessel, monkeypatch):
    monkeypatch.setattr(gu, "_jetzt", lambda: datetime(2026, 12, 12, 23, 59, tzinfo=gu.WIEN))
    assert gu.freigabe("tts", None) == "gemini-3.8-flash-tts"
    monkeypatch.setattr(gu, "_jetzt", lambda: datetime(2026, 12, 13, 0, 0, tzinfo=gu.WIEN))
    with pytest.raises(HTTPException) as e:
        gu.freigabe("tts", None)
    assert e.value.status_code == 503
    assert "2026-12-13" in e.value.detail["grund"]


def test_unlesbares_ablaufdatum_sperrt(schluessel, monkeypatch):
    monkeypatch.setenv("GOOGLE_ULTRA_BIS", "irgendwann")
    with pytest.raises(HTTPException) as e:
        gu.freigabe("transcribe", None)
    assert e.value.status_code == 503


def test_nur_freigegebene_modelle_kein_rueckfall(schluessel):
    with pytest.raises(HTTPException) as e:
        gu.freigabe("transcribe", "gemini-2.5-flash")
    assert e.value.status_code == 422
    assert e.value.detail["available"] == ["gemini-3.5-flash-lite", "gemini-3.5-transcribe"]
    # Text, Bild, Video gibt es ueber diesen Weg gar nicht
    with pytest.raises(HTTPException) as e:
        gu.freigabe("text", "gemini-3.8-flash")
    assert e.value.status_code == 503


def test_schluessel_nie_in_der_prozessumgebung(schluessel):
    """CLI-Kindprozesse bekommen os.environ.copy(). Der Schluessel darf dort
    nie auftauchen, auch nicht nach dem Lesen."""
    assert gu.schluessel() == "testschluessel"
    assert "testschluessel" not in os.environ.values()


# --- Deckel -------------------------------------------------------------------

def test_tagesdeckel_greift_vor_dem_aufruf(schluessel, monkeypatch):
    monkeypatch.setenv("GOOGLE_ULTRA_TAG_USD", "0.10")
    # 10 min gemini-3.5-transcribe = 0,051 $
    gu.reservieren("transcribe", "gemini-3.5-transcribe", 10)
    with pytest.raises(HTTPException) as e:
        gu.reservieren("transcribe", "gemini-3.5-transcribe", 10)
    assert e.value.status_code == 429
    assert e.value.detail["error"] == "google_budget_erschoepft"
    # die abgewiesene Anfrage steht nicht im Buch
    assert len(_buchungen()) == 1


def test_gesamtdeckel_ueber_tage(schluessel, monkeypatch):
    monkeypatch.setenv("GOOGLE_ULTRA_GESAMT_USD", "0.20")
    gu.reservieren("tts", "gemini-3.8-flash-tts", 10)  # 0,135
    monkeypatch.setattr(gu, "_jetzt", lambda: datetime(2026, 11, 17, 10, 0, tzinfo=gu.WIEN))
    with pytest.raises(HTTPException) as e:
        gu.reservieren("tts", "gemini-3.8-flash-tts", 10)
    assert e.value.status_code == 429


def test_storno_und_berichtigung(schluessel):
    r = gu.reservieren("tts", "gemini-3.8-flash-tts", 10)
    gu.stornieren(r, "fehler")
    r2 = gu.reservieren("tts", "gemini-3.8-flash-tts", 10)
    assert gu.abrechnen(r2, 2.0) == 0.027
    s = gu.stand()
    assert s["gesamt_usd"] == 0.027
    assert s["je_art"]["tts"]["aufrufe"] == 1
    assert s["freigegeben"] is True


# --- Transkription ------------------------------------------------------------

class _Modelle:
    def __init__(self, antwort=None, fehler=None):
        self.antwort, self.fehler, self.aufrufe = antwort, fehler, []

    def generate_content(self, **kw):
        self.aufrufe.append(kw)
        if self.fehler:
            raise self.fehler
        return self.antwort


def _genai(monkeypatch, modelle):
    gesehen = {}

    class Client:
        def __init__(self, api_key=None):
            gesehen["api_key"] = api_key
            self.models = modelle
            self.files = types.SimpleNamespace(
                upload=lambda file: types.SimpleNamespace(name="files/x"),
                delete=lambda name: None)

    import google.genai as g
    monkeypatch.setattr(g, "Client", Client)
    return gesehen


def _upload():
    return UploadFile(filename="a.mp3", file=io.BytesIO(b"audio"))


# Echte Antwort von gemini-3.5-transcribe am 16.11. (gekuerzt): der Text
# steht in audioTranscription, nicht in text. Das SDK 2.6.0 verwarf ihn.
ANTWORT_TRANSCRIBE = {"candidates": [{"content": {"parts": [{"audioTranscription": {
    "text": "Guten Morgen. Das ist ein kurzer Test."}}], "role": "model"},
    "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 146, "candidatesTokenCount": 24}}
ANTWORT_FLASH_LITE = {"candidates": [{"content": {"parts": [
    {"text": "gedacht", "thought": True}, {"text": "Guten Morgen."}]}}]}


def test_text_aus_beiden_antwortformen():
    assert gu.text_aus_antwort(ANTWORT_TRANSCRIBE) == "Guten Morgen. Das ist ein kurzer Test."
    assert gu.text_aus_antwort(ANTWORT_FLASH_LITE) == "Guten Morgen."
    assert gu.text_aus_antwort({}) == ""


def _rest(monkeypatch, antwort=None, fehler=None):
    aufrufe = []

    async def generate(modell, rumpf, timeout=300.0):
        aufrufe.append((modell, rumpf))
        if fehler:
            raise fehler
        return antwort

    monkeypatch.setattr(gu, "generate", generate)
    return aufrufe


def test_transkription_bucht_nach_dauer(schluessel, monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: 120.0)
    aufrufe = _rest(monkeypatch, ANTWORT_TRANSCRIBE)
    r = asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-3.5-transcribe", aufrufer="Comm"))
    assert r["text"] == "Guten Morgen. Das ist ein kurzer Test." and r["kosten_usd"] == 0.0102
    modell, rumpf = aufrufe[0]
    assert modell == "gemini-3.5-transcribe"
    teile = rumpf["contents"][0]["parts"]
    assert teile[1]["inline_data"]["mime_type"] == "audio/mpeg"
    assert gu.stand()["gesamt_usd"] == 0.0102
    assert _buchungen()[0]["aufrufer"] == "Comm"
    assert _buchungen()[1]["tokens_in"] == 146


def test_transkription_altes_modell_422_ohne_aufruf(schluessel, monkeypatch):
    aufrufe = _rest(monkeypatch)
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-2.0-flash-exp"))
    assert e.value.status_code == 422 and aufrufe == []


def test_transkription_fehler_storniert(schluessel, monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: 60.0)
    _rest(monkeypatch, fehler=RuntimeError("quota"))
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-3.5-transcribe"))
    assert e.value.status_code == 502
    assert gu.stand()["gesamt_usd"] == 0.0


def test_transkription_leere_antwort_storniert(schluessel, monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: 60.0)
    _rest(monkeypatch, {"candidates": [{"content": {"parts": [{}]}}]})
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-3.5-transcribe"))
    assert e.value.status_code == 502
    assert gu.stand()["gesamt_usd"] == 0.0


def test_transkription_ohne_dauer_kein_aufruf(schluessel, monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: None)
    aufrufe = _rest(monkeypatch)
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-3.5-transcribe"))
    assert e.value.status_code == 422 and aufrufe == []


def test_schluessel_nur_im_kopf_nie_in_der_adresse(schluessel, monkeypatch):
    import httpx
    gesehen = {}

    class C:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None, json=None):
            gesehen.update(url=url, headers=headers)
            return httpx.Response(200, json=ANTWORT_TRANSCRIBE)

    monkeypatch.setattr(httpx, "AsyncClient", C)
    asyncio.run(gu.generate("gemini-3.5-transcribe", {}))
    assert "testschluessel" not in gesehen["url"]
    assert gesehen["headers"]["x-goog-api-key"] == "testschluessel"


# --- TTS ------------------------------------------------------------------------

def _pcm(sekunden, rate=24000):
    return struct.pack("<%dh" % int(rate * sekunden), *([0] * int(rate * sekunden)))


def test_tts_spricht_wandelt_und_bucht_ist_dauer(schluessel, monkeypatch):
    teil = types.SimpleNamespace(inline_data=types.SimpleNamespace(
        data=_pcm(3.0), mime_type="audio/L16;codec=pcm;rate=24000"))
    antwort = types.SimpleNamespace(candidates=[types.SimpleNamespace(
        content=types.SimpleNamespace(parts=[teil]))])
    modelle = _Modelle(antwort=antwort)
    _genai(monkeypatch, modelle)
    abgelegt = {}

    async def ablage(**kw):
        abgelegt.update(kw)
        return types.SimpleNamespace(id=7, file_url="https://x/7")

    import ai.clients.storage_client as sc
    monkeypatch.setattr(sc, "save_file_and_record", ablage)
    req = n.GeminiTTSRequest(text="Guten Morgen.", confirm_api_billing=True)
    r = asyncio.run(n.tts_gemini(req, authorization=None, x_agent_name="Story", api_key=""))
    assert r["duration_s"] == 3.0 and r["kosten_usd"] == 0.000675
    assert abgelegt["data"][:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3") or abgelegt["data"][0] == 0xFF
    cfg = modelle.aufrufe[0]["config"]
    assert cfg.response_modalities == ["AUDIO"]
    assert gu.stand()["gesamt_usd"] == 0.000675


def test_tts_ohne_bestaetigung_403(schluessel):
    with pytest.raises(HTTPException) as e:
        asyncio.run(n.tts_gemini(n.GeminiTTSRequest(text="x"), authorization=None,
                                 x_agent_name=None, api_key=""))
    assert e.value.status_code == 403


def test_tts_nach_ablauf_503(schluessel, monkeypatch):
    monkeypatch.setattr(gu, "_jetzt", lambda: datetime(2026, 12, 13, 8, 0, tzinfo=gu.WIEN))
    with pytest.raises(HTTPException) as e:
        asyncio.run(n.tts_gemini(n.GeminiTTSRequest(text="x", confirm_api_billing=True),
                                 authorization=None, x_agent_name=None, api_key=""))
    assert e.value.status_code == 503


def test_stand_nur_fuer_betreiber():
    from ai import auth_guard as ag
    route = next(r for r in n.router.routes if getattr(r, "path", "") == "/tts/gemini/stand")
    assert any(d.call is ag.betreiber for d in route.dependant.dependencies)
