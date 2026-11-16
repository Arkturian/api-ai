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


def test_transkription_bucht_nach_dauer(schluessel, monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: 120.0)
    modelle = _Modelle(antwort=types.SimpleNamespace(text="hallo welt", usage_metadata=None))
    gesehen = _genai(monkeypatch, modelle)
    r = asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-3.5-transcribe", aufrufer="Comm"))
    assert r["text"] == "hallo welt" and r["kosten_usd"] == 0.0102
    assert gesehen["api_key"] == "testschluessel"
    assert modelle.aufrufe[0]["model"] == "gemini-3.5-transcribe"
    assert gu.stand()["gesamt_usd"] == 0.0102
    assert _buchungen()[0]["aufrufer"] == "Comm"


def test_transkription_altes_modell_422_ohne_aufruf(schluessel, monkeypatch):
    modelle = _Modelle()
    _genai(monkeypatch, modelle)
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-2.0-flash-exp"))
    assert e.value.status_code == 422 and modelle.aufrufe == []


def test_transkription_fehler_storniert(schluessel, monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: 60.0)
    _genai(monkeypatch, _Modelle(fehler=RuntimeError("quota")))
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-3.5-transcribe"))
    assert e.value.status_code == 502
    assert gu.stand()["gesamt_usd"] == 0.0


def test_transkription_ohne_dauer_kein_aufruf(schluessel, monkeypatch):
    monkeypatch.setattr(a, "_audio_duration_seconds", lambda d: None)
    modelle = _Modelle()
    _genai(monkeypatch, modelle)
    with pytest.raises(HTTPException) as e:
        asyncio.run(a._transcribe_with_gemini(_upload(), "gemini-3.5-transcribe"))
    assert e.value.status_code == 422 and modelle.aufrufe == []


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
