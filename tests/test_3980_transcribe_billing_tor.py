"""#3980: /ai/transcribe mit OpenAI-Modellen braucht confirm_api_billing.

Schalter TRANSCRIBE_BILLING_TOR: `beobachten` (Vorgabe) sperrt nicht, markiert
aber die Antwort; jeder andere Wert sperrt mit 403 und Preis, bevor Audio
abgerufen oder OpenAI gerufen wird. Gemini bleibt unveraendert.
"""
import types

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import audio_ai_routes as ar


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_ACCESS_KEY", raising=False)
    monkeypatch.delenv("TRANSCRIBE_BILLING_TOR", raising=False)
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def whisper(monkeypatch):
    aufrufe = []

    class _Tr:
        async def create(self, **kw):
            aufrufe.append(kw)
            return types.SimpleNamespace(text="hallo", language="german", duration=60.0)

    class _Client:
        def __init__(self, **kw):
            self.audio = types.SimpleNamespace(transcriptions=_Tr())

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ar, "AsyncOpenAI", _Client)
    monkeypatch.setattr(ar, "_normalize_audio_for_openai", lambda data, base: (data, ".mp3"))
    monkeypatch.setattr(ar, "_audio_duration_seconds", lambda data: 60.0)
    return aufrufe


def _post(client, **form):
    return client.post("/ai/transcribe", data=form,
                       files={"file": ("a.mp3", b"ID3-nicht-leer", "audio/mpeg")})


def test_beobachten_sperrt_nicht_und_markiert(client, whisper, caplog):
    with caplog.at_level("WARNING"):
        r = _post(client)
    assert r.status_code == 200, r.text
    h = r.json()["billing_hinweis"]
    assert h["status"] == "beobachtet" and h["usd_per_minute"] == 0.006 and h["model"] == "whisper-1"
    assert len(whisper) == 1
    assert any("TRANSCRIBE OHNE confirm_api_billing" in m for m in caplog.messages)


def test_beobachten_mit_flag_ohne_hinweis(client, whisper):
    r = _post(client, confirm_api_billing="true")
    assert r.status_code == 200 and "billing_hinweis" not in r.json()


@pytest.mark.parametrize("modus", ["an", "AN", "scharf", "tippfehler"])
def test_scharf_ohne_flag_403_vor_anbieter(client, whisper, monkeypatch, modus):
    monkeypatch.setenv("TRANSCRIBE_BILLING_TOR", modus)
    r = _post(client)
    assert r.status_code == 403
    d = r.json()["detail"]
    assert d["error"] == "api_billing_confirmation_required" and d["usd_per_minute"] == 0.006
    assert whisper == []


def test_scharf_vor_dem_abruf_einer_url(client, whisper, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BILLING_TOR", "an")
    abrufe = []

    async def fake_fetch(url):
        abrufe.append(url)
        raise AssertionError("darf nicht abgerufen werden")
    monkeypatch.setattr(ar, "_fetch_audio_from_url", fake_fetch)
    r = client.post("/ai/transcribe", data={"file_url": "https://example.org/a.mp3"})
    assert r.status_code == 403 and abrufe == [] and whisper == []


@pytest.mark.parametrize("wert", ["true", "1", "yes", "True"])
def test_scharf_mit_flag_laeuft(client, whisper, monkeypatch, wert):
    monkeypatch.setenv("TRANSCRIBE_BILLING_TOR", "an")
    r = _post(client, confirm_api_billing=wert)
    assert r.status_code == 200 and r.json()["text"] == "hallo" and len(whisper) == 1


def test_scharf_preis_je_modell(client, whisper, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BILLING_TOR", "an")
    r = _post(client, model="gpt-4o-mini-transcribe")
    assert r.status_code == 403 and r.json()["detail"]["usd_per_minute"] == 0.003


def test_gemini_unveraendert_ohne_flag_403(client, whisper, monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_BILLING_TOR", raising=False)
    r = _post(client, model="gemini-3.5-transcribe")
    assert r.status_code == 403
    assert r.json()["detail"]["endpoint"] == "transcribe-gemini"
    assert whisper == []


@pytest.mark.parametrize("wert", ["false", "0", "no", ""])
def test_scharf_flag_false_ist_kein_ja(client, whisper, monkeypatch, wert):
    """Formularfelder sind Text: "false" waere in Python wahr."""
    monkeypatch.setenv("TRANSCRIBE_BILLING_TOR", "an")
    r = _post(client, confirm_api_billing=wert)
    assert r.status_code == 403 and whisper == []
