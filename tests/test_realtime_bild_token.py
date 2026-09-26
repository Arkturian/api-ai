"""Bild-Token im Realtime-Zaehler (CloudV2/Alex 26.09.2026: Arcturian sieht
den Bildschirm nativ). Gemessen: gpt-realtime und 2.1 melden je 640x360-JPEG
323 image_tokens; Preis beider 5 USD/1M, zwischengespeichert 0,50."""
import pytest

from ai.routes import realtime_routes as rr
from ai.services import openai_realtime_cost_tracker as k


@pytest.mark.parametrize("modell,voll,gecacht", [
    ("gpt-realtime", 5.0, 0.50), ("gpt-realtime-2.1", 5.0, 0.50), ("gpt-realtime-2.1-mini", 0.80, 0.08)])
def test_bildpreise(modell, voll, gecacht):
    p = k.OPENAI_REALTIME_PRICING[modell]
    assert p["image_input_per_1m"] == voll and p["image_input_cached_per_1m"] == gecacht


@pytest.fixture
def zaehler(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_REALTIME_COST_TRACKER_DATA_DIR", str(tmp_path))
    z = object.__new__(k.OpenAIRealtimeCostTracker)
    z._initialized = False
    z.__init__()
    z.master_url = None
    z._reset_monthly_data()
    z._save_data()
    return z


def test_kosten_mit_bild_und_cache(zaehler):
    usd, _ = zaehler._cost_for_session("gpt-realtime-2.1", 0, 0, 0, 10, image_input_tokens=1000,
                                        cached_image_input_tokens=600)
    assert usd == pytest.approx((400 * 5.0 + 600 * 0.5 + 10 * 24.0) / 1e6)


def test_nur_bild_eingabe_wird_angenommen_und_gezaehlt(zaehler):
    r = zaehler.track_session(model="gpt-realtime", image_input_tokens=323, text_input_tokens=146,
                              text_output_tokens=20, voice_session_id="v1", usage_event_id="r1")
    assert r["accepted"]
    st = zaehler._usage_data["by_model"]["gpt-realtime"]
    assert st["image_input_tokens"] == 323 and st["cached_image_input_tokens"] == 0


def test_21_bildantwort_schaetzt_nur_text_nicht_bild(zaehler):
    """2.1 meldet bei Textausgabe nur image_tokens, text 0 — der Text-Kontext
    wird wie gehabt geschaetzt, das Bild zaehlt wie gemeldet."""
    zaehler.track_session(model="gpt-realtime-2.1", text_input_tokens=5000, audio_input_tokens=200,
                          audio_output_tokens=100, voice_session_id="v2", usage_event_id="r1")
    r = zaehler.track_session(model="gpt-realtime-2.1", image_input_tokens=323, text_output_tokens=20,
                              voice_session_id="v2", usage_event_id="r2")
    assert r["input_estimated"] is True
    st = zaehler._usage_data["by_model"]["gpt-realtime-2.1"]
    assert st["image_input_tokens"] == 323
    assert st["estimated_input_tokens"] == 5200


def test_usage_bericht_kennt_bildfelder():
    f = rr.RealtimeUsageReport.model_fields
    assert "image_input_tokens" in f and "cached_image_input_tokens" in f


def test_prompt_image_attached():
    z = rr._screen_tools_addendum("de")
    assert "image_attached" in z
    assert "image_attached" in rr._screen_tool_defs()[1]["description"]
