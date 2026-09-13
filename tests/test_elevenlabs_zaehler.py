"""ElevenLabs-Zaehler (Story, 13.09.): Sperre VOR dem Sprechen, in jedem Pfad.

Bis heute stand ElevenLabs in keinem Zaehler — 10 000 Zeichen inklusive,
danach payg, und bei uns bremste nichts. Die Tonengine der Story-Engine
wuerde je Szene fuenf Spuren auf diesen Topf legen.

Ziel: Deckel in Zeichen, 429 bevor gesprochen wird. Gegenfall: unter dem
Deckel geht es durch, Geraeusche/Musik zaehlen sichtbar, aber nicht in
den Zeichendeckel. Nachbarfall: Kontingent beim Anbieter aufgebraucht
-> 402, auch wenn unser Deckel noch Luft hat. Verdrahtung: alle fuenf
Aufrufstellen rufen pre_check vor dem Anbieter und track danach.
"""

import inspect

import pytest
from fastapi import HTTPException

from ai.services import elevenlabs_cost_tracker as ez


@pytest.fixture
def frisch(monkeypatch, tmp_path):
    """Frischer Singleton mit eigener Ablage und Deckel 1000 Zeichen."""
    monkeypatch.setenv("ELEVENLABS_COST_TRACKER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ELEVENLABS_MONTHLY_CHAR_CAP", "1000")
    monkeypatch.setenv("ELEVENLABS_BLOCK_BEYOND_INCLUDED", "false")
    monkeypatch.delenv("ELEVENLABS_COST_TRACKER_MASTER_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    ez.ElevenLabsCostTracker._instance = None
    t = ez.ElevenLabsCostTracker()
    yield t
    ez.ElevenLabsCostTracker._instance = None


def test_unter_dem_deckel_geht_es_durch_und_wird_gezaehlt(frisch):
    frisch.pre_check(400, endpoint="narrate")
    frisch.track_tts(400, caller="narrate", audio_seconds=4.1)
    st = frisch.get_status()
    assert st["chars_used"] == 400 and st["tts_calls"] == 1
    assert st["audio_seconds_total"] == 4.1
    assert st["usage_percentage"] == 40.0
    assert st["by_caller"]["narrate"]["chars"] == 400


def test_deckel_sperrt_vor_dem_sprechen(frisch):
    frisch.track_tts(900, caller="tts_service")
    with pytest.raises(HTTPException) as exc:
        frisch.pre_check(200, endpoint="narrate")      # 900 + 200 > 1000
    assert exc.value.status_code == 429
    assert exc.value.detail["error"] == "monthly_api_cap_reached"
    assert exc.value.detail["chars_planned"] == 200
    frisch.pre_check(100, endpoint="narrate")          # genau bis zum Deckel: frei


def test_der_ganze_text_zaehlt_nicht_nur_das_erste_stueck():
    """Verdrahtung in tts_service: pre_check mit len(text) VOR der Schleife."""
    from ai.services import tts_service
    q = inspect.getsource(tts_service.generate_elevenlabs_tts)
    assert "elevenlabs_cost_tracker.pre_check(len(text)" in q
    assert q.index("pre_check(len(text)") < q.index("for i, chunk_text_str in enumerate(text_chunks)")
    assert q.count("elevenlabs_cost_tracker.track_tts(") == 2   # beide Pfade


def test_geraeusche_und_musik_zaehlen_sichtbar_aber_nicht_im_deckel(frisch):
    frisch.track_sfx(caller="gensfx")
    frisch.track_music(caller="genmusic_eleven", seconds_requested=30)
    st = frisch.get_status()
    assert st["sfx_calls"] == 1 and st["music_calls"] == 1
    assert st["music_seconds_requested"] == 30.0
    assert st["chars_used"] == 0 and st["sfx_music_in_char_cap"] is False


def test_hard_cap_sperrt_auch_geraeusche(frisch):
    frisch.trip_hard_cap("manuell")
    with pytest.raises(HTTPException) as exc:
        frisch.pre_check(0, endpoint="gensfx")
    assert exc.value.status_code == 429
    assert frisch.clear_hard_cap()["was_active"] is True
    frisch.pre_check(0, endpoint="gensfx")


def test_kontingent_beim_anbieter_aufgebraucht_gibt_402(frisch, monkeypatch):
    frisch.block_beyond_included = True
    monkeypatch.setattr(frisch, "subscription_snapshot", lambda force=False: {
        "tier": "payg", "character_count": 9990, "character_limit": 10000, "remaining": 10,
    })
    with pytest.raises(HTTPException) as exc:
        frisch.pre_check(50, endpoint="narrate")       # unser Deckel haette Luft
    assert exc.value.status_code == 402
    assert exc.value.detail["error"] == "elevenlabs_quota_exhausted"
    frisch.pre_check(5, endpoint="narrate")            # passt noch ins Kontingent


def test_ohne_preis_kein_erfundener_euro(frisch):
    frisch.track_tts(500, caller="narrate")
    st = frisch.get_status()
    assert st["price_per_1k_chars_usd"] is None and st["total_cost_eur"] is None


def test_mit_preis_wird_gerechnet(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_COST_TRACKER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ELEVENLABS_PRICE_PER_1K_CHARS_USD", "0.30")
    monkeypatch.setenv("ELEVENLABS_BLOCK_BEYOND_INCLUDED", "false")
    ez.ElevenLabsCostTracker._instance = None
    t = ez.ElevenLabsCostTracker()
    t.track_tts(2000, caller="narrate")
    assert round(t.get_status()["total_cost_eur"], 4) == round(0.60 / ez.EUR_USD_RATE, 4)
    ez.ElevenLabsCostTracker._instance = None


@pytest.mark.parametrize("modul,funktion,endpunkt", [
    ("ai.services.narration_service", "NarrationService._generate_tts", "narrate"),
    ("ai.routes.audio_ai_routes", "generate_sfx_endpoint", "gensfx"),
    ("ai.services.audio_drama_service", "AudioDramaGenerator._source_single_sfx", "dialog-sfx"),
    ("ai.routes.music_generation", "generate_music_elevenlabs", "genmusic_eleven"),
])
def test_jede_aufrufstelle_sperrt_vor_dem_anbieter(modul, funktion, endpunkt):
    import importlib
    m = importlib.import_module(modul)
    obj = m
    for teil in funktion.split("."):
        obj = getattr(obj, teil)
    q = inspect.getsource(obj)
    assert f'pre_check(' in q and f'endpoint="{endpunkt}"' in q, funktion
    # Sperre steht VOR dem Anbieteraufruf
    marker = "AsyncElevenLabs(" if "AsyncElevenLabs(" in q else "music/compose"
    assert q.index("pre_check(") < q.index(marker), funktion


def test_status_route_existiert():
    from ai.routes import narration_routes as r
    pfade = [getattr(x, "path", "") for x in r.router.routes]
    assert "/tts/elevenlabs/cost-status" in pfade


def test_client_status_zeigt_den_master(monkeypatch, tmp_path):
    """Story-Codex, 13.09.: am oeffentlichen Ziel (Client) stand
    chars_used=0 nach vier echten Aufrufen — der Master hatte 276."""
    monkeypatch.setenv("ELEVENLABS_COST_TRACKER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ELEVENLABS_COST_TRACKER_MASTER_URL", "https://master.example")
    monkeypatch.setenv("COST_TRACKER_SHARED_SECRET", "s")
    monkeypatch.setenv("ELEVENLABS_BLOCK_BEYOND_INCLUDED", "false")
    ez.ElevenLabsCostTracker._instance = None
    t = ez.ElevenLabsCostTracker()
    monkeypatch.setattr(t, "_fetch_master_status", lambda: {"chars_used": 276, "tts_calls": 4, "monthly_char_cap": 10000})
    st = t.get_status()
    assert st["chars_used"] == 276 and st["tts_calls"] == 4 and st["view"] == "master"
    # Master nicht erreichbar -> lokale Sicht, ehrlich markiert
    def kaputt():
        raise RuntimeError("down")
    monkeypatch.setattr(t, "_fetch_master_status", kaputt)
    st2 = t.get_status()
    assert st2["view"] == "local" and st2["chars_used"] == 0
    ez.ElevenLabsCostTracker._instance = None
