"""#1884: GPT-Live-1 als Prototyp-Provider. Gemessen 15.09.2026 (WS-Sonde):
session.start mit Responses-Delegation, Deutsch und Slowenisch verstanden
und in der Sprache beantwortet, Werkzeugaufruf ueber das Backend (auto und
required), erste Transkriptausgabe ~0,9 s nach Sprachende, Abrechnung in
Sekunden (usage.seconds), Backend-Token je Delegation (818 in / 49 out)."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai.routes import realtime_routes as rr
from ai.services import openai_realtime_cost_tracker as k


# --- Preise + Kosten ------------------------------------------------------

def test_live_und_backend_preise_vom_15_09():
    assert k.OPENAI_LIVE_PRICING["gpt-live-1"]["per_minute_usd"] == 0.05
    assert k.OPENAI_BACKEND_TEXT_PRICING["gpt-5.6-luna"] == {"input_per_1m": 0.20, "cached_input_per_1m": 0.02, "output_per_1m": 1.20}
    assert k.OPENAI_BACKEND_TEXT_PRICING["gpt-6-astra"]["output_per_1m"] == 50.0
    assert k.OPENAI_BACKEND_TEXT_PRICING["default"] == k.OPENAI_BACKEND_TEXT_PRICING["gpt-6-astra"]   # teuerster Rueckfall


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


def test_live_kosten_sekunden_plus_backend(zaehler):
    usd, eur = zaehler._cost_for_live(120.0, "gpt-5.6-luna", 10_000, 8_000, 500)
    erwartet = 2 * 0.05 + (2_000 * 0.20 + 8_000 * 0.02 + 500 * 1.20) / 1e6
    assert usd == pytest.approx(erwartet)
    assert eur == pytest.approx(erwartet / k.EUR_USD_RATE)


def test_track_live_bucht_und_dedupt(zaehler):
    z = zaehler
    r = z.track_live(live_seconds=10.0, backend_model="gpt-5.6-luna", backend_input_tokens=818,
                     backend_output_tokens=49, voice_session_id="v1", usage_event_id="live_u0_x")
    assert r["accepted"] and not r["deduped"]
    st = z._usage_data["by_model"]["gpt-live-1"]
    assert st["modality"] == "live" and st["live_seconds"] == 10.0
    assert st["backend"]["gpt-5.6-luna"]["input_tokens"] == 818 and st["backend"]["gpt-5.6-luna"]["output_tokens"] == 49
    assert z._usage_data["total_cost_usd"] == pytest.approx(10 / 60 * 0.05 + (818 * 0.20 + 49 * 1.20) / 1e6)
    r2 = z.track_live(live_seconds=10.0, voice_session_id="v1", usage_event_id="live_u0_x")
    assert r2["deduped"] is True
    assert z.track_live(live_seconds=0, voice_session_id="v1", usage_event_id="leer")["accepted"] is False


# --- Endpunkt -------------------------------------------------------------

class _Grant:
    monthly_budget_eur = None
    daily_budget_eur = 10.0
    max_parallel_sessions = 2
    profile_id = "p-test"
    sub = "u-test"
    scopes = ("mint", "usage")
    host_key = "arkserver"
    grant_id = "g-test"
    expires_at = 0


@pytest.fixture
def client(monkeypatch, tmp_path):
    gesehen = {}

    class _Antwort:
        status_code = 201
        text = ""
        headers = {"content-type": "application/json"}
        def json(self):
            return {"session": {"id": "live_test_1", "model": "gpt-live-1", "expires_at": 1789486403},
                    "transport": {"type": "webrtc", "sdp": "v=0\r\no=- answer\r\n"}}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None, **k):
            gesehen["url"] = url; gesehen["json"] = json; gesehen["headers"] = headers
            return _Antwort()

    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-nicht-echt")
    monkeypatch.setenv("OPENAI_REALTIME_COST_TRACKER_DATA_DIR", str(tmp_path))
    from ai.services import realtime_budget_guard as guard
    monkeypatch.setattr(guard, "RESERVATIONS_PATH", tmp_path / "res.json")
    app = FastAPI()
    app.include_router(rr.router, prefix="/ai")
    app.dependency_overrides[rr.get_api_key] = lambda: "test-key"
    for route in app.routes:
        for dep in getattr(getattr(route, "dependant", None), "dependencies", []):
            if getattr(dep.call, "__qualname__", "").startswith("require_realtime_grant"):
                app.dependency_overrides[dep.call] = lambda: _Grant()
    c = TestClient(app, raise_server_exceptions=True)
    c.gesehen = gesehen
    return c


def test_sdp_endpunkt_baut_live_sitzung_mit_delegation(client):
    r = client.post("/ai/realtime/live/sdp", json={"sdp": "v=0\r\no=- offer\r\n", "language": "de",
                                                   "confirm_api_billing": True, "session_id": "s-1"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provider"] == "openai-live" and d["model"] == "gpt-live-1"
    assert d["sdp"].startswith("v=0") and d["live_session_id"] == "live_test_1"
    assert d["backend_model"] == "gpt-5.6-luna" and "sende_hinweis" not in d["tools"]
    g = client.gesehen
    assert g["url"] == "https://api.openai.com/v1/live/sessions"
    assert g["json"]["transport"] == {"type": "webrtc", "sdp": "v=0\r\no=- offer\r\n"}
    s = g["json"]["session"]
    assert s["model"] == "gpt-live-1" and s["audio"]["output"]["voice"] == "marin"
    assert "format" not in s["audio"]                       # WebRTC: Format ueber SDP
    assert s["delegation"]["type"] == "responses"
    assert s["delegation"]["responses"]["model"] == "gpt-5.6-luna"
    assert s["delegation"]["responses"]["tool_choice"] == "auto"
    assert len(s["delegation"]["responses"]["tools"]) == len(rr._all_tool_defs())
    assert len(s["instructions"]) < 4000                    # Sprech-Persona kurz, Verfahren im Backend
    assert len(s["delegation"]["responses"]["instructions"]) > len(s["instructions"])
    assert d["voice_session_id"]


def test_sdp_endpunkt_ohne_kostenzustimmung_ist_403(client):
    r = client.post("/ai/realtime/live/sdp", json={"sdp": "v=0", "confirm_api_billing": False})
    assert r.status_code == 403


def test_sdp_endpunkt_unbekanntes_backend_ist_400(client):
    r = client.post("/ai/realtime/live/sdp", json={"sdp": "v=0", "confirm_api_billing": True, "backend_model": "gpt-9"})
    assert r.status_code == 400 and r.json()["detail"]["error"] == "unsupported_live_backend_model"


def test_modelliste_fuehrt_live_als_prototyp():
    import asyncio
    d = asyncio.run(rr.list_realtime_models())
    m = {x["id"]: x for x in d["models"]}["gpt-live-1"]
    assert m["provider"] == "openai-live" and m["tier"] == "prototype" and m["default"] is False


def test_usage_report_nimmt_live_felder(client, monkeypatch):
    from ai.services.openai_realtime_cost_tracker import openai_realtime_cost_tracker as t
    aufrufe = []
    monkeypatch.setattr(t, "track_live", lambda **kw: (aufrufe.append(kw) or {"accepted": True, "deduped": False}))
    r = client.post("/ai/realtime/usage", json={"model": "gpt-live-1", "live_seconds": 12.0, "backend_model": "gpt-5.6-luna",
                                                "backend_input_tokens": 818, "backend_output_tokens": 49,
                                                "voice_session_id": "v9", "usage_event_id": "live_u0_9"})
    assert r.status_code == 200, r.text
    assert aufrufe and aufrufe[0]["live_seconds"] == 12.0 and aufrufe[0]["backend_model"] == "gpt-5.6-luna"
