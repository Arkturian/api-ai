"""#2938 (Befund 59): Betreiber-Schalter mit Platzhalter-Pruefung, offene
Status-/Konto-Routen, Storage-Schluessel auf Aufrufer-Adressen, ungedeckelte
Szenenbilder, GCP-Webhook, CORS mit Credentials, uncodierte Agentennamen."""
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from ai import auth_guard as ag

SCHLUESSEL = "betreiber-2938"


@pytest.fixture
def c(monkeypatch):
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", SCHLUESSEL)
    monkeypatch.delenv("GCP_WEBHOOK_AUDIENCE", raising=False)
    return TestClient(main.app)


@pytest.mark.parametrize("methode,pfad", [
    ("post", "/ai/gemini/cost-status/reset-hard-cap"),
    ("post", "/ai/notifications/reset/gcp_budget_alert"),
    ("post", "/ai/gemini/send-report"),
    ("post", "/ai/status/smoketest"),
    ("get", "/internal/cli-pressure"),
    ("get", "/ai/tts/elevenlabs/cost-status"),
])
def test_betreiber_routen_ohne_schluessel_401(c, methode, pfad):
    assert getattr(c, methode)(pfad).status_code == 401
    assert getattr(c, methode)(pfad, headers={"X-API-KEY": "falsch"}).status_code == 401


def test_reset_mit_schluessel_geht(c, monkeypatch):
    from ai.services.cost_tracker import cost_tracker
    monkeypatch.setattr(cost_tracker, "clear_gcp_hard_cap", lambda: {"was_active": False})
    r = c.post("/ai/gemini/cost-status/reset-hard-cap", headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 200


@pytest.mark.parametrize("pfad", ["/ai/status", "/ai/status.json", "/ai/chatgpt/irgendwas"])
def test_status_nur_lokal_oder_betreiber(c, pfad):
    # TestClient ist nicht Loopback -> 401; ueber nginx (X-Real-IP fremd) ebenso
    assert c.get(pfad).status_code == 401
    assert c.get(pfad, headers={"X-Real-IP": "1.2.3.4"}).status_code == 401


@pytest.mark.parametrize("host,kopf,erwartet", [
    ("127.0.0.1", {}, True),
    ("127.0.0.1", {"x-real-ip": "127.0.0.1"}, True),     # lokaler Dienst ueber lokales nginx
    ("127.0.0.1", {"x-real-ip": "1.2.3.4"}, False),      # fremd ueber nginx
    ("127.0.0.1", {"x-forwarded-for": "127.0.0.1"}, False),  # XFF allein zaehlt nie
    ("10.0.0.5", {}, False),
    # uvicorn: alle XFF-Eintraege vertraut -> client None (lokales nginx, 15.11. live 401)
    (None, {"x-real-ip": "127.0.0.1", "x-forwarded-for": "127.0.0.1"}, True),
    (None, {"x-real-ip": "1.2.3.4", "x-forwarded-for": "1.2.3.4"}, False),
    (None, {}, False),
])
def test_von_diesem_host(host, kopf, erwartet):
    req = SimpleNamespace(client=SimpleNamespace(host=host), headers=kopf)
    assert ag.von_diesem_host(req) is erwartet


@pytest.mark.parametrize("pfad", ["/ai/genvideo/flow/account", "/ai/genvideo/kling/account"])
def test_konten_nur_angemeldet(c, pfad):
    assert c.get(pfad).status_code == 401


def test_flow_startbild_ohne_dienstschluessel():
    import inspect
    from ai.routes import flow_routes as fr
    quelle = inspect.getsource(fr._bild_hochladen)
    assert "mit_schluessel=False" in quelle and "mit_schluessel=True" not in quelle


def test_szenenbilder_gedeckelt(monkeypatch):
    from fastapi import HTTPException
    from ai.routes import image_generation_routes as ig
    monkeypatch.setenv("SCENE_MAX_IMAGES", "5")
    woerter = [{"word": "a", "start": i, "end": i + 1} for i in range(100)]
    with pytest.raises(HTTPException) as e:
        ig._segment_timestamps(woerter, 1.0)
    assert e.value.status_code == 422
    assert ig._segment_timestamps(woerter[:5], 1.0)


def test_webhook_mit_audience_ohne_token_401(c, monkeypatch):
    monkeypatch.setenv("GCP_WEBHOOK_AUDIENCE", "https://api-ai.arkturian.com/ai/gemini/gcp-budget-webhook")
    from ai.services.cost_tracker import cost_tracker
    ausgeloest = []
    monkeypatch.setattr(cost_tracker, "trip_gcp_hard_cap", lambda reason: ausgeloest.append(reason))
    import base64, json
    nachricht = base64.b64encode(json.dumps({"alertThresholdExceeded": 1.0, "costAmount": 1, "budgetAmount": 1}).encode()).decode()
    r = c.post("/ai/gemini/gcp-budget-webhook", json={"message": {"data": nachricht}})
    assert r.status_code == 401 and ausgeloest == []


def test_cors_ohne_credentials():
    mw = [m for m in main.app.user_middleware if "CORS" in str(m.cls)]
    assert mw and mw[0].kwargs.get("allow_credentials") is False


def test_bezahlpfade_vollstaendiger():
    for p in ("/ai/translate", "/ai/translate/batch", "/ai/upscale", "/ai/gendepth",
              "/ai/gendepth/result/x", "/ai/audio/mix", "/ai/realtime/token"):
        assert main._ist_bezahlpfad(p), p


def test_agentenname_als_ein_segment():
    from ai.routes import realtime_routes as rr
    assert rr._quote_seg("../x?y#z") == "..%2Fx%3Fy%23z"
