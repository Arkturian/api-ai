"""30.10.2026: /ai/claude meldet "kein Claude-Konto frei" als 503 statt 500
mit Anbietertext (Cloud/Alex); /ai/gemini kann per GEMINI_FORWARD_URL an
einen anderen api-ai weiterreichen (arkturian: "User location is not
supported"). Die Meldungen unten sind die echten Texte vom 30.10."""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t


@pytest.mark.parametrize("text,grund,reset", [
    ("You've hit your weekly limit · resets Nov 1, 1pm (Europe/Vienna)", "limit", "Nov 1, 1pm (Europe/Vienna)"),
    ("Failed to authenticate: OAuth token revoked. Please log in again or contact your administrator.", "anmeldung", None),
])
def test_konto_fehler_werden_503(text, grund, reset):
    with pytest.raises(HTTPException) as e:
        t._claude_konto_pruefen(text)
    d = e.value.detail
    assert e.value.status_code == 503 and d["error"] == "claude_account_unavailable"
    assert d["grund"] == grund and d["resets"] == reset and d["fallback"] is None


def test_andere_fehler_bleiben_unberuehrt():
    t._claude_konto_pruefen("Prompt is too long")


def test_endpunkt_liefert_503(monkeypatch):
    import json, subprocess

    class R:
        returncode = 1
        stdout = json.dumps({"is_error": True, "result": "You've hit your weekly limit · resets Nov 1, 1pm (Europe/Vienna)"})
        stderr = ""
    monkeypatch.setattr(t, "_run_cli_with_pgid", lambda *a, **k: R())
    r = TestClient(main.app).post("/ai/claude", json={"prompt": "ok"})
    assert r.status_code == 503, r.text
    assert r.json()["detail"]["grund"] == "limit"


class _Antwort:
    status_code = 200
    def json(self):
        return {"response": "ok", "model": "gemini-3.8-flash"}


def _fake_client(gesehen):
    class C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, params=None, headers=None, json=None):
            gesehen.update(url=url, params=params, headers=headers, json=json)
            return _Antwort()
    return C


def test_gemini_wird_weitergereicht(monkeypatch):
    import httpx
    gesehen = {}
    monkeypatch.setenv("GEMINI_FORWARD_URL", "https://ziel.example/")
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(gesehen))
    r = TestClient(main.app).post("/ai/gemini?model=gemini-3.8-flash", json={"prompt": "ok", "effort": "low"},
                                  headers={"X-Agent-Name": "SWFME"})
    assert r.status_code == 200 and r.json()["forwarded_to"] == "https://ziel.example"
    assert gesehen["url"] == "https://ziel.example/ai/gemini" and gesehen["params"]["model"] == "gemini-3.8-flash"
    assert gesehen["headers"]["X-Api-Ai-Forwarded"] == "1" and gesehen["headers"]["x-agent-name"] == "SWFME"
    assert gesehen["json"]["effort"] == "low"


@pytest.mark.parametrize("kopf,body", [
    ({"X-Api-Ai-Forwarded": "1"}, {"prompt": "ok"}),            # nie zweimal weiterreichen
    ({}, {"prompt": "ok", "image_paths": ["/tmp/x.png"]}),     # lokale Datei sieht der Ziel-Host nicht
])
def test_gemini_bleibt_lokal(monkeypatch, kopf, body):
    import httpx
    gesehen = {}
    monkeypatch.setenv("GEMINI_FORWARD_URL", "https://ziel.example")
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(gesehen))

    class Stop(Exception):
        pass
    def halt(*a, **k):
        raise Stop
    monkeypatch.setattr(t, "_weise_response_format_ab", halt)
    with pytest.raises(Stop):
        TestClient(main.app).post("/ai/gemini", json=body, headers=kopf)
    assert gesehen == {}
