"""Bildschirm-Werkzeuge fuer die Portal-Sprachsitzung (CloudV2 26.09.2026,
Auftrag Alex). Opt-in per Mint-Feld `screen_tools`, damit der freigegebene
Arcturian-Vertrag (ein Modellwerkzeug) ohne Opt-in unveraendert bleibt.
look_at_screen holt das Bild mit dem Nutzer-JWT ueber cloud-api (nicht mit
dem Storage-Dienstschluessel: private_media ist fuer ihn gesperrt) und
beschreibt es ueber das ChatGPT-Abo."""
import asyncio
import types

import pytest

from ai.routes import realtime_routes as rr
from ai.routes import text_ai_routes as t


def test_werkzeugdefinitionen():
    namen = [w["name"] for w in rr._screen_tool_defs()]
    assert namen == ["screen_capture", "look_at_screen"]
    sc, look = rr._screen_tool_defs()
    assert sc["parameters"]["properties"] == {}
    assert look["parameters"]["required"] == ["storage_id", "question"]
    assert look["parameters"]["properties"]["storage_id"]["type"] == "integer"
    assert "look_at_screen" in rr.READ_TOOL_NAMES
    assert "screen_capture" not in rr.READ_TOOL_NAMES      # Client-Werkzeug


def test_mint_feld_ist_opt_in():
    assert rr.RealtimeTokenRequest(session_id="s").screen_tools is False


class _Antwort:
    def __init__(self, status, ctype="image/jpeg", body=b"\xff\xd8\xff\xe0jpeg"):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.content = body


def _verdrahten(monkeypatch, antwort):
    gesehen = {}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None, **k):
            gesehen["url"] = url; gesehen["headers"] = headers
            return antwort

    monkeypatch.setattr(rr.httpx, "AsyncClient", _Client)

    async def fake_chatgpt(prompt, model, api_key):
        gesehen["prompt"] = prompt; gesehen["model"] = model
        import os
        gesehen["bild_da"] = all(os.path.exists(p) for p in (prompt.image_paths or []))
        return types.SimpleNamespace(response="Links ein rotes Feld, Text PORTAL TEST 4711.", model=model)

    monkeypatch.setattr(t, "_chatgpt_einmal", fake_chatgpt)
    return gesehen


def test_look_at_screen_holt_mit_nutzer_jwt_und_beschreibt(monkeypatch):
    g = _verdrahten(monkeypatch, _Antwort(200))
    r = asyncio.run(rr._tool_look_at_screen({"storage_id": 126294, "question": "Was ist links?"}, "Bearer nutzer-jwt"))
    assert g["url"].endswith("/api/media/126294")
    assert g["headers"]["Authorization"] == "Bearer nutzer-jwt"
    assert g["bild_da"] is True and g["model"] == "gpt-5.6-luna"
    assert g["prompt"].sandbox == "read-only"
    assert "Was ist links?" in str(g["prompt"].prompt)
    assert r["description"].startswith("Links ein rotes Feld") and r["storage_id"] == 126294


def test_ohne_jwt_kein_abruf(monkeypatch):
    g = _verdrahten(monkeypatch, _Antwort(200))
    r = asyncio.run(rr._tool_look_at_screen({"storage_id": 1, "question": "x"}, None))
    assert r["error"] == "user_jwt_required" and "url" not in g


@pytest.mark.parametrize("args,fehler", [
    ({"storage_id": "abc", "question": "x"}, "invalid_storage_id"),
    ({"storage_id": 0, "question": "x"}, "invalid_storage_id"),
    ({"storage_id": True, "question": "x"}, "invalid_storage_id"),
])
def test_ungueltige_id(monkeypatch, args, fehler):
    g = _verdrahten(monkeypatch, _Antwort(200))
    assert asyncio.run(rr._tool_look_at_screen(args, "Bearer j"))["error"] == fehler
    assert "url" not in g


def test_fremdes_bild_wird_nicht_beschrieben(monkeypatch):
    g = _verdrahten(monkeypatch, _Antwort(403, "application/json", b'{"detail":"forbidden"}'))
    r = asyncio.run(rr._tool_look_at_screen({"storage_id": 5, "question": "x"}, "Bearer j"))
    assert r["error"] == "media_not_accessible" and r["status"] == 403 and "prompt" not in g


def test_kein_bild_wird_nicht_beschrieben(monkeypatch):
    g = _verdrahten(monkeypatch, _Antwort(200, "application/pdf", b"%PDF"))
    r = asyncio.run(rr._tool_look_at_screen({"storage_id": 5, "question": "x"}, "Bearer j"))
    assert r["error"] == "not_an_image" and "prompt" not in g


def test_prompt_nennt_knopf_und_fehler():
    """CloudV2 26.09.: getDisplayMedia braucht einen echten Klick; das
    Modell muss den Knopf nennen und keine_freigabe kennen."""
    z = rr._screen_tools_addendum("de")
    assert "Bild zeigen" in z and "keine_freigabe" in z
    sc = rr._screen_tool_defs()[0]["description"]
    assert "Bild zeigen" in sc and "keine_freigabe" in sc
