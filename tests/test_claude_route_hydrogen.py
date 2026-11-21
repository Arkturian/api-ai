"""CLAUDE_ROUTE=hydrogen (Alex, 21.11., #3652): /ai/claude laeuft ueber agy,
markiert, mit eigenem Tagesdeckel, und nichts weicht still aus."""
import asyncio
import json
import os
import subprocess

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import ki_wege as kw

AGY_OK = json.dumps({"status": "SUCCESS", "response": "pong", "usage": {"input_tokens": 1, "output_tokens": 1}})


@pytest.fixture(autouse=True)
def umgebung(monkeypatch):
    def verboten(*a, **k):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **k):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    for n in ("REALTIME_AUTH_ISSUER", "REALTIME_AUTH_JWKS_URL", "REALTIME_AUTH_GRANT_URL", "API_ACCESS_KEY",
              "GEMINI_FORWARD_URL", "AGY_RUN_AS_USER", kw.CLAUDE_HYDROGEN_MAX_ENV,
              "CLAUDE_ROUTE_HYDROGEN_MODEL", "CLAUDE_ROUTE_HYDROGEN_MODEL_STARK"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("CLAUDE_ROUTE", "hydrogen")


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def agy(monkeypatch):
    gesehen = []

    async def fake(prompt, model=None, api_key=None, request=None):
        gesehen.append({"model": model, "prompt": prompt})
        return t.AIResponse(response="pong", model=model)

    async def nie(*a, **k):
        raise AssertionError("codex darf nicht laufen")
    monkeypatch.setattr(t, "gemini_endpoint", fake)
    monkeypatch.setattr(t, "chatgpt_endpoint", nie)
    return gesehen


@pytest.mark.parametrize("modell,ziel", [(None, "gemini-4-hydrogen-low"), ("sonnet", "gemini-4-hydrogen-low"),
                                         ("haiku", "gemini-4-hydrogen-low"), ("opus", "gemini-4-hydrogen-high"),
                                         ("claude-fable-5-1", "gemini-4-hydrogen-high")])
def test_modell_und_markierung(client, agy, modell, ziel):
    b = {"prompt": "x", "effort": "high", "tools": False}
    if modell:
        b["model"] = modell
    r = client.post("/ai/claude", json=b)
    d = r.json()
    assert r.status_code == 200 and (d["routed_to"], d["model"]) == ("agy", ziel)
    assert d["model_requested"] == (modell or "claude-default")
    p = agy[0]["prompt"]
    assert (p.effort, p.tools, p.model, p.request_id) == (None, None, None, None)


def test_modell_aus_umgebung(monkeypatch, client, agy):
    monkeypatch.setenv("CLAUDE_ROUTE_HYDROGEN_MODEL", "gemini-4-hydrogen-medium")
    assert client.post("/ai/claude", json={"prompt": "x"}).json()["model"] == "gemini-4-hydrogen-medium"


def test_echter_gemini_weg_startet_agy_mit_hydrogen(monkeypatch, client):
    gesehen = []

    def lauf(cmd, env=None, timeout=300, cwd="/", input=None, phasen=None):
        gesehen.append((list(cmd), input))
        return subprocess.CompletedProcess(cmd, 0, AGY_OK, "")
    monkeypatch.setattr(t, "_run_cli_with_pgid", lauf)
    r = client.post("/ai/claude", json={"prompt": "--help sag pong", "system": "kurz"})
    assert r.status_code == 200 and r.json()["response"] == "pong" and r.json()["routed_to"] == "agy"
    cmd, eingabe = gesehen[0]
    assert cmd[cmd.index("--model") + 1] == "gemini-4-hydrogen-low" and "--effort" not in cmd
    assert not any("--help" in a for a in cmd) and "--help sag pong" in (eingabe or "")
    assert "kurz" in eingabe


@pytest.mark.parametrize("b,code,fehler", [
    ({"prompt": "x", "require_provider": "claude"}, 503, "claude_routed_to_agy"),
    ({"prompt": "x", "response_format": {"type": "json_object"}}, 503, "claude_routed_to_agy"),
    ({"prompt": "x", "tools": True}, 403, "tools_disabled"),
])
def test_abgewiesen(client, agy, b, code, fehler):
    r = client.post("/ai/claude", json=b)
    assert (r.status_code, r.json()["detail"]["error"]) == (code, fehler) and agy == []


def test_meldung_nennt_hydrogen(client, agy):
    d = client.post("/ai/claude", json={"prompt": "x", "require_provider": "claude"}).json()["detail"]
    assert "Hydrogen" in d["message"] and "Codex" not in d["message"] and d["routed_to"] == "agy"


def test_lokaler_pfad_von_aussen_bleibt_gesperrt(client, agy):
    r = client.post("/ai/claude", json={"prompt": "x", "image_paths": ["/etc/passwd"]},
                    headers={"X-Real-IP": "203.0.113.9"})
    assert r.status_code == 403 and agy == []


def test_vorgabe_deckel_85():
    assert kw.hydrogen_tag_max() == 85


@pytest.mark.parametrize("wert", ["", "0", "-3", "abc"])
def test_unsinniger_deckel_gilt_als_vorgabe(monkeypatch, wert):
    monkeypatch.setenv(kw.CLAUDE_HYDROGEN_MAX_ENV, wert)
    assert kw.hydrogen_tag_max() == 85


def test_deckel_429_ohne_aufruf(monkeypatch, client, agy):
    monkeypatch.setenv(kw.CLAUDE_HYDROGEN_MAX_ENV, "2")
    assert [client.post("/ai/claude", json={"prompt": "x"}).status_code for _ in range(2)] == [200, 200]
    r = client.post("/ai/claude", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["weg"], d["grund"]) == (429, "kontingent", "agy", "tagesdeckel")
    assert d["resets_at"] and "quota" in d["hint"] and int(r.headers["retry-after"]) > 0
    assert len(agy) == 2


def test_zaehler_kaputt_geht_nichts_hinaus(monkeypatch, client, agy):
    monkeypatch.setenv("VERBRAUCH_DIR", "/proc/nicht-schreibbar")
    r = client.post("/ai/claude", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "hydrogen_zaehler_fehlt") and agy == []


def test_kaputte_zaehlerdatei_geht_nichts_hinaus(client, agy):
    from datetime import datetime
    d = kw.Path(os.environ["VERBRAUCH_DIR"]); d.mkdir(parents=True, exist_ok=True)
    (d / f"claude_hydrogen_{datetime.now().astimezone().date().isoformat()}.json").write_text("{kaputt")
    assert client.post("/ai/claude", json={"prompt": "x"}).status_code == 503 and agy == []


def test_zaehlerdatei_nur_fuer_den_dienst(client, agy):
    client.post("/ai/claude", json={"prompt": "geheimer-text"})
    for f in kw.Path(os.environ["VERBRAUCH_DIR"]).glob("claude_hydrogen_*.json"):
        assert oct(f.stat().st_mode & 0o777) == "0o600" and "geheimer" not in f.read_text()


def test_idempotenz(monkeypatch, tmp_path, client, agy):
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))
    b = {"prompt": "x", "request_id": "claude-hydrogen-1"}
    assert client.post("/ai/claude", json=b).status_code == 200
    assert client.post("/ai/claude", json=b).json()["response"] == "pong"
    assert len(agy) == 1
    assert client.post("/ai/claude", json={**b, "prompt": "y"}).status_code == 409


def test_codex_weg_unveraendert(monkeypatch, client):
    monkeypatch.setenv("CLAUDE_ROUTE", "codex")
    gesehen = []

    async def fake(prompt, model=None, api_key=None):
        gesehen.append(model)
        return t.AIResponse(response="ok", model="codex-default")

    async def nie(*a, **k):
        raise AssertionError("agy darf nicht laufen")
    monkeypatch.setattr(t, "chatgpt_endpoint", fake)
    monkeypatch.setattr(t, "gemini_endpoint", nie)
    assert client.post("/ai/claude", json={"prompt": "x"}).json()["routed_to"] == "codex" and gesehen == [None]
    d = client.post("/ai/claude", json={"prompt": "x", "require_provider": "claude"}).json()["detail"]
    assert d["error"] == "claude_routed_to_codex" and "Codex" in d["message"]
