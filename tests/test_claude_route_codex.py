"""CLAUDE_ROUTE=codex (Alex/Cloud, 30.10.2026): /ai/claude laeuft serverseitig
auf Codex, die Antwort sagt es; require_provider=claude, response_format und
persona mit MCP -> 503; nichts weicht still aus. Dazu das Aufrufer-Protokoll."""
import json

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t


@pytest.fixture
def codex(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_ROUTE", "codex")
    monkeypatch.setenv("AUFRUFER_LOG_DIR", str(tmp_path))
    gesehen = []

    async def fake(prompt, model=None, api_key=None):
        gesehen.append({"model": model, "prompt": prompt})
        return t.AIResponse(response="ok", model=model or "codex-default")
    monkeypatch.setattr(t, "chatgpt_endpoint", fake)

    def nie(*a, **k):
        raise AssertionError("Claude-CLI darf nicht starten")
    monkeypatch.setattr(t, "_run_cli_with_pgid", nie)
    c = TestClient(main.app)
    c.gesehen, c.logdir = gesehen, tmp_path
    return c


@pytest.mark.parametrize("modell,ziel", [
    (None, None), ("sonnet", None), ("opus", "gpt-6.1-sol"), ("claude-fable-5-1", "gpt-6.1-sol"),
    ("haiku", None)])  # None = Codex-Vorgabe gpt-6.1-terra (Alex, 30.10.)
def test_modellabbildung_und_markierung(codex, modell, ziel):
    body = {"prompt": "ok", "effort": "high"}
    if modell:
        body["model"] = modell
    r = codex.post("/ai/claude", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["routed_to"] == "codex" and d["model_requested"] == (modell or "claude-default")
    assert codex.gesehen[0]["model"] == ziel and codex.gesehen[0]["prompt"].effort == "high"


@pytest.mark.parametrize("body,grund", [
    ({"prompt": "ok", "require_provider": "claude"}, "require_provider"),
    ({"prompt": "ok", "response_format": {"type": "json_schema", "json_schema": {}}}, "response_format"),
])
def test_ablehnungen_503(codex, body, grund):
    r = codex.post("/ai/claude", json=body)
    assert r.status_code == 503 and r.json()["detail"]["grund"] == grund and codex.gesehen == []


def test_persona_mit_mcp_503_ohne_mcp_laeuft(codex, monkeypatch):
    async def bundle(ep, v):
        return {"rendered": "P", "allowed_tools": ["mcp__x__y"] if v == "mcp" else []}
    monkeypatch.setattr(t, "get_persona_bundle", bundle)
    r = codex.post("/ai/claude", json={"prompt": "ok", "persona_variant": "mcp"})
    assert r.status_code == 503 and r.json()["detail"]["grund"] == "persona_mcp"
    r = codex.post("/ai/claude", json={"prompt": "ok", "persona_variant": "plain"})
    assert r.status_code == 200 and r.json()["routed_to"] == "codex"


def test_ohne_schalter_kein_umweg(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_ROUTE", raising=False)
    (tmp_path / ".credentials.json").write_text('{"agentosNoAccount": true}')
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    async def nie(*a, **k):
        raise AssertionError("ohne Schalter kein Codex")
    monkeypatch.setattr(t, "chatgpt_endpoint", nie)
    r = TestClient(main.app).post("/ai/claude", json={"prompt": "ok"})
    assert r.status_code == 503 and r.json()["detail"]["grund"] == "kein_konto"


def test_aufrufer_wird_protokolliert(codex):
    codex.post("/ai/claude", json={"prompt": "ok"}, headers={"X-Agent-Name": "storage-api"})
    codex.post("/ai/claude", json={"prompt": "ok"}, headers={"X-Real-IP": "10.0.0.7"})
    zeilen = [json.loads(z) for f in codex.logdir.glob("*.jsonl") for z in f.read_text().splitlines()]
    assert [(z["caller"], z["quelle"], z["status"]) for z in zeilen] == [
        ("storage-api", "header", 200), ("10.0.0.7", "ip", 200)]
