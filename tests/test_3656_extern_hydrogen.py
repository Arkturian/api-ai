"""#3656: P3 ruft Hydrogen ueber unseren Kern, nur mit eigenem Schluessel,
nur Text, nur Hydrogen, mit eigenem Tagesdeckel."""
import asyncio
import hashlib
import json
import os
import subprocess

import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import extern_zugang as ez

SCHLUESSEL = "p3-schluessel-0123456789abcdef"
PFAD = "/ai/extern/hydrogen"


@pytest.fixture(autouse=True)
def keine_cli(monkeypatch):
    def verboten(*a, **k):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **k):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    monkeypatch.delenv("GEMINI_FORWARD_URL", raising=False)


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def zugang(tag_max=100, **weitere):
    eintraege = {"p3": {"sha256": hashlib.sha256(SCHLUESSEL.encode()).hexdigest(), "tag_max": tag_max}}
    eintraege.update(weitere)
    with open(os.environ[ez.DATEI_ENV], "w") as f:
        json.dump(eintraege, f)


@pytest.fixture
def agy(monkeypatch):
    gesehen = []

    async def fake(prompt, model=None, api_key=None, request=None):
        gesehen.append({"model": model, "prompt": prompt})
        return t.AIResponse(response="pong", model=model)
    monkeypatch.setattr(t, "gemini_endpoint", fake)
    return gesehen


K = {"X-API-KEY": SCHLUESSEL}


def test_ohne_datei_aus(client, agy):
    r = client.post(PFAD, json={"prompt": "x"}, headers=K)
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "extern_aus") and agy == []


@pytest.mark.parametrize("kopf", [{}, {"X-API-KEY": ""}, {"X-API-KEY": "falsch"},
                                  {"X-API-KEY": SCHLUESSEL + "x"}])
def test_ohne_gueltigen_schluessel_401(client, agy, kopf):
    zugang()
    r = client.post(PFAD, json={"prompt": "x"}, headers=kopf)
    assert (r.status_code, r.json()["detail"]["error"]) == (401, "extern_schluessel_required") and agy == []


def test_schluessel_vor_dem_rumpf(client, agy):
    """Kaputtes JSON ohne Schluessel ergibt 401, nicht 422."""
    zugang()
    r = client.post(PFAD, content=b"{kaputt", headers={"Content-Type": "application/json"})
    assert r.status_code == 401 and agy == []


def test_nicht_ascii_schluessel_401_statt_500(client, agy):
    zugang()
    r = client.post(PFAD, json={"prompt": "x"}, headers={"X-API-KEY": "schlüssel".encode()})
    assert r.status_code == 401 and agy == []


def test_erfolg_mit_vorgabemodell(client, agy):
    zugang()
    r = client.post(PFAD, json={"prompt": "sag pong", "system": "kurz"}, headers=K)
    d = r.json()
    assert r.status_code == 200 and (d["response"], d["routed_to"], d["model"]) == ("pong", "agy", "gemini-4-hydrogen-low")
    p = agy[0]["prompt"]
    assert (p.prompt, p.system, p.tools, p.image_paths, p.persona_variant) == ("sag pong", "kurz", None, None, None)


@pytest.mark.parametrize("modell", ["gemini-4-hydrogen", "gemini-4-hydrogen-medium", "gemini-4-hydrogen-xhigh",
                                    "gemini-4-carbon", "gemini-4-carbon-medium", "gemini-4-carbon-high"])
def test_hydrogen_modelle(client, agy, modell):
    zugang()
    assert client.post(PFAD, json={"prompt": "x", "model": modell}, headers=K).json()["model"] == modell


@pytest.mark.parametrize("rumpf", [
    {"prompt": "x", "model": "gemini-4-carbon-xhigh"},
    {"prompt": "x", "model": "gemini-4-neon-low"},
    {"prompt": "x", "model": "gemini-4-hydrogen-ultra"},
    {"prompt": "x", "model": "gpt-6-luna"},
    {"prompt": "x", "image_paths": ["/etc/passwd"]},
    {"prompt": "x", "image_paths": ["https://example.com/a.png"]},
    {"prompt": "x", "tools": True},
    {"prompt": "x", "persona_variant": "default"},
    {"prompt": "x", "conversation_history": [{"role": "user", "content": "y"}]},
    {"prompt": {"text": "x", "image_paths": ["/etc/passwd"]}},
    {"prompt": ""},
    {"prompt": "x" * 40001},
    {"prompt": "x", "system": "s" * 8001},
])
def test_nur_text_und_hydrogen(client, agy, rumpf):
    zugang()
    r = client.post(PFAD, json=rumpf, headers=K)
    assert r.status_code == 422 and agy == []


def test_tagesdeckel_je_aufrufer(client, agy):
    zugang(tag_max=2)
    assert [client.post(PFAD, json={"prompt": "x"}, headers=K).status_code for _ in range(2)] == [200, 200]
    r = client.post(PFAD, json={"prompt": "x"}, headers=K)
    d = r.json()["detail"]
    assert r.status_code == 429 and (d["error"], d["grund"], d["weg"]) == ("kontingent", "tagesdeckel", "agy")
    assert d["resets_at"] and int(r.headers["Retry-After"]) > 0
    assert len(agy) == 2


def test_deckel_getrennt_vom_claude_weg(client, agy, monkeypatch):
    """P3 zaehlt eigens, /ai/claude bleibt davon unberuehrt."""
    zugang(tag_max=1)
    assert client.post(PFAD, json={"prompt": "x"}, headers=K).status_code == 200
    ordner = os.environ["VERBRAUCH_DIR"]
    assert sorted(n for n in os.listdir(ordner) if n.endswith(".json"))[0].startswith("extern_hydrogen_p3_")
    assert not any(n.startswith("claude_hydrogen_") for n in os.listdir(ordner))


def test_widerruf_ohne_neustart(client, agy):
    zugang()
    assert client.post(PFAD, json={"prompt": "x"}, headers=K).status_code == 200
    with open(os.environ[ez.DATEI_ENV], "w") as f:
        json.dump({}, f)
    assert client.post(PFAD, json={"prompt": "x"}, headers=K).status_code == 503


@pytest.mark.parametrize("inhalt", ["{kaputt", "[]", '{"P3": {"sha256": "ab", "tag_max": 1}}',
                                    '{"p3": {"sha256": "' + "a" * 64 + '", "tag_max": 0}}',
                                    '{"p3": {"sha256": "' + "a" * 64 + '", "tag_max": 5000}}'])
def test_kaputte_datei_ist_aus(client, agy, inhalt):
    with open(os.environ[ez.DATEI_ENV], "w") as f:
        f.write(inhalt)
    r = client.post(PFAD, json={"prompt": "x"}, headers=K)
    assert r.status_code == 503 and agy == []


def test_fremder_schluessel_trifft_nur_sich(client, agy):
    zugang(andere={"sha256": hashlib.sha256(b"anderer").hexdigest(), "tag_max": 1})
    assert client.post(PFAD, json={"prompt": "x"}, headers={"X-API-KEY": "anderer"}).status_code == 200
    assert client.post(PFAD, json={"prompt": "x"}, headers={"X-API-KEY": "anderer"}).status_code == 429
    assert client.post(PFAD, json={"prompt": "x"}, headers=K).status_code == 200


def test_echter_weg_startet_agy_mit_hydrogen(monkeypatch, client):
    zugang()
    gesehen = []
    ok = json.dumps({"status": "SUCCESS", "response": "pong", "usage": {"input_tokens": 1, "output_tokens": 1}})

    def lauf(cmd, env=None, timeout=300, cwd="/", input=None, phasen=None):
        gesehen.append((list(cmd), input))
        return subprocess.CompletedProcess(cmd, 0, ok, "")
    monkeypatch.setattr(t, "_run_cli_with_pgid", lauf)
    r = client.post(PFAD, json={"prompt": "--help sag pong"}, headers=K)
    assert r.status_code == 200 and r.json()["response"] == "pong"
    cmd, eingabe = gesehen[0]
    assert cmd[cmd.index("--model") + 1] == "gemini-4-hydrogen-low"
    assert not any("--help" in a for a in cmd) and "--help sag pong" in (eingabe or "")
