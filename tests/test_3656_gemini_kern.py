"""#3656: /ai/gemini auf P3 reicht Text ueber den Kern weiter (/ai/extern/hydrogen).

Kern und Instanz laufen hier in derselben App: Die Instanz ruft ueber einen
ASGI-Transport sich selbst als Kern auf.
"""
import asyncio
import hashlib
import json
import os
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ai.routes import text_ai_routes as t
from ai.services import extern_zugang as ez
from ai.services import gemini_kern as gk

SCHLUESSEL = "p3-kern-schluessel-abcdef0123456789"


@pytest.fixture(autouse=True)
def umgebung(monkeypatch, tmp_path):
    def verboten(*a, **k):
        raise AssertionError("CLI gestartet")

    async def verboten_async(*a, **k):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)
    for n in ("GEMINI_FORWARD_URL", "API_ACCESS_KEY", "REALTIME_AUTH_ISSUER", "REALTIME_AUTH_JWKS_URL",
              "REALTIME_AUTH_GRANT_URL"):
        monkeypatch.delenv(n, raising=False)
    datei = tmp_path / "kern.key"
    datei.write_text(SCHLUESSEL + "\n")
    monkeypatch.setenv(gk.URL_ENV, "http://kern.test")
    monkeypatch.setenv(gk.AUFRUFER_ENV, "Cloud")
    monkeypatch.setenv(gk.DATEI_ENV, str(datei))
    with open(os.environ[ez.DATEI_ENV], "w") as f:
        json.dump({"p3": {"sha256": hashlib.sha256(SCHLUESSEL.encode()).hexdigest(), "tag_max": 3}}, f)


@pytest.fixture
def netz(monkeypatch):
    """Leitet den Aufruf an den Kern in dieselbe App. Zaehlt die Aufrufe."""
    aufrufe = []
    echt = httpx.AsyncClient

    def client(*a, **k):
        k.pop("timeout", None)
        return echt(transport=httpx.ASGITransport(app=main.app), base_url="http://kern.test")

    orig_post = httpx.AsyncClient.post

    async def post(self, url, *a, **k):
        aufrufe.append((str(url), k.get("json"), dict(k.get("headers") or {})))
        return await orig_post(self, url, *a, **k)
    monkeypatch.setattr(gk.httpx if hasattr(gk, "httpx") else httpx, "AsyncClient", client, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(echt, "post", post)
    return aufrufe


@pytest.fixture
def agy(monkeypatch):
    """Die Seite des Kerns: agy liefert pong (die Route /ai/gemini selbst bleibt echt)."""
    gesehen = []

    async def fake(prompt, model=None, api_key=None, request=None):
        gesehen.append({"model": model, "prompt": prompt})
        return t.AIResponse(response="pong", model=model)
    monkeypatch.setattr(t, "gemini_endpoint", fake)
    return gesehen


@pytest.fixture
def client():
    """Aufrufer ist die P3-cloud-api (Sprechstile), wie freigegeben."""
    return TestClient(main.app, raise_server_exceptions=False, headers={"X-Agent-Name": "Cloud"})


def test_text_geht_ueber_den_kern(client, netz, agy):
    r = client.post("/ai/gemini", json={"prompt": "sag pong", "system": "kurz"})
    d = r.json()
    assert r.status_code == 200 and (d["response"], d["routed_to"], d["model"]) == ("pong", "kern", "gemini-4-hydrogen-low")
    url, rumpf, kopf = netz[0]
    assert url.endswith("/ai/extern/hydrogen") and kopf["X-API-KEY"] == SCHLUESSEL
    assert kopf["X-Extern-Aufrufer"] == "Cloud"
    assert rumpf == {"prompt": "sag pong", "system": "kurz", "model": "gemini-4-hydrogen-low"}
    assert agy[0]["model"] == "gemini-4-hydrogen-low"


@pytest.mark.parametrize("modell", ["gemini-4-carbon-medium", "gemini-4-hydrogen-high"])
def test_modell_geht_mit(client, netz, agy, modell):
    r = client.post("/ai/gemini", params={"model": modell}, json={"prompt": "x"})
    assert r.status_code == 200 and r.json()["model"] == modell and agy[0]["model"] == modell


@pytest.mark.parametrize("rumpf,grund", [
    ({"prompt": "x", "image_paths": ["https://example.com/a.png"]}, "bilder"),
    ({"prompt": {"text": "x", "image_paths": ["https://example.com/a.png"]}}, "bilder"),
    ({"prompt": "x", "conversation_history": [{"role": "user", "content": "y"}]}, "verlauf"),
    ({"prompt": "x", "persona_variant": "default"}, "persona"),
    ({"prompt": "x", "model": "gemini-3.5-flash"}, "modell"),
    ({"prompt": "x", "model": "gemini-4-neon-low"}, "modell"),
])
def test_ablehnung_ohne_netz(client, netz, agy, rumpf, grund):
    r = client.post("/ai/gemini", json=rumpf)
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["grund"]) == (422, "gemini_kern_nicht_moeglich", grund)
    assert netz == [] and agy == []


def test_ohne_schluesseldatei_503(client, netz, agy, monkeypatch, tmp_path):
    monkeypatch.setenv(gk.DATEI_ENV, str(tmp_path / "fehlt.key"))
    r = client.post("/ai/gemini", json={"prompt": "x"})
    d = r.json()["detail"]
    assert (r.status_code, d["error"], d["grund"]) == (503, "cli_fehlt", "kern_schluessel_fehlt") and netz == []


def test_falscher_schluessel_kommt_als_401_zurueck(client, netz, agy, tmp_path, monkeypatch):
    datei = tmp_path / "falsch.key"
    datei.write_text("falsch")
    monkeypatch.setenv(gk.DATEI_ENV, str(datei))
    r = client.post("/ai/gemini", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (401, "extern_schluessel_required") and agy == []


def test_deckel_des_kerns_kommt_unveraendert_zurueck(client, netz, agy):
    codes = [client.post("/ai/gemini", json={"prompt": "x"}).status_code for _ in range(3)]
    r = client.post("/ai/gemini", json={"prompt": "x"})
    d = r.json()["detail"]
    assert codes == [200, 200, 200] and r.status_code == 429
    assert (d["error"], d["grund"], d["routed_to"]) == ("kontingent", "tagesdeckel", "kern") and d["resets_at"]
    assert int(r.headers["Retry-After"]) > 0


def test_kern_nicht_erreichbar(client, agy, monkeypatch):
    class Kaputt:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise httpx.ConnectError("weg")
    monkeypatch.setattr(httpx, "AsyncClient", Kaputt)
    r = client.post("/ai/gemini", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (503, "gemini_kern_nicht_erreichbar") and agy == []


@pytest.mark.parametrize("kopf", [{}, {"X-Agent-Name": "Buchhaltung"}, {"X-Agent-Name": "SWFME"},
                                  {"X-Agent-Name": "cloud"}])
def test_nicht_freigegebener_aufrufer_bleibt_auf_der_instanz(netz, agy, kopf):
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.post("/ai/gemini", json={"prompt": "Beleg 4711"}, headers=kopf)
    assert (r.status_code, r.json()["detail"]["error"]) == (403, "gemini_kern_aufrufer_gesperrt")
    assert netz == [] and agy == []


def test_ohne_freigabeliste_geht_niemand(client, netz, agy, monkeypatch):
    monkeypatch.delenv(gk.AUFRUFER_ENV)
    r = client.post("/ai/gemini", json={"prompt": "x"})
    assert r.status_code == 403 and netz == [] and agy == []


def test_freigabe_ist_ein_eintrag(netz, agy, monkeypatch):
    monkeypatch.setenv(gk.AUFRUFER_ENV, "Cloud, SWFME")
    c = TestClient(main.app, raise_server_exceptions=False, headers={"X-Agent-Name": "SWFME"})
    assert c.post("/ai/gemini", json={"prompt": "x"}).status_code == 200


def test_kern_eigene_aufruferliste(client, netz, agy):
    with open(os.environ[ez.DATEI_ENV], "w") as f:
        json.dump({"p3": {"sha256": hashlib.sha256(SCHLUESSEL.encode()).hexdigest(), "tag_max": 3,
                          "aufrufer": ["Andere"]}}, f)
    r = client.post("/ai/gemini", json={"prompt": "x"})
    assert (r.status_code, r.json()["detail"]["error"]) == (403, "extern_aufrufer_gesperrt") and agy == []


def test_ohne_kern_url_bleibt_alles_lokal(client, netz, monkeypatch):
    monkeypatch.delenv(gk.URL_ENV)
    gesehen = []

    def lauf(cmd, env=None, timeout=300, cwd="/", input=None, phasen=None):
        gesehen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": "SUCCESS", "response": "lokal"}), "")
    monkeypatch.setattr(t, "_run_cli_with_pgid", lauf)
    r = client.post("/ai/gemini", json={"prompt": "x"})
    assert r.status_code == 200 and r.json()["response"] == "lokal" and netz == [] and gesehen


def test_kern_reicht_nie_weiter(client, netz, monkeypatch):
    """Ist GEMINI_KERN_URL auch auf dem Kern gesetzt, laeuft der Extern-Weg trotzdem
    lokal: keine Schleife."""
    gesehen = []

    def lauf(cmd, env=None, timeout=300, cwd="/", input=None, phasen=None):
        gesehen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": "SUCCESS", "response": "pong"}), "")
    monkeypatch.setattr(t, "_run_cli_with_pgid", lauf)
    r = client.post("/ai/extern/hydrogen", json={"prompt": "x"}, headers={"X-API-KEY": SCHLUESSEL})
    assert r.status_code == 200 and r.json()["response"] == "pong" and netz == [] and len(gesehen) == 1
