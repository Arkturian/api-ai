"""#3607: Auf einer Kundeninstanz startet niemand ohne Schluessel eine Modell-CLI."""
import asyncio
import itertools
import os
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ai.services import auth_konfig as ak

LOKAL = {
    ak.ISSUER_ENV: "kunde.example.com",
    ak.JWKS_ENV: "http://127.0.0.1:8010/api/v1/auth/.well-known/jwks.json",
    ak.GRANT_ENV: "http://127.0.0.1:8010/api/v1/auth/realtime-grant",
}
DREI = tuple(LOKAL)
HALB = [c for r in (1, 2) for c in itertools.combinations(DREI, r)]
ROUTEN = ["/ai/chatgpt", "/ai/claude", "/ai/gemini", "/ai/grok"]
SCHLUESSEL = "instanz-schluessel"


@pytest.fixture(autouse=True)
def umgebung(monkeypatch):
    for n in (*DREI, *ak.ZUSATZ_ENVS, "API_ACCESS_KEY"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr(ak, "_gemeldet", False)


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def keine_cli(monkeypatch):
    """Fuer JEDEN Test: Auch ein wieder eingebauter Fehler startet keine echte
    CLI (die Routen nutzen subprocess.Popen, nicht asyncio)."""
    def verboten(*a, **k):
        raise AssertionError("CLI gestartet")
    async def verboten_async(*a, **k):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(subprocess, "Popen", verboten)
    monkeypatch.setattr(subprocess, "run", verboten)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten_async)


def kunde(monkeypatch, schluessel=SCHLUESSEL):
    for k, v in LOKAL.items():
        monkeypatch.setenv(k, v)
    if schluessel:
        monkeypatch.setenv("API_ACCESS_KEY", schluessel)


@pytest.mark.parametrize("route", ROUTEN)
@pytest.mark.parametrize("kopf", [{}, {"X-API-KEY": "falsch"}, {"X-API-KEY": ""}])
def test_kunde_ohne_gueltigen_schluessel(monkeypatch, client, keine_cli, route, kopf):
    kunde(monkeypatch)
    r = client.post(route, headers=kopf, json={"prompt": "x"})
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "api_key_required"


@pytest.mark.parametrize("route", ROUTEN)
def test_kunde_mit_schluessel_kommt_durch(monkeypatch, client, route):
    kunde(monkeypatch)
    assert client.post(route, headers={"X-API-KEY": SCHLUESSEL}, json={}).status_code == 422


@pytest.mark.parametrize("route", ROUTEN)
def test_kunde_ohne_konfigurierten_schluessel_bleibt_zu(monkeypatch, client, keine_cli, route):
    kunde(monkeypatch, schluessel=None)
    r = client.post(route, json={"prompt": "x"})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "api_key_not_configured"


@pytest.mark.parametrize("namen", HALB)
@pytest.mark.parametrize("route", ROUTEN)
def test_halbe_konfiguration_ist_auch_kunde(monkeypatch, client, keine_cli, route, namen):
    for n in namen:
        monkeypatch.setenv(n, LOKAL[n])
    monkeypatch.setenv("API_ACCESS_KEY", SCHLUESSEL)
    assert client.post(route, json={"prompt": "x"}).status_code == 401


@pytest.mark.parametrize("pfad", ["/ai/chatgpt/", "/ai/gemini/vision", "/ai/chatgpt/abc",
                                  "/ai/chat%67pt", "/ai/%63laude", "/ai/gemin%69", "/ai/gro%6B"])
def test_unterpfade_und_kodierung(monkeypatch, client, keine_cli, pfad):
    kunde(monkeypatch)
    assert client.post(pfad, json={"prompt": "x"}, follow_redirects=False).status_code == 401


def test_aehnlicher_pfad_faellt_nicht_ins_tor(monkeypatch):
    kunde(monkeypatch)
    assert main._modelltor_gilt("/ai/chatgptx") is False


def test_webhook_bleibt_offen(monkeypatch, client):
    kunde(monkeypatch)
    r = client.post("/ai/gemini/gcp-budget-webhook", json={})
    assert r.status_code != 401 or "api_key_required" not in r.text


def test_realtime_token_bleibt_ohne_schluessel(monkeypatch, client):
    kunde(monkeypatch)
    r = client.post("/ai/realtime/token", json={})
    assert r.status_code == 401
    assert "realtime_user_jwt_required" in r.text


def test_bezahlpfad_bleibt_im_tor(monkeypatch, client):
    kunde(monkeypatch)
    r = client.post("/ai/deepseek", json={"prompt": "x"})
    assert r.status_code == 401 and "api_key_required" in r.text


@pytest.mark.parametrize("route", ROUTEN)
@pytest.mark.parametrize("schluessel", [None, SCHLUESSEL])
def test_unser_kern_bleibt_wie_bisher(monkeypatch, client, route, schluessel):
    if schluessel:
        monkeypatch.setenv("API_ACCESS_KEY", schluessel)
    assert client.post(route, json={}).status_code == 422


# --- Modellname: nie eine Option fuer die CLI ------------------------------

@pytest.mark.parametrize("route", ROUTEN)
@pytest.mark.parametrize("name", ["--sandbox", "-c", "a b", "x;id", "x\nz", "a" * 81, "../x y"])
def test_ungueltiger_modellname_im_rumpf(client, route, name):
    r = client.post(route, json={"prompt": "x", "model": name})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_model_name"


@pytest.mark.parametrize("route", ROUTEN)
def test_ungueltiger_modellname_in_der_query(client, route):
    r = client.post(route, params={"model": "--config=x"}, json={"prompt": "x"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_model_name"


@pytest.mark.parametrize("name", [None, "", "gpt-6-luna", "gemini-4-hydrogen-low", "gpt-5.6-terra",
                                  "claude-opus-5-5", "codex-default", "org/modell:1"])
def test_gueltige_modellnamen(name):
    from ai.routes.text_ai_routes import _modellname_pruefen
    assert _modellname_pruefen(name) == name


# --- Betreiber-Routen: im Code zu, nicht nur in nginx -----------------------

@pytest.mark.parametrize("pfad,rumpf", [
    ("/ai/status/smoketest", None),
    ("/internal/notify-cli-update", {"host": "probe"}),
])
@pytest.mark.parametrize("dienstschluessel", [None, "dienst"])
@pytest.mark.parametrize("kopf", [{}, {"X-API-KEY": "falsch"}])
def test_betreiberrouten_anonym_zu(monkeypatch, client, pfad, rumpf, dienstschluessel, kopf):
    monkeypatch.delenv("REALTIME_GRANT_SERVICE_KEY", raising=False)
    if dienstschluessel:
        monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", dienstschluessel)
    r = client.post(pfad, headers=kopf, json=rumpf)
    assert r.status_code in (401, 403, 503), r.text


# --- Prompt, Systemprompt und effort werden nie zu CLI-Optionen (Review) ----

BOESE = '--settings={"apiKeyHelper":"id"}'


@pytest.fixture
def cli_fang(monkeypatch):
    from ai.routes import text_ai_routes as t
    faenge = []

    def fang(cmd, env=None, timeout=300, cwd="/", input=None, phasen=None):
        eintrag = {"cmd": list(cmd), "input": input, "datei": None, "modus": None}
        if "--prompt-file" in cmd:
            pfad = cmd[cmd.index("--prompt-file") + 1]
            eintrag["pfad"] = pfad
            eintrag["datei"] = open(pfad, encoding="utf-8").read()
            eintrag["modus"] = os.stat(pfad).st_mode & 0o777
        faenge.append(eintrag)
        return subprocess.CompletedProcess(cmd, 1, "", "abgebrochen im Test")

    monkeypatch.setattr(t, "_run_cli_with_pgid", fang)
    monkeypatch.setattr(t, "_claude_konto_vorab", lambda: None)
    monkeypatch.delenv("CLAUDE_ROUTE", raising=False)
    monkeypatch.delenv("GEMINI_FORWARD_URL", raising=False)
    return faenge


def test_claude_prompt_geht_ueber_stdin(client, cli_fang):
    client.post("/ai/claude", json={"prompt": BOESE})
    assert len(cli_fang) == 1
    f = cli_fang[0]
    assert f["cmd"][:2] == ["claude", "-p"] and f["input"] == BOESE
    assert not any(BOESE in a for a in f["cmd"])


def test_claude_systemprompt_ist_ein_argument(client, cli_fang):
    client.post("/ai/claude", json={"prompt": "x", "system": "--mcp-config=/tmp/x"})
    cmd = cli_fang[0]["cmd"]
    assert "--system-prompt=--mcp-config=/tmp/x" in cmd
    assert "--mcp-config=/tmp/x" not in cmd and "--system-prompt" not in cmd


def test_grok_prompt_geht_ueber_datei(client, cli_fang):
    client.post("/ai/grok", json={"prompt": "--always-approve"})
    assert len(cli_fang) == 1
    f = cli_fang[0]
    assert "-p" not in f["cmd"] and "--always-approve" not in f["cmd"]
    assert f["datei"] == "--always-approve" and f["modus"] == 0o600
    assert not os.path.exists(f["pfad"])


@pytest.mark.parametrize("route", ROUTEN)
@pytest.mark.parametrize("effort", ["--evil", "-x", "a b", "x;id", "a" * 25])
def test_ungueltiger_effort(client, cli_fang, route, effort):
    r = client.post(route, json={"prompt": "x", "effort": effort})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_effort"
    assert cli_fang == []


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh", " High "])
def test_gueltiger_effort(effort):
    from ai.routes.text_ai_routes import Prompt, _cli_eingaben_pruefen
    _cli_eingaben_pruefen(Prompt(prompt="x", effort=effort))


@pytest.mark.parametrize("route,env,wert", [
    ("/ai/claude", "CLAUDE_ROUTE", "codex"),
    ("/ai/gemini", "GEMINI_FORWARD_URL", "http://127.0.0.1:9/nie"),
])
@pytest.mark.parametrize("feld,wert_feld,fehler", [
    ("model", "--sandbox", "invalid_model_name"), ("effort", "--evil", "invalid_effort")])
def test_pruefung_vor_umleitung(monkeypatch, client, cli_fang, route, env, wert, feld, wert_feld, fehler):
    monkeypatch.setenv(env, wert)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: pytest.fail("weitergeleitet"))
    r = client.post(route, json={"prompt": "x", feld: wert_feld})
    assert r.status_code == 422 and r.json()["detail"]["error"] == fehler


def test_vision_prueft_den_modellnamen(monkeypatch, client):
    from ai.routes import text_ai_routes as t
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    r = client.post("/ai/gemini/vision", headers={"X-API-KEY": "dienst"},
                    json={"prompt": "x", "model": "--sandbox"})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "invalid_model_name"


def test_leerzeichen_als_schluessel_gilt_als_fehlend(monkeypatch, client):
    kunde(monkeypatch, schluessel="   ")
    r = client.post("/ai/chatgpt", headers={"X-API-KEY": "   "}, json={"prompt": "x"})
    assert r.status_code == 503


# --- Das Tor greift VOR der Validierung (Direktprobe von Cloud-release) -----

ZAEHLER = ["cost", "minimax-cost", "deepseek-cost", "elevenlabs-cost", "openai-cost", "openai-realtime-cost"]


@pytest.mark.parametrize("name", ZAEHLER)
@pytest.mark.parametrize("geheimnis", [None, "zaehler"])
def test_zaehlerrouten_antworten_nie_mit_422(monkeypatch, client, name, geheimnis):
    monkeypatch.delenv("COST_TRACKER_SHARED_SECRET", raising=False)
    if geheimnis:
        monkeypatch.setenv("COST_TRACKER_SHARED_SECRET", geheimnis)
    pfad = f"/internal/{name}-shared-state"
    assert client.post(pfad, json={}).status_code == (401 if geheimnis else 503)
    assert client.get(pfad).status_code == (401 if geheimnis else 503)


def test_alle_internen_post_routen_vor_der_validierung(monkeypatch, client):
    from fastapi.routing import APIRoute
    monkeypatch.delenv("COST_TRACKER_SHARED_SECRET", raising=False)
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    pfade = [r.path for r in main.app.routes if isinstance(r, APIRoute)
             and "POST" in r.methods and r.path.startswith("/internal/")]
    assert len(pfade) >= 7
    for pfad in pfade:
        assert client.post(pfad, json={}).status_code in (401, 403, 503), pfad


@pytest.mark.parametrize("route", ROUTEN)
def test_kunde_leerer_rumpf_401_vor_validierung(monkeypatch, client, route):
    kunde(monkeypatch)
    assert client.post(route, json={}).status_code == 401


def test_openapi_bleibt_lesbar(monkeypatch, client):
    kunde(monkeypatch)
    assert client.get("/openapi.json").status_code == 200


@pytest.fixture
def smoke_fang(monkeypatch, tmp_path):
    from ai.routes import dashboard_routes as d
    gelaufen = []
    # Der Dienst laeuft als root; der Testbenutzer darf /var/lib/api-ai nicht lesen.
    monkeypatch.setattr(d, "_MODELS_JSON", tmp_path / "models.json")
    monkeypatch.setattr(d, "_LAST_RUN", tmp_path / "last_run")

    async def probe(provider, model):
        gelaufen.append(provider)
        return {"provider": provider}

    monkeypatch.setattr(d, "_smoke_probe", probe)
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    return gelaufen


@pytest.mark.parametrize("kopf", [{}, {"X-API-KEY": "falsch"}, {"X-API-KEY": SCHLUESSEL}])
def test_smoketest_laeuft_nicht_ohne_betreiberschluessel(monkeypatch, client, smoke_fang, kopf):
    kunde(monkeypatch)
    assert client.post("/ai/status/smoketest", headers=kopf).status_code == 401
    assert smoke_fang == []


def test_smoketest_mit_betreiberschluessel(monkeypatch, client, smoke_fang):
    kunde(monkeypatch)
    assert client.post("/ai/status/smoketest", headers={"X-API-KEY": "dienst"}).status_code == 200
    assert smoke_fang


def test_smoke_probe_schickt_den_instanzschluessel(monkeypatch):
    from ai.routes import dashboard_routes as d
    koepfe = []

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            koepfe.append((url, headers))
            return httpx.Response(200, text="pong")

    monkeypatch.setattr(d.httpx, "AsyncClient", Client)
    monkeypatch.setenv("API_ACCESS_KEY", SCHLUESSEL)
    asyncio.run(d._smoke_probe("codex", None))
    assert koepfe == [("http://127.0.0.1:8000/ai/chatgpt", {"X-API-KEY": SCHLUESSEL})]
