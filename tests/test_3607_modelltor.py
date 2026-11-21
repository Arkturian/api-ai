"""#3607: Auf einer Kundeninstanz startet niemand ohne Schluessel eine Modell-CLI."""
import asyncio
import itertools

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


@pytest.fixture
def keine_cli(monkeypatch):
    async def verboten(*a, **k):
        raise AssertionError("CLI gestartet")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", verboten)


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
