"""#3607: Auf einer Kundeninstanz startet KEINE Route ohne Schluessel eine CLI
oder einen Netzaufruf an einen Anbieter.

Durchlauf ueber alle Routen der App, nicht ueber eine Liste: Eine neue Route,
die ohne Tor eine CLI oder einen bezahlten Anbieter erreicht, faellt hier auf.
"""
import asyncio
import subprocess

import httpx
import pytest
import requests
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import main
from ai.services import auth_konfig as ak

LOKAL = {
    ak.ISSUER_ENV: "kunde.example.com",
    ak.JWKS_ENV: "http://127.0.0.1:8010/api/v1/auth/.well-known/jwks.json",
    ak.GRANT_ENV: "http://127.0.0.1:8010/api/v1/auth/realtime-grant",
}
SCHLUESSEL = "instanz-schluessel"

# Rumpf, der die Pflichtfelder der meisten Modelle erfuellt, damit ein fehlendes
# Tor nicht hinter einer 422 verschwindet.
RUMPF = {"prompt": "x", "text": "x", "input": "x", "messages": [{"role": "user", "content": "x"}],
         "state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}},
         "model": "x", "voice": "x", "url": "https://example.com/a.png",
         "image_url": "https://example.com/a.png", "target_language": "en"}


def _routen():
    for r in main.app.routes:
        if not isinstance(r, APIRoute):
            continue
        for m in sorted(r.methods - {"HEAD", "OPTIONS"}):
            yield m, r.path


ROUTEN = sorted(set(_routen()))


def _pfad(p: str) -> str:
    # Pfadparameter mit einem harmlosen Wert fuellen.
    import re
    return re.sub(r"\{[^}]+\}", "test-1", p)


def _nur_version(a) -> bool:
    """`<cli> --version` (Health) startet kein Modell und liest keinen Prompt."""
    argv = a[0] if a else None
    if isinstance(argv, (list, tuple)):
        return len(argv) == 2 and argv[1] == "--version"
    return False


@pytest.fixture(params=["mit_schluessel", "ohne_schluessel"])
def instanz(monkeypatch, request):
    for n in (*LOKAL, *ak.ZUSATZ_ENVS, "API_ACCESS_KEY"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr(ak, "_gemeldet", False)
    for k, v in LOKAL.items():
        monkeypatch.setenv(k, v)
    if request.param == "mit_schluessel":
        monkeypatch.setenv("API_ACCESS_KEY", SCHLUESSEL)
    gesehen = []

    def prozess(*a, **k):
        if _nur_version(a):
            raise RuntimeError("Versionsprobe im Test aus")
        gesehen.append(("prozess", repr(a[:1])[:80]))
        raise RuntimeError("Prozessstart verboten")

    async def prozess_async(*a, **k):
        if _nur_version(a):
            raise RuntimeError("Versionsprobe im Test aus")
        gesehen.append(("prozess", repr(a[:1])[:80]))
        raise RuntimeError("Prozessstart verboten")

    # Echte Transporte blockieren, nicht Client.send: Der TestClient laeuft
    # selbst ueber httpx, aber mit eigenem Transport.
    def netz(self, request, *a, **k):
        gesehen.append(("netz", str(request.url)[:80]))
        raise httpx.ConnectError("Netz verboten", request=request)

    async def netz_async(self, request, *a, **k):
        gesehen.append(("netz", str(request.url)[:80]))
        raise httpx.ConnectError("Netz verboten", request=request)

    def netz_requests(self, method, url, *a, **k):
        gesehen.append(("netz", str(url)[:80]))
        raise requests.ConnectionError("Netz verboten")

    monkeypatch.setattr(subprocess, "Popen", prozess)
    monkeypatch.setattr(subprocess, "run", prozess)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", prozess_async)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", prozess_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", netz)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", netz_async)
    monkeypatch.setattr(requests.Session, "request", netz_requests)
    return gesehen


GESCHUETZT = sorted((m, p) for m, p in ROUTEN
                    if p not in main._OFFENE_PFADE
                    and (main._ist_bezahlpfad(p)
                         or any(p == k or p.startswith(k + "/") for k in main._MODELLPFADE)))


@pytest.mark.parametrize("methode,pfad", GESCHUETZT)
def test_geschuetzte_routen_weisen_ab(instanz, request, methode, pfad):
    """Ohne Instanzschluessel 503, mit Schluessel aber ohne Kopf 401, und zwar
    vor dem Handler. Prueft am Status, weil dem Test die Anbieterschluessel
    fehlen: Ein offener Bezahlpfad scheitert hier sonst still im Handler."""
    c = TestClient(main.app, raise_server_exceptions=False)
    kw = {} if methode in ("GET", "DELETE") else {"json": RUMPF}
    r = c.request(methode, _pfad(pfad), **kw)
    ohne = request.node.callspec.params["instanz"] == "ohne_schluessel"
    soll = (503, "api_key_not_configured") if ohne else (401, "api_key_required")
    if pfad.startswith("/ai/openai"):
        soll = (503, "operator_key_not_configured") if ohne else (401, "operator_key_required")
    assert (r.status_code, r.json()["detail"]["error"]) == soll, f"{methode} {pfad}"


def test_kern_bleibt_offen(monkeypatch):
    """Auf unserem Kern (vorgabe, kein Schluessel) aendert #3607 nichts."""
    for n in (*LOKAL, *ak.ZUSATZ_ENVS, "API_ACCESS_KEY"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr(ak, "_gemeldet", False)
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.get("/ai/genimage/models")
    assert r.status_code not in (401, 503) or "api_key" not in r.text


@pytest.mark.parametrize("methode,pfad", ROUTEN)
def test_ohne_schluessel_kein_prozess_und_kein_netz(instanz, methode, pfad):
    c = TestClient(main.app, raise_server_exceptions=False)
    kw = {} if methode in ("GET", "DELETE") else {"json": RUMPF}
    try:
        c.request(methode, _pfad(pfad), **kw)
    except Exception:
        pass
    assert not instanz, f"{methode} {pfad} ohne Schluessel: {instanz}"
