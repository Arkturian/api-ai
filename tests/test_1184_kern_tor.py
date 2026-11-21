"""#1184/#3607: Die Beobachtung fuer ein Tor auf dem Kern sperrt niemanden."""
import asyncio
import json
import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from ai import auth_guard as ag
from ai.services import auth_konfig, jwt_pruefung, kern_tor as kt

AUTH_ENVS = ("REALTIME_AUTH_ISSUER", "REALTIME_AUTH_JWKS_URL", "REALTIME_AUTH_GRANT_URL",
             "AUTH_API_ME_URL", "AGENTOS_JWT_REVOCATION_URL")
SCHLUESSEL = "instanzschluessel-1234"
TOKEN = "kopf.nutzlast.signatur-geheim"


@pytest.fixture(autouse=True)
def kern(monkeypatch):
    for n in AUTH_ENVS + ("API_ACCESS_KEY", kt.NUR_REALTIME_ENV, "REALTIME_GRANT_SERVICE_KEY",
                          "COST_TRACKER_SHARED_SECRET"):
        monkeypatch.delenv(n, raising=False)


def kunde(monkeypatch):
    monkeypatch.setenv("REALTIME_AUTH_ISSUER", "kunde.example.com")
    monkeypatch.setenv("REALTIME_AUTH_JWKS_URL", "http://127.0.0.1:8010/j")
    monkeypatch.setenv("REALTIME_AUTH_GRANT_URL", "http://127.0.0.1:8010/g")


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def claims(monkeypatch, wert):
    def pruefen(authorization):
        if not authorization:
            return None
        if isinstance(wert, Exception):
            raise wert
        return wert
    monkeypatch.setattr(jwt_pruefung, "pruefen", pruefen)


AGENT = {"type": "agent", "sub": "agent:Comm", "tenant": "arkturian"}
DIENST = {"type": "service", "sub": "service:storage-api", "tenant": "arkturian"}
NUTZER = {"type": "user", "sub": "person@example.com", "tenant": "arkturian"}
DELEGIERT = {"type": "delegated", "sub": "person@example.com", "tenant": "arkturian"}
WAVE = {"type": "agent", "sub": "agent:wave-3d", "tenant": "arkturian"}
FINDER = {"type": "agent", "sub": "agent:productfinder-internal-demo", "tenant": "oneal"}


@pytest.mark.parametrize("c,lokal,erwartet,grund", [
    (AGENT, False, kt.DURCH, "jwt"),
    (DIENST, False, kt.DURCH, "jwt"),
    (NUTZER, False, kt.WUERDE_SPERREN, "typ_nicht_zugelassen"),
    (DELEGIERT, False, kt.WUERDE_SPERREN, "typ_nicht_zugelassen"),
    (NUTZER, True, kt.DURCH, "lokal"),
    (WAVE, False, kt.WUERDE_SPERREN, "nur_realtime"),
    (WAVE, True, kt.WUERDE_SPERREN, "nur_realtime"),
    (FINDER, True, kt.WUERDE_SPERREN, "nur_realtime"),
])
def test_urteil_nach_tokentyp(monkeypatch, c, lokal, erwartet, grund):
    claims(monkeypatch, c)
    e = kt.urteil("", "", f"Bearer {TOKEN}", lokal)
    assert (e["urteil"], e["grund"]) == (erwartet, grund)


def test_personen_stehen_nie_als_sub_im_urteil(monkeypatch):
    for c in (NUTZER, DELEGIERT):
        claims(monkeypatch, c)
        assert kt.urteil("", "", f"Bearer {TOKEN}", False)["sub"] is None


@pytest.mark.parametrize("soll,ist,lokal,erwartet,grund", [
    ("", "", False, kt.WUERDE_SPERREN, "anonym"),
    ("", "", True, kt.DURCH, "lokal"),
    (SCHLUESSEL, SCHLUESSEL, False, kt.DURCH, "schluessel"),
    (SCHLUESSEL, "falsch", False, kt.WUERDE_SPERREN, "schluessel_falsch"),
    (SCHLUESSEL, "äöü", False, kt.WUERDE_SPERREN, "schluessel_falsch"),
    ("", "irgendwas", False, kt.UNKLAR, "schluessel_nicht_pruefbar"),
])
def test_urteil_nach_schluessel(soll, ist, lokal, erwartet, grund):
    e = kt.urteil(soll, ist, None, lokal)
    assert (e["urteil"], e["grund"]) == (erwartet, grund)


def test_falsches_token_wuerde_gesperrt(monkeypatch):
    claims(monkeypatch, HTTPException(status_code=401))
    e = kt.urteil("", "", f"Bearer {TOKEN}", False)
    assert (e["urteil"], e["grund"]) == (kt.WUERDE_SPERREN, "jwt_ungueltig")


def test_ausfall_der_auth_api_ist_kein_sperrtreffer(monkeypatch):
    claims(monkeypatch, HTTPException(status_code=503))
    e = kt.urteil("", "", f"Bearer {TOKEN}", False)
    assert (e["urteil"], e["grund"]) == (kt.UNKLAR, "jwt_nicht_pruefbar")


def test_nur_realtime_liste_aus_der_umgebung(monkeypatch):
    claims(monkeypatch, AGENT)
    monkeypatch.setenv(kt.NUR_REALTIME_ENV, "agent:Comm, agent:x")
    assert kt.urteil("", "", f"Bearer {TOKEN}", False)["grund"] == "nur_realtime"
    claims(monkeypatch, WAVE)
    assert kt.urteil("", "", f"Bearer {TOKEN}", False)["urteil"] == kt.DURCH


@pytest.mark.parametrize("pfad,erwartet", [
    ("/ai/chatgpt", True), ("/ai/claude", True), ("/ai/gemini", True), ("/ai/grok", True),
    ("/ai/gemini/vision", True), ("/ai/jev", True), ("/ai/decisions", True),
    ("/ai/genimage", True), ("/ai/m3", True),
    ("/ai/m3x", False), ("/ai/realtime/token", False), ("/ai/realtime/live", False),
    ("/ai/realtime/live/sdp", False), ("/ai/gemini/gcp-budget-webhook", False),
    ("/health", False), ("/internal/cli-pressure", False),
])
def test_geltungsbereich(pfad, erwartet):
    assert kt.gilt("POST", pfad, main._KERN_TOR_PFADE, main._KERN_TOR_OFFEN) is erwartet


def test_cors_vorabfrage_zaehlt_nicht():
    assert kt.gilt("OPTIONS", "/ai/chatgpt", main._KERN_TOR_PFADE, main._KERN_TOR_OFFEN) is False


def test_kundeninstanz_wird_nicht_beobachtet(monkeypatch):
    kunde(monkeypatch)
    assert auth_konfig.zustand() == auth_konfig.EIGEN
    assert kt.gilt("POST", "/ai/chatgpt", main._KERN_TOR_PFADE, main._KERN_TOR_OFFEN) is False


@pytest.fixture
def fang(monkeypatch):
    gesehen = []
    monkeypatch.setattr(kt, "_starten", gesehen.append)
    return gesehen


@pytest.mark.parametrize("route", ["/ai/chatgpt", "/ai/claude", "/ai/gemini", "/ai/grok"])
def test_sperrt_niemanden(client, fang, route):
    # 422 kommt aus der Validierung: Die Anfrage ist am Tor vorbeigekommen.
    assert client.post(route, json={}).status_code == 422
    assert len(fang) == 1


def test_fehler_in_der_beobachtung_trifft_die_anfrage_nicht(monkeypatch, client):
    def kaputt(anfrage):
        raise RuntimeError("kaputt")
    monkeypatch.setattr(kt, "_starten", kaputt)
    assert client.post("/ai/chatgpt", json={}).status_code == 422


def test_externer_aufruf_gilt_nicht_als_lokal(client, fang):
    client.post("/ai/chatgpt", json={}, headers={"X-Real-IP": "203.0.113.9"})
    assert fang[0]["lokal"] is False
    assert fang[0]["quelle"] == "203.0.113.9"


def test_hostausnahme_oeffnet_keine_betreiberroute(monkeypatch, client):
    monkeypatch.setenv("REALTIME_GRANT_SERVICE_KEY", "dienst")
    assert ag.von_diesem_host(None) is True
    for methode, pfad in (("post", "/ai/status/smoketest"), ("post", "/internal/notify-cli-update"),
                          ("get", "/internal/cli-pressure")):
        assert getattr(client, methode)(pfad).status_code == 401, pfad


def anfrage(**k):
    return {"methode": "POST", "pfad": "/ai/chatgpt", "schluessel": "", "authorization": None,
            "lokal": False, "quelle": "203.0.113.9", "ua": "t", "etikett": None,
            "weitergereicht": False, **k}


def zeilen(pfad):
    return [json.loads(z) for d in pfad.glob("*.jsonl") for z in d.read_text().splitlines()]


def test_protokoll_ohne_token_und_schluessel(monkeypatch, tmp_path):
    monkeypatch.setenv(kt.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("API_ACCESS_KEY", SCHLUESSEL)
    claims(monkeypatch, NUTZER)
    kt._auswerten_sync(anfrage(schluessel="falsch-geheim", authorization=f"Bearer {TOKEN}"))
    roh = "".join(d.read_text() for d in tmp_path.glob("*.jsonl"))
    for geheim in (TOKEN, "falsch-geheim", SCHLUESSEL, "person@example.com"):
        assert geheim not in roh
    z = zeilen(tmp_path)[0]
    assert (z["urteil"], z["typ"], z["hat_bearer"], z["hat_schluessel"]) == (
        kt.WUERDE_SPERREN, "user", True, True)


def test_protokolldatei_nur_fuer_den_dienst(monkeypatch, tmp_path):
    monkeypatch.setenv(kt.LOG_DIR_ENV, str(tmp_path / "neu"))
    kt._auswerten_sync(anfrage())
    datei = next((tmp_path / "neu").glob("*.jsonl"))
    assert oct(datei.stat().st_mode & 0o777) == "0o600"


def test_tagesgrenze(monkeypatch, tmp_path):
    monkeypatch.setenv(kt.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(kt, "_MAX_BYTES_JE_TAG", 10)
    monkeypatch.setattr(kt, "_voll_gemeldet", None)
    for _ in range(3):
        kt._auswerten_sync(anfrage())
    assert len(zeilen(tmp_path)) == 1


@pytest.mark.kern_tor_echt
def test_starten_begrenzt_und_zaehlt(monkeypatch, tmp_path):
    monkeypatch.setenv(kt.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(kt, "_laufend", kt._MAX_GLEICHZEITIG)
    monkeypatch.setattr(kt, "_uebersprungen", 0)

    async def lauf():
        kt._starten(anfrage())
        assert kt._uebersprungen == 1
        kt._laufend = 0
        kt._starten(anfrage())
        await asyncio.gather(*list(kt._aufgaben))

    asyncio.run(lauf())
    assert kt._laufend == 0
    assert zeilen(tmp_path)[0]["uebersprungen"] == 1


@pytest.mark.kern_tor_echt
def test_fehler_im_hintergrund_gibt_den_platz_frei(monkeypatch):
    monkeypatch.setattr(kt, "_laufend", 0)

    def kaputt(a):
        raise RuntimeError("kaputt")
    monkeypatch.setattr(kt, "_auswerten_sync", kaputt)

    async def lauf():
        kt._starten(anfrage())
        await asyncio.gather(*list(kt._aufgaben))

    asyncio.run(lauf())
    assert kt._laufend == 0


# --- Vertrag mit der Direktprobe von Cloud-release: Texte nicht aendern ------

def test_vertrag_operator_key_not_configured(monkeypatch, client):
    kunde(monkeypatch)
    r = client.post("/internal/notify-cli-update", json={})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "operator_key_not_configured"


def test_vertrag_api_key_not_configured(monkeypatch, client):
    kunde(monkeypatch)
    r = client.post("/ai/chatgpt", json={})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "api_key_not_configured"


@pytest.mark.parametrize("name", ["cost", "minimax-cost", "deepseek-cost", "elevenlabs-cost",
                                  "openai-cost", "openai-realtime-cost"])
def test_vertrag_zaehler_text(monkeypatch, client, name):
    kunde(monkeypatch)
    r = client.get(f"/internal/{name}-shared-state")
    assert r.status_code == 503
    assert r.json()["detail"] == "shared-counter master is not configured on this host"
