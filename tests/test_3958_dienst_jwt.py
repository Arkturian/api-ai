"""#3958: Dienst-JWT-Weg fuer /ai/transcribe und /ai/chatgpt.

Gegenproben aus AuthAPI, Kommentar 12940: dasselbe Token an /ai/genimage,
/ai/tts und /ai/jev -> 403; ein Token ohne das Recht der Route -> 403;
Agenten-Tokens gehen diesen Weg nicht.
"""
import json
import time
import types

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

import main
from ai.routes import audio_ai_routes as ar
from ai.routes import text_ai_routes as tr
from ai.services import dienst_jwt as dj
from ai.services import jwt_pruefung as jp

ISS = "auth-api.arkturian.com"
SCHLUESSEL = rsa.generate_private_key(public_exponent=65537, key_size=2048)
FREMD = rsa.generate_private_key(public_exponent=65537, key_size=2048)
YT_RECHTE = ["content:posts:create", "ai:transcribe", "ai:chatgpt"]


def _jwks():
    k = RSAAlgorithm.to_jwk(SCHLUESSEL.public_key(), as_dict=True)
    k.update(kid="auth-primary", use="sig", alg="RS256")
    return {"keys": [k]}


def _token(typ="service", sub="service:youtube-api", permissions=None, key=SCHLUESSEL,
           tenant="arkturian", **extra):
    jetzt = int(time.time())
    claims = {"sub": sub, "iss": ISS, "iat": jetzt - 10, "exp": jetzt + 600, "jti": f"j-{sub}",
              "type": typ, "tenant": tenant,
              "permissions": YT_RECHTE if permissions is None else permissions, **extra}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "auth-primary"})


def _kopf(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def netz(monkeypatch, tmp_path):
    zustand = {"down": False}

    def fetch(url, params):
        if zustand["down"]:
            raise OSError("auth-api weg")
        if url.endswith("jwks.json"):
            return _jwks()
        if url.endswith("revoked-agent-jtis"):
            return {"revoked_jtis": [], "as_of": "2026-11-25T09:00:00Z",
                    "min_agent_iat": None, "owner_min_iat": {}}
        raise AssertionError(url)

    echt = jp.Verifier

    class MitAttrappe(echt):
        def __init__(self, **kw):
            super().__init__(fetch=fetch, **kw)

    monkeypatch.setattr(jp, "Verifier", MitAttrappe)
    jp._zuruecksetzen()
    yield zustand
    jp._zuruecksetzen()


@pytest.fixture
def client(netz, monkeypatch, tmp_path):
    # Das Schluesseltor scharf, wie auf einer Instanz mit API_ACCESS_KEY: so zeigt
    # sich, dass das Dienst-JWT den Schluessel ersetzt und das Agenten-JWT nicht.
    monkeypatch.setenv("API_ACCESS_KEY", "instanz-schluessel-nie-ausgeben")
    monkeypatch.delenv("DIENST_TAGESGRENZEN", raising=False)
    monkeypatch.delenv("DIENST_JWT_MANDANTEN", raising=False)
    monkeypatch.setenv("NARRATE_JOBS_DIR", str(tmp_path / "jobs"))
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def whisper(monkeypatch):
    """OpenAI-Attrappe: zaehlt Aufrufe; Dauer fest 600 s."""
    aufrufe = []

    class _Tr:
        async def create(self, **kw):
            aufrufe.append(kw)
            return types.SimpleNamespace(text="hallo", language="german", duration=600.0)

    class _Client:
        def __init__(self, **kw):
            self.audio = types.SimpleNamespace(transcriptions=_Tr())

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ar, "AsyncOpenAI", _Client)
    monkeypatch.setattr(ar, "_normalize_audio_for_openai", lambda data, base: (data, ".mp3"))
    monkeypatch.setattr(ar, "_audio_duration_seconds", lambda data: 600.0)
    return aufrufe


@pytest.fixture
def chatgpt(monkeypatch):
    aufrufe = []

    async def fake(prompt, model, api_key):
        aufrufe.append(prompt)
        return tr.AIResponse(response="ok", model="gpt-5.6-luna")

    monkeypatch.setattr(tr, "_chatgpt_mit_rechten", fake)
    return aufrufe


def _transcribe(client, kopf):
    return client.post("/ai/transcribe", headers=kopf,
                       files={"file": ("a.mp3", b"ID3-nicht-leer", "audio/mpeg")})


# --- Routen und Rechte ------------------------------------------------------------

def test_recht_fuer_routen():
    assert dj.recht_fuer("POST", "/ai/transcribe") == "ai:transcribe"
    assert dj.recht_fuer("POST", "/ai/chatgpt") == "ai:chatgpt"
    assert dj.recht_fuer("GET", "/ai/chatgpt/abc-123") == "ai:chatgpt"
    assert dj.recht_fuer("GET", "/ai/chatgpt/cost-status") is None
    assert dj.recht_fuer("GET", "/ai/chatgpt/a/b") is None
    assert dj.recht_fuer("GET", "/ai/transcribe") is None
    assert dj.recht_fuer("POST", "/ai/chatgptx") is None


def test_transcribe_mit_dienst_jwt_statt_schluessel(client, whisper):
    r = _transcribe(client, _kopf(_token()))
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "hallo" and len(whisper) == 1


def test_ohne_alles_bleibt_das_schluesseltor(client, whisper):
    r = _transcribe(client, {})
    assert r.status_code == 401 and r.json()["detail"]["error"] == "api_key_required"
    assert whisper == []


def test_agenten_jwt_geht_diesen_weg_nicht(client, whisper):
    r = _transcribe(client, _kopf(_token(typ="agent", sub="agent:YouTube")))
    assert r.status_code == 401 and r.json()["detail"]["error"] == "api_key_required"
    assert whisper == []


def test_chatgpt_mit_dienst_jwt(client, chatgpt):
    r = client.post("/ai/chatgpt", headers=_kopf(_token()), json={"prompt": "hi"})
    assert r.status_code == 200, r.text
    assert len(chatgpt) == 1


@pytest.mark.parametrize("pfad", ["/ai/genimage", "/ai/tts", "/ai/jev", "/ai/decisions",
                                  "/ai/claude", "/ai/kette", "/ai/openai/chat"])
def test_dasselbe_token_an_anderen_routen_403(client, pfad):
    r = client.post(pfad, headers=_kopf(_token()), json={"prompt": "x"})
    assert r.status_code == 403, (pfad, r.text)
    assert r.json()["detail"]["error"] == "dienst_route_nicht_freigegeben"


def test_dasselbe_token_an_cost_status_403(client):
    r = client.get("/ai/chatgpt/cost-status", headers=_kopf(_token()))
    assert r.status_code == 403


def test_token_ohne_recht_der_route_403(client, whisper, chatgpt):
    nur_chat = _kopf(_token(permissions=["ai:chatgpt"]))
    r = _transcribe(client, nur_chat)
    assert r.status_code == 403 and r.json()["detail"] == {"error": "dienst_recht_fehlt",
                                                           "required": "ai:transcribe"}
    nur_trans = _kopf(_token(permissions=["ai:transcribe"]))
    r = client.post("/ai/chatgpt", headers=nur_trans, json={"prompt": "hi"})
    assert r.status_code == 403 and r.json()["detail"]["required"] == "ai:chatgpt"
    ohne_ai = _kopf(_token(sub="service:issue-api", permissions=["content:posts:create"]))
    assert _transcribe(client, ohne_ai).status_code == 403
    assert whisper == [] and chatgpt == []


def test_anderer_dienst_ohne_ai_rechte_an_jev_unveraendert(client, netz, monkeypatch):
    """storage-api, tools-api, conversation-api rufen /ai/jev mit Dienst-JWT."""
    seen = []
    monkeypatch.setattr(main, "JSONResponse", main.JSONResponse)
    from ai.routes import jev_routes as jr

    def _pruefen(req):
        seen.append(req)
        raise main.HTTPException(418, "bis hierher")
    monkeypatch.setattr(jr, "_pruefen", _pruefen)
    tok = _token(sub="service:storage-api", permissions=["storage:read"])
    r = client.post("/ai/jev", headers={**_kopf(tok), "X-API-KEY": "instanz-schluessel-nie-ausgeben"},
                    json={"state": "x", "questions": {}})
    assert r.status_code == 418 and len(seen) == 1


def test_fremder_mandant_403(client, whisper):
    r = _transcribe(client, _kopf(_token(tenant="anderer")))
    assert r.status_code == 403 and r.json()["detail"]["error"] == "dienst_mandant_nicht_freigegeben"
    assert whisper == []


def test_gefaelschte_signatur_401(client, whisper):
    r = _transcribe(client, _kopf(_token(key=FREMD)))
    assert r.status_code == 401 and whisper == []


def test_audience_token_ohne_rechte_401(client, whisper):
    r = _transcribe(client, _kopf(_token(aud="ai-api")))
    assert r.status_code == 401 and whisper == []


def test_auth_api_weg_503_nicht_durch(client, netz, whisper):
    netz["down"] = True
    r = _transcribe(client, _kopf(_token()))
    assert r.status_code == 503 and whisper == []


def test_status_abfrage_mit_dienst_jwt_von_aussen(client):
    # TestClient ist nicht loopback: ohne Recht 401, mit Recht weiter bis zur Route.
    assert client.get("/ai/chatgpt/req-unbekannt").status_code == 401
    r = client.get("/ai/chatgpt/req-unbekannt", headers=_kopf(_token()))
    assert r.status_code == 404 and r.json()["detail"]["error"] == "unknown_request_id"
    r = client.get("/ai/chatgpt/req-unbekannt", headers=_kopf(_token(permissions=["ai:transcribe"])))
    assert r.status_code == 403


# --- Tagesgrenzen -----------------------------------------------------------------

def test_transcribe_minuten_grenze_vor_dem_anbieter(client, whisper, monkeypatch):
    monkeypatch.setenv("DIENST_TAGESGRENZEN", json.dumps(
        {"service:youtube-api": {"transcribe_minuten": 15}}))
    assert _transcribe(client, _kopf(_token())).status_code == 200      # 10 min
    r = _transcribe(client, _kopf(_token()))                           # 20 > 15
    assert r.status_code == 429 and r.json()["detail"]["error"] == "dienst_tagesgrenze"
    assert "Retry-After" in r.headers
    assert len(whisper) == 1


def test_chatgpt_anfragen_grenze(client, chatgpt, monkeypatch):
    monkeypatch.setenv("DIENST_TAGESGRENZEN", json.dumps(
        {"service:youtube-api": {"chatgpt_anfragen": 2}}))
    for _ in range(2):
        assert client.post("/ai/chatgpt", headers=_kopf(_token()), json={"prompt": "hi"}).status_code == 200
    r = client.post("/ai/chatgpt", headers=_kopf(_token()), json={"prompt": "hi"})
    assert r.status_code == 429 and len(chatgpt) == 2
    # Statusabfragen zaehlen nicht
    assert client.get("/ai/chatgpt/req-x", headers=_kopf(_token())).status_code == 404


def test_vorgabe_grenzen_und_kaputtes_json(monkeypatch):
    monkeypatch.setenv("DIENST_TAGESGRENZEN", "{kaputt")
    assert dj.grenzen("service:youtube-api") == dj.VORGABE_GRENZEN
    monkeypatch.setenv("DIENST_TAGESGRENZEN", json.dumps({"service:youtube-api": {"transcribe_minuten": True}}))
    assert dj.grenzen("service:youtube-api")["transcribe_minuten"] == dj.VORGABE_GRENZEN["transcribe_minuten"]


def test_ohne_dienst_aufruf_bucht_transcribe_nichts():
    dj.transcribe_buchen(None)       # kein Dienst im Kontext: kein Fehler, nichts gebucht


def test_dienst_ohne_dauer_422():
    marke = dj.AKTUELL.set({"sub": "service:youtube-api", "recht": "ai:transcribe"})
    try:
        with pytest.raises(main.HTTPException) as e:
            dj.transcribe_buchen(None)
        assert e.value.status_code == 422
    finally:
        dj.AKTUELL.reset(marke)
